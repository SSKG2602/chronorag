"""Pluggable LLM backend loader supporting local and remote execution modes."""

from typing import Protocol, List, Dict, Optional
import os

import httpx


class LLMBackend(Protocol):
    def generate(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        stop: Optional[list] = None,
    ) -> str:
        ...


class OpenAICompat:
    def __init__(self, endpoint: str, api_key: str, model: str):
        self.endpoint, self.api_key, self.model = endpoint, api_key, model

    def generate(self, messages, max_tokens, temperature, stop=None) -> str:
        """Invoke any OpenAI-compatible /chat/completions endpoint."""
        if not self.endpoint or not self.api_key:
            raise RuntimeError("OpenAI-compatible not configured")
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = httpx.post(
            f"{self.endpoint}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class LlamaCppBackend:
    def __init__(self, gguf_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        from llama_cpp import Llama  # local import

        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(self, messages, max_tokens, temperature, stop=None) -> str:
        out = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or [],
        )
        return out["choices"][0]["message"]["content"]


class OllamaBackend:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model, self.host = model, host

    def generate(self, messages, max_tokens, temperature, stop=None) -> str:
        response = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": stop or [],
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class LocalHFBackend:
    def __init__(self, model_path: str, dtype: str = "bfloat16", device_map: Optional[str] = "auto"):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
        import torch

        if hasattr(torch, dtype):
            torch_dtype = getattr(torch, dtype)
        else:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    def generate(self, messages, max_tokens, temperature, stop=None) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 1e-5),
            "do_sample": temperature > 0,
        }
        outputs = self.model.generate(
            **inputs,
            **gen_kwargs,
            use_cache=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        completion = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
        if stop:
            for token in stop:
                if token in text:
                    text = text.split(token)[0]
        return text


def load_backend(cfg) -> tuple[LLMBackend, str]:
    """Return the first working backend defined by the strategy order."""
    order = cfg["strategy_order"]
    for name in order:
        try:
            if name == "openai_compat":
                entry = cfg["openai_compat"]
                endpoint = os.getenv(entry["endpoint_env"] or "")
                api_key = os.getenv(entry["api_key_env"] or "")
                if endpoint and api_key:
                    return OpenAICompat(endpoint, api_key, entry["model"]), "openai_compat"
            elif name == "local_hf":
                entry = cfg["local_hf"]
                path = entry["model_path"]
                if os.path.exists(path):
                    backend = LocalHFBackend(
                        path,
                        entry.get("dtype", "bfloat16"),
                        entry.get("device_map", "auto"),
                    )
                    return backend, "local_hf"
            elif name == "llama_cpp":
                entry = cfg["llama_cpp"]
                path = entry["gguf_path"]
                if os.path.exists(path):
                    return LlamaCppBackend(path, entry["n_ctx"], entry["n_gpu_layers"]), "llama_cpp"
            elif name == "ollama":
                entry = cfg["ollama"]
                return OllamaBackend(entry["model"], entry["host"]), "ollama"
        except Exception:
            continue
    raise RuntimeError("No LLM backend available")
