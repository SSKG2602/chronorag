#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from app.deps import get_models_cfg
from core.generator.llm_loader import load_backend

cfg = get_models_cfg().get("llm", {})
try:
    backend, name = load_backend(cfg)
except RuntimeError as exc:
    print(f"[chronorag] ERROR: {exc}")
    print("Make sure you've run ./install.sh or set LLM_ENDPOINT/LLM_API_KEY for the fallback backend.")
    raise SystemExit(1)

print(f"[chronorag] LLM backend ready: {name}")
messages = [
    {"role": "system", "content": "You are a concise diagnostic assistant."},
    {"role": "user", "content": "Say hello in one sentence."},
]
sample = backend.generate(messages, max_tokens=32, temperature=0.15, stop=None)
print("[chronorag] sample output:", sample.strip())
PY
