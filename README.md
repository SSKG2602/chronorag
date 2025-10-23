# TimeGuard ChronoRAG (Research Scaffold)

ChronoRAG is a research-grade, repo-grade scaffold for building a temporal-safe retrieval augmented generation (RAG) system. It implements a lightweight yet complete ingest → retrieve → answer pipeline with ChronoGuard controls, designed to run on free Colab T4 instances and M-series Macs with ~7 GB GPU VRAM.

## Quickstart

### Local (macOS / WSL)
```bash
bash scripts/setup_dev.sh
bash scripts/run_api.sh
# in a new terminal
python -m cli.chronorag_cli ingest data/sample/docs
python -m cli.chronorag_cli answer --query "Who is the CEO today?" --mode HARD
```
FastAPI serves at http://127.0.0.1:8000 with `/healthz`, `/ingest`, `/retrieve`, `/answer`, `/policy`, `/incident` endpoints.

> Tip: The repository enables `CHRONORAG_LIGHT=1` in `pytest.ini` so tests run with lightweight stubs instead of downloading large models. Disable it (`export CHRONORAG_LIGHT=0`) when you want full model execution.

### Colab
1. Open `notebooks/ChronoRAG_Colab.ipynb`.
2. Run all cells — it installs deps, spins up the API, ingests samples, and issues a smoke `/answer` call.

## Hardware & Models
- Runs on CPU by default; optional GPU acceleration when available.
- Embeddings: `bge-base-en-v1.5` (CPU/GPU via sentence-transformers) with in-repo fallback stub.
- Reranker: `bge-reranker-base` cross-encoder (batch ≤16, CPU-friendly fallback).
- LLM Answerer: defaults to local HuggingFace `microsoft/Phi-3-mini-4k-instruct` with OpenAI-compatible fallback (see `config/models.yaml`).
- Python 3.11 preferred; if you remain on Python 3.9 (e.g., Anaconda base), leave LIGHT mode enabled to avoid incompatible binary wheels.
- Optional LLM Judge reranker is available (light-mode heuristic when stubs active); toggle via `config/models.yaml`.
- ChronoSanity Gate enforces overlap blocks (`chronosanity.overlap_threshold`), returning evidence-only cards when conflicts trigger.

### Switching LLM Backends
1. **Local HuggingFace (default)**: the loader now accepts remote repo IDs. Accept the license for `microsoft/Phi-3-mini-4k-instruct` on Hugging Face, export `HF_HOME` if you need a custom cache path, and the weights will download on first run. On Kaggle P100/T4 runtimes install `bitsandbytes` for optional 4-bit loading and ensure the GPU is selected (`torch.cuda.is_available()`).
2. **OpenAI-compatible**: export `LLM_ENDPOINT` and `LLM_API_KEY` to point at a v1 `/chat/completions` endpoint.
3. **llama.cpp**: drop a GGUF file into `models_bin/gguf/` matching the config path; the loader auto-switches when the file exists.
4. **Ollama**: run `ollama serve` locally; the loader will call `http://localhost:11434` when higher-priority options are unavailable.

> Kaggle GPU tip: enable the T4/P100 accelerator, `pip install bitsandbytes`, run `huggingface-cli login --token $HF_TOKEN`, then launch `python -m app.uvicorn_runner`. The first invocation downloads the Phi-3 Mini weights into the Kaggle working directory.

## Smoke Tests
- `curl http://127.0.0.1:8000/healthz`
- `python -m cli.chronorag_cli retrieve --query "Q2 revenue" --mode INTELLIGENT`
- `pytest -q`
- Admin policy change: `curl -XPOST http://127.0.0.1:8000/policy/apply -H 'Authorization: Bearer chronorag-admin' -H 'Content-Type: application/json' -d '{"policy_version":"v1.1.2","changes":{"chronosanity":{"overlap_threshold":0.7}},"idempotency_key":"demo"}'`

## Design Choices
- **Temporal Pre-mask**: all recall candidates are filtered against the requested time window (HARD) or decayed (INTELLIGENT) before reranking.
- **Monotone Time-safe Fusion**: final passage scores never improve when time compliance worsens, ensuring temporal monotonicity.
- **ChronoCards**: answers return structured attribution cards with windows, authority ladder info, confidence bands, and counterfactual scaffolds.

Refer to `chronorag.md` for the full specification that this scaffold implements.
