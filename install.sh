#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[chronorag] upgrading pip and core dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade -r "${ROOT_DIR}/requirements.txt"
python3 -m pip install --upgrade "transformers>=4.39.0" accelerate safetensors sentencepiece huggingface_hub

echo "[chronorag] pre-downloading embedding and reranker models..."
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
targets = {
    "models_bin/phi3-mini-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "models_bin/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "models_bin/bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}

for rel_path, repo_id in targets.items():
    dest = root / rel_path
    if dest.exists():
        print(f"[chronorag] {repo_id} already available at {dest}")
        continue
    print(f"[chronorag] downloading {repo_id} → {dest}")
    snapshot_download(repo_id=repo_id, local_dir=dest, local_dir_use_symlinks=False, revision=None)
print("[chronorag] model assets ready.")
PY

echo "[chronorag] install complete. Activate your venv and run ingestion next."
