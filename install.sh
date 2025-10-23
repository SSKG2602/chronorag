#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[chronorag] upgrading pip and core dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade -r "${ROOT_DIR}/requirements.txt"
python3 -m pip install --upgrade "transformers>=4.39.0" accelerate safetensors sentencepiece huggingface_hub hf_xet

echo "[chronorag] pre-downloading embedding and reranker models..."
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
targets = [
    ("models_bin/bge-base-en-v1.5", "BAAI/bge-base-en-v1.5"),
    ("models_bin/bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3"),
    ("models_bin/qwen-0_5-chat", "Qwen/Qwen1.5-0.5B-Chat"),
]

for rel_path, repo_id in targets:
    dest = root / rel_path
    if dest.exists():
        print(f"[chronorag] {repo_id} already available at {dest}")
        continue
    print(f"[chronorag] downloading {repo_id} → {dest}")
    snapshot_download(repo_id=repo_id, local_dir=dest, local_dir_use_symlinks=False)

qwen_path = root / "models_bin" / "qwen-0_5-chat"
if not qwen_path.exists() or not any(qwen_path.iterdir()):
    qwen_path.mkdir(parents=True, exist_ok=True)
    print("[chronorag] snapshot download failed; attempting wget fallback for Qwen1.5-0.5B-Chat")
    import subprocess
    urls = [
        "https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/resolve/main/model.safetensors",
        "https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/resolve/main/tokenizer.model",
        "https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/resolve/main/config.json",
        "https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/resolve/main/tokenizer.json",
    ]
    for url in urls:
        target = qwen_path / Path(url).name
        if target.exists():
            continue
        subprocess.run(["wget", "-c", url, "-O", str(target)], check=True)
print("[chronorag] model assets ready.")
PY

echo "[chronorag] install complete. Activate your venv and run ingestion next."
