#!/usr/bin/env bash
set -e

echo "[*] Creating conda env (if available) or pip fallback"
if command -v conda >/dev/null 2>&1; then
  conda env update -f environment.yml || conda env create -f environment.yml
  echo "Activate with: conda activate chronorag"
else
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
fi

echo "[*] Done."
