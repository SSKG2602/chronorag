#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "ChronoRAG interactive query mode. Type 'quit' to exit."
while true; do
  read -rp "chronorag> " prompt || break
  if [[ "$prompt" == "quit" ]]; then
    echo "bye!"
    break
  fi
  if [[ -z "$prompt" ]]; then
    continue
  fi
  python -m cli.chronorag_cli answer --query "$prompt" --mode INTELLIGENT --axis valid
done
