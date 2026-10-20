#!/usr/bin/env bash
# CortaAI - instala (na primeira vez) e abre o editor em http://127.0.0.1:8000
set -e
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "Instale o ffmpeg primeiro (Mac: brew install ffmpeg | Ubuntu: sudo apt install ffmpeg)"; exit 1; }
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -e ".[whisper,parakeet,llm]"
  cortaai modelos
else
  . .venv/bin/activate
fi
( sleep 3; (command -v open >/dev/null && open http://127.0.0.1:8000) || (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8000) || true ) &
cortaai servidor --porta 8000
