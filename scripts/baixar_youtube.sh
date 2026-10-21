#!/usr/bin/env bash
# CortaAI - baixa videos do YouTube (e outros sites) para ~/Downloads/CortaAI
# Rode, cole o link, Enter. Depois arraste o arquivo para o cortaai.html.
set -u
OUT="$HOME/Downloads/CortaAI"; mkdir -p "$OUT"
DIR="$(cd "$(dirname "$0")" && pwd)"
if command -v yt-dlp >/dev/null; then YTDLP=yt-dlp
else
  case "$(uname -s)" in Darwin) BIN=yt-dlp_macos ;; *) BIN=yt-dlp_linux ;; esac
  YTDLP="$DIR/yt-dlp"
  if [ ! -x "$YTDLP" ]; then
    echo "Baixando o yt-dlp (só na primeira vez)..."
    curl -fL -o "$YTDLP" "https://github.com/yt-dlp/yt-dlp/releases/latest/download/$BIN" && chmod +x "$YTDLP" || { echo "Não consegui baixar o yt-dlp."; exit 1; }
  fi
fi
if command -v ffmpeg >/dev/null; then FMT='bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b'
else FMT='b[ext=mp4]/b'; echo "Sem ffmpeg: vou baixar em 360p. Para 720p: brew install ffmpeg (Mac) ou sudo apt install ffmpeg."; fi
while true; do
  printf '\nCole o link do vídeo e aperte Enter (vazio para sair): '; read -r URL || exit 0
  [ -z "$URL" ] && exit 0
  if ! "$YTDLP" --no-playlist -f "$FMT" --merge-output-format mp4 -o "$OUT/%(title).80s.%(ext)s" "$URL"; then
    echo "O YouTube pediu login. Tentando com a sessão do seu navegador..."
    ok=0
    for B in chrome firefox safari edge brave; do
      "$YTDLP" --no-playlist --cookies-from-browser "$B" -f "$FMT" --merge-output-format mp4 -o "$OUT/%(title).80s.%(ext)s" "$URL" && { ok=1; break; }
    done
    [ $ok = 1 ] || { echo "Não deu certo. Feche o navegador e tente de novo."; continue; }
  fi
  echo "Vídeo salvo em $OUT — arraste o arquivo para o CortaAI."
  (command -v open >/dev/null && open "$OUT") || (command -v xdg-open >/dev/null && xdg-open "$OUT") || true
done
