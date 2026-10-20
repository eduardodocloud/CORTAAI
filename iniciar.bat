@echo off
REM CortaAI - instala (na primeira vez) e abre o editor em http://127.0.0.1:8000
cd /d "%~dp0"
where ffmpeg >nul 2>nul || (
  echo Instalando ffmpeg...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  echo Feche esta janela e abra o iniciar.bat de novo para o ffmpeg ser reconhecido.
  pause
  exit /b
)
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -e ".[whisper,parakeet,llm]"
  cortaai modelos
) else (
  call .venv\Scripts\activate.bat
)
start "" http://127.0.0.1:8000
cortaai servidor --porta 8000
pause
