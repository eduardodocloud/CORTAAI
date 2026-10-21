@echo off
REM CortaAI - baixa videos do YouTube (e outros sites) para Downloads\CortaAI
REM Dois cliques, cole o link, Enter. Depois arraste o arquivo para o cortaai.html.
chcp 65001 >nul
setlocal EnableDelayedExpansion
title CortaAI - baixar video
cd /d "%~dp0"
set "YTDLP=%~dp0yt-dlp.exe"
set "OUT=%USERPROFILE%\Downloads\CortaAI"
if not exist "%OUT%" mkdir "%OUT%"

if not exist "%YTDLP%" (
  echo Baixando o yt-dlp ^(so na primeira vez^)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe' -OutFile '%YTDLP%'"
  if not exist "%YTDLP%" ( echo Nao consegui baixar o yt-dlp. Verifique a internet. & pause & exit /b 1 )
) else (
  "%YTDLP%" -U >nul 2>nul
)

REM com ffmpeg: video ate 720p + audio juntos; sem ffmpeg: arquivo unico (geralmente 360p)
set "FMT=b[ext=mp4]/b"
where ffmpeg >nul 2>nul && set "FMT=bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b"
where ffmpeg >nul 2>nul || (
  echo.
  echo O ffmpeg nao esta instalado: vou baixar em qualidade menor ^(360p^).
  echo Para 720p, instalo agora o ffmpeg e voce abre este arquivo de novo.
  choice /c SN /m "Instalar o ffmpeg agora"
  if !errorlevel! == 1 (
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    echo Pronto. Feche esta janela e abra o baixar_youtube.bat de novo.
    pause & exit /b 0
  )
)

:loop
echo.
set "URL="
set /p "URL=Cole o link do video e aperte Enter (vazio para sair): "
if "!URL!"=="" exit /b 0
"%YTDLP%" --no-playlist -f "!FMT!" --merge-output-format mp4 -o "%OUT%\%%(title).80s.%%(ext)s" "!URL!"
if errorlevel 1 (
  echo.
  echo O YouTube pediu login. Tentando com a sessao do seu navegador...
  for %%B in (edge chrome firefox brave) do (
    "%YTDLP%" --no-playlist --cookies-from-browser %%B -f "!FMT!" --merge-output-format mp4 -o "%OUT%\%%(title).80s.%%(ext)s" "!URL!" && goto ok
  )
  echo Nao deu certo. Feche o navegador e tente de novo ^(o Chrome trava os cookies enquanto aberto^).
  goto loop
)
:ok
echo.
echo Video salvo em %OUT% - arraste o arquivo para o CortaAI.
start "" "%OUT%"
goto loop
