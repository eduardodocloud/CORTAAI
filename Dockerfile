# CortaAI em contêiner: editor web + modelos já embutidos na imagem.
FROM python:3.12-slim

# ffmpeg (com libass para as legendas) e bibliotecas que o OpenCV precisa
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY fonts ./fonts
RUN pip install --no-cache-dir -e ".[parakeet,llm]"

# baixa os modelos durante o build (~700 MB) para o app subir sem espera
RUN cortaai modelos

# Parakeet roda bem em CPU; data em volume persistente quando houver
ENV CORTAAI_ASR=parakeet \
    CORTAAI_DATA_DIR=/data \
    PORT=8000
RUN mkdir -p /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://127.0.0.1:${PORT}/saude || exit 1
CMD ["sh", "-c", "cortaai servidor --host 0.0.0.0 --porta ${PORT}"]
