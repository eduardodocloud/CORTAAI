#!/usr/bin/env bash
# Baixa os modelos locais usados pelo CortaAI para ./models
#  - detector de rosto SSD ResNet-10 (OpenCV)
#  - Parakeet TDT 0.6B v3 int8 (transcrição multilíngue via sherpa-onnx, ~650 MB)
#  - voz Piper pt_BR (só para gerar mídia de teste)
# O faster-whisper baixa o próprio modelo do HuggingFace na primeira execução.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
cd models

REL=https://github.com/k2-fsa/sherpa-onnx/releases/download

[ -f face_deploy.prototxt ] || curl -fsSL -o face_deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt
[ -f face_res10.caffemodel ] || curl -fsSL -o face_res10.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

if [ ! -d sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 ]; then
  curl -fSL -o parakeet.tar.bz2 "$REL/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
  tar xjf parakeet.tar.bz2 && rm parakeet.tar.bz2
fi

if [ "${WITH_TTS:-0}" = "1" ] && [ ! -d vits-piper-pt_BR-faber-medium ]; then
  curl -fSL -o piper.tar.bz2 "$REL/tts-models/vits-piper-pt_BR-faber-medium.tar.bz2"
  tar xjf piper.tar.bz2 && rm piper.tar.bz2
fi

# imagens de rosto para os testes de integração (não versionadas)
mkdir -p ../tests/assets
[ -f ../tests/assets/face1.jpg ] || curl -fsSL -o ../tests/assets/face1.jpg \
  https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg
if [ ! -f ../tests/assets/face2.jpg ]; then
  curl -fsSL -o /tmp/messi5.jpg https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg
  python -c "import cv2; im=cv2.imread('/tmp/messi5.jpg'); cv2.imwrite('../tests/assets/face2.jpg', cv2.resize(im[40:220,150:330],(360,360),interpolation=cv2.INTER_CUBIC))"
fi
echo "modelos prontos em $(pwd)"
