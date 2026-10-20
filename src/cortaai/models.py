"""Download dos modelos locais, multiplataforma (substitui o scripts/download_models.sh no Windows)."""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

FACE_PROTO = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL = ("https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/"
              "res10_300x300_ssd_iter_140000.caffemodel")
PARAKEET = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2")


def _fetch(url: str, dest: Path) -> None:
    print(f"baixando {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)


def download_models(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    if not (models_dir / "face_deploy.prototxt").exists():
        _fetch(FACE_PROTO, models_dir / "face_deploy.prototxt")
    if not (models_dir / "face_res10.caffemodel").exists():
        _fetch(FACE_MODEL, models_dir / "face_res10.caffemodel")
    if not (models_dir / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" / "encoder.int8.onnx").exists():
        arc = models_dir / "parakeet.tar.bz2"
        _fetch(PARAKEET, arc)
        print("extraindo ...", flush=True)
        with tarfile.open(arc, "r:bz2") as t:
            try:
                t.extractall(models_dir, filter="data")
            except TypeError:  # Python < 3.10.12 não tem o parâmetro filter
                t.extractall(models_dir)
        arc.unlink()
    print(f"modelos prontos em {models_dir}")
