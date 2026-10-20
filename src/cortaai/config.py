"""Configuração via variáveis de ambiente (todas opcionais)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Settings:
    models_dir: Path = field(default_factory=lambda: _path("CORTAAI_MODELS_DIR", PROJECT_ROOT / "models"))
    fonts_dir: Path = field(default_factory=lambda: _path("CORTAAI_FONTS_DIR", PROJECT_ROOT / "fonts"))
    data_dir: Path = field(default_factory=lambda: _path("CORTAAI_DATA_DIR", PROJECT_ROOT / "data"))
    # "auto" tenta faster-whisper e cai para Parakeet (sherpa-onnx) se não houver.
    asr_backend: str = field(default_factory=lambda: os.environ.get("CORTAAI_ASR", "auto"))
    whisper_model: str = field(default_factory=lambda: os.environ.get("CORTAAI_WHISPER_MODEL", "large-v3-turbo"))
    language: str | None = field(default_factory=lambda: os.environ.get("CORTAAI_LANGUAGE") or None)
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY") or None)
    llm_model: str = field(default_factory=lambda: os.environ.get("CORTAAI_LLM_MODEL", "claude-opus-5-5"))
    threads: int = field(default_factory=lambda: int(os.environ.get("CORTAAI_THREADS", str(os.cpu_count() or 4))))

    @property
    def parakeet_dir(self) -> Path:
        return self.models_dir / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"

    @property
    def face_prototxt(self) -> Path:
        return self.models_dir / "face_deploy.prototxt"

    @property
    def face_caffemodel(self) -> Path:
        return self.models_dir / "face_res10.caffemodel"


def get_settings() -> Settings:
    return Settings()
