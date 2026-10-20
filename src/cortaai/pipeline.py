"""Orquestra: vídeo longo + estilo (referência ou preset) -> N cortes prontos."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from .config import Settings, get_settings
from .media import download, load_audio, probe, thumbnail
from .selection import select_clips
from .selection.candidates import _energy_curve
from .render import render_clip
from .style.profile import PRESETS, StyleProfile
from .transcribe import transcribe
from .vision import analyze_video

log = logging.getLogger(__name__)
Progress = Callable[[str, float], None]


def _noop(_m: str, _f: float) -> None:
    pass


def resolve_source(source: str, work_dir: Path) -> Path:
    if source.startswith(("http://", "https://")):
        return download(source, work_dir / "downloads")
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(source)
    return p


def make_clips(
    source: str | Path,
    out_dir: str | Path,
    profile: StyleProfile | None = None,
    n_clips: int = 5,
    settings: Settings | None = None,
    use_llm: bool = True,
    progress: Progress = _noop,
) -> dict:
    settings = settings or get_settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = settings.data_dir / "cache"
    profile = profile or PRESETS["padrao"]

    progress("preparando vídeo", 0.01)
    src = resolve_source(str(source), settings.data_dir)
    info = probe(src)

    progress("transcrevendo", 0.05)
    transcript = transcribe(src, cache_dir=cache, settings=settings)
    if not transcript.words:
        raise RuntimeError("não encontrei fala no vídeo para escolher cortes")

    progress("escolhendo os melhores momentos", 0.3)
    audio = load_audio(src) if info.has_audio else None
    clips = select_clips(transcript, profile, n_clips, audio=audio, settings=settings, use_llm=use_llm)
    energy = _energy_curve(audio)

    progress("rastreando rostos", 0.4)
    vis = None
    if profile.format.layout in ("crop", "split"):
        vis = analyze_video(src, fps=4.0, width=640, info=info, cache_dir=cache, settings=settings)

    results = []
    for k, clip in enumerate(clips):
        base = 0.5 + 0.5 * k / max(1, len(clips))
        span = 0.5 / max(1, len(clips))
        progress(f"renderizando corte {k + 1}/{len(clips)}", base)
        out = out_dir / f"corte_{k + 1:02d}.mp4"
        meta = render_clip(
            src, info, transcript, clip.start, clip.end, profile, out, vis=vis, title=clip.title, settings=settings, energy=energy,
            progress=lambda f, base=base, span=span, k=k: progress(f"renderizando corte {k + 1}/{len(clips)}", base + span * f),
        )
        poster = thumbnail(out, out.with_suffix(".jpg"), t=min(1.0, clip.duration / 2))
        results.append({**clip.to_json(), **meta, "arquivo": out.name, "poster_arquivo": poster.name})

    manifest = {
        "fonte": str(src),
        "duracao_fonte_s": round(info.duration, 1),
        "transcricao": transcript.backend,
        "perfil": profile.model_dump(),
        "cortes": results,
    }
    (out_dir / "cortes.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    progress("pronto", 1.0)
    return manifest
