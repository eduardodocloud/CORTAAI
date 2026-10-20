"""API + editor web.

Fila de 1 trabalhador (render é pesado em CPU). Estado em memória e em JSON no disco,
então a lista de estilos e cortes sobrevive a reinícios.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..media import thumbnail
from ..pipeline import make_clips
from ..style.analyzer import analyze_reference, merge_profiles
from ..style.profile import PRESETS, StyleProfile

log = logging.getLogger(__name__)
settings = get_settings()
DATA = settings.data_dir
STYLES = DATA / "estilos"
JOBS = DATA / "jobs"
for d in (STYLES, JOBS):
    d.mkdir(parents=True, exist_ok=True)

WEB = Path(__file__).resolve().parents[1] / "web"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

app = FastAPI(title="CortaAI")
app.mount("/files", StaticFiles(directory=DATA), name="files")
if settings.fonts_dir.exists():
    app.mount("/fonts", StaticFiles(directory=settings.fonts_dir), name="fonts")

_executor = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _update(path: Path, **fields) -> None:
    with _lock:
        data = _read(path)
        data.update(fields)
        _write(path, data)


def _save_upload(up: UploadFile, dest_dir: Path) -> Path:
    ext = Path(up.filename or "video.mp4").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"formato não suportado: {ext}")
    dest = dest_dir / f"{uuid.uuid4().hex[:8]}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(up.file, f)
    return dest


# --------------------------------------------------------------------------- páginas

@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/presets")
def presets():
    return {k: v.model_dump() for k, v in PRESETS.items()}


# --------------------------------------------------------------------------- estilos

def _analyze_job(style_id: str, videos: list[Path], name: str, use_llm: bool) -> None:
    meta = STYLES / style_id / "estilo.json"
    _update(meta, status="processando")
    try:
        profiles = []
        for i, v in enumerate(videos):
            def prog(msg, frac, i=i):
                _update(meta, progress=(i + frac) / len(videos), message=f"referência {i + 1}/{len(videos)}: {msg}")
            profiles.append(analyze_reference(v, name=name, cache_dir=DATA / "cache", settings=settings, use_llm=use_llm, progress=prog))
        prof = merge_profiles(profiles, name=name)
        _update(meta, status="pronto", progress=1.0, message="estilo aprendido", profile=prof.model_dump())
    except Exception as exc:
        log.exception("análise falhou")
        _update(meta, status="erro", message=str(exc), trace=traceback.format_exc()[-2000:])


@app.post("/api/estilos")
def create_style(arquivos: list[UploadFile] = File(...), nome: str = Form("meu estilo"), usar_ia: bool = Form(True)):
    style_id = uuid.uuid4().hex[:10]
    d = STYLES / style_id
    d.mkdir(parents=True)
    videos = [_save_upload(f, d) for f in arquivos]
    try:
        thumbnail(videos[0], d / "poster.jpg")
    except Exception:
        log.warning("não consegui gerar miniatura da referência")
    _write(d / "estilo.json", {
        "poster": f"/files/estilos/{style_id}/poster.jpg",
        "id": style_id, "nome": nome, "status": "na fila", "progress": 0.0, "message": "aguardando",
        "referencias": [f"/files/estilos/{style_id}/{v.name}" for v in videos], "profile": None,
    })
    _executor.submit(_analyze_job, style_id, videos, nome, usar_ia)
    return {"id": style_id}


@app.get("/api/estilos")
def list_styles():
    out = []
    for meta in sorted(STYLES.glob("*/estilo.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        d = _read(meta)
        out.append({k: d.get(k) for k in ("id", "nome", "status", "progress", "message")})
    return out


@app.get("/api/estilos/{style_id}")
def get_style(style_id: str):
    meta = STYLES / style_id / "estilo.json"
    if not meta.exists():
        raise HTTPException(404, "estilo não encontrado")
    return _read(meta)


@app.put("/api/estilos/{style_id}")
def update_style(style_id: str, profile: dict):
    meta = STYLES / style_id / "estilo.json"
    if not meta.exists():
        raise HTTPException(404, "estilo não encontrado")
    prof = StyleProfile.model_validate(profile)  # valida antes de salvar
    _update(meta, profile=prof.model_dump())
    return {"ok": True}


# --------------------------------------------------------------------------- cortes

def _clip_job(job_id: str, source: str, profile: StyleProfile, n: int, use_llm: bool) -> None:
    d = JOBS / job_id
    meta = d / "job.json"
    _update(meta, status="processando")
    try:
        manifest = make_clips(
            source, d / "cortes", profile, n, settings, use_llm=use_llm,
            progress=lambda msg, frac: _update(meta, progress=frac, message=msg),
        )
        for c in manifest["cortes"]:
            c["url"] = f"/files/jobs/{job_id}/cortes/{c['arquivo']}"
            c["poster"] = f"/files/jobs/{job_id}/cortes/{c['poster_arquivo']}"
        _update(meta, status="pronto", progress=1.0, message="cortes prontos", cortes=manifest["cortes"])
    except Exception as exc:
        log.exception("corte falhou")
        _update(meta, status="erro", message=str(exc), trace=traceback.format_exc()[-2000:])


@app.post("/api/cortes")
def create_job(
    arquivo: UploadFile | None = File(None),
    url: str = Form(""),
    estilo_id: str = Form(""),
    preset: str = Form("padrao"),
    quantidade: int = Form(5),
    usar_ia: bool = Form(True),
):
    if estilo_id:
        st = get_style(estilo_id)
        if not st.get("profile"):
            raise HTTPException(409, "o estilo ainda está sendo analisado")
        profile = StyleProfile.model_validate(st["profile"])
    else:
        if preset not in PRESETS:
            raise HTTPException(400, "preset desconhecido")
        profile = PRESETS[preset]
    job_id = uuid.uuid4().hex[:10]
    d = JOBS / job_id
    d.mkdir(parents=True)
    if arquivo is not None and arquivo.filename:
        source = str(_save_upload(arquivo, d))
    elif url.strip():
        source = url.strip()
    else:
        raise HTTPException(400, "envie um arquivo ou cole um link")
    _write(d / "job.json", {
        "id": job_id, "status": "na fila", "progress": 0.0, "message": "aguardando",
        "estilo": estilo_id or preset, "quantidade": quantidade, "cortes": [],
    })
    _executor.submit(_clip_job, job_id, source, profile, max(1, min(20, quantidade)), usar_ia)
    return {"id": job_id}


@app.get("/api/cortes/{job_id}")
def get_job(job_id: str):
    meta = JOBS / job_id / "job.json"
    if not meta.exists():
        raise HTTPException(404, "trabalho não encontrado")
    return _read(meta)


@app.get("/api/status")
def status():
    return {
        "ia": bool(settings.anthropic_api_key),
        "modelo_ia": settings.llm_model if settings.anthropic_api_key else None,
        "transcricao": settings.asr_backend,
    }
