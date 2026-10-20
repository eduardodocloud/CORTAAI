"""Utilitários de mídia: ffprobe, extração de áudio, leitura de frames, download."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


class MediaError(RuntimeError):
    pass


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, **kwargs)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[-2000:] if isinstance(proc.stderr, bytes) else str(proc.stderr)[-2000:]
        raise MediaError(f"comando falhou ({cmd[0]}): {err}")
    return proc


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool

    @property
    def aspect(self) -> float:
        return self.width / self.height


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    out = run([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,r_frame_rate:stream_tags=rotate:stream_side_data=rotation:format=duration",
        "-of", "json", str(path),
    ]).stdout
    data = json.loads(out)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"{path} não tem trilha de vídeo")
    w, h = int(video["width"]), int(video["height"])
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        rotation = int(sd.get("rotation", 0) or 0)
    rotation = int((video.get("tags") or {}).get("rotate", rotation) or 0)
    if abs(rotation) % 180 == 90:
        w, h = h, w
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate")) or 30.0
    duration = float(data.get("format", {}).get("duration") or 0.0)
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    return VideoInfo(path, w, h, fps, duration, has_audio)


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "0"):
        return 0.0
    num, _, den = rate.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def extract_audio(src: str | Path, dst: str | Path, sample_rate: int = 16000) -> Path:
    dst = Path(dst)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)])
    return dst


def load_audio(src: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Áudio mono float32 direto do ffmpeg (sem arquivo temporário)."""
    out = run(["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def iter_frames(
    src: str | Path,
    fps: float | None = None,
    width: int | None = None,
    start: float | None = None,
    end: float | None = None,
    info: VideoInfo | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Itera (timestamp, frame BGR) decodificando via ffmpeg.

    `fps` amostra a uma taxa fixa; `width` redimensiona mantendo proporção.
    """
    info = info or probe(src)
    out_w = width or info.width
    out_h = int(round(info.height * out_w / info.width / 2)) * 2
    out_w = int(round(out_w / 2)) * 2
    filters = []
    if fps:
        filters.append(f"fps={fps}")
    filters.append(f"scale={out_w}:{out_h}:flags=bilinear")
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{max(0.0, end - (start or 0.0)):.3f}"]
    cmd += ["-an", "-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    step = 1.0 / (fps or info.fps)
    frame_bytes = out_w * out_h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        i = 0
        base = start or 0.0
        while True:
            buf = proc.stdout.read(frame_bytes)  # type: ignore[union-attr]
            if len(buf) < frame_bytes:
                break
            yield base + i * step, np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 3)
            i += 1
    finally:
        proc.kill()
        proc.wait()


def download(url: str, out_dir: str | Path) -> Path:
    """Baixa um vídeo (YouTube, Twitch, Kick...) com yt-dlp, no máximo 1080p."""
    import yt_dlp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():
        mp4 = path.with_suffix(".mp4")
        if mp4.exists():
            return mp4
        raise MediaError(f"download não gerou arquivo: {url}")
    return path


def thumbnail(src: str | Path, dst: str | Path, t: float = 1.0, width: int = 540) -> Path:
    """Miniatura JPEG de um frame (poster para o player)."""
    dst = Path(dst)
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
         "-vf", f"scale={width}:-2", "-q:v", "4", str(dst)])
    return dst
