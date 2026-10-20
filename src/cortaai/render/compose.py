"""Renderiza um corte: reenquadra (rosto/tela dividida/fundo desfocado), aplica zoom e queima legendas.

Os frames passam por Python (OpenCV) e vão direto para um único ffmpeg que
junta o áudio editado e as legendas ASS, então o vídeo é codificado uma vez só.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings, get_settings
from ..media import VideoInfo, iter_frames, run
from ..style.profile import StyleProfile
from ..transcribe import Transcript
from ..vision import VisualAnalysis
from .captions import build_ass
from .edit import EditList, build_edit

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- trilha de rostos

@dataclass
class Track:
    times: np.ndarray
    cx: np.ndarray  # centro x do rosto principal (0-1), NaN sem rosto
    cy: np.ndarray
    fh: np.ndarray  # altura do rosto (0-1)
    two: np.ndarray  # bool: duas pessoas lado a lado
    cx2: np.ndarray  # centro x da segunda pessoa
    cy2: np.ndarray
    fh2: np.ndarray


def build_track(vis: VisualAnalysis) -> Track:
    """Rosto principal com 'aderência': não pula para outro rosto por um frame só."""
    n = len(vis.samples)
    cx, cy, fh = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    two = np.zeros(n, dtype=bool)
    cx2, cy2, fh2 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    prev = None
    cuts = set(round(t, 3) for t in vis.cuts)
    for i, s in enumerate(vis.samples):
        if round(s.t, 3) in cuts:
            prev = None
        faces = [f for f in s.faces if f.h > 0.04]
        if not faces:
            continue
        faces.sort(key=lambda f: -f.area)
        main = faces[0]
        if prev is not None:
            near = min(faces, key=lambda f: abs(f.cx - prev))
            if abs(near.cx - prev) < 0.12 and near.area > 0.5 * main.area:
                main = near
        cx[i], cy[i], fh[i] = main.cx, main.cy, main.h
        prev = main.cx
        others = [f for f in faces if f is not main and f.area > 0.2 * main.area and abs(f.cx - main.cx) > 0.2]
        if others:
            o = others[0]
            two[i] = True
            left, right = (main, o) if main.cx < o.cx else (o, main)
            cx[i], cy[i], fh[i] = left.cx, left.cy, left.h
            cx2[i], cy2[i], fh2[i] = right.cx, right.cy, right.h
    times = np.array([s.t for s in vis.samples])
    return Track(times, cx, cy, fh, two, cx2, cy2, fh2)


def _fill_and_smooth(times: np.ndarray, values: np.ndarray, cuts: list[float], default: float, deadzone: float) -> np.ndarray:
    """Preenche buracos, filtra e aplica zona morta por plano (câmera não fica 'nervosa')."""
    out = values.copy()
    if np.all(np.isnan(out)):
        return np.full_like(out, default)
    # preenche para frente/trás
    idx = np.where(~np.isnan(out))[0]
    out = np.interp(np.arange(len(out)), idx, out[idx])
    bounds = [0] + [int(np.searchsorted(times, c)) for c in cuts] + [len(out)]
    res = out.copy()
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        seg = out[a:b]
        k = min(len(seg), 5)
        if k >= 3:
            pad = np.pad(seg, (k // 2, k // 2), mode="edge")
            seg = np.array([np.median(pad[j:j + k]) for j in range(len(seg))])
        cur = seg[0]
        sm = np.empty_like(seg)
        for j, v in enumerate(seg):
            if abs(v - cur) > deadzone:
                cur = cur + (v - cur) * 0.35
            sm[j] = cur
        res[a:b] = sm
    return res


class Reframer:
    def __init__(self, info: VideoInfo, vis: VisualAnalysis | None, profile: StyleProfile):
        self.info = info
        self.profile = profile
        self.W, self.H = profile.format.width, profile.format.height
        self.layout = profile.format.layout
        self.track = build_track(vis) if vis and vis.samples else None
        if self.track is not None:
            cuts = vis.cuts if vis else []
            t = self.track
            self.sx = _fill_and_smooth(t.times, t.cx, cuts, 0.5, 0.04)
            self.sy = _fill_and_smooth(t.times, t.cy, cuts, 0.4, 0.05)
            self.sx2 = _fill_and_smooth(t.times, t.cx2, cuts, 0.75, 0.04)
            self.sy2 = _fill_and_smooth(t.times, t.cy2, cuts, 0.4, 0.05)
            self.sh = _fill_and_smooth(t.times, t.fh, cuts, 0.3, 0.03)
            self.sh2 = _fill_and_smooth(t.times, t.fh2, cuts, 0.3, 0.03)
            # split só onde há duas pessoas de forma estável (janela de ~2s)
            win = 9
            two = t.two.astype(float)
            pad = np.pad(two, (win // 2, win // 2), mode="edge")
            self.two_stable = np.array([pad[j:j + win].mean() > 0.6 for j in range(len(two))])

    def _at(self, arr: np.ndarray, t: float) -> float:
        return float(np.interp(t, self.track.times, arr))

    def _crop(self, frame: np.ndarray, cx: float, cy: float, out_w: int, out_h: int, zoom: float) -> np.ndarray:
        fh, fw = frame.shape[:2]
        aspect = out_w / out_h
        ch = fh / zoom
        cw = ch * aspect
        if cw > fw:
            cw = fw / zoom
            ch = cw / aspect
        # rosto um pouco acima do centro (regra dos terços)
        x0 = np.clip(cx * fw - cw / 2, 0, fw - cw)
        y0 = np.clip(cy * fh - ch * 0.38, 0, fh - ch)
        x0, y0, cw, ch = int(x0), int(y0), int(cw), int(ch)
        crop = frame[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA if cw > out_w else cv2.INTER_CUBIC)

    def _fit(self, frame: np.ndarray, zoom: float, blur: bool) -> np.ndarray:
        W, H = self.W, self.H
        fh, fw = frame.shape[:2]
        if blur:
            scale = max(W / fw, H / fh)
            small = cv2.resize(frame, (max(1, int(fw * scale / 8)), max(1, int(fh * scale / 8))), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 6)
            bg = cv2.resize(small, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_LINEAR)
            bh, bw = bg.shape[:2]
            y0, x0 = (bh - H) // 2, (bw - W) // 2
            canvas = (bg[y0:y0 + H, x0:x0 + W] * 0.6).astype(np.uint8)
        else:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
        fg_w = int(W * zoom)
        fg_h = int(fh * fg_w / fw)
        fg = cv2.resize(frame, (fg_w, fg_h), interpolation=cv2.INTER_AREA)
        if fg_w > W:
            off = (fg_w - W) // 2
            fg = fg[:, off:off + W]
            fg_w = W
        y = max(0, (H - fg_h) // 2)
        fg = fg[: H - y]
        canvas[y:y + fg.shape[0], (W - fg_w) // 2:(W - fg_w) // 2 + fg_w] = fg
        return canvas

    def render(self, frame: np.ndarray, src_t: float, zoom: float) -> np.ndarray:
        if self.layout in ("fit_blur", "fit_black"):
            return self._fit(frame, zoom, self.layout == "fit_blur")
        if self.track is None:
            return self._crop(frame, 0.5, 0.4, self.W, self.H, zoom)
        cx, cy = self._at(self.sx, src_t), self._at(self.sy, src_t)
        if self.layout == "split":
            j = int(np.clip(np.searchsorted(self.track.times, src_t), 0, len(self.two_stable) - 1))
            if self.two_stable[j]:
                half = self.H // 2
                # cada pessoa ocupa ~40% da altura da sua metade
                z1 = float(np.clip(0.4 / max(0.05, self._at(self.sh, src_t)), 1.0, 3.0)) * zoom
                z2 = float(np.clip(0.4 / max(0.05, self._at(self.sh2, src_t)), 1.0, 3.0)) * zoom
                top = self._crop(frame, cx, cy, self.W, half, z1)
                bot = self._crop(frame, self._at(self.sx2, src_t), self._at(self.sy2, src_t), self.W, self.H - half, z2)
                out = np.vstack([top, bot])
                out[half - 2:half + 2] = 0
                return out
        return self._crop(frame, cx, cy, self.W, self.H, zoom)


# --------------------------------------------------------------------------- render

def _audio_filter(edit: EditList) -> str:
    parts, labels = [], []
    for k, s in enumerate(edit.segments):
        fade = min(0.012, s.duration / 4)
        parts.append(
            f"[0:a]atrim=start={s.src_start:.3f}:end={s.src_end:.3f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:d={fade:.3f},afade=t=out:st={max(0, s.duration - fade):.3f}:d={fade:.3f}[a{k}]"
        )
        labels.append(f"[a{k}]")
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1,loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
    return ";".join(parts)


def render_clip(
    source: Path,
    info: VideoInfo,
    transcript: Transcript,
    start: float,
    end: float,
    profile: StyleProfile,
    out_path: Path,
    vis: VisualAnalysis | None = None,
    title: str | None = None,
    settings: Settings | None = None,
    progress=None,
    energy: np.ndarray | None = None,
) -> dict:
    settings = settings or get_settings()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edit = build_edit(transcript.words, start, end, profile, energy=energy)
    reframer = Reframer(info, vis, profile)
    fps = profile.format.fps
    W, H = profile.format.width, profile.format.height
    decode_w = min(info.width, 1920)

    work = Path(tempfile.mkdtemp(prefix="cortaai-"))
    try:
        (work / "subs.ass").write_text(build_ass(edit.words, edit.duration, profile, title), encoding="utf-8")
        if info.has_audio:
            run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-filter_complex", _audio_filter(edit),
                 "-map", "[aout]", "-ar", "48000", "-ac", "2", str(work / "audio.wav")])
        fonts_dir = settings.fonts_dir.resolve()
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0"]
        if info.has_audio:
            cmd += ["-i", "audio.wav"]
        cmd += ["-vf", f"ass=subs.ass:fontsdir={fonts_dir}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if info.has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [str(out_path.resolve())]
        enc = subprocess.Popen(cmd, cwd=work, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        written = 0
        total_frames = max(1, int(edit.duration * fps))
        try:
            for seg in edit.segments:
                n_frames = int(round(seg.out_end * fps)) - int(round(seg.out_start * fps))
                count = 0
                last = None
                for _, frame in iter_frames(source, fps=fps, width=decode_w, start=seg.src_start, end=seg.src_end + 0.2, info=info):
                    if count >= n_frames:
                        break
                    out_t = seg.out_start + count / fps
                    src_t = seg.src_start + count / fps
                    last = reframer.render(frame, src_t, edit.zoom_at(out_t))
                    enc.stdin.write(last.tobytes())  # type: ignore[union-attr]
                    count += 1
                    written += 1
                # completa com o último frame se o decode entregou menos
                while last is not None and count < n_frames:
                    enc.stdin.write(last.tobytes())  # type: ignore[union-attr]
                    count += 1
                    written += 1
                if progress:
                    progress(written / total_frames)
        finally:
            enc.stdin.close()  # type: ignore[union-attr]
            err = enc.stderr.read().decode(errors="replace")  # type: ignore[union-attr]
            enc.wait()
        if enc.returncode != 0:
            raise RuntimeError(f"ffmpeg falhou ao codificar o corte: {err[-1500:]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    return {
        "path": str(out_path),
        "duracao_s": round(edit.duration, 2),
        "trechos": len(edit.segments),
        "pausas_removidas_s": round((end - start) - edit.duration, 2),
        "planos": len(edit.shots),
        "edicao": edit.stats,
    }
