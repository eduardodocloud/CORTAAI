"""Análise visual: rostos, cortes de cena e zooms (punch-ins), numa passada só."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Settings, get_settings
from .media import VideoInfo, iter_frames, probe

log = logging.getLogger(__name__)


@dataclass
class Face:
    # coordenadas normalizadas (0-1) no frame original
    x: float
    y: float
    w: float
    h: float
    conf: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class FrameSample:
    t: float
    faces: list[Face] = field(default_factory=list)
    diff: float = 0.0  # diferença de conteúdo para a amostra anterior
    scale: float = 1.0  # escala estimada em relação à amostra anterior (só com detect_zoom)
    inliers: float = 0.0


@dataclass
class VisualEvent:
    t: float
    kind: str  # "cut" (troca de plano) | "zoom" (punch-in/out no mesmo plano)
    scale: float = 1.0


@dataclass
class VisualAnalysis:
    fps: float
    samples: list[FrameSample]
    events: list[VisualEvent]
    slow_zoom_ratio: float = 0.0  # fração dos planos com zoom lento contínuo

    @property
    def cuts(self) -> list[float]:
        return [e.t for e in self.events if e.kind == "cut"]

    def to_json(self) -> dict:
        return {
            "fps": self.fps,
            "samples": [{**asdict(s), "faces": [asdict(f) for f in s.faces]} for s in self.samples],
            "events": [asdict(e) for e in self.events],
            "slow_zoom_ratio": self.slow_zoom_ratio,
        }

    @classmethod
    def from_json(cls, d: dict) -> "VisualAnalysis":
        return cls(
            fps=d["fps"],
            samples=[FrameSample(**{**s, "faces": [Face(**f) for f in s["faces"]]}) for s in d["samples"]],
            events=[VisualEvent(**e) for e in d["events"]],
            slow_zoom_ratio=d.get("slow_zoom_ratio", 0.0),
        )


class FaceDetector:
    """SSD ResNet-10 do OpenCV (robusto a perfil/iluminação); cai para Haar se faltar o modelo."""

    def __init__(self, settings: Settings | None = None, min_conf: float = 0.55):
        settings = settings or get_settings()
        self.min_conf = min_conf
        self.net = None
        if settings.face_prototxt.exists() and settings.face_caffemodel.exists():
            self.net = cv2.dnn.readNetFromCaffe(str(settings.face_prototxt), str(settings.face_caffemodel))
        else:
            log.warning("modelo SSD de rosto ausente, usando Haar cascade (menos preciso)")
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        if self.net is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
            self.net.setInput(blob)
            det = self.net.forward()[0, 0]
            faces = []
            for row in det:
                conf = float(row[2])
                if conf < self.min_conf:
                    continue
                x1, y1, x2, y2 = np.clip(row[3:7], 0, 1)
                if x2 - x1 < 0.01 or y2 - y1 < 0.01:
                    continue
                faces.append(Face(float(x1), float(y1), float(x2 - x1), float(y2 - y1), conf))
            return faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(max(24, w // 30), max(24, w // 30)))
        return [Face(x / w, y / h, fw / w, fh / h, 0.8) for (x, y, fw, fh) in rects]


def _content_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Diferença média HSV (0-255), mesma ideia do ContentDetector do PySceneDetect."""
    ha = cv2.cvtColor(a, cv2.COLOR_BGR2HSV).astype(np.int16)
    hb = cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype(np.int16)
    return float(np.mean(np.abs(ha - hb)))


_ORB = None


def _fit_scale(pa: np.ndarray, pb: np.ndarray) -> tuple[float, np.ndarray] | None:
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl is None:
        return None
    return float(np.hypot(M[0, 0], M[1, 0])), inl.ravel().astype(bool)


def estimate_scale(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Escala do conteúdo de b em relação a a e fração de pontos que a sustentam.

    Legendas, títulos e logos ficam parados enquanto a imagem dá zoom; então,
    se o melhor modelo for "parado", tiramos esses pontos e procuramos um
    segundo movimento só no conteúdo.
    """
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(1000)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ka, da = _ORB.detectAndCompute(ga, None)
    kb, db = _ORB.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return 1.0, 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 15:
        return 1.0, 0.0
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    first = _fit_scale(pa, pb)
    if first is None:
        return 1.0, 0.0
    scale, inl = first
    if abs(scale - 1.0) < 0.03 and (~inl).sum() >= 15:
        second = _fit_scale(pa[~inl], pb[~inl])
        if second is not None:
            s2, inl2 = second
            if inl2.sum() >= 12 and inl2.mean() >= 0.3 and abs(s2 - 1.0) >= 0.06:
                return s2, float(inl2.sum()) / len(matches)
    return scale, float(inl.sum()) / len(matches)


def face_zoom(a: np.ndarray, b: np.ndarray, fa: Face, fb: Face) -> tuple[float, float] | None:
    """Punch-in no mesmo rosto? Amplia `a` em torno do rosto e procura o resultado em `b`.

    A caixa do detector é imprecisa, então testa escalas perto da razão dos rostos
    e deixa o template deslizar. Retorna (escala, correlação) se for o mesmo conteúdo.
    """
    ratio = fb.h / max(1e-6, fa.h)
    if abs(ratio - 1.0) < 0.06:
        return None
    h, w = a.shape[:2]
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    sx0, sx1 = int(max(0, (fb.x - fb.w * 0.6) * w)), int(min(w, (fb.x + fb.w * 1.6) * w))
    sy0, sy1 = int(max(0, (fb.y - fb.h * 0.6) * h)), int(min(h, (fb.y + fb.h * 1.6) * h))
    search = gb[sy0:sy1, sx0:sx1]
    best = (0.0, 1.0)
    for sc in ratio * np.linspace(0.85, 1.18, 9):
        big = cv2.resize(ga, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_LINEAR)
        cx, cy = fa.cx * w * sc, fa.cy * h * sc
        tw, th = fa.w * w * sc * 0.9, fa.h * h * sc * 0.9
        x0, y0 = int(cx - tw / 2), int(cy - th / 2)
        if x0 < 0 or y0 < 0 or x0 + int(tw) > big.shape[1] or y0 + int(th) > big.shape[0]:
            continue
        tpl = big[y0:y0 + int(th), x0:x0 + int(tw)]
        if tpl.shape[0] < 12 or tpl.shape[1] < 12 or tpl.shape[0] > search.shape[0] or tpl.shape[1] > search.shape[1]:
            continue
        score = float(cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED).max())
        if score > best[0]:
            best = (score, float(sc))
    return (best[1], best[0]) if best[0] > 0.7 and abs(best[1] - 1.0) >= 0.06 else None


def analyze_video(
    path: str | Path,
    fps: float = 4.0,
    width: int = 480,
    detect_faces: bool = True,
    detect_zoom: bool = False,
    info: VideoInfo | None = None,
    cache_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> VisualAnalysis:
    """Amostra frames a `fps` e extrai rostos + cortes (+ zooms se `detect_zoom`)."""
    info = info or probe(path)
    cache_file = None
    if cache_dir:
        from .transcribe import file_hash

        cache_file = Path(cache_dir) / f"visual-{file_hash(path)}-{fps}-{width}-{int(detect_zoom)}.json"
        if cache_file.exists():
            return VisualAnalysis.from_json(json.loads(cache_file.read_text()))

    detector = FaceDetector(settings) if detect_faces else None
    samples: list[FrameSample] = []
    events: list[VisualEvent] = []
    prev = small_prev = None
    for t, frame in iter_frames(path, fps=fps, width=width, info=info):
        small = cv2.resize(frame, (96, int(96 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
        sample = FrameSample(t, detector.detect(frame) if detector else [])
        if small_prev is not None:
            sample.diff = _content_diff(small_prev, small)
        if detect_zoom and prev is not None:
            sample.scale, sample.inliers = estimate_scale(prev, frame)
            if samples and len(samples[-1].faces) == 1 and len(sample.faces) == 1:
                z = face_zoom(prev, frame, samples[-1].faces[0], sample.faces[0])
                if z is not None:
                    sample.scale, sample.inliers = z
        samples.append(sample)
        prev, small_prev = frame, small

    # limiar adaptativo: picos bem acima do movimento típico do vídeo
    diffs = np.array([s.diff for s in samples]) if samples else np.zeros(0)
    if len(diffs) > 2:
        base = float(np.median(diffs[1:]))
        thr = max(22.0, base * 3.5)
        min_gap = max(1, int(round(fps * 0.4)))
        last = -10**9
        for i in range(1, len(samples)):
            local = diffs[max(1, i - 2): i + 3]
            if diffs[i] >= thr and diffs[i] == local.max() and i - last >= min_gap:
                events.append(VisualEvent(samples[i].t, "cut"))
                last = i

    slow_ratio = 0.0
    if detect_zoom and samples:
        events, slow_ratio = _classify_zooms(samples, events)

    va = VisualAnalysis(fps, samples, events, slow_ratio)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(va.to_json()))
    return va


def _classify_zooms(samples: list[FrameSample], events: list[VisualEvent]) -> tuple[list[VisualEvent], float]:
    """Separa trocas de plano de punch-ins e mede zoom lento por plano.

    Um punch-in de editor gera um salto de conteúdo grande, mas os pontos do
    frame continuam casando com uma transformação de escala pura.
    """
    cut_times = {e.t for e in events}
    out: list[VisualEvent] = []
    for s in samples[1:]:
        zoomed = abs(s.scale - 1.0) >= 0.06 and s.inliers >= 0.12
        if s.t in cut_times:
            out.append(VisualEvent(s.t, "zoom", s.scale) if zoomed else VisualEvent(s.t, "cut"))
        elif zoomed and s.diff > 6:
            out.append(VisualEvent(s.t, "zoom", s.scale))

    # zoom lento: produto das escalas passo a passo dentro de cada plano
    boundaries = {e.t for e in out}
    shots, slow, drift, steps = 0, 0, 1.0, 0
    for s in samples[1:] + [None]:
        if s is None or s.t in boundaries:
            if steps >= 6:
                shots += 1
                slow += abs(drift - 1.0) >= 0.04
            drift, steps = 1.0, 0
            continue
        if s.inliers >= 0.1:
            drift *= s.scale
        steps += 1
    return out, (slow / shots if shots else 0.0)
