"""Lista de edição de um corte: trechos mantidos (jump cuts), mapa de tempo e planos com zoom."""

from __future__ import annotations

from dataclasses import dataclass

from ..style.profile import StyleProfile
from ..transcribe import Word


@dataclass
class Segment:
    src_start: float
    src_end: float
    out_start: float

    @property
    def duration(self) -> float:
        return self.src_end - self.src_start

    @property
    def out_end(self) -> float:
        return self.out_start + self.duration


@dataclass
class Shot:
    out_start: float
    out_end: float
    zoom: float  # zoom no início do plano
    zoom_end: float  # zoom no fim (≠ zoom quando há zoom lento)


@dataclass
class EditList:
    segments: list[Segment]
    shots: list[Shot]
    words: list[Word]  # palavras já no tempo de saída

    @property
    def duration(self) -> float:
        return self.segments[-1].out_end if self.segments else 0.0

    def src_time(self, out_t: float) -> float:
        for s in self.segments:
            if out_t < s.out_end or s is self.segments[-1]:
                return s.src_start + max(0.0, out_t - s.out_start)
        return 0.0

    def zoom_at(self, out_t: float) -> float:
        for sh in self.shots:
            if sh.out_start <= out_t < sh.out_end or sh is self.shots[-1]:
                if sh.zoom_end == sh.zoom:
                    return sh.zoom
                f = (out_t - sh.out_start) / max(1e-6, sh.out_end - sh.out_start)
                return sh.zoom + (sh.zoom_end - sh.zoom) * min(1.0, max(0.0, f))
        return 1.0


PAD_BEFORE = 0.06
PAD_AFTER = 0.12
MIN_SEGMENT = 0.25


def keep_segments(words: list[Word], start: float, end: float, profile: StyleProfile) -> list[tuple[float, float]]:
    """Trechos da fonte que ficam no corte. Com jump cuts, remove pausas longas."""
    ws = [w for w in words if w.start >= start - 0.05 and w.end <= end + 0.05]
    if not profile.pacing.jump_cuts or not ws:
        return [(start, end)]
    max_pause = profile.pacing.max_pause_s
    spans: list[list[float]] = []
    for w in ws:
        a, b = max(start, w.start - PAD_BEFORE), min(end, w.end + PAD_AFTER)
        if spans and a - spans[-1][1] <= max_pause:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    # trechos minúsculos viram parte do vizinho (evita "piscadas")
    merged: list[list[float]] = []
    for sp in spans:
        if merged and (sp[1] - sp[0] < MIN_SEGMENT or merged[-1][1] - merged[-1][0] < MIN_SEGMENT):
            merged[-1][1] = sp[1]
        else:
            merged.append(sp)
    return [(a, b) for a, b in merged if b - a > 0.05]


def build_edit(words: list[Word], start: float, end: float, profile: StyleProfile, has_cuts_in_source: bool = False) -> EditList:
    spans = keep_segments(words, start, end, profile)
    segments: list[Segment] = []
    t = 0.0
    for a, b in spans:
        segments.append(Segment(a, b, t))
        t += b - a
    total = t

    # palavras no tempo de saída
    out_words: list[Word] = []
    for w in words:
        for s in segments:
            if s.src_start - 0.01 <= w.start < s.src_end:
                ws = s.out_start + (w.start - s.src_start)
                we = s.out_start + (min(w.end, s.src_end) - s.src_start)
                out_words.append(Word(w.text, ws, max(we, ws + 0.05)))
                break

    shots = _plan_shots(segments, total, profile, has_cuts_in_source)
    return EditList(segments, shots, out_words)


def _plan_shots(segments: list[Segment], total: float, profile: StyleProfile, has_cuts_in_source: bool) -> list[Shot]:
    """Distribui punch-ins para chegar no ritmo (trocas/min) da referência.

    Jump cuts viram troca de zoom (esconde o pulo); se ainda faltar ritmo,
    adiciona trocas periódicas no meio das falas longas.
    """
    scale = profile.pacing.punch_in_scale
    cpm = profile.pacing.cuts_per_min
    if total <= 0:
        return [Shot(0, 0, 1.0, 1.0)]
    if scale <= 1.01 or cpm <= 0:
        return [Shot(0, total, 1.0, 1.04 if profile.pacing.slow_zoom else 1.0)]
    interval = 60.0 / cpm
    min_len = max(0.7, interval * 0.6)
    bounds = [0.0]
    for s in segments[1:]:
        if s.out_start - bounds[-1] >= min_len:
            bounds.append(s.out_start)
    # preenche trechos longos com trocas periódicas
    filled = [0.0]
    for a, b in zip(bounds, bounds[1:] + [total]):
        n_extra = int((b - a) / interval) - 1
        if n_extra > 0:
            step = (b - a) / (n_extra + 1)
            filled.extend(a + step * k for k in range(1, n_extra + 1))
        if b < total:
            filled.append(b)
    filled = sorted(set(round(x, 3) for x in filled))
    shots: list[Shot] = []
    for k, a in enumerate(filled):
        b = filled[k + 1] if k + 1 < len(filled) else total
        if b - a <= 0.01:
            continue
        z = 1.0 if k % 2 == 0 else scale
        z_end = z * 1.04 if profile.pacing.slow_zoom else z
        shots.append(Shot(a, b, z, z_end))
    return shots or [Shot(0, total, 1.0, 1.0)]
