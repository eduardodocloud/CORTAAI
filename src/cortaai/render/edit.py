"""Edição contextual de um corte: pausas, planos e zoom decididos pela fala.

A referência dá só a BASE do estilo (ritmo de trocas, força do punch-in, tamanho das pausas).
Cada corte ajusta isso ao que está sendo dito:
- hesitação no meio da frase é cortada seca; pergunta e punchline ganham respiro;
- frase nova volta ao plano aberto; ênfase (número, palavra forte, exclamação, voz alta)
  puxa punch-in, e ênfase sobre ênfase aperta mais;
- fala rápida pede mais trocas, fala calma menos; trecho longo e calmo ganha zoom lento.
(Mesma lógica do cortaai.html.)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from ..style.profile import StyleProfile
from ..transcribe import SENTENCE_END, Word

FILLER = re.compile(r"^(h?[ãâa]+h*n*|h?[éeê]{2,}|eh+|hu?m+|hmm+|u+h+m*|uh+m*|er+m*)[.,!?…]*$", re.I)
STRONG = set(
    "nunca ninguem segredo verdade mentira absurdo erro erros dinheiro milhao milhoes rico pobre morte morri medo "
    "louco insano incrivel chocante polemica proibido pior melhor maior jamais sempre todo mundo cuidado atencao "
    "problema resultado simples facil dificil odeio amo chorei briga demitido falencia sucesso fracasso deus mano "
    "cara never secret truth lie money million worst best mistake crazy insane".split()
)


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
    zoom: float
    zoom_end: float
    kind: str = "inicio"  # inicio | frase | enfase | emenda | ritmo


@dataclass
class EditList:
    segments: list[Segment]
    shots: list[Shot]
    words: list[Word]  # palavras já no tempo de saída (sem hesitações removidas)
    stats: dict = field(default_factory=dict)

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


def _key(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return re.sub(r"[^a-z0-9%]", "", "".join(c for c in t if not unicodedata.combining(c)))


def _energy_at(energy: np.ndarray | None, w: Word) -> float:
    if energy is None or not len(energy):
        return 0.0
    a, b = int(w.start * 10), max(int(w.start * 10) + 1, int(np.ceil(w.end * 10)))
    seg = energy[a:b]
    return float(seg.mean()) if len(seg) else 0.0


def emphasis_scores(ws: list[Word], energy: np.ndarray | None) -> list[float]:
    """Quanto cada palavra pesa na fala: conteúdo forte, número, exclamação, resposta, volume, alongamento."""
    durs = [(w.end - w.start) / max(1, len(w.text)) for w in ws]
    typical = float(np.median(durs)) if durs else 0.06
    out = []
    for i, w in enumerate(ws):
        t, k = w.text, _key(w.text)
        s = 0.0
        s += 1 if k in STRONG else 0
        s += 1 if re.search(r"\d", t) else 0
        s += 1 if t.endswith("!") else 0
        s += 0.5 if t.endswith("?") else 0
        s += 0.8 if i > 0 and ws[i - 1].text.endswith("?") else 0  # resposta logo depois da pergunta
        e = _energy_at(energy, w)
        s += 1.5 if e > 1.8 else 1.0 if e > 1.0 else 0
        s += 0.5 if len(t) >= 4 and durs[i] > typical * 1.6 else 0  # palavra esticada
        s -= 0.5 if len(k) <= 2 and not re.search(r"\d", t) else 0  # artigos/preposições
        out.append(s)
    return out


def build_edit(
    words: list[Word], start: float, end: float, profile: StyleProfile, energy: np.ndarray | None = None
) -> EditList:
    ref = profile.pacing
    ws = [w for w in words if w.start >= start - 0.05 and w.end <= end + 0.05]
    stats = {"hesitacoes": 0, "pausas_cortadas": 0, "respiros": 0, "enfases": 0}

    # 1) hesitações soltas ("ééé", "hum") saem
    if ref.jump_cuts:
        kept = []
        for i, w in enumerate(ws):
            prev = ws[i - 1] if i else None
            nxt = ws[i + 1] if i + 1 < len(ws) else None
            isolated = (prev is None or w.start - prev.end > 0.12) and (nxt is None or nxt.start - w.end > 0.12)
            if FILLER.match(w.text.strip()) and isolated:
                stats["hesitacoes"] += 1
            else:
                kept.append(w)
        ws = kept
    emph = emphasis_scores(ws, energy)

    # 2) quanto silêncio cada intervalo pode manter depende de onde ele cai na fala
    base = min(0.45, max(0.1, ref.max_pause_s or 0.3))

    def allowed(i: int) -> float:
        t = ws[i].text
        if t.endswith("?"):
            return min(0.65, max(0.3, max(base, 0.45)))  # respiro antes da resposta
        if SENTENCE_END.search(t) and i + 1 < len(emph) and emph[i + 1] >= 2:
            return min(0.55, max(0.25, max(base, 0.38)))  # antes da punchline
        if SENTENCE_END.search(t):
            return base
        if t.endswith((",", ";", ":")):
            return min(base, 0.2)
        return min(base, 0.1)  # hesitação no meio da frase

    spans: list[list[float]] = [[start, end]]
    if ref.jump_cuts and ws:
        spans = []
        a = max(start, ws[0].start - 0.06)
        for i in range(len(ws) - 1):
            gap, al = ws[i + 1].start - ws[i].end, allowed(i)
            if gap > al + 0.04:
                spans.append([a, min(end, ws[i].end + al * 0.6)])
                a = ws[i + 1].start - al * 0.4
                stats["pausas_cortadas"] += 1
                stats["respiros"] += al >= 0.3
            elif al >= 0.3 and gap >= 0.3:
                stats["respiros"] += 1
        spans.append([a, min(end, ws[-1].end + 0.15)])
        merged: list[list[float]] = []
        for sp in spans:
            if merged and (sp[1] - sp[0] < 0.25 or merged[-1][1] - merged[-1][0] < 0.25):
                merged[-1][1] = sp[1]
            else:
                merged.append(sp)
        spans = [s for s in merged if s[1] - s[0] > 0.05]

    segments: list[Segment] = []
    t = 0.0
    for a, b in spans:
        segments.append(Segment(a, b, t))
        t += b - a

    out_words: list[Word] = []
    out_emph: list[float] = []
    for w, e in zip(ws, emph):
        for s in segments:
            if s.src_start - 0.01 <= w.start < s.src_end:
                ws_ = s.out_start + (w.start - s.src_start)
                we = s.out_start + (min(w.end, s.src_end) - s.src_start)
                out_words.append(Word(w.text, ws_, max(we, ws_ + 0.05)))
                out_emph.append(e)
                break

    shots = plan_shots(segments, out_words, out_emph, t, profile, stats)
    stats["trocas_por_min"] = round((len(shots) - 1) / max(1.0, t) * 60, 1)
    return EditList(segments, shots, out_words, stats)


def plan_shots(
    segments: list[Segment], words: list[Word], emph: list[float], total: float, profile: StyleProfile, stats: dict
) -> list[Shot]:
    ref = profile.pacing
    if total <= 0:
        return [Shot(0, 0, 1.0, 1.0)]
    base_zoom = ref.punch_in_scale if ref.punch_in_scale > 1.01 else 1.0
    # ritmo base da referência, acelerado/desacelerado pela fala deste trecho
    wps = len(words) / max(1.0, total)
    speed = min(1.4, max(0.7, wps / max(1.0, ref.words_per_sec or 2.8)))
    target = min(6.0, max(0.9, 60.0 / max(2.0, ref.cuts_per_min or 12) / speed))
    min_gap, max_gap = max(0.7, target * 0.55), max(2.5, target * 1.9)

    cands: list[dict] = []
    sent_start = 0
    for i, w in enumerate(words):
        prev = words[i - 1] if i else None
        if prev is not None and SENTENCE_END.search(prev.text):
            sent_start = i
            lead = max(emph[i], emph[i + 1] if i + 1 < len(emph) else -9)
            if lead >= 1:  # frase que já abre com ênfase entra com punch-in
                cands.append({"t": w.start - 0.04, "score": 2.1 + lead * 0.3, "kind": "enfase", "lead": lead})
            else:  # ideia nova: volta ao aberto
                # (vale mais que uma ênfase colada nela, senão o plano apertado "vaza" para a ideia calma seguinte)
                cands.append({"t": w.start - 0.04, "score": 2.1 + (0.4 if prev.text.endswith("?") else 0), "kind": "frase"})
        elif emph[i] >= 1 and i - sent_start >= 2:
            cands.append({"t": max(0.0, w.start - 0.05), "score": 0.9 + emph[i] * 0.5, "kind": "enfase"})
    for s in segments[1:]:
        cands.append({"t": s.out_start, "score": 0.8, "kind": "emenda"})  # esconde o pulo do jump cut

    chosen: list[dict] = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if c["t"] < 0.6 or c["t"] > total - 0.6:
            continue
        if all(abs(o["t"] - c["t"]) >= min_gap for o in chosen):
            chosen.append(c)
    chosen.sort(key=lambda c: c["t"])

    # nada de plano parado demais: completa com o melhor candidato do trecho (ou o meio dele)
    filled: list[dict] = []
    prev_t = 0.0
    for c in chosen + [{"t": total, "kind": "fim"}]:
        while c["t"] - prev_t > max_gap:
            mids = [x for x in cands if prev_t + min_gap < x["t"] < min(c["t"] - min_gap, prev_t + max_gap)]
            add = max(mids, key=lambda x: x["score"]) if mids else {"t": prev_t + min(max_gap, (c["t"] - prev_t) / 2), "score": 0.5, "kind": "ritmo"}
            filled.append(add)
            prev_t = add["t"]
        if c["kind"] != "fim":
            filled.append(c)
            prev_t = c["t"]

    def zoom_for(c: dict, prev_z: float) -> float:
        if base_zoom <= 1.01:
            return 1.0 if prev_z > 1.0 else 1.05  # referência sem zoom: troca discreta só nas emendas
        punch = 1 + (base_zoom - 1) * min(1.2, max(0.7, 0.7 + 0.2 * (c.get("lead", c["score"]) - 1)))
        tight = min(1.4, 1 + (base_zoom - 1) * 1.4)
        if c["kind"] == "enfase":
            if prev_z <= 1.02:
                return punch
            return punch if prev_z >= tight - 0.03 else tight  # ênfase sobre ênfase aperta mais
        if c["kind"] == "frase":
            return 1.0 if prev_z > 1.02 else 1 + (base_zoom - 1) * 0.45
        return 1.0 if prev_z > 1.02 else 1 + (base_zoom - 1) * 0.5

    first_emph = bool(emph) and emph[0] >= 2.5
    z = 1 + (base_zoom - 1) * 0.8 if first_emph else 1.0
    pts = [{"t": 0.0, "kind": "inicio"}] + filled
    shots: list[Shot] = []
    for k, c in enumerate(pts):
        if k > 0:
            z = zoom_for(c, z)
        a = c["t"]
        b = pts[k + 1]["t"] if k + 1 < len(pts) else total
        if b - a <= 0.01:
            continue
        long_calm = b - a > 3.5 and c["kind"] != "enfase"
        z2 = z * 1.04 if (ref.slow_zoom or long_calm) else z  # trecho longo e calmo: zoom lento
        shots.append(Shot(a, b, z, z2, c["kind"]))
        stats["enfases"] += c["kind"] == "enfase"
    return shots or [Shot(0, total, 1.0, 1.0)]
