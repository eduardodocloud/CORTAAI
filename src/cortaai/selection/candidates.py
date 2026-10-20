"""Gera janelas candidatas alinhadas a frases e escolhe os melhores cortes.

O pontuador local funciona sem nenhuma API. Com ANTHROPIC_API_KEY o Claude
escolhe entre os candidatos (e o pontuador local vira plano B).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import Settings, get_settings
from ..style.analyzer import QUESTION_START
from ..style.profile import StyleProfile
from ..transcribe import Sentence, Transcript

log = logging.getLogger(__name__)


@dataclass
class Clip:
    start: float
    end: float
    score: float
    title: str
    hook: str
    reason: str = ""
    first_sentence: int = 0
    last_sentence: int = 0
    scores: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_json(self) -> dict:
        return {**asdict(self), "duration": round(self.duration, 2)}


# palavras que costumam segurar atenção em cortes (PT + EN)
STRONG = {
    "nunca", "ninguem", "segredo", "verdade", "mentira", "absurdo", "erro", "erros", "dinheiro", "milhao",
    "milhoes", "rico", "pobre", "morte", "morri", "medo", "louco", "insano", "incrivel", "chocante", "polemica",
    "proibido", "ninguém", "pior", "melhor", "maior", "jamais", "sempre", "todo", "mundo", "ninguem", "cuidado",
    "atencao", "problema", "resultado", "simples", "facil", "dificil", "odeio", "amo", "chorei", "briga",
    "demitido", "falencia", "sucesso", "fracasso", "deus", "puta", "caralho", "porra", "mano", "cara",
    "never", "secret", "truth", "lie", "money", "million", "worst", "best", "mistake", "crazy", "insane",
}
# começar com pronome que aponta para algo dito antes deixa o corte sem contexto
ANAPHORA = {"isso", "isto", "esse", "essa", "esses", "essas", "aquilo", "ele", "ela", "eles", "elas", "la", "this", "that", "it", "they", "he", "she"}
CONNECTIVES = {"e", "mas", "entao", "ai", "né", "ne", "tipo", "porque", "pois", "que", "and", "but", "so", "because", "or", "ou"}


def _norm(word: str) -> str:
    w = unicodedata.normalize("NFKD", word.lower())
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in w if not unicodedata.combining(c)))


def build_candidates(transcript: Transcript, profile: StyleProfile, max_candidates: int = 4000) -> list[tuple[int, int]]:
    """Pares (frase_inicial, frase_final) cuja duração cabe no perfil."""
    sents = transcript.sentences()
    lo, hi = profile.duration.min_s, profile.duration.max_s
    out: list[tuple[int, int]] = []
    for i, s in enumerate(sents):
        for j in range(i, len(sents)):
            d = sents[j].end - s.start
            if d > hi:
                break
            if d >= lo:
                out.append((i, j))
    if len(out) > max_candidates:
        step = len(out) / max_candidates
        out = [out[int(k * step)] for k in range(max_candidates)]
    return out


def _energy_curve(audio: np.ndarray | None, sr: int = 16000, hop: float = 0.1) -> np.ndarray | None:
    if audio is None or not len(audio):
        return None
    n = int(sr * hop)
    frames = len(audio) // n
    rms = np.sqrt(np.mean(audio[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    return (db - np.median(db)) / (np.std(db) + 1e-6)


def _hook_score(s: Sentence, profile: StyleProfile) -> tuple[float, dict]:
    words = [_norm(w.text) for w in s.words]
    text = s.text
    feats = {
        "pergunta": "?" in text or bool(QUESTION_START.match(text.lower())),
        "numero": bool(re.search(r"\d", text)),
        "palavras_fortes": sum(1 for w in words if w in STRONG),
        "curta": len(words) <= 14,
        "comeca_com_conectivo": bool(words) and words[0] in CONNECTIVES,
        "comeca_sem_contexto": bool(words) and words[0] in ANAPHORA,
        "exclamacao": "!" in text,
    }
    score = 0.0
    score += 1.2 if feats["pergunta"] else 0
    score += 0.8 if feats["numero"] else 0
    score += min(1.5, 0.5 * feats["palavras_fortes"])
    score += 0.4 if feats["curta"] else -0.3
    score += 0.5 if feats["exclamacao"] else 0
    score -= 1.5 if feats["comeca_com_conectivo"] else 0
    score -= 1.2 if feats["comeca_sem_contexto"] else 0
    want = profile.content.hook_type
    if want == "question" and feats["pergunta"]:
        score += 1.0
    elif want == "number" and feats["numero"]:
        score += 1.0
    elif want == "statement" and not feats["pergunta"] and feats["palavras_fortes"]:
        score += 0.6
    return score, feats


def score_window(
    sents: list[Sentence], i: int, j: int, profile: StyleProfile, energy: np.ndarray | None
) -> tuple[float, dict]:
    first, last = sents[i], sents[j]
    start, end = first.start, last.end
    d = end - start
    hook, feats = _hook_score(first, profile)
    window_words = [_norm(w.text) for s in sents[i: j + 1] for w in s.words]
    strong_density = sum(1 for w in window_words if w in STRONG) / max(1, len(window_words))
    ending = 0.6 if last.words[-1].is_sentence_end else -0.6
    if last.words[-1].text.endswith(("?", "!")):
        ending += 0.3
    target = profile.duration.target_s
    dur_fit = -abs(d - target) / max(5.0, target)
    wps = len(window_words) / max(1.0, d)
    pace = -abs(wps - profile.pacing.words_per_sec) / 2.0
    en = 0.0
    if energy is not None:
        a, b = int(start / 0.1), max(int(start / 0.1) + 1, int(end / 0.1))
        seg = energy[a:b]
        hook_seg = energy[a: a + 30]
        if len(seg):
            en = 0.5 * float(np.mean(seg)) + 0.5 * float(np.percentile(seg, 90)) * 0.3
            if len(hook_seg):
                en += 0.3 * float(np.mean(hook_seg))
    parts = {
        "gancho": round(hook * 1.6, 2),
        "palavras_fortes": round(strong_density * 25, 2),
        "final": round(ending, 2),
        "duracao": round(dur_fit * 1.5, 2),
        "ritmo": round(pace, 2),
        "energia": round(en, 2),
    }
    return sum(parts.values()), {**parts, **{f"gancho_{k}": v for k, v in feats.items()}}


def heuristic_select(
    transcript: Transcript, profile: StyleProfile, n: int, audio: np.ndarray | None = None
) -> list[Clip]:
    sents = transcript.sentences()
    cands = build_candidates(transcript, profile)
    if not cands:
        return _fallback_windows(transcript, profile, n)
    energy = _energy_curve(audio)
    scored = []
    for i, j in cands:
        s, parts = score_window(sents, i, j, profile, energy)
        scored.append((s, i, j, parts))
    scored.sort(key=lambda x: -x[0])
    chosen: list[tuple[float, int, int, dict]] = []
    for item in scored:
        _, i, j, _ = item
        if all(j < ci or i > cj for _, ci, cj, _ in chosen):
            chosen.append(item)
        if len(chosen) >= n:
            break
    raw = np.array([c[0] for c in scored])
    lo, hi = float(np.percentile(raw, 5)), float(raw.max())
    clips = []
    for s, i, j, parts in chosen:
        pct = 100 * (s - lo) / max(1e-6, hi - lo)
        clips.append(Clip(
            start=sents[i].start,
            end=sents[j].end,
            score=round(float(np.clip(40 + pct * 0.58, 0, 99)), 1),
            title=_title_from(sents[i].text),
            hook=sents[i].text,
            reason=_explain(parts),
            first_sentence=i,
            last_sentence=j,
            scores=parts,
        ))
    return sorted(clips, key=lambda c: -c.score)


def _fallback_windows(transcript: Transcript, profile: StyleProfile, n: int) -> list[Clip]:
    """Sem frases suficientes (vídeo curto/sem fala): janelas fixas."""
    dur = transcript.duration or (transcript.words[-1].end if transcript.words else 0)
    size = min(profile.duration.target_s, dur)
    clips = []
    t = 0.0
    while t + size <= dur + 0.01 and len(clips) < n:
        clips.append(Clip(t, t + size, 50.0, "Corte", "", "janela fixa (pouca fala)"))
        t += size
    return clips or [Clip(0.0, dur, 50.0, "Corte", "", "vídeo inteiro")]


def _title_from(text: str, limit: int = 42) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _explain(parts: dict) -> str:
    bits = []
    if parts.get("gancho_pergunta"):
        bits.append("abre com pergunta")
    if parts.get("gancho_numero"):
        bits.append("abre com número")
    if parts.get("gancho_palavras_fortes"):
        bits.append("gancho com palavras fortes")
    if parts.get("energia", 0) > 0.3:
        bits.append("fala com energia acima da média")
    if parts.get("final", 0) > 0:
        bits.append("termina com frase completa")
    return ", ".join(bits) or "melhor equilíbrio de gancho, ritmo e duração"


def select_clips(
    transcript: Transcript,
    profile: StyleProfile,
    n: int = 5,
    audio: np.ndarray | None = None,
    settings: Settings | None = None,
    use_llm: bool = True,
) -> list[Clip]:
    settings = settings or get_settings()
    local = heuristic_select(transcript, profile, max(n * 3, 10), audio)
    if use_llm and settings.anthropic_api_key and transcript.sentences():
        try:
            from .llm import llm_select

            clips = llm_select(transcript, profile, n, settings, hints=local)
            if clips:
                return clips[:n]
        except Exception as exc:
            log.warning("seleção com Claude falhou, usando pontuador local: %s", exc)
    return local[:n]
