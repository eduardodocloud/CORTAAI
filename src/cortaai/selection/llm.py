"""Claude como editor: escolhe os momentos e descreve o estilo da referência."""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ..config import Settings
from ..style.profile import StyleProfile
from ..transcribe import Transcript
from .candidates import Clip, _title_from

log = logging.getLogger(__name__)


class _ClipChoice(BaseModel):
    first_sentence: int = Field(description="índice [n] da frase onde o corte começa")
    last_sentence: int = Field(description="índice [n] da frase onde o corte termina (inclusive)")
    title: str = Field(description="título curto e chamativo, no idioma do vídeo, até 40 caracteres")
    score: int = Field(description="potencial viral de 0 a 100")
    reason: str = Field(description="por que esse trecho prende a atenção, em uma frase")


class _Selection(BaseModel):
    clips: list[_ClipChoice]


class _ReferenceNotes(BaseModel):
    hook_type: str = Field(description="question | statement | number | story | controversy | any")
    tone: str
    content_type: str
    editing_notes: str
    selection_guidance: str = Field(
        description="instruções concretas (2-4 frases) para escolher trechos de outro vídeo longo no mesmo estilo"
    )


SYSTEM = (
    "Você é um editor de cortes virais para TikTok, Reels e Shorts. Você conhece o que segura "
    "a atenção nos primeiros 3 segundos e o que faz alguém assistir até o fim."
)


def _client(settings: Settings):
    import anthropic

    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _fmt_ts(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:04.1f}"


def llm_select(
    transcript: Transcript, profile: StyleProfile, n: int, settings: Settings, hints: list[Clip] | None = None
) -> list[Clip]:
    sents = transcript.sentences()
    lines = [f"[{s.index}] ({_fmt_ts(s.start)}–{_fmt_ts(s.end)}) {s.text}" for s in sents]
    hint_txt = ""
    if hints:
        hint_txt = "\n\nCandidatos que um pontuador automático gostou (use só como pista):\n" + "\n".join(
            f"- frases {c.first_sentence}–{c.last_sentence}: {c.reason}" for c in hints[: n * 2]
        )
    prompt = (
        f"Transcrição numerada por frase:\n\n" + "\n".join(lines) + hint_txt + "\n\n"
        f"Escolha os {n} melhores cortes, sem sobreposição.\n"
        f"Regras:\n"
        f"- Cada corte deve durar entre {profile.duration.min_s:.0f}s e {profile.duration.max_s:.0f}s "
        f"(ideal ~{profile.duration.target_s:.0f}s), somando as frases escolhidas.\n"
        f"- Comece numa frase que funcione sozinha como gancho, sem depender de contexto anterior.\n"
        f"- Termine com conclusão, punchline ou virada.\n"
        f"- Estilo a imitar (aprendido do vídeo de referência): {profile.content.guidance}\n"
        f"Ordene do maior para o menor potencial viral."
    )
    resp = _client(settings).messages.parse(
        model=settings.llm_model,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
        output_format=_Selection,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        log.warning("Claude não retornou seleção (stop_reason=%s)", resp.stop_reason)
        return []
    out: list[Clip] = []
    used: list[tuple[int, int]] = []
    lo, hi = profile.duration.min_s * 0.8, profile.duration.max_s * 1.2
    for c in resp.parsed_output.clips:
        i, j = c.first_sentence, c.last_sentence
        if not (0 <= i <= j < len(sents)):
            continue
        d = sents[j].end - sents[i].start
        if not (lo <= d <= hi) or any(not (j < a or i > b) for a, b in used):
            continue
        used.append((i, j))
        out.append(Clip(
            start=sents[i].start,
            end=sents[j].end,
            score=float(max(0, min(100, c.score))),
            title=_title_from(c.title),
            hook=sents[i].text,
            reason=c.reason,
            first_sentence=i,
            last_sentence=j,
            scores={"fonte": "claude"},
        ))
    return out


def _jpeg_b64(frame: np.ndarray, width: int = 512) -> str:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.standard_b64encode(buf.tobytes()).decode()


def describe_reference(frames: list[np.ndarray], transcript: Transcript | None, settings: Settings) -> dict | None:
    """Claude olha 6 frames + transcrição da referência e descreve o critério editorial."""
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, num=min(6, len(frames))).astype(int)
    content: list[dict] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg_b64(frames[k])}}
        for k in idx
    ]
    text = transcript.text[:6000] if transcript else "(sem fala)"
    content.append({
        "type": "text",
        "text": (
            "Estes são frames em ordem de um corte viral de referência, e abaixo a transcrição dele.\n\n"
            f"Transcrição: {text}\n\n"
            "Descreva o padrão editorial: tipo de gancho, tom, tipo de conteúdo, observações de edição "
            "e, principalmente, como escolher trechos de OUTRO vídeo longo para ficar igual a este."
        ),
    })
    resp = _client(settings).messages.parse(
        model=settings.llm_model,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": content}],
        output_format=_ReferenceNotes,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        return None
    return resp.parsed_output.model_dump()
