"""Aprende o estilo de um vídeo de referência e gera um StyleProfile.

Mede, de forma objetiva:
- formato e layout (rosto em tela cheia, fundo desfocado, faixas pretas, tela dividida)
- duração, ritmo de fala, pausas (jump cuts)
- cortes por minuto, punch-ins (zoom seco) e zoom lento
- legendas via OCR: posição, tamanho, palavras por bloco, caixa alta, cores, destaque
- título fixo na tela (gancho no topo)
- tipo de gancho da primeira frase
Opcionalmente o Claude olha alguns frames + transcrição e descreve o critério de escolha.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import Settings, get_settings
from ..media import VideoInfo, iter_frames, probe
from ..transcribe import Transcript, transcribe
from ..vision import VisualAnalysis, analyze_video
from .profile import CaptionStyle, ContentStyle, DurationStyle, FormatStyle, PacingStyle, StyleProfile, TitleStyle

log = logging.getLogger(__name__)
Progress = Callable[[str, float], None]


def _noop(_msg: str, _frac: float) -> None:
    pass


# --------------------------------------------------------------------------- OCR

@dataclass
class TextBox:
    t: float
    text: str
    x: float  # normalizado
    y: float
    w: float
    h: float
    score: float
    frame_idx: int

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


_OCR = None


def _ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR = RapidOCR()
    return _OCR


def run_ocr(path: Path, info: VideoInfo, fps: float = 3.0, width: int = 720) -> tuple[list[TextBox], list[np.ndarray]]:
    """OCR em frames amostrados. Retorna caixas de texto e os frames (para medir cores)."""
    boxes: list[TextBox] = []
    frames: list[np.ndarray] = []
    for idx, (t, frame) in enumerate(iter_frames(path, fps=fps, width=width, info=info)):
        frames.append(frame)
        h, w = frame.shape[:2]
        result, _ = _ocr()(frame)
        for pts, text, score in result or []:
            score = float(score)
            if score < 0.6 or not text.strip():
                continue
            pts = np.array(pts, dtype=np.float32)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            boxes.append(TextBox(t, text.strip(), x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h, score, idx))
    return boxes, frames


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def _matches_speech(box: TextBox, transcript: Transcript | None, window: float = 2.5) -> bool:
    if transcript is None or not transcript.words:
        return True
    spoken = {_norm(w.text) for w in transcript.words_between(box.t - window, box.t + window)}
    tokens = [tok for tok in _norm(box.text).split() if tok]
    if not tokens:
        return False
    hits = sum(1 for tok in tokens if tok in spoken or any(tok in s or s in tok for s in spoken if len(s) > 3))
    return hits / len(tokens) >= 0.5


# --------------------------------------------------------------------------- cores

def _hex(bgr: np.ndarray) -> str:
    b, g, r = (int(round(float(v))) for v in bgr)
    return f"#{r:02X}{g:02X}{b:02X}"


def _box_pixels(frame: np.ndarray, b: TextBox, pad: float = 0.15) -> np.ndarray:
    h, w = frame.shape[:2]
    px, py = b.w * pad, b.h * pad
    x0, y0 = int(max(0, (b.x - px) * w)), int(max(0, (b.y - py) * h))
    x1, y1 = int(min(w, (b.x + b.w + px) * w)), int(min(h, (b.y + b.h + py) * h))
    return frame[y0:y1, x0:x1]


def _dominant_hue_color(bgr: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """Cor mais frequente por matiz (não a média, que mistura cores diferentes)."""
    bins = (hsv[:, 0] // 10).astype(int)
    top = np.bincount(bins, minlength=18).argmax()
    return np.median(bgr[bins == top], axis=0)


def analyze_text_colors(items: list[tuple[np.ndarray, TextBox]]) -> dict:
    """Separa preenchimento, destaque, contorno e caixa de fundo de um texto sobreposto.

    - Texto com contorno: só conta pixels claros/coloridos colados no contorno escuro,
      assim o rosto/cenário atrás da legenda não contamina a cor.
    - Caixa de fundo: a cor que domina o interior da caixa do OCR e é diferente do
      que está nas laterais da tela na mesma altura.
    """
    fills, accents, box_cols, text_on_box = [], [], [], []
    dark_frac, box_votes, strokes = [], [], []
    highlight_frames = 0
    for frame, b in items:
        crop = _box_pixels(frame, b, pad=0.0)
        if crop.size == 0 or min(crop.shape[:2]) < 6:
            continue
        hsv_img = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv = hsv_img.reshape(-1, 3).astype(np.float32)
        bgr = crop.reshape(-1, 3).astype(np.float32)

        # caixa de fundo? a borda da caixa do OCR é fundo; se ela é uniforme e diferente
        # das laterais da tela na mesma altura, há uma caixa atrás do texto
        ring = _box_pixels(frame, b, pad=0.02)
        border = np.concatenate([ring[:2].reshape(-1, 3), ring[-2:].reshape(-1, 3),
                                 ring[:, :2].reshape(-1, 3), ring[:, -2:].reshape(-1, 3)]).astype(np.float32)
        border_col = np.median(border, axis=0)
        uniform = float(np.mean(np.abs(border - border_col) < 30)) > 0.85
        H, W = frame.shape[:2]
        y0, y1 = int(b.y * H), int((b.y + b.h) * H)
        side = np.concatenate([frame[y0:y1, : max(1, int(W * 0.04))], frame[y0:y1, -max(1, int(W * 0.04)):]]).reshape(-1, 3)
        side_col = np.median(side, axis=0) if len(side) else border_col
        is_box = uniform and float(np.abs(border_col - side_col).mean()) > 35
        box_votes.append(is_box)
        if is_box:
            box_cols.append(border_col)
            far = np.abs(bgr - border_col).mean(axis=1) > 80
            if far.sum() > 10:
                text_on_box.append(np.median(bgr[far], axis=0))
            continue

        dark = (hsv_img[:, :, 2] < 50).astype(np.uint8)
        dark_frac.append(float(dark.mean()))
        k = max(3, int(crop.shape[0] * 0.12)) | 1
        near_outline = cv2.dilate(dark, np.ones((k, k), np.uint8)).reshape(-1).astype(bool)
        cand = near_outline & (hsv[:, 2] >= 50)
        if dark.mean() < 0.05:  # sem contorno: usa tudo
            cand = hsv[:, 2] >= 50
        # espessura do contorno: 2x a distância máxima até a borda dentro da faixa escura
        # colada no texto (transformada de distância)
        near_dark = cv2.dilate(dark, np.ones((k, k), np.uint8)).astype(bool)
        bright = ((hsv_img[:, :, 2] > 150) & ~dark.astype(bool) & near_dark).astype(np.uint8)
        ring_dark = dark & cv2.dilate(bright, np.ones((k, k), np.uint8))
        if ring_dark.sum() > 30 and dark.mean() >= 0.05:
            dt = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
            strokes.append(2 * float(np.percentile(dt[ring_dark > 0], 90)) / frame.shape[0])
        white = cand & (hsv[:, 1] < 50) & (hsv[:, 2] > 190)
        colored = cand & (hsv[:, 1] > 120) & (hsv[:, 2] > 150)
        thr = max(25, 0.02 * len(hsv))
        if white.sum() > thr:
            fills.append(np.median(bgr[white], axis=0))
        if colored.sum() > thr:
            accents.append(_dominant_hue_color(bgr[colored], hsv[colored]))
            if white.sum() > thr:
                highlight_frames += 1  # branco + cor na mesma legenda = palavra destacada
    n = max(1, len(items))
    box = sum(box_votes) / n > 0.5
    acc = None
    if accents:
        acc_arr = np.array(accents)
        acc = _hex(_dominant_hue_color(acc_arr, cv2.cvtColor(acc_arr.astype(np.uint8)[None], cv2.COLOR_BGR2HSV)[0].astype(np.float32)))
    return {
        "primary": _hex(np.median(text_on_box, axis=0)) if box and text_on_box else (_hex(np.median(fills, axis=0)) if fills else None),
        "accent": acc,
        "highlight_ratio": highlight_frames / n,
        "outline": bool(float(np.median(dark_frac)) > 0.08) if dark_frac else False,
        "box": bool(box),
        "box_color": _hex(np.median(box_cols, axis=0)) if box_cols else "#FFFFFF",
        "stroke": float(np.median(strokes)) if strokes else 0.0,  # fração da altura do frame
    }


# --------------------------------------------------------------------------- layout

def detect_layout(frames: list[np.ndarray], vis: VisualAnalysis, info: VideoInfo) -> tuple[str, dict]:
    if info.aspect > 0.9:
        return "crop", {"motivo": "referência não é vertical; usando rosto em tela cheia"}
    rows_mean, rows_sharp = [], []
    for f in frames[:: max(1, len(frames) // 12)]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        rows_mean.append(g.mean(axis=1))
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
        rows_sharp.append(lap.mean(axis=1))
    mean = np.median(rows_mean, axis=0)
    sharp = np.median(rows_sharp, axis=0)
    h = len(mean)
    band = int(h * 0.12)
    mid = sharp[int(h * 0.4): int(h * 0.6)].mean() + 1e-6
    top_dark, bot_dark = mean[:band].mean() < 18, mean[-band:].mean() < 18
    top_soft = sharp[:band].mean() / mid < 0.35
    bot_soft = sharp[-band:].mean() / mid < 0.35
    ev = {"brilho_topo": round(float(mean[:band].mean()), 1), "nitidez_topo_rel": round(float(sharp[:band].mean() / mid), 2)}
    if top_dark and bot_dark:
        return "fit_black", ev
    if top_soft and bot_soft:
        return "fit_blur", ev
    # duas pessoas, uma em cada metade, em boa parte do vídeo
    two = sum(
        1 for s in vis.samples
        if len(s.faces) >= 2 and min(f.cy for f in s.faces) < 0.5 < max(f.cy for f in s.faces)
    )
    if vis.samples and two / len(vis.samples) > 0.3:
        ev["amostras_com_2_rostos"] = two
        return "split", ev
    return "crop", ev


# --------------------------------------------------------------------------- principal

def _font_for(char_aspect: float) -> str:
    # largura média por caractere / altura da caixa do OCR (calibrado renderizando com libass):
    # Anton 0.33-0.41, Montserrat Black 0.59-0.79, Poppins Black 0.56-0.67
    if char_aspect < 0.5:
        return "Anton"
    return "Montserrat Black"


# altura da caixa do OCR / Fontsize do ASS, medida com libass para cada fonte
OCR_HEIGHT_RATIO = {"Anton": 0.62, "Montserrat Black": 0.68, "Poppins Black": 0.56}


def _font_size_from_box(box_h: float, font: str) -> float:
    return round(box_h / OCR_HEIGHT_RATIO.get(font, 0.64), 4)


QUESTION_START = re.compile(
    r"^\s*(por que|porque|pq|como|qual|quais|quem|quando|onde|o que|quanto|quantos|voc[eê] sabia|sabia que|"
    r"why|how|what|who|when|where|did you know)\b"
)


def _hook_type(transcript: Transcript | None) -> str:
    if not transcript or not transcript.words:
        return "any"
    first = transcript.sentences()[0].text if transcript.sentences() else ""
    # o ASR às vezes não pontua; palavra interrogativa no início também conta
    if "?" in first or QUESTION_START.match(first.lower()):
        return "question"
    if re.search(r"\d", first):
        return "number"
    if re.search(r"\b(eu|quando|um dia|era uma vez|uma vez)\b", first.lower()):
        return "story"
    return "statement"


HOOK_GUIDANCE = {
    "question": "Comece o corte com uma PERGUNTA provocativa que gere curiosidade.",
    "number": "Comece o corte com um NÚMERO ou dado concreto e surpreendente.",
    "story": "Comece o corte no início de uma HISTÓRIA pessoal, com tensão logo de cara.",
    "statement": "Comece o corte com uma AFIRMAÇÃO forte e direta, sem introdução.",
    "controversy": "Comece o corte com uma opinião polêmica.",
    "any": "Comece o corte com um gancho forte nos primeiros 3 segundos.",
}


def analyze_reference(
    path: str | Path,
    name: str | None = None,
    cache_dir: str | Path | None = None,
    settings: Settings | None = None,
    use_llm: bool = True,
    progress: Progress = _noop,
) -> StyleProfile:
    settings = settings or get_settings()
    path = Path(path)
    info = probe(path)
    ev: dict = {"arquivo": path.name, "resolucao": f"{info.width}x{info.height}", "duracao_s": round(info.duration, 1)}

    # formato de saída: o mais próximo entre 9:16, 4:5 e 1:1
    if info.aspect < 0.7:
        fmt = FormatStyle(width=1080, height=1920)
    elif info.aspect < 0.9:
        fmt = FormatStyle(width=1080, height=1350)
    elif info.aspect < 1.2:
        fmt = FormatStyle(width=1080, height=1080)
    else:
        fmt = FormatStyle(width=1080, height=1920)
        ev["aviso"] = "referência horizontal: o formato de saída continua 9:16"
    fmt.fps = int(round(min(60, max(24, info.fps))))

    d = max(5.0, info.duration)
    duration = DurationStyle(target_s=round(d, 1), min_s=round(max(10.0, d * 0.6), 1), max_s=round(min(180.0, d * 1.4), 1))

    progress("transcrevendo referência", 0.1)
    transcript = None
    if info.has_audio:
        try:
            transcript = transcribe(path, cache_dir=cache_dir, settings=settings)
        except Exception as exc:
            log.warning("transcrição da referência falhou: %s", exc)

    pacing = PacingStyle()
    if transcript and len(transcript.words) > 5:
        speech_time = sum(w.end - w.start for w in transcript.words)
        gaps = np.array([b.start - a.end for a, b in zip(transcript.words, transcript.words[1:])])
        gaps = gaps[gaps >= 0]
        span = transcript.words[-1].end - transcript.words[0].start
        pacing.words_per_sec = round(len(transcript.words) / max(1.0, span), 2)
        p90 = float(np.percentile(gaps, 90)) if len(gaps) else 0.0
        pacing.jump_cuts = p90 < 0.45
        pacing.max_pause_s = round(float(np.clip(np.percentile(gaps, 95) if len(gaps) else 0.35, 0.12, 0.8)), 2)
        ev["fala"] = {
            "palavras_por_s": pacing.words_per_sec,
            "pausa_p90_s": round(p90, 2),
            "fala_ocupa": round(speech_time / max(1.0, span), 2),
        }

    progress("analisando cortes, zooms e rostos", 0.35)
    vis = analyze_video(path, fps=8.0, width=480, detect_faces=True, detect_zoom=True, info=info, cache_dir=cache_dir, settings=settings)
    n_cuts = sum(1 for e in vis.events if e.kind == "cut")
    zooms = [e for e in vis.events if e.kind == "zoom"]
    minutes = d / 60.0
    pacing.cuts_per_min = round((n_cuts + len(zooms)) / minutes, 1)
    if zooms:
        pacing.punch_in_scale = round(float(np.clip(np.median([max(z.scale, 1 / z.scale) for z in zooms]), 1.05, 1.5)), 2)
    pacing.slow_zoom = vis.slow_zoom_ratio > 0.4
    ev["edicao"] = {
        "trocas_de_plano": n_cuts,
        "punch_ins": len(zooms),
        "trocas_por_min": pacing.cuts_per_min,
        "duracao_media_plano_s": round(d / (n_cuts + len(zooms) + 1), 2),
        "planos_com_zoom_lento": round(vis.slow_zoom_ratio, 2),
    }

    progress("lendo legendas (OCR)", 0.6)
    boxes, frames = run_ocr(path, info)
    layout, layout_ev = detect_layout(frames, vis, info)
    fmt.layout = layout  # type: ignore[assignment]
    ev["layout"] = {"tipo": layout, **layout_ev}

    captions, title, cap_ev = _infer_text_styles(boxes, frames, transcript, info)
    ev["legendas"] = cap_ev

    hook = _hook_type(transcript)
    content = ContentStyle(hook_type=hook, language=transcript.language if transcript else None)  # type: ignore[arg-type]
    content.guidance = HOOK_GUIDANCE[hook] + " " + ContentStyle().guidance
    if transcript and transcript.sentences():
        ev["gancho"] = {"tipo": hook, "primeira_frase": transcript.sentences()[0].text[:160]}

    if use_llm and settings.anthropic_api_key:
        progress("Claude analisando o estilo", 0.85)
        try:
            from ..selection.llm import describe_reference

            extra = describe_reference(frames, transcript, settings)
            if extra:
                content.guidance = extra.get("selection_guidance") or content.guidance
                if extra.get("hook_type") in HOOK_GUIDANCE:
                    content.hook_type = extra["hook_type"]
                ev["claude"] = extra
        except Exception as exc:
            log.warning("análise com Claude falhou: %s", exc)

    progress("pronto", 1.0)
    return StyleProfile(
        name=name or path.stem,
        source="referencia",
        format=fmt,
        duration=duration,
        pacing=pacing,
        captions=captions,
        title=title,
        content=content,
        evidence=_plain(ev),
    )


def _infer_text_styles(
    boxes: list[TextBox], frames: list[np.ndarray], transcript: Transcript | None, info: VideoInfo
) -> tuple[CaptionStyle, TitleStyle, dict]:
    n_frames = max(1, len(frames))
    title = TitleStyle(enabled=False)
    ev: dict = {}

    # texto fixo: mesmo texto na mesma altura em boa parte dos frames -> título/gancho
    by_text = Counter((_norm(b.text), round(b.cy, 1)) for b in boxes)
    static_keys = {k for k, c in by_text.items() if c >= max(3, 0.35 * n_frames) and k[0]}
    static = [b for b in boxes if (_norm(b.text), round(b.cy, 1)) in static_keys]
    if static:
        ys = [b.cy for b in static]
        t_aspect = float(np.median([b.w * info.width / max(1, len(b.text)) / (b.h * info.height) for b in static]))
        t_font = _font_for(t_aspect)
        colors = analyze_text_colors([(frames[b.frame_idx], b) for b in static[:20]])
        title = TitleStyle(
            enabled=True,
            font=t_font,
            position_y=round(float(np.median(ys)), 3),
            font_size=_font_size_from_box(float(np.median([b.h for b in static])), t_font),
            uppercase=_upper_ratio([b.text for b in static]) > 0.7,
            text_color=colors["primary"] or ("#000000" if colors["box"] else "#FFFFFF"),
            box_color=colors["box_color"] if colors["box"] else "#00000000",
            duration_s=None if len({b.frame_idx for b in static}) > 0.7 * n_frames else round(
                (max(b.t for b in static) - min(b.t for b in static)) + 0.5, 1),
        )
        ev["titulo"] = {"texto": static[0].text, "posicao_y": title.position_y}

    dynamic = [b for b in boxes if (_norm(b.text), round(b.cy, 1)) not in static_keys]
    caps = [b for b in dynamic if _matches_speech(b, transcript)]
    ev["caixas_ocr"] = len(boxes)
    ev["caixas_legenda"] = len(caps)
    frames_with_caps = len({b.frame_idx for b in caps})
    if frames_with_caps < max(3, 0.15 * n_frames):
        ev["conclusao"] = "sem legendas detectadas"
        return CaptionStyle(enabled=False), title, ev

    # uma legenda pode virar várias caixas (linhas); junta por frame
    per_frame: dict[int, list[TextBox]] = {}
    for b in caps:
        per_frame.setdefault(b.frame_idx, []).append(b)
    words_per = [sum(len(b.text.split()) for b in bs) for bs in per_frame.values()]
    ys = [float(np.mean([b.cy for b in bs])) for bs in per_frame.values()]
    heights = [b.h for b in caps if len(b.text) >= 2]
    char_aspect = float(np.median([b.w * info.width / max(1, len(b.text)) / (b.h * info.height) for b in caps if len(b.text) >= 3] or [0.6]))
    colors = analyze_text_colors([(frames[b.frame_idx], b) for b in caps[:120]])

    wpc = int(np.clip(round(float(np.median(words_per))), 1, 8))
    # "pop": o mesmo texto aparece com alturas bem diferentes (palavra crescendo)
    same_text_h: dict[str, list[float]] = {}
    for b in caps:
        same_text_h.setdefault(_norm(b.text), []).append(b.h)
    pop_votes = [max(v) / max(1e-6, min(v)) > 1.15 for v in same_text_h.values() if len(v) >= 2]
    pop = bool(pop_votes) and sum(pop_votes) / len(pop_votes) > 0.4

    if colors["highlight_ratio"] > 0.25:
        animation = "pop" if pop else "karaoke"
    elif wpc == 1:
        animation = "word"
    elif pop:
        animation = "pop"
    else:
        animation = "static"

    primary = colors["primary"] or "#FFFFFF"
    accent = colors["accent"]
    if colors["highlight_ratio"] <= 0.25 and accent and not colors["primary"]:
        primary = accent  # legenda inteira colorida, sem branco
    font = _font_for(char_aspect)
    caption_size = _font_size_from_box(float(np.median(heights)), font) if heights else 0.055
    caption = CaptionStyle(
        enabled=True,
        font=font,
        font_size=caption_size,
        position_y=round(float(np.median(ys)), 3),
        words_per_caption=wpc,
        uppercase=_upper_ratio([b.text for b in caps]) > 0.7,
        primary_color=primary,
        highlight_color=accent or "#FFE600",
        outline_color="#000000",
        outline_width=round(float(np.clip(colors["stroke"] / max(1e-6, caption_size), 0.02, 0.15)), 3) if colors["outline"] else 0.0,
        animation=animation,  # type: ignore[arg-type]
        background_box=colors["box"],
    )
    ev.update({
        "palavras_por_legenda": wpc,
        "posicao_y": caption.position_y,
        "altura_rel": caption.font_size,
        "proporcao_caractere": round(char_aspect, 2),
        "cor_principal": primary,
        "cor_destaque": accent,
        "frames_com_destaque": round(colors["highlight_ratio"], 2),
        "contorno": colors["outline"],
        "espessura_contorno": caption.outline_width,
        "caixa_fundo": colors["box"],
        "animacao": animation,
    })
    return caption, title, ev


def _plain(obj):
    """Converte tipos numpy em tipos Python (para JSON)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _upper_ratio(texts: list[str]) -> float:
    letters = [c for t in texts for c in t if c.isalpha()]
    return sum(c.isupper() for c in letters) / max(1, len(letters))


def merge_profiles(profiles: list[StyleProfile], name: str = "combinado") -> StyleProfile:
    """Combina várias referências: mediana do numérico, voto da maioria no categórico."""
    if len(profiles) == 1:
        return profiles[0]
    base = profiles[0].model_copy(deep=True)
    base.name, base.source = name, "referencias"

    def med(get):
        return float(np.median([get(p) for p in profiles]))

    def vote(get):
        return Counter(get(p) for p in profiles).most_common(1)[0][0]

    base.duration.target_s = round(med(lambda p: p.duration.target_s), 1)
    base.duration.min_s = round(med(lambda p: p.duration.min_s), 1)
    base.duration.max_s = round(med(lambda p: p.duration.max_s), 1)
    base.pacing.cuts_per_min = round(med(lambda p: p.pacing.cuts_per_min), 1)
    base.pacing.punch_in_scale = round(med(lambda p: p.pacing.punch_in_scale), 2)
    base.pacing.max_pause_s = round(med(lambda p: p.pacing.max_pause_s), 2)
    base.pacing.jump_cuts = vote(lambda p: p.pacing.jump_cuts)
    base.format.layout = vote(lambda p: p.format.layout)
    with_caps = [p for p in profiles if p.captions.enabled] or profiles
    c = base.captions
    c.enabled = vote(lambda p: p.captions.enabled)
    c.position_y = round(float(np.median([p.captions.position_y for p in with_caps])), 3)
    c.font_size = round(float(np.median([p.captions.font_size for p in with_caps])), 4)
    c.words_per_caption = int(round(float(np.median([p.captions.words_per_caption for p in with_caps]))))
    c.animation = Counter(p.captions.animation for p in with_caps).most_common(1)[0][0]
    c.highlight_color = Counter(p.captions.highlight_color for p in with_caps).most_common(1)[0][0]
    base.title.enabled = vote(lambda p: p.title.enabled)
    base.evidence = {"referencias": [p.evidence for p in profiles]}
    return base
