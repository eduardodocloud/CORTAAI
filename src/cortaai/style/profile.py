"""StyleProfile: tudo que define "o jeito" de um corte, aprendido da referência ou de um preset."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FormatStyle(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    # crop: rosto preenche a tela | fit_blur: vídeo inteiro com fundo desfocado |
    # fit_black: faixas pretas | split: duas pessoas empilhadas
    layout: Literal["crop", "fit_blur", "fit_black", "split"] = "crop"


class DurationStyle(BaseModel):
    min_s: float = 20.0
    max_s: float = 60.0
    target_s: float = 35.0


class PacingStyle(BaseModel):
    # trocas visuais por minuto (cortes + punch-ins); o renderizador usa punch-ins
    # para imitar o ritmo quando a fonte tem um plano só.
    cuts_per_min: float = 14.0
    punch_in_scale: float = 1.15
    slow_zoom: bool = False
    # corte seco nas pausas (jump cut): pausas maiores que max_pause_s são removidas
    jump_cuts: bool = True
    max_pause_s: float = 0.35
    words_per_sec: float = 2.8  # ritmo de fala observado (informativo)


class CaptionStyle(BaseModel):
    enabled: bool = True
    font: str = "Montserrat Black"
    font_size: float = 0.055  # fração da altura do vídeo
    position_y: float = 0.70  # centro vertical da legenda (0 topo, 1 base)
    words_per_caption: int = 3
    uppercase: bool = True
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFE600"
    outline_color: str = "#000000"
    outline_width: float = 0.05  # espessura do contorno, fração do tamanho da fonte
    shadow: bool = True
    # karaoke: grupo fixo, palavra falada muda de cor | pop: palavra falada cresce
    # | word: uma palavra por vez | static: grupo sem destaque
    animation: Literal["karaoke", "pop", "word", "static"] = "karaoke"
    background_box: bool = False


class TitleStyle(BaseModel):
    """Título/gancho fixo na tela (muito comum nos cortes BR: faixa no topo)."""

    enabled: bool = False
    font: str = "Montserrat Black"
    position_y: float = 0.12
    font_size: float = 0.042
    uppercase: bool = True
    text_color: str = "#000000"
    box_color: str = "#FFFFFF"
    duration_s: float | None = None  # None = o corte inteiro


class ContentStyle(BaseModel):
    """Como escolher os momentos (vai para o prompt do LLM e para o pontuador local)."""

    hook_type: Literal["question", "statement", "number", "story", "controversy", "any"] = "any"
    language: str | None = None
    guidance: str = (
        "Escolha trechos com gancho forte nos primeiros 3 segundos, uma ideia completa "
        "e final com conclusão ou virada. Evite começar no meio de uma frase."
    )


class StyleProfile(BaseModel):
    name: str = "padrao"
    source: str = "preset"
    format: FormatStyle = Field(default_factory=FormatStyle)
    duration: DurationStyle = Field(default_factory=DurationStyle)
    pacing: PacingStyle = Field(default_factory=PacingStyle)
    captions: CaptionStyle = Field(default_factory=CaptionStyle)
    title: TitleStyle = Field(default_factory=TitleStyle)
    content: ContentStyle = Field(default_factory=ContentStyle)
    # evidências e confiança de cada medida (para mostrar ao usuário)
    evidence: dict = Field(default_factory=dict)


PRESETS: dict[str, StyleProfile] = {
    "padrao": StyleProfile(),
    "podcast_br": StyleProfile(
        name="podcast_br",
        title=TitleStyle(enabled=True),
        captions=CaptionStyle(words_per_caption=3, highlight_color="#00FF5A", position_y=0.68),
        pacing=PacingStyle(cuts_per_min=12.0),
    ),
    "hormozi": StyleProfile(
        name="hormozi",
        captions=CaptionStyle(
            font="Anton", words_per_caption=2, highlight_color="#FFE600", animation="pop", position_y=0.62, font_size=0.07
        ),
        pacing=PacingStyle(cuts_per_min=24.0, punch_in_scale=1.2, max_pause_s=0.2),
    ),
    "minimalista": StyleProfile(
        name="minimalista",
        captions=CaptionStyle(
            font="Poppins Black", uppercase=False, words_per_caption=5, animation="static", font_size=0.042, position_y=0.78
        ),
        pacing=PacingStyle(cuts_per_min=4.0, jump_cuts=False, punch_in_scale=1.0),
        format=FormatStyle(layout="fit_blur"),
    ),
}
