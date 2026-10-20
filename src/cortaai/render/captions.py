"""Legendas ASS (libass) a partir do perfil de estilo: karaokê, pop, palavra a palavra, título fixo."""

from __future__ import annotations

from ..style.profile import CaptionStyle, StyleProfile, TitleStyle
from ..transcribe import Word


def ass_color(hex_color: str) -> str:
    """#RRGGBB ou #RRGGBBAA -> &HAABBGGRR (alfa do ASS: 00 opaco, FF transparente)."""
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    a = "00"
    if len(h) == 8:
        a = f"{255 - int(h[6:8], 16):02X}"
    return f"&H{a}{b}{g}{r}".upper()


def _inline_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def _ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def group_words(words: list[Word], n: int, max_gap: float = 0.6) -> list[list[Word]]:
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        if cur and (len(cur) >= n or w.start - cur[-1].end > max_gap):
            groups.append(cur)
            cur = []
        cur.append(w)
        if w.is_sentence_end and len(cur) >= max(1, n // 2):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


FONT_FILES = {
    "Montserrat Black": "Montserrat Black",
    "Montserrat ExtraBold": "Montserrat ExtraBold",
    "Anton": "Anton",
    "Poppins Black": "Poppins Black",
    "Bangers": "Bangers",
}


def build_ass(words: list[Word], duration: float, profile: StyleProfile, title: str | None = None) -> str:
    W, H = profile.format.width, profile.format.height
    cap: CaptionStyle = profile.captions
    ttl: TitleStyle = profile.title
    size = max(12, int(round(cap.font_size * H)))
    outline = round(cap.outline_width * size, 1)
    border_style = 3 if cap.background_box else 1
    back = ass_color("#000000A0") if cap.background_box else ass_color("#00000080")
    t_size = max(12, int(round(ttl.font_size * H)))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{FONT_FILES.get(cap.font, cap.font)},{size},{ass_color(cap.primary_color)},{ass_color(cap.highlight_color)},{ass_color(cap.outline_color)},{back},-1,0,0,0,100,100,0,0,{border_style},{outline},{2 if cap.shadow else 0},5,{int(W*0.08)},{int(W*0.08)},0,1
Style: Title,{FONT_FILES.get(ttl.font, ttl.font)},{t_size},{ass_color(ttl.text_color)},{ass_color(ttl.text_color)},{ass_color(ttl.box_color)},{ass_color(ttl.box_color)},-1,0,0,0,100,100,0,0,3,{max(6, int(t_size*0.35))},0,5,{int(W*0.1)},{int(W*0.1)},0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines: list[str] = []
    x, y = W // 2, int(cap.position_y * H)
    pos = f"\\pos({x},{y})"

    if ttl.enabled and title:
        t_end = duration if ttl.duration_s is None else min(duration, ttl.duration_s)
        txt = title.upper() if ttl.uppercase else title
        lines.append(f"Dialogue: 1,{_ts(0)},{_ts(t_end)},Title,,0,0,0,,{{\\pos({x},{int(ttl.position_y * H)})}}{_esc(txt)}")

    if cap.enabled and words:
        def fmt(w: Word) -> str:
            return _esc(w.text.upper() if cap.uppercase else w.text)

        hl = _inline_color(cap.highlight_color)
        base = _inline_color(cap.primary_color)
        n = 1 if cap.animation == "word" else max(1, cap.words_per_caption)
        groups = group_words(words, n)
        for gi, group in enumerate(groups):
            g_start = group[0].start
            g_end = group[-1].end + 0.15
            if gi + 1 < len(groups):
                g_end = min(g_end, groups[gi + 1][0].start)  # nunca dois grupos na tela ao mesmo tempo
            if cap.animation in ("static",):
                text = " ".join(fmt(w) for w in group)
                lines.append(f"Dialogue: 0,{_ts(g_start)},{_ts(g_end)},Caption,,0,0,0,,{{{pos}}}{text}")
                continue
            if cap.animation == "word":
                w = group[0]
                pop = "\\fscx70\\fscy70\\t(0,90,\\fscx100\\fscy100)"
                lines.append(f"Dialogue: 0,{_ts(w.start)},{_ts(g_end)},Caption,,0,0,0,,{{{pos}{pop}\\c{hl}}}{fmt(w)}")
                continue
            # karaoke/pop: grupo inteiro na tela, palavra falada em destaque
            for k, w in enumerate(group):
                a = g_start if k == 0 else w.start
                b = group[k + 1].start if k + 1 < len(group) else g_end
                parts = []
                for m, other in enumerate(group):
                    if m == k:
                        grow = "\\fscx118\\fscy118" if cap.animation == "pop" else ""
                        parts.append(f"{{\\c{hl}{grow}}}{fmt(other)}{{\\c{base}\\fscx100\\fscy100}}")
                    else:
                        parts.append(fmt(other))
                lines.append(f"Dialogue: 0,{_ts(a)},{_ts(b)},Caption,,0,0,0,,{{{pos}}}{' '.join(parts)}")
    return header + "\n".join(lines) + "\n"
