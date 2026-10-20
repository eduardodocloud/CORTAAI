"""Testes rápidos (sem modelos): edição, legendas, seleção, cores, perfis."""

import cv2
import numpy as np
import pytest

from cortaai.render.captions import ass_color, build_ass, group_words
from cortaai.render.edit import build_edit, keep_segments
from cortaai.selection.candidates import build_candidates, heuristic_select
from cortaai.style.analyzer import TextBox, analyze_text_colors, merge_profiles
from cortaai.style.profile import PRESETS, CaptionStyle, DurationStyle, PacingStyle, StyleProfile
from cortaai.transcribe import Transcript, Word, _merge_tokens


def words_from(text: str, start: float = 0.0, step: float = 0.3, gaps: dict | None = None) -> list[Word]:
    out, t = [], start
    for i, w in enumerate(text.split()):
        t += (gaps or {}).get(i, 0.0)
        out.append(Word(w, t, t + step * 0.9))
        t += step
    return out


# --------------------------------------------------------------------------- transcrição

def test_merge_tokens_joins_subwords_and_space_tokens():
    tokens = [" Você", " sab", "ia", " que", " ", "9", "0", "%", " das"]
    stamps = [0.0, 0.24, 0.4, 0.56, 0.72, 0.88, 1.04, 1.2, 1.52]
    words = _merge_tokens(tokens, stamps, [0.16] * len(tokens), offset=10.0)
    assert [w.text for w in words] == ["Você", "sabia", "que", "90%", "das"]
    assert words[0].start == pytest.approx(10.0)
    assert all(a.end <= b.start + 1e-9 for a, b in zip(words, words[1:]))


def test_sentences_split_on_punctuation_and_long_pause():
    ws = words_from("Oi tudo bem? Hoje eu vou falar", gaps={3: 2.0})
    sents = Transcript(ws).sentences()
    assert [s.text for s in sents] == ["Oi tudo bem?", "Hoje eu vou falar"]


# --------------------------------------------------------------------------- edição

def test_jump_cuts_remove_long_pauses_and_remap_words():
    ws = words_from("um dois três quatro", gaps={2: 1.5})
    prof = StyleProfile(pacing=PacingStyle(jump_cuts=True, max_pause_s=0.3))
    spans = keep_segments(ws, 0.0, ws[-1].end, prof)
    assert len(spans) == 2
    edit = build_edit(ws, 0.0, ws[-1].end, prof)
    assert edit.duration < ws[-1].end - 1.0  # a pausa de 1,5s sumiu (com folgas)
    assert [w.text for w in edit.words] == ["um", "dois", "três", "quatro"]
    assert all(b.start >= a.start for a, b in zip(edit.words, edit.words[1:]))
    # tempo de saída volta para o tempo da fonte
    assert edit.src_time(edit.words[2].start) == pytest.approx(ws[2].start, abs=1e-6)


def test_no_jump_cuts_keeps_everything():
    ws = words_from("um dois três quatro", gaps={2: 1.5})
    prof = StyleProfile(pacing=PacingStyle(jump_cuts=False))
    assert keep_segments(ws, 0.0, 5.0, prof) == [(0.0, 5.0)]


def test_shots_follow_reference_rhythm():
    ws = words_from(" ".join(["palavra"] * 100), step=0.3)  # 30s de fala contínua
    prof = StyleProfile(pacing=PacingStyle(cuts_per_min=20, punch_in_scale=1.2, jump_cuts=False))
    edit = build_edit(ws, 0.0, 30.0, prof)
    assert 8 <= len(edit.shots) <= 12  # ~10 trocas em 30s a 20/min
    zooms = {round(s.zoom, 2) for s in edit.shots}
    assert zooms == {1.0, 1.2}


# --------------------------------------------------------------------------- legendas

def test_ass_color_conversion():
    assert ass_color("#FFE600") == "&H0000E6FF"
    assert ass_color("#00000080") == "&H7F000000"


def test_caption_groups_never_overlap():
    ws = words_from("um dois três quatro cinco seis sete", step=0.25)
    prof = StyleProfile(captions=CaptionStyle(words_per_caption=2, animation="karaoke"))
    ass = build_ass(ws, 3.0, prof)
    events = [line for line in ass.splitlines() if line.startswith("Dialogue: 0")]
    assert len(events) == len(ws)  # karaokê: um evento por palavra

    def secs(ts):
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    spans = [(secs(e.split(",")[1]), secs(e.split(",")[2])) for e in events]
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(spans, spans[1:]))
    assert "UM" in ass  # caixa alta


def test_group_words_respects_size_and_sentence_end():
    ws = words_from("Isso é real. Agora outra coisa importante")
    groups = group_words(ws, 3)
    assert [len(g) for g in groups] == [3, 3, 1]


# --------------------------------------------------------------------------- seleção

def _story_transcript() -> Transcript:
    filler = "então a gente foi almoçar e estava tudo normal lá. " * 6
    hook = "Você sabia que 90% das pessoas desistem? Isso é absurdo e ninguém te conta a verdade. "
    body = "O segredo é repetir todo dia sem desculpa. " * 4
    return Transcript(words_from(filler + hook + body + filler, step=0.35))


def test_candidates_respect_duration():
    t = _story_transcript()
    prof = StyleProfile(duration=DurationStyle(min_s=8, max_s=20, target_s=14))
    sents = t.sentences()
    for i, j in build_candidates(t, prof):
        assert 8 <= sents[j].end - sents[i].start <= 20


def test_heuristic_prefers_hook_and_avoids_overlap():
    t = _story_transcript()
    prof = StyleProfile(duration=DurationStyle(min_s=8, max_s=20, target_s=14))
    clips = heuristic_select(t, prof, 3)
    assert clips[0].hook.startswith("Você sabia")
    for a in clips:
        for b in clips:
            if a is not b:
                assert a.end <= b.start or b.end <= a.start


# --------------------------------------------------------------------------- cores / perfis

def _caption_frame(fill=(255, 255, 255), highlight=(0, 230, 255), box=None):
    frame = np.full((640, 360, 3), (120, 90, 160), np.uint8)  # fundo colorido (simula cenário)
    frame[:, :120] = (60, 140, 200)
    if box is not None:
        cv2.rectangle(frame, (60, 300), (300, 350), box, -1)
    for text, x, color in (("ISSO", 70, fill), ("AQUI", 190, highlight)):
        if box is None:
            cv2.putText(frame, text, (x, 340), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), 9, cv2.LINE_AA)
        cv2.putText(frame, text, (x, 340), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 3, cv2.LINE_AA)
    return frame, TextBox(0, "ISSO AQUI", 60 / 360, 300 / 640, 240 / 360, 50 / 640, 0.9, 0)


def test_text_colors_find_highlight_not_background():
    frame, box = _caption_frame()
    c = analyze_text_colors([(frame, box)] * 3)
    r, g, b = (int(c["accent"][i:i + 2], 16) for i in (1, 3, 5))
    assert r > 200 and g > 180 and b < 80  # amarelo, não o roxo/laranja do fundo
    assert c["highlight_ratio"] > 0.5
    assert c["outline"] is True and c["box"] is False


def test_text_colors_detect_background_box():
    frame, box = _caption_frame(fill=(0, 0, 0), highlight=(0, 0, 0), box=(255, 255, 255))
    box = TextBox(0, "ISSO AQUI", 62 / 360, 302 / 640, 236 / 360, 46 / 640, 0.9, 0)
    c = analyze_text_colors([(frame, box)] * 3)
    assert c["box"] is True
    assert c["box_color"].upper().startswith("#F")
    assert c["primary"] in ("#000000",) or int(c["primary"][1:3], 16) < 60


def test_merge_profiles_uses_median_and_vote():
    a = StyleProfile(pacing=PacingStyle(cuts_per_min=10), captions=CaptionStyle(words_per_caption=2))
    b = StyleProfile(pacing=PacingStyle(cuts_per_min=20), captions=CaptionStyle(words_per_caption=2))
    c = StyleProfile(pacing=PacingStyle(cuts_per_min=40), captions=CaptionStyle(words_per_caption=4))
    m = merge_profiles([a, b, c])
    assert m.pacing.cuts_per_min == 20
    assert m.captions.words_per_caption == 2


def test_presets_roundtrip_json():
    for p in PRESETS.values():
        assert StyleProfile.model_validate_json(p.model_dump_json()) == p
