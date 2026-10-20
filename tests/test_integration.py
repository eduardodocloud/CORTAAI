"""Integração: gera mídia sintética com estilo CONHECIDO, aprende o estilo e corta.

Precisa dos modelos (scripts/download_models.sh). Rode com: pytest -m slow
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cortaai.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    s = get_settings()
    if not (s.parakeet_dir / "encoder.int8.onnx").exists() or not (ROOT / "models" / "vits-piper-pt_BR-faber-medium").exists():
        pytest.skip("modelos não baixados")
    faces = ROOT / "tests" / "assets"
    if not (faces / "face1.jpg").exists():
        pytest.skip("imagens de rosto de teste ausentes (tests/assets)")
    out = tmp_path_factory.mktemp("media")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_media.py"), str(out),
                    str(faces / "face1.jpg"), str(faces / "face2.jpg")], check=True)
    return out


def test_reference_style_is_recovered(media, monkeypatch, tmp_path):
    monkeypatch.setenv("CORTAAI_ASR", "parakeet")
    from cortaai.style.analyzer import analyze_reference

    p = analyze_reference(media / "referencia.mp4", cache_dir=tmp_path, settings=get_settings(), use_llm=False)
    c = p.captions
    assert c.enabled and c.words_per_caption == 2 and c.uppercase
    assert c.font == "Anton"
    assert abs(c.position_y - 0.62) < 0.03
    assert c.animation == "karaoke"
    r, g, b = (int(c.highlight_color[i:i + 2], 16) for i in (1, 3, 5))
    assert r > 220 and g > 200 and b < 60  # amarelo
    assert p.title.enabled and abs(p.title.position_y - 0.12) < 0.03
    assert p.title.box_color.upper().startswith("#F") and int(p.title.text_color[1:3], 16) < 60  # texto preto em caixa branca
    assert 22 <= p.pacing.cuts_per_min <= 34  # troca a cada 2s
    assert 1.15 <= p.pacing.punch_in_scale <= 1.35  # punch-in de 1.25
    assert p.content.hook_type == "question"


def test_clips_are_rendered(media, monkeypatch, tmp_path):
    monkeypatch.setenv("CORTAAI_ASR", "parakeet")
    monkeypatch.setenv("CORTAAI_DATA_DIR", str(tmp_path / "data"))
    from cortaai.pipeline import make_clips
    from cortaai.style.profile import PRESETS

    man = make_clips(media / "fonte_longa.mp4", tmp_path / "out", PRESETS["podcast_br"], 2, get_settings(), use_llm=False)
    assert len(man["cortes"]) == 2
    for c in man["cortes"]:
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "json", c["path"]],
            capture_output=True, check=True).stdout)
        assert info["streams"][0]["width"] == 1080 and info["streams"][0]["height"] == 1920
    assert "90%" in man["cortes"][0]["hook"] or "erro" in man["cortes"][0]["hook"]
