"""Transcrição com timestamps por palavra.

Backends:
- faster-whisper (melhor qualidade; baixa modelos do HuggingFace)
- Parakeet TDT v3 via sherpa-onnx (multilíngue, inclui PT; roda bem em CPU)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings, get_settings
from .media import load_audio

log = logging.getLogger(__name__)
SR = 16000
SENTENCE_END = re.compile(r"[.!?…]+[\"'”»)]*$")


@dataclass
class Word:
    text: str
    start: float
    end: float

    @property
    def is_sentence_end(self) -> bool:
        return bool(SENTENCE_END.search(self.text))


@dataclass
class Sentence:
    index: int
    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class Transcript:
    words: list[Word]
    language: str | None = None
    duration: float = 0.0
    backend: str = ""
    _sentences: list[Sentence] | None = field(default=None, repr=False, compare=False)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def words_between(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if w.end > start and w.start < end]

    def sentences(self, max_gap: float = 1.2, max_words: int = 40) -> list[Sentence]:
        """Agrupa palavras em frases por pontuação, pausas longas ou tamanho."""
        if self._sentences is not None:
            return self._sentences
        out: list[Sentence] = []
        cur: list[Word] = []
        for i, w in enumerate(self.words):
            cur.append(w)
            nxt = self.words[i + 1] if i + 1 < len(self.words) else None
            gap = (nxt.start - w.end) if nxt else 0.0
            if nxt is None or w.is_sentence_end or gap > max_gap or len(cur) >= max_words:
                out.append(Sentence(len(out), cur))
                cur = []
        self._sentences = out
        return out

    def to_json(self) -> dict:
        return {
            "language": self.language,
            "duration": self.duration,
            "backend": self.backend,
            "words": [asdict(w) for w in self.words],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Transcript":
        return cls(
            words=[Word(**w) for w in data["words"]],
            language=data.get("language"),
            duration=data.get("duration", 0.0),
            backend=data.get("backend", ""),
        )


# --------------------------------------------------------------------------- backends


class _Backend:
    name = "base"

    def transcribe(self, audio: np.ndarray, language: str | None) -> tuple[list[Word], str | None]:
        raise NotImplementedError


class FasterWhisperBackend(_Backend):
    name = "faster-whisper"

    def __init__(self, settings: Settings):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(settings.whisper_model, device="auto", compute_type="int8", cpu_threads=settings.threads)

    def transcribe(self, audio, language):
        segments, info = self.model.transcribe(
            audio, language=language, word_timestamps=True, vad_filter=True, beam_size=5
        )
        words = [
            Word(w.word.strip(), float(w.start), float(w.end))
            for seg in segments
            for w in (seg.words or [])
            if w.word.strip()
        ]
        return words, info.language


class ParakeetBackend(_Backend):
    """NVIDIA Parakeet TDT 0.6B v3 (25 idiomas europeus, inclui português)."""

    name = "parakeet"
    chunk_seconds = 45.0

    def __init__(self, settings: Settings):
        import sherpa_onnx

        d = settings.parakeet_dir
        if not (d / "encoder.int8.onnx").exists():
            raise FileNotFoundError(f"modelo Parakeet não encontrado em {d} (rode scripts/download_models.sh)")
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=settings.threads,
        )

    def transcribe(self, audio, language):
        words: list[Word] = []
        for offset, chunk in _split_on_silence(audio, self.chunk_seconds):
            stream = self.rec.create_stream()
            stream.accept_waveform(SR, chunk)
            self.rec.decode_stream(stream)
            r = stream.result
            durations = list(getattr(r, "durations", []) or [])
            words.extend(_merge_tokens(list(r.tokens), list(r.timestamps), durations, offset))
        return words, language


def _merge_tokens(tokens: list[str], stamps: list[float], durations: list[float], offset: float) -> list[Word]:
    """Junta subpalavras (BPE) em palavras; token iniciado por espaço abre palavra nova."""
    words: list[Word] = []
    pending_space = False
    for i, tok in enumerate(tokens):
        start = offset + stamps[i]
        end = start + (durations[i] if i < len(durations) and durations[i] > 0 else 0.08)
        piece = tok.replace("▁", " ")
        if not piece.strip():
            pending_space = True  # token só de espaço: a próxima peça abre palavra
            continue
        starts_word = piece.startswith(" ") or pending_space or not words
        pending_space = False
        piece = piece.strip()
        if starts_word:
            words.append(Word(piece, start, end))
        else:
            words[-1].text += piece
            words[-1].end = end
    # fim da palavra não passa do início da próxima
    for a, b in zip(words, words[1:]):
        if b.start > a.start:
            a.end = min(max(a.end, a.start + 0.05), b.start)
    return words


def _split_on_silence(audio: np.ndarray, target: float) -> list[tuple[float, np.ndarray]]:
    """Divide áudio em blocos de ~target s cortando no trecho mais silencioso perto do limite."""
    n = len(audio)
    if n <= target * SR * 1.3:
        return [(0.0, audio)]
    win = int(0.05 * SR)
    rms = np.sqrt(np.convolve(audio.astype(np.float64) ** 2, np.ones(win) / win, mode="same"))
    chunks, pos = [], 0
    while pos < n:
        ideal = pos + int(target * SR)
        if ideal >= n - int(5 * SR):
            chunks.append((pos / SR, audio[pos:]))
            break
        lo, hi = ideal - int(8 * SR), min(n, ideal + int(8 * SR))
        cut = lo + int(np.argmin(rms[lo:hi]))
        chunks.append((pos / SR, audio[pos:cut]))
        pos = cut
    return chunks


def refine_word_times(words: list[Word], audio: np.ndarray) -> list[Word]:
    """O ASR costuma esticar o fim da palavra para dentro do silêncio seguinte. Apara início e fim
    onde o áudio cai ao nível do fundo, para as pausas reais aparecerem (o jump cut depende disso)."""
    if not words or audio is None or not len(audio):
        return words
    hop = SR // 100
    n = len(audio) // hop
    frames = audio[: n * hop].astype(np.float64).reshape(n, hop)
    db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-10)
    voiced = db > np.percentile(db[::5], 10) + 9
    out = []
    for w in words:
        a, b = int(w.start * 100), min(n - 1, int(np.ceil(w.end * 100)))
        while b > a + 6 and not voiced[b] and not voiced[b - 1]:
            b -= 1
        while a < b - 6 and not voiced[a] and not voiced[a + 1]:
            a += 1
        out.append(Word(w.text, a / 100, min(w.end, (b + 3) / 100)))
    return out


def _make_backend(settings: Settings) -> _Backend:
    order = {
        "auto": [FasterWhisperBackend, ParakeetBackend],
        "whisper": [FasterWhisperBackend],
        "faster-whisper": [FasterWhisperBackend],
        "parakeet": [ParakeetBackend],
    }.get(settings.asr_backend, [FasterWhisperBackend, ParakeetBackend])
    errors = []
    for cls in order:
        try:
            return cls(settings)
        except Exception as exc:  # modelo ausente, rede bloqueada etc.
            errors.append(f"{cls.name}: {exc}")
            log.warning("backend %s indisponível: %s", cls.name, exc)
    raise RuntimeError("nenhum backend de transcrição disponível:\n" + "\n".join(errors))


_BACKEND_CACHE: dict[str, _Backend] = {}


def get_backend(settings: Settings | None = None) -> _Backend:
    settings = settings or get_settings()
    key = settings.asr_backend
    if key not in _BACKEND_CACHE:
        _BACKEND_CACHE[key] = _make_backend(settings)
    return _BACKEND_CACHE[key]


def file_hash(path: str | Path, nbytes: int = 8 << 20) -> str:
    p = Path(path)
    h = hashlib.sha1(str(p.stat().st_size).encode())
    with p.open("rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def transcribe(path: str | Path, cache_dir: str | Path | None = None, settings: Settings | None = None) -> Transcript:
    settings = settings or get_settings()
    cache_file = None
    if cache_dir:
        cache_file = Path(cache_dir) / f"transcript-{file_hash(path)}.json"
        if cache_file.exists():
            return Transcript.from_json(json.loads(cache_file.read_text()))
    audio = load_audio(path, SR)
    backend = get_backend(settings)
    words, lang = backend.transcribe(audio, settings.language)
    words = refine_word_times(words, audio)
    t = Transcript(words=words, language=lang or settings.language, duration=len(audio) / SR, backend=backend.name)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(t.to_json(), ensure_ascii=False))
    return t
