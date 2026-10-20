"""Gera mídia sintética para testes (sem depender de YouTube):

- fonte_longa.mp4: 16:9, fala PT-BR (voz Piper), rosto que se move, trecho com 2 pessoas, pausas longas
- referencia.mp4: 9:16 com estilo CONHECIDO (legenda Anton 2 palavras, destaque amarelo em y=0.62,
  título branco no topo, troca de plano a cada 2s com punch-in), para validar o analisador.

Uso: python scripts/make_test_media.py <pasta_saida> <img_rosto1> <img_rosto2>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import sherpa_onnx
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SR_OUT = 22050

LONG_SCRIPT = [
    "Bom dia pessoal, sejam bem-vindos a mais um episódio do nosso podcast.",
    "Hoje a gente vai conversar sobre carreira, dinheiro e algumas coisas que aconteceram comigo nesse ano.",
    "Antes de começar, deixa eu ajeitar aqui o microfone.",
    "Então, como eu estava dizendo, o assunto de hoje é bem tranquilo.",
    "Você sabia que noventa por cento das pessoas desistem antes do primeiro resultado?",
    "Isso é absurdo, mas é a verdade que ninguém te conta.",
    "Eu perdi todo o meu dinheiro aos vinte e dois anos, e foi o melhor erro da minha vida.",
    "Porque ali eu aprendi que sucesso não é sorte, é repetição.",
    "Todo dia, mesmo sem vontade, eu fazia o simples.",
    "E em dois anos eu ganhei meu primeiro milhão.",
    "Pois é, e aí a gente foi almoçar depois, foi bem legal.",
    "O restaurante era perto da casa da minha mãe, um lugar bem calmo.",
    "Tinha uma comida boa, nada demais, preço normal.",
    "Mas olha só, qual é o maior erro de quem quer ficar rico?",
    "É achar que precisa de uma ideia genial.",
    "Ninguém fica rico com ideia, fica rico com execução.",
    "Eu vi gente muito mais inteligente que eu quebrar porque nunca começou.",
    "Então se você está esperando o momento perfeito, ele nunca vai chegar.",
    "Bom, agora vamos falar um pouco sobre o tempo.",
    "Choveu bastante essa semana aqui na cidade.",
    "Eu fiquei em casa a maior parte dos dias, lendo e descansando.",
    "Nada muito interessante para contar sobre isso.",
    "Agora presta atenção, porque essa é a parte mais importante do episódio.",
    "Existem três hábitos que mudaram minha vida completamente.",
    "Primeiro, acordar cedo e não olhar o celular na primeira hora.",
    "Segundo, escrever todo dia o que eu quero conquistar.",
    "E terceiro, cortar da minha vida pessoas que só reclamam.",
    "Parece simples, mas quase ninguém faz.",
    "Enfim, era isso que eu queria compartilhar hoje.",
    "Obrigado por assistir e até o próximo episódio.",
]

REF_SCRIPT = [
    "Por que ninguém te conta isso?",
    "O segredo do dinheiro não é ganhar mais.",
    "É parar de perder com bobagem todo santo dia.",
    "Eu sei porque eu já fui assim, gastando tudo que entrava.",
    "Delivery, assinatura que eu nem usava, presente caro para impressionar.",
    "Um dia eu sentei e somei tudo, e quase caí da cadeira.",
    "Eram mais de mil reais por mês jogados fora.",
    "Então faz o teste hoje: anota cada centavo que sai.",
    "Faz isso por trinta dias e me agradece depois!",
]


def tts_engine():
    d = ROOT / "models" / "vits-piper-pt_BR-faber-medium"
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=str(d / "pt_BR-faber-medium.onnx"), tokens=str(d / "tokens.txt"),
                                                   data_dir=str(d / "espeak-ng-data")), num_threads=4))
    return sherpa_onnx.OfflineTts(cfg)


def synth(tts, sentences, pauses, speed=1.0):
    """Retorna áudio e lista de (inicio, fim, frase)."""
    chunks, spans, t = [], [], 0.0
    for s, pause in zip(sentences, pauses):
        a = np.array(tts.generate(s, sid=0, speed=speed).samples, dtype=np.float32)
        # corta silêncio das pontas que o TTS deixa
        idx = np.where(np.abs(a) > 0.01)[0]
        a = a[max(0, idx[0] - 200): idx[-1] + 200] if len(idx) else a
        chunks.append(a)
        spans.append((t, t + len(a) / SR_OUT, s))
        t += len(a) / SR_OUT
        chunks.append(np.zeros(int(pause * SR_OUT), dtype=np.float32))
        t += pause
    return np.concatenate(chunks), spans


def word_times(spans):
    out = []
    for a, b, s in spans:
        words = s.split()
        weights = np.array([len(w) + 2 for w in words], dtype=float)
        edges = a + np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * (b - a)
        out += [(edges[i], edges[i + 1], w) for i, w in enumerate(words)]
    return out


def encoder(path, w, h, fps, audio_path):
    return subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
                             "-i", "pipe:0", "-i", str(audio_path), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], stdin=subprocess.PIPE)


def paste(canvas, img, cx, cy, height):
    h0, w0 = img.shape[:2]
    nh = int(height)
    nw = int(w0 * nh / h0)
    im = cv2.resize(img, (nw, nh))
    H, W = canvas.shape[:2]
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + nw), min(H, y0 + nh)
    canvas[ya:yb, xa:xb] = im[ya - y0:yb - y0, xa - x0:xb - x0]


def make_long(out: Path, face1: np.ndarray, face2: np.ndarray, tts):
    pauses = [0.9 if i % 3 else 1.4 for i in range(len(LONG_SCRIPT))]
    audio, spans = synth(tts, LONG_SCRIPT, pauses)
    wav = out / "fonte_longa.wav"
    sf.write(wav, audio, SR_OUT)
    dur = len(audio) / SR_OUT
    W, H, fps = 1280, 720, 25
    enc = encoder(out / "fonte_longa.mp4", W, H, fps, wav)
    yy = np.linspace(0, 1, H)[:, None, None]
    bg = (np.array([60, 40, 30]) * (1 - yy) + np.array([90, 70, 50]) * yy).astype(np.uint8)
    bg = np.repeat(bg, W, axis=1)
    two_start, two_end = dur * 0.55, dur * 0.75
    for i in range(int(dur * fps)):
        t = i / fps
        frame = bg.copy()
        if two_start <= t < two_end:
            paste(frame, face1, W * 0.25, H * 0.5, H * 0.6)
            paste(frame, face2, W * 0.75, H * 0.5, H * 0.6)
        else:
            cx = W * (0.3 + 0.35 * (0.5 + 0.5 * np.sin(t / 12)))
            paste(frame, face1, cx, H * 0.48, H * 0.7)
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    enc.wait()
    wav.unlink()
    return dur


def make_reference(out: Path, face1: np.ndarray, face2: np.ndarray, tts):
    audio, spans = synth(tts, REF_SCRIPT, [0.15] * len(REF_SCRIPT), speed=1.1)
    wav = out / "referencia.wav"
    sf.write(wav, audio, SR_OUT)
    dur = len(audio) / SR_OUT
    words = word_times(spans)
    W, H, fps = 1080, 1920, 30
    font = ImageFont.truetype(str(ROOT / "fonts" / "Anton-Regular.ttf"), int(0.065 * H))
    tfont = ImageFont.truetype(str(ROOT / "fonts" / "Montserrat-Black.ttf"), int(0.03 * H))
    groups = [words[i:i + 2] for i in range(0, len(words), 2)]
    enc = encoder(out / "referencia.mp4", W, H, fps, wav)
    shot_len = 2.0
    for i in range(int(dur * fps)):
        t = i / fps
        shot = int(t / shot_len)
        src = face1 if (shot // 2) % 2 == 0 else face2
        zoom = 1.25 if shot % 2 else 1.0  # punch-in a cada troca ímpar
        frame = np.full((H, W, 3), (40, 30, 25), dtype=np.uint8)
        paste(frame, src, W / 2, H * 0.42, H * 0.75 * zoom)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        # título fixo: caixa branca, texto preto, topo
        title = "O SEGREDO DO DINHEIRO"
        tb = d.textbbox((0, 0), title, font=tfont)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = int(0.12 * H)
        d.rectangle([W / 2 - tw / 2 - 30, ty - th / 2 - 25, W / 2 + tw / 2 + 30, ty + th / 2 + 30], fill=(255, 255, 255))
        d.text((W / 2, ty), title, font=tfont, fill=(0, 0, 0), anchor="mm")
        # legenda: grupo de 2 palavras, palavra falada em amarelo
        g = next((g for g in groups if g[0][0] <= t < g[-1][1] + 0.1), None)
        if g:
            texts = [w[2].upper() for w in g]
            widths = [d.textlength(x + " ", font=font) for x in texts]
            x = W / 2 - sum(widths) / 2
            for (a, b, _), txt, wd in zip(g, texts, widths):
                color = (255, 230, 0) if a <= t < b else (255, 255, 255)
                d.text((x, int(0.62 * H)), txt, font=font, fill=color, anchor="lm", stroke_width=8, stroke_fill=(0, 0, 0))
                x += wd
        enc.stdin.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR).tobytes())
    enc.stdin.close()
    enc.wait()
    wav.unlink()
    return dur


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    f1, f2 = cv2.imread(sys.argv[2]), cv2.imread(sys.argv[3])
    tts = tts_engine()
    print("referencia.mp4", round(make_reference(out, f1, f2, tts), 1), "s")
    print("fonte_longa.mp4", round(make_long(out, f1, f2, tts), 1), "s")
