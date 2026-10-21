# CortaAI

Editor de cortes virais: transforma vídeos longos (podcast, live, entrevista, aula) em shorts 9:16
prontos para TikTok, Reels e Shorts — **imitando o estilo de um vídeo de referência**.

Você envia um corte viral que quer copiar. O CortaAI mede como ele foi editado e aplica o mesmo
padrão nos seus cortes:

| O que ele aprende da referência | Como mede |
|---|---|
| Fonte, tamanho, posição e cor das legendas, cor de destaque, contorno, caixa de fundo | OCR (RapidOCR) em frames amostrados + análise de cor só nos pixels colados no contorno do texto |
| Palavras por legenda e animação (karaokê, pop, palavra a palavra, estática) | agrupamento das caixas de texto por frame, comparação com a transcrição |
| Título fixo na tela (faixa do topo) | texto que fica parado na mesma posição na maior parte do vídeo |
| Ritmo de edição: trocas de plano por minuto, punch-ins (zoom seco) e escala, zoom lento | diferença de conteúdo entre frames + teste de "mesmo rosto ampliado" (template matching em várias escalas) |
| Jump cuts e pausa máxima | intervalos entre palavras na transcrição |
| Layout: rosto em tela cheia, tela dividida, fundo desfocado, faixas pretas | nitidez/brilho por faixa da tela + rostos detectados |
| Duração e tipo de gancho (pergunta, número, história, afirmação) | duração da referência + primeira frase |

Com vários vídeos de referência, ele combina o padrão (mediana dos números, voto nas escolhas).

## Versão de um arquivo só (navegador, sem instalar nada)

Abra **`cortaai.html`** com dois cliques no Chrome ou Edge. Tudo roda no navegador:

- transcrição com Whisper ([transformers.js](https://huggingface.co/docs/transformers.js); usa a GPU via WebGPU quando existe)
- rostos com [MediaPipe](https://ai.google.dev/edge/mediapipe), legendas da referência lidas com [Tesseract.js](https://tesseract.projectnaptha.com/)
- cortes desenhados em canvas e gravados pelo próprio navegador (MP4 no Chrome/Edge, WebM em outros)

O vídeo longo pode ser o **arquivo** ou um **link do YouTube** (watch, youtu.be, shorts, live) ou de um arquivo
.mp4/.webm. Como o navegador não pode baixar direto do YouTube, o link é baixado por servidores públicos do
[Invidious](https://invidious.io/) (tenta vários até um responder; até 720p, ou 360p em vídeos com mais de 1 h).
Esses servidores às vezes ficam fora do ar: aí baixe o vídeo e arraste o arquivo, ou use a versão Python
(`cortaai servidor`), que baixa qualquer link com o yt-dlp.

Precisa de internet só para baixar as bibliotecas e os modelos na primeira vez (o navegador guarda em cache).
Os cortes são gravados em tempo real (30 s de corte ≈ 30 s), então deixe a aba aberta e visível enquanto gera.
A análise da referência usa até 90 s do vídeo.

## Como funciona

```
vídeo longo ─► transcrição com tempo por palavra (faster-whisper ou Parakeet)
            ─► escolha dos momentos (Claude, ou pontuador local sem API)
            ─► rastreio de rostos (OpenCV SSD) ─► reenquadramento 9:16 suavizado
            ─► jump cuts + punch-ins no ritmo da referência
            ─► legendas ASS (libass) no estilo da referência ─► MP4 (um único encode)
```

- **Seleção de momentos**: com `ANTHROPIC_API_KEY` o Claude lê a transcrição numerada por frase e escolhe os
  cortes seguindo o critério aprendido da referência (ele também olha frames da referência para descrever o
  estilo editorial). Sem chave, um pontuador local avalia gancho (pergunta, número, palavras fortes, sem começar
  com conectivo/pronome solto), energia da voz, final com frase completa, duração e ritmo.
- **Tudo roda local** (CPU basta). Só a seleção pelo Claude usa a internet.

## Instalação

Requer Python 3.10+ e `ffmpeg` com libass.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[whisper,parakeet,llm]"
./scripts/download_models.sh          # detector de rosto + Parakeet (~650 MB)
export ANTHROPIC_API_KEY=...          # opcional: seleção de momentos pelo Claude
```

## Deploy na nuvem (link público)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/eduardodocloud/CORTAAI)

1. Clique no botão acima e entre com o GitHub (se o repositório for privado, autorize o Render a lê-lo).
2. Confirme o Blueprint (`render.yaml`): serviço Docker no plano **Standard** (2 GB de RAM) com disco de 10 GB em `/data`.
3. Opcional: cole sua `ANTHROPIC_API_KEY` para o Claude escolher os momentos.
4. Ao terminar o build (~10 min, os modelos são embutidos na imagem), abra o link `https://cortaai-xxxx.onrender.com`.
   O navegador pede uma senha: ela está em **Environment → `CORTAAI_SENHA`** no painel do Render (qualquer usuário serve).

A mesma imagem roda em qualquer lugar com Docker:

```bash
docker build -t cortaai .
docker run -p 8000:8000 -e CORTAAI_SENHA=minhasenha -v cortaai-data:/data cortaai
```

## Uso

### Editor web

```bash
cortaai servidor            # abre em http://127.0.0.1:8000
```

1. **Estilo**: arraste um ou mais cortes de referência (ou escolha um preset). Ele mostra o que aprendeu e deixa
   ajustar tudo (fonte, cores, posição, ritmo, zoom, duração) com prévia ao vivo.
2. **Vídeo longo**: arraste o arquivo ou cole um link do YouTube/Twitch/Kick, escolha quantos cortes e gere.

### Linha de comando

```bash
cortaai estilo referencia1.mp4 referencia2.mp4 -o meu_estilo.json   # aprende e salva o estilo
cortaai cortar podcast.mp4 --estilo meu_estilo.json -n 8 -o cortes/  # gera 8 cortes
cortaai cortar "https://youtube.com/watch?v=..." --referencia ref.mp4  # aprende e corta de uma vez
cortaai cortar podcast.mp4 --preset hormozi                           # presets: padrao, podcast_br, hormozi, minimalista
```

Cada execução gera `cortes/corte_XX.mp4`, uma miniatura `.jpg` e `cortes.json` com nota, título, gancho,
motivo da escolha e o perfil de estilo usado.

## Configuração (variáveis de ambiente)

| Variável | Padrão | |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | liga a seleção de momentos pelo Claude |
| `CORTAAI_LLM_MODEL` | `claude-opus-5-5` | modelo do Claude |
| `CORTAAI_ASR` | `auto` | `auto` (faster-whisper → Parakeet), `whisper` ou `parakeet` |
| `CORTAAI_WHISPER_MODEL` | `large-v3-turbo` | modelo do faster-whisper |
| `CORTAAI_LANGUAGE` | detectado | força o idioma da transcrição (ex.: `pt`) |
| `CORTAAI_DATA_DIR` | `./data` | uploads, cache de transcrição/rostos e cortes do editor web |
| `CORTAAI_SENHA` | — | exige senha (HTTP Basic) para abrir o editor; use sempre que ele estiver na internet |

## Testes

```bash
pytest -m "not slow"     # unitários (segundos)
WITH_TTS=1 ./scripts/download_models.sh && pytest   # inclui integração
```

O teste de integração gera uma referência sintética com estilo **conhecido** (legenda Anton de 2 palavras,
destaque amarelo em 62% da altura, título com caixa branca, troca de plano a cada 2 s com punch-in de 1,25×) e
confere se o analisador recupera cada parâmetro; depois gera cortes de um vídeo longo sintético.

## Estrutura

```
src/cortaai/
  transcribe.py        transcrição com tempo por palavra (faster-whisper / Parakeet)
  vision.py            rostos, cortes de cena, punch-ins e zoom lento
  style/profile.py     StyleProfile (o "DNA" do corte) + presets
  style/analyzer.py    aprende o StyleProfile de vídeos de referência
  selection/           candidatos por frase, pontuador local e seleção pelo Claude
  render/edit.py       jump cuts, mapa de tempo, planos com punch-in
  render/captions.py   legendas ASS (karaokê, pop, palavra, título)
  render/compose.py    reenquadramento (rosto, tela dividida, fundo desfocado) e encode
  pipeline.py          orquestra tudo
  api/server.py        API FastAPI + fila de trabalhos
  web/index.html       editor web
```

## Licenças

Fontes em `fonts/` sob SIL Open Font License 1.1 (Montserrat, Anton, Bangers, Poppins).
