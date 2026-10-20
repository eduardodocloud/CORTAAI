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
