"""Linha de comando.

    cortaai estilo referencia.mp4 [outra.mp4 ...] -o estilo.json
    cortaai cortar video.mp4 --estilo estilo.json -n 5 -o saida/
    cortaai cortar https://youtube.com/... --preset hormozi
    cortaai servidor --porta 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _progress(msg: str, frac: float) -> None:
    sys.stderr.write(f"\r[{frac*100:5.1f}%] {msg:<50}")
    sys.stderr.flush()
    if frac >= 1.0:
        sys.stderr.write("\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cortaai", description="Cortes virais com o estilo de um vídeo de referência")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("estilo", help="aprende o estilo de um ou mais vídeos de referência")
    a.add_argument("referencias", nargs="+")
    a.add_argument("-o", "--saida", default="estilo.json")
    a.add_argument("--nome")
    a.add_argument("--sem-ia", action="store_true", help="não usar o Claude mesmo com ANTHROPIC_API_KEY")

    c = sub.add_parser("cortar", help="gera cortes de um vídeo longo (arquivo ou link)")
    c.add_argument("fonte")
    c.add_argument("--estilo", help="JSON gerado por 'cortaai estilo'")
    c.add_argument("--referencia", nargs="*", help="aprende o estilo na hora a partir destes vídeos")
    c.add_argument("--preset", default="padrao")
    c.add_argument("-n", "--quantidade", type=int, default=5)
    c.add_argument("-o", "--saida", default="cortes")
    c.add_argument("--sem-ia", action="store_true")

    sub.add_parser("modelos", help="baixa os modelos locais (rosto + transcrição)")

    s = sub.add_parser("servidor", help="abre o editor web")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--porta", type=int, default=8000)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from .config import get_settings

    settings = get_settings()
    cache = settings.data_dir / "cache"

    if args.cmd == "estilo":
        from .style.analyzer import analyze_reference, merge_profiles

        profiles = [
            analyze_reference(r, name=args.nome, cache_dir=cache, settings=settings, use_llm=not args.sem_ia, progress=_progress)
            for r in args.referencias
        ]
        prof = merge_profiles(profiles, name=args.nome or "combinado")
        Path(args.saida).write_text(prof.model_dump_json(indent=2))
        print(json.dumps(prof.evidence, ensure_ascii=False, indent=2))
        print(f"\nestilo salvo em {args.saida}")
        return 0

    if args.cmd == "cortar":
        from .pipeline import make_clips
        from .style.profile import PRESETS, StyleProfile

        if args.estilo:
            profile = StyleProfile.model_validate_json(Path(args.estilo).read_text())
        elif args.referencia:
            from .style.analyzer import analyze_reference, merge_profiles

            profile = merge_profiles([
                analyze_reference(r, cache_dir=cache, settings=settings, use_llm=not args.sem_ia, progress=_progress)
                for r in args.referencia
            ])
        else:
            if args.preset not in PRESETS:
                p.error(f"preset desconhecido; opções: {', '.join(PRESETS)}")
            profile = PRESETS[args.preset]
        manifest = make_clips(args.fonte, args.saida, profile, args.quantidade, settings, use_llm=not args.sem_ia, progress=_progress)
        for c in manifest["cortes"]:
            print(f"{c['arquivo']}  nota {c['score']:>5}  {c['duracao_s']:>5.1f}s  {c['title']}")
        return 0

    if args.cmd == "modelos":
        from .models import download_models

        download_models(settings.models_dir)
        return 0

    if args.cmd == "servidor":
        import uvicorn

        print(f"\n  Editor aberto em: http://{args.host}:{args.porta}\n")
        uvicorn.run("cortaai.api.server:app", host=args.host, port=args.porta)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
