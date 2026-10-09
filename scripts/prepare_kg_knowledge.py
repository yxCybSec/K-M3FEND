"""
One-shot: expand KG from corpus + build structured knowledge npy for train/val/test.

Replaces LLM *_knowledge.npy with KG-MFEND-style vectors (*_kg_knowledge.npy).

Usage (project root):
  python scripts/prepare_kg_knowledge.py --dataset ch
  python scripts/prepare_kg_knowledge.py --dataset ch --limit 50   # smoke test
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd: list) -> None:
    print("[prepare_kg]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare KG + kg_knowledge npy for KM3FEND")
    parser.add_argument("--dataset", choices=["ch", "en"], default="ch")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--kg_dir", default="./data/kg")
    parser.add_argument("--entity_file", default=None)
    parser.add_argument("--triple_file", default=None)
    parser.add_argument("--kg_backend", choices=["local", "cndbpedia", "ownthink"], default="local")
    parser.add_argument("--cndbpedia_base_url", default="http://shuyantech.com/api/cndbpedia/avpair")
    parser.add_argument("--cndbpedia_apikey", default=None)
    parser.add_argument("--cndbpedia_cache", default=None)
    parser.add_argument("--cndbpedia_timeout", type=float, default=8.0)
    parser.add_argument("--ownthink_base_url", default="https://api.ownthink.com/kg")
    parser.add_argument("--ownthink_cache", default=None)
    parser.add_argument("--ownthink_timeout", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=None, help="Debug: only first N rows per split")
    parser.add_argument("--skip_corpus_kg", action="store_true", help="Skip build_kg_from_corpus")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(_ROOT, "data", args.dataset)
    entity_file = args.entity_file or os.path.join(_ROOT, args.kg_dir, "entities.txt")
    triple_file = args.triple_file or os.path.join(_ROOT, args.kg_dir, "triples.tsv")
    py = sys.executable

    if args.kg_backend == "local" and (not args.skip_corpus_kg) and args.dataset == "ch":
        _run(
            [
                py,
                os.path.join(_ROOT, "scripts", "build_kg_from_corpus.py"),
                "--data_dir",
                data_dir,
                "--kg_dir",
                os.path.join(_ROOT, args.kg_dir),
            ]
        )

    for split in ("train", "val", "test"):
        pkl = os.path.join(data_dir, f"{split}.pkl")
        if not os.path.exists(pkl):
            print(f"[prepare_kg] skip missing {pkl}")
            continue
        cmd = [
            py,
            os.path.join(_ROOT, "scripts", "build_knowledge_from_kg.py"),
            "--dataset",
            args.dataset,
            "--pkl",
            pkl,
            "--entity_file",
            entity_file,
            "--triple_file",
            triple_file,
        ]
        if args.kg_backend == "cndbpedia":
            cmd.extend(
                [
                    "--kg_backend",
                    "cndbpedia",
                    "--cndbpedia_base_url",
                    args.cndbpedia_base_url,
                    "--cndbpedia_timeout",
                    str(args.cndbpedia_timeout),
                ]
            )
            if args.cndbpedia_apikey:
                cmd.extend(["--cndbpedia_apikey", args.cndbpedia_apikey])
            if args.cndbpedia_cache:
                cmd.extend(["--cndbpedia_cache", args.cndbpedia_cache])
        elif args.kg_backend == "ownthink":
            cmd.extend(
                [
                    "--kg_backend",
                    "ownthink",
                    "--ownthink_base_url",
                    args.ownthink_base_url,
                    "--ownthink_timeout",
                    str(args.ownthink_timeout),
                ]
            )
            if args.ownthink_cache:
                cmd.extend(["--ownthink_cache", args.ownthink_cache])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.device:
            cmd.extend(["--device", args.device])
        _run(cmd)

    print("[prepare_kg] done. Training will load *_kg_knowledge.npy (see utils/dataloader.py).")


if __name__ == "__main__":
    main()
