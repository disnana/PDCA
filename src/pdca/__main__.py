"""PDCA CLI: JSON-value lossless compression with reusable profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .core import assert_roundtrip, canonical, decode, encode, optimize, token_count, train
from .providers import propose_atoms


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str, data: object) -> None:
    value = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    if path == "-":
        print(value)
    else:
        Path(path).write_text(value + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PDCA profile-driven lossless JSON compression")
    subs = ap.add_subparsers(dest="command", required=True)
    tr = subs.add_parser("train", help="Build a deterministic reusable profile")
    tr.add_argument("input")
    tr.add_argument("-o", "--output", required=True)
    tr.add_argument("--no-atoms", action="store_true")
    en = subs.add_parser("encode", help="Encode with an existing profile")
    en.add_argument("input"); en.add_argument("--profile", required=True)
    en.add_argument("-o", "--output", required=True)
    en.add_argument("--external-profile", action="store_true")
    de = subs.add_parser("decode", help="Decode a PDCA packet")
    de.add_argument("input"); de.add_argument("-o", "--output", required=True)
    de.add_argument("--profile", help="Profile file, required for external-profile packets")
    op = subs.add_parser("optimize", help="Find a lossless profile with lower OpenAI reference token count")
    op.add_argument("input"); op.add_argument("-o", "--output", required=True)
    op.add_argument("--profile-out", help="Optional location to save the profile")
    op.add_argument("--encoding", default="o200k_base")
    op.add_argument("--ai-provider", choices=("openai", "gemini"))
    op.add_argument("--ai-model", help="AI model identifier, e.g. gpt-6 or a Gemini model available in your API account")
    be = subs.add_parser("benchmark", help="Measure tokens in minified input and packet")
    be.add_argument("input")
    be.add_argument("--encoding", default="o200k_base")
    args = ap.parse_args(argv)
    if args.command == "optimize" and args.ai_provider == "gemini" and args.ai_model and args.ai_model.upper() in {"GEMINI_MODEL", "YOUR_AVAILABLE_GEMINI_MODEL"}:\n        ap.error("GEMINI_MODEL is a placeholder, not a real model ID. Try --ai-model gemini-3.8-flash or check https://ai.google.dev/gemini-api/docs/models")\n    if args.command == "train":
        profile = train(load_json(args.input), atoms=not args.no_atoms)
        save_json(args.output, profile)
    elif args.command == "encode":
        data = load_json(args.input)
        profile = load_json(args.profile)
        assert_roundtrip(data, profile)
        save_json(args.output, encode(data, profile, embed_profile=not args.external_profile))
    elif args.command == "decode":
        packet = load_json(args.input)
        profile = load_json(args.profile) if args.profile else None
        save_json(args.output, decode(packet, profile))
    elif args.command == "optimize":
        data = load_json(args.input)
        candidates = []
        if args.ai_provider:
            if not args.ai_model:
                ap.error("--ai-model required when --ai-provider is used")
            print("Warning: selected AI provider receives a sample of your input JSON", file=sys.stderr)
            candidates = propose_atoms(data, args.ai_provider, args.ai_model)
        profile, metrics = optimize(data, candidate_atoms=candidates, encoding=args.encoding)
        save_json(args.output, encode(data, profile))
        if args.profile_out:
            save_json(args.profile_out, profile)
        print(json.dumps(metrics, ensure_ascii=False), file=sys.stderr)
    elif args.command == "benchmark":
        data = load_json(args.input)
        profile, metrics = optimize(data, encoding=args.encoding)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())