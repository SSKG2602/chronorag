"""Command-line utility for ingesting, retrieving, answering, and purging data."""

from __future__ import annotations

import argparse
import json

from app.deps import get_app_state
from app.services.answer_service import answer
from app.services.ingest_service import ingest
from app.services.retrieve_service import retrieve
from app.services.maintenance_service import purge_system


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest the provided documents and print a summary of ingested chunk ids."""
    docs = ingest(args.paths or [], args.text or [], args.provenance)
    summary = {
        "ingested_chunks": len(docs),
        "source_files": args.paths or [],
        "chunk_ids": docs,
    }
    print(json.dumps(summary, indent=2))


def cmd_retrieve(args: argparse.Namespace) -> None:
    """Run retrieval-only pipeline to inspect top-k chunks for a query."""
    state = get_app_state()
    decision = state.router.route(args.query, None, signals=None)
    data = retrieve(args.query, decision.window, args.mode, top_k=args.top_k, axis=args.axis)
    print(json.dumps(data, indent=2, default=str))


def cmd_answer(args: argparse.Namespace) -> None:
    """Execute the full answer pipeline and pretty-print the response."""
    response = answer(args.query, None, args.mode, args.axis)

    print("Answer:")
    print(response["answer"] or "(no generated text)")

    print("\nEvidence Only:")
    print("yes" if response.get("evidence_only") else "no")
    if response.get("reason"):
        print(f"Reason: {response['reason']}")

    print("\nHops Used:")
    print(response["controller_stats"]["hops_used"])

    print("\nAttribution Card:")
    card = response["attribution_card"]
    print(f"  Mode: {card['mode']}")
    print(f"  Axis: {card['time_axis']}")
    print(f"  Window: {card['window']['from']} → {card['window']['to']}")
    print("  Sources:")
    for idx, src in enumerate(card["sources"], start=1):
        print(f"    {idx}. {src['uri']} (score {src['score']:.2f})")
        print(f"       Quote: {src['quote'].strip()}")
        print(f"       Interval: {src['interval']['from']} → {src['interval']['to']}")

    print("\nController Stats:")
    stats = response["controller_stats"]
    print(f"  hops_used: {stats['hops_used']}")
    print(f"  coverage: {stats['signals']['coverage']}")
    print(f"  authority: {stats['signals']['authority']}")
    print(f"  latency_ms: {stats['latency_ms']}")
    print(f"  rerank_method: {stats['rerank_method']}")

    if response.get("audit_trail"):
        print("\nAudit Trail:")
        for event in response["audit_trail"]:
            print(f"  - {event['event']}")
            if "conflicts" in event:
                for conflict in event["conflicts"]:
                    print(f"      {conflict['first']} ↔ {conflict['second']} (overlap {conflict['overlap']:.3f})")


def cmd_purge(_args: argparse.Namespace) -> None:
    """Clear PVDB chunks and caches to ensure a clean slate."""
    payload = purge_system()
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(prog="chronorag")
    sub = parser.add_subparsers(dest="command")

    ingest_p = sub.add_parser("ingest")
    ingest_p.add_argument("paths", nargs="*")
    ingest_p.add_argument("--text", nargs="*", help="Inline text blobs")
    ingest_p.add_argument("--provenance")
    ingest_p.set_defaults(func=cmd_ingest)

    retrieve_p = sub.add_parser("retrieve")
    retrieve_p.add_argument("--query", required=True)
    retrieve_p.add_argument("--mode", default="INTELLIGENT")
    retrieve_p.add_argument("--axis", default="valid")
    retrieve_p.add_argument("--top-k", dest="top_k", type=int, default=5)
    retrieve_p.set_defaults(func=cmd_retrieve)

    purge_p = sub.add_parser("purge")
    purge_p.set_defaults(func=cmd_purge)

    answer_p = sub.add_parser("answer")
    answer_p.add_argument("--query", required=True)
    answer_p.add_argument("--mode", default="INTELLIGENT")
    answer_p.add_argument("--axis", default="valid")
    answer_p.set_defaults(func=cmd_answer)

    return parser


def main() -> None:
    """CLI entry point invoked via `python -m cli.chronorag_cli ...`."""
    get_app_state()  # ensure initialization
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
