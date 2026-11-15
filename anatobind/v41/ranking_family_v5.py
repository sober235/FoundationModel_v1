"""The comparisons of several ranking runs as one family (amendment of 2026-11-15 to the ranking audit).

Each checkpoint is read by its own `rank` command (real_unified_v5), so the adjustment over the family can only be
made afterwards. This entry reads the summaries of those commands (counts and distributions, no identifier), and
the summaries of their preflights when given, checks that they were read alike, and writes the family: effect,
patient interval, raw and Holm-adjusted p and the reading of every development comparison
(evaluation.findings_ranking_v5.ranking_family_v5). It reads no sample and builds no model.

  python -m anatobind.v41.ranking_family_v5 --entry R1=<summary.json> ... [--preflight R1=<summary.json> ...] \\
      --out <new private directory>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .contracts_v5 import V5Error
from .evaluation.findings_ranking_v5 import FAMILY_SIZE, ranking_family_v5, render_ranking_family_md_v5
from .paths import resolve_private_output, write_private_json

SETTINGS = ("n_placements", "n_bootstrap", "seed", "roles")


def build_ranking_family_v5(summaries: dict, preflights: dict | None = None, *,
                            family_size: int = FAMILY_SIZE) -> dict:
    """`summaries`: label -> the summary of a `rank` command; `preflights`: label -> the summary of its preflight,
    or None when no preflight is required. The commands must have been read with the same settings and the same
    code, each under another checkpoint, and must name the same single box-supervised member."""
    if not summaries or any(summary.get("kind") != "unified_ranking" for summary in summaries.values()):
        raise V5Error("FAMILY_ENTRY_INVALID", "$.entries")
    first = next(iter(summaries.values()))
    for summary in summaries.values():
        if any(summary["evaluation"][key] != first["evaluation"][key] for key in SETTINGS) \
                or summary["evaluation"]["code"] != first["evaluation"]["code"] \
                or summary["evaluation"].get("preflight_scans") is not None:
            raise V5Error("FAMILY_SETTINGS_DIFFER", "$.entries")
    digests = [summary["checkpoint"]["sha256"] for summary in summaries.values()]
    if len(set(digests)) != len(digests):
        raise V5Error("FAMILY_CHECKPOINT_REPEATED", "$.entries")
    members = {tuple(sorted(summary["members"])) for summary in summaries.values() if summary["members"]}
    if len(members) > 1 or any(len(names) != 1 for names in members):
        raise V5Error("FAMILY_MEMBERS_DIFFER", "$.entries")
    entries = []
    for label, summary in summaries.items():
        problems = []
        if summary["status"] != "COMPLETED" or not summary["members"]:
            problems.append("JOB_NOT_COMPLETED")
        if summary["checkpoint"]["weights_unchanged"] is not True:
            problems.append("WEIGHTS_NOT_VERIFIED")
        if preflights is not None:
            checked = preflights.get(label)
            passed = checked is not None and checked.get("kind") == "unified_ranking_preflight" \
                and checked["status"] == "COMPLETED" and checked["checkpoint"]["sha256"] == summary["checkpoint"]["sha256"] \
                and checked["checkpoint"]["weights_unchanged"] is True and bool(checked["members"]) \
                and all(item["preflight"]["passed"] for item in checked["members"].values())
            if not passed:
                problems.append("PREFLIGHT_NOT_PASSED")
        ranking = None
        if summary["status"] == "COMPLETED" and summary["members"]:
            by_role = next(iter(summary["members"].values()))
            ranking = {role: by_role[role]["ranking"] for role in summary["evaluation"]["roles"]}
        entries.append({"label": label, "checkpoint_sha256": summary["checkpoint"]["sha256"],
                        "step": summary["checkpoint"]["step"], "plan_id": summary["plan_id"], "problems": problems,
                        "ranking": ranking})
    family = ranking_family_v5(entries, family_size=family_size)
    family.update(settings={key: first["evaluation"][key] for key in SETTINGS},
                  code=dict(first["evaluation"]["code"]), preflight_required=preflights is not None,
                  scientific_conclusion="NOT_EVALUATED")
    return family


def _labelled(values: list, path: str) -> dict:
    out = {}
    for value in values:
        label, _, file = value.partition("=")
        if not label or not file or label in out:
            raise V5Error("FAMILY_ENTRY_INVALID", path)
        out[label] = json.loads(Path(file).read_text(encoding="utf-8"))
    return out


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anatobind.v41.ranking_family_v5")
    parser.add_argument("--entry", action="append", required=True, help="LABEL=summary.json of a rank command")
    parser.add_argument("--preflight", action="append", default=[], help="LABEL=summary.json of its preflight")
    parser.add_argument("--no-preflight", action="store_true", help="no preflight is required of the entries")
    parser.add_argument("--out", required=True, help="a new private directory")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    try:
        if args.no_preflight and args.preflight:
            raise V5Error("FAMILY_ENTRY_INVALID", "$.preflight")
        family = build_ranking_family_v5(_labelled(args.entry, "$.entries"),
                                         None if args.no_preflight else _labelled(args.preflight, "$.preflight"))
        out = Path(args.out)
        if out.exists():
            raise V5Error("EVAL_OUT_EXISTS", "$.out")
        target = resolve_private_output(out / "family.json", out, repo)
        out.mkdir(mode=0o700)                                           # never overwritten
        write_private_json(target, family)
        fd = os.open(out / "ranking_family.md", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(render_ranking_family_md_v5(family))
    except V5Error as error:
        print(json.dumps({"status": "FAILED", "code": error.code, "path": error.path}))
        return 1
    print(json.dumps({"status": "COMPLETED", "comparisons": sum(1 for row in family["comparisons"] if row["family"]),
                      "same_support": family["same_support"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
