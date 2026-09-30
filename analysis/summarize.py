"""Aggregate eval JSONs into a cross-seed summary table for the report.

Groups the standard Team-A-versus-baseline evaluations by condition and
opponent, then reports mean ± sample standard deviation across seeds.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

METRICS = ("win_rate_A", "win_rate_B", "draw_rate", "mean_return_A", "var_return_A", "mean_ep_len")
SEED_RE = re.compile(r"_seed\d+")


def condition_of(run: str) -> str:
    """'2v2_seed1_1M' -> '2v2_1M'; the seed is the aggregation axis."""
    return SEED_RE.sub("", run)


def group_evals(results_dir: str = "results") -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
        with open(path) as f:
            data = json.load(f)
        if "run" not in data or "opponent" not in data:
            continue  # legacy records without the grouping keys
        # Paired Team-A-baseline files are valuable for side diagnostics but
        # are not independent seeds and must not double-count the headline
        # Team-A-versus-Team-B baseline summary.
        if data["opponent"] != "self" and data.get("opponent_team") == "A":
            continue
        key = (condition_of(data["run"]), data["opponent"])
        groups.setdefault(key, []).append(data)
    return groups


def aggregate(groups: dict[tuple[str, str], list[dict]]) -> dict[tuple[str, str], dict]:
    out = {}
    for key, runs in groups.items():
        agg: dict[str, object] = {"n_seeds": len(runs)}
        for m in METRICS:
            vals = np.asarray([r[m] for r in runs if m in r], dtype=np.float64)
            if len(vals):
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                agg[m] = (float(vals.mean()), std)
        overlaps = [
            r["specialization"]["A"]["coverage_overlap"]
            for r in runs
            if "coverage_overlap" in r.get("specialization", {}).get("A", {})
        ]
        if overlaps:
            vals = np.asarray(overlaps, dtype=np.float64)
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            agg["coverage_overlap_A"] = (float(vals.mean()), std)
        out[key] = agg
    return out


def format_table(agg: dict[tuple[str, str], dict]) -> str:
    header = (
        "| condition | opponent | seeds | win A | win B | draw | return A | " "ep len | A overlap |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|"]
    for (cond, opp), a in sorted(agg.items()):

        def cell(metric, a=a):
            if metric not in a:
                return "-"
            mean, std = a[metric]
            return f"{mean:.2f} ± {std:.2f}" if a["n_seeds"] > 1 else f"{mean:.2f}"

        lines.append(
            f"| {cond} | {opp} | {a['n_seeds']} | {cell('win_rate_A')} | "
            f"{cell('win_rate_B')} | {cell('draw_rate')} | {cell('mean_return_A')} | "
            f"{cell('mean_ep_len')} | {cell('coverage_overlap_A')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=os.path.join("results", "summary.md"))
    args = ap.parse_args()
    table = format_table(aggregate(group_evals(args.results_dir)))
    print(table)
    with open(args.out, "w") as f:
        f.write(table)


if __name__ == "__main__":
    main()
