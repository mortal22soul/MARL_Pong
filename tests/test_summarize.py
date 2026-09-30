"""Tests for cross-seed summary aggregation of eval records."""

import json
import os

from analysis.summarize import aggregate, condition_of, format_table, group_evals


def _write_eval(results_dir, run, opponent, **overrides):
    data = {
        "mode": "2v2",
        "episodes": 60,
        "win_rate_A": 0.5,
        "win_rate_B": 0.4,
        "draw_rate": 0.1,
        "mean_return_A": 0.2,
        "var_return_A": 0.9,
        "mean_ep_len": 1500.0,
        "specialization": {"A": {"coverage_overlap": 0.3}, "B": {}},
        "run": run,
        "opponent": opponent,
    }
    data.update(overrides)
    suffix = "" if opponent == "self" else f"_vs_{opponent}"
    if data.get("opponent_team") == "A":
        suffix += "_teamA"
    path = os.path.join(str(results_dir), f"eval_{run}{suffix}.json")
    with open(path, "w") as f:
        json.dump(data, f)


def test_condition_of_strips_seed():
    assert condition_of("2v2_seed1_1M") == "2v2_1M"
    assert condition_of("1v1_seed0") == "1v1"
    assert condition_of("2v2_seed0_cont1M") == "2v2_cont1M"


def test_group_aggregate_and_format(tmp_path):
    _write_eval(tmp_path, "2v2_seed0_1M", "self", win_rate_A=0.4)
    _write_eval(tmp_path, "2v2_seed1_1M", "self", win_rate_A=0.6)
    _write_eval(tmp_path, "2v2_seed0_1M", "random", win_rate_A=0.8)
    # Legacy record (no run/opponent keys) must be ignored, not crash.
    with open(os.path.join(str(tmp_path), "eval_old.json"), "w") as f:
        json.dump({"win_rate_A": 0.9}, f)

    groups = group_evals(str(tmp_path))
    assert sorted(groups.keys()) == [("2v2_1M", "random"), ("2v2_1M", "self")]
    assert len(groups[("2v2_1M", "self")]) == 2

    agg = aggregate(groups)
    self_agg = agg[("2v2_1M", "self")]
    assert self_agg["n_seeds"] == 2
    mean, std = self_agg["win_rate_A"]
    assert mean == 0.5 and round(std, 3) == 0.141
    assert self_agg["coverage_overlap_A"] == (0.3, 0.0)

    table = format_table(agg)
    assert "| 2v2_1M | self | 2 | 0.50 ± 0.14 |" in table
    assert "| 2v2_1M | random | 1 | 0.80 |" in table


def test_group_evals_excludes_paired_team_a_baseline(tmp_path):
    _write_eval(tmp_path, "2v2_seed0_1M", "random", opponent_team="B")
    _write_eval(tmp_path, "2v2_seed0_1M", "random", opponent_team="A")
    groups = group_evals(str(tmp_path))
    assert len(groups[("2v2_1M", "random")]) == 1
