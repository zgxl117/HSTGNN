from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from score_algorithm_results import (
    ScoreConfig,
    add_solution_scores,
    clamp,
    count_register_rows,
    hypervolume_2d_min,
)


OBJECTIVE_COLUMNS = ["residual_ratio", "area_overhead_pct", "power_overhead_pct"]


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        dominated = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        dominated[index] = False
        keep[index] = not np.any(dominated)
    return keep


def prepare_candidates(path: Path, config: ScoreConfig) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    frame = raw.copy()
    frame["run"] = frame.get("source_group", "")
    frame["solution_index"] = np.arange(len(frame))
    frame["solution_id"] = frame.get("candidate_id", frame["solution_index"])
    frame["source_feasible"] = frame.get("ppa_feasible", False)
    frame["area_overhead"] = pd.to_numeric(frame["area_overhead_pct"], errors="coerce") / 100.0
    frame["power_overhead"] = pd.to_numeric(frame["power_overhead_pct"], errors="coerce") / 100.0
    frame["genes"] = np.nan
    return add_solution_scores(frame, config)


def summarize_algorithm(
    part: pd.DataFrame,
    config: ScoreConfig,
    greedy_register_file: Path | None = None,
) -> tuple[dict, pd.DataFrame, pd.Series]:
    feasible = part[part["ppa_feasible"]].copy()
    if feasible.empty:
        raise ValueError(f"{part['algorithm'].iloc[0]} has no PPA-feasible candidate")

    best = feasible.sort_values(
        ["solution_score", "ser_reduction_pct", "area_overhead_pct", "power_overhead_pct"],
        ascending=[False, False, True, True],
    ).iloc[0]

    unique_feasible = feasible.sort_values("solution_score", ascending=False).drop_duplicates(
        OBJECTIVE_COLUMNS, keep="first"
    )
    points_3d = np.column_stack(
        [
            unique_feasible["residual_ratio"].to_numpy(dtype=float),
            unique_feasible["area_overhead_pct"].to_numpy(dtype=float) / config.area_limit_pct,
            unique_feasible["power_overhead_pct"].to_numpy(dtype=float) / config.power_limit_pct,
        ]
    )
    finite = np.all(np.isfinite(points_3d), axis=1)
    unique_feasible = unique_feasible.iloc[np.flatnonzero(finite)].copy()
    points_3d = points_3d[finite]
    front = unique_feasible.iloc[np.flatnonzero(nondominated_mask(points_3d))].copy()

    hv_points = np.column_stack(
        [
            front["area_overhead_pct"].to_numpy(dtype=float) / config.area_limit_pct,
            front["residual_ratio"].to_numpy(dtype=float),
        ]
    )
    hypervolume = hypervolume_2d_min(hv_points, ref=(1.0, 1.0))
    diversity_score = clamp(len(front) / config.diversity_ref_count)
    best_score = float(best["solution_score"])
    pareto_score = (
        config.pareto_best_weight * clamp(best_score / 100.0)
        + config.pareto_hv_weight * hypervolume
        + config.pareto_diversity_weight * diversity_score
        + config.pareto_feasible_ratio_weight
    )

    if best["algorithm"] == "greedy" and greedy_register_file is not None:
        source_file = greedy_register_file
    else:
        source_file = Path(str(best["source_file"]))
    actual_count = count_register_rows(source_file, str(best["algorithm"]))
    expected_count = int(best["tmr_count"])
    count_matches = actual_count == expected_count
    integrity_score = 100.0 if count_matches else (50.0 if source_file.exists() else 0.0)
    final_list_score = best_score * integrity_score / 100.0
    overall_score = (
        config.overall_pareto_weight * pareto_score
        + config.overall_final_weight * final_list_score
    )

    summary = {
        "algorithm": best["algorithm"],
        "overall_score": overall_score,
        "pareto_score": pareto_score,
        "final_solution_score": best_score,
        "final_register_list_score": final_list_score,
        "register_integrity_score": integrity_score,
        "candidate_count": len(part),
        "ppa_feasible_count": len(feasible),
        "raw_ppa_feasible_ratio": len(feasible) / len(part),
        "unique_feasible_count": len(unique_feasible),
        "feasible_pareto_count": len(front),
        "hv2d_area_residual": hypervolume,
        "best_candidate_id": best["candidate_id"],
        "best_tmr_count": expected_count,
        "best_ser_reduction_pct": best["ser_reduction_pct"],
        "best_residual_ratio": best["residual_ratio"],
        "best_area_overhead_pct": best["area_overhead_pct"],
        "best_power_overhead_pct": best["power_overhead_pct"],
        "best_worst_slack": best["worst_slack"],
        "best_source_file": str(source_file),
        "actual_register_count": actual_count,
        "register_count_match": count_matches,
    }
    return summary, front, best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score PPA-feasible Pareto fronts from unified candidate metrics."
    )
    parser.add_argument("--unified_candidates", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--area_limit_pct", type=float, default=20.0)
    parser.add_argument("--power_limit_pct", type=float, default=50.0)
    parser.add_argument("--slack_min_ns", type=float, default=0.0)
    parser.add_argument("--greedy_register_file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScoreConfig(
        area_limit_pct=args.area_limit_pct,
        power_limit_pct=args.power_limit_pct,
        slack_min_ns=args.slack_min_ns,
    )
    candidates = prepare_candidates(args.unified_candidates, config)

    summaries = []
    fronts = []
    best_rows = []
    for _, part in candidates.groupby("algorithm", sort=False):
        summary, front, best = summarize_algorithm(part, config, args.greedy_register_file)
        summaries.append(summary)
        fronts.append(front)
        best_rows.append(best)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries).sort_values("overall_score", ascending=False)
    front_df = pd.concat(fronts, ignore_index=True)
    best_df = pd.DataFrame(best_rows).sort_values("solution_score", ascending=False)
    summary_df.to_csv(args.out_dir / "unified_algorithm_scores.csv", index=False)
    front_df.to_csv(args.out_dir / "unified_feasible_pareto_fronts.csv", index=False)
    best_df.to_csv(args.out_dir / "unified_best_solutions.csv", index=False)

    columns = [
        "algorithm",
        "overall_score",
        "pareto_score",
        "final_solution_score",
        "hv2d_area_residual",
        "feasible_pareto_count",
        "best_ser_reduction_pct",
        "best_area_overhead_pct",
        "best_power_overhead_pct",
        "best_worst_slack",
        "best_tmr_count",
    ]
    print(summary_df[columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
