"""Summarize adaptive/narrow/fixed operator ablations under one score."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ACTION_COLUMNS = [
    "mutation_flips_spea2",
    "pc_spea2",
    "mutation_flips_nsga2",
    "pc_nsga2",
    "spea2_budget_share",
    "elite_injection_ratio",
    "local_search_strength",
    "restart_ratio",
]


def summarize_one(label: str, log_dir: Path, score_dir: Path) -> dict:
    history = pd.read_csv(log_dir / "all_parameters_history.csv")
    score = pd.read_csv(score_dir / "unified_algorithm_scores.csv")
    sac_score = score.loc[score["algorithm"].eq("sac_moea")].iloc[0]
    greedy_score = score.loc[score["algorithm"].eq("greedy")].iloc[0]

    attribution = pd.read_csv(log_dir.parent.parent / "results" / log_dir.name / "search_attribution.csv")
    convergence = pd.read_csv(log_dir / "convergence_history.csv")
    actions = history[ACTION_COLUMNS].astype(float)

    return {
        "variant": label,
        "log_dir": str(log_dir),
        "score_dir": str(score_dir),
        "overall_score": float(sac_score["overall_score"]),
        "pareto_score": float(sac_score["pareto_score"]),
        "final_solution_score": float(sac_score["final_solution_score"]),
        "hv2d_area_residual": float(sac_score["hv2d_area_residual"]),
        "feasible_pareto_count": int(sac_score["feasible_pareto_count"]),
        "best_ser_reduction_pct": float(sac_score["best_ser_reduction_pct"]),
        "best_area_overhead_pct": float(sac_score["best_area_overhead_pct"]),
        "best_power_overhead_pct": float(sac_score["best_power_overhead_pct"]),
        "best_tmr_count": int(sac_score["best_tmr_count"]),
        "greedy_overall_score": float(greedy_score["overall_score"]),
        "overall_gain_vs_greedy": float(sac_score["overall_score"] - greedy_score["overall_score"]),
        "pareto_gain_vs_greedy": float(sac_score["pareto_score"] - greedy_score["pareto_score"]),
        "total_moea_generations": int(history["moea_generations"].sum()),
        "sac_updates": int(history["sac_update_step"].max()),
        "mean_action_quality": float(history["action_quality"].mean()),
        "positive_action_quality_steps": int((history["action_quality"] > 0).sum()),
        "action_std_mean": float(actions.std(ddof=0).mean()),
        "action_std_max": float(actions.std(ddof=0).max()),
        "final_archive_hv": float(history["archive_hv"].iloc[-1]),
        "final_episode_hv": float(convergence["final_hv"].iloc[-1]),
        "mean_episode_generations": float(convergence["moea_generations"].mean()),
        "search_hv_gain": float(attribution["search_hv_gain"].iloc[0]),
        "nonbaseline_front_points": int(attribution["nonbaseline_front_points"].iloc[0]),
        "exploration_front_points": int(attribution["exploration_step_front_points"].iloc[0]),
        "policy_front_points": int(attribution["policy_step_front_points"].iloc[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "LOG_DIR", "SCORE_DIR"),
        required=True,
    )
    args = parser.parse_args()

    rows = []
    for label, log_name, score_name in args.run:
        rows.append(
            summarize_one(
                label,
                args.root / "logs" / log_name,
                args.root / "results" / score_name,
            )
        )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
