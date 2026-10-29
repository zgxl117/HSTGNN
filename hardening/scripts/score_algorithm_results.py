from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SAC_ROOT = Path(r"D:\OneDrive\Desktop\SAC_parallel_gpu_bak")
DEFAULT_GREEDY_DIR = Path(
    r"D:\OneDrive\Desktop\greedy_hardening_package\runs\current_no_gnn_greedy\greedy\seed_deterministic"
)
DEFAULT_PURE_MOEA_OUTPUT = Path(r"D:\OneDrive\Desktop\rv_data\output")
DEFAULT_REGISTER_CSV = DEFAULT_SAC_ROOT / "data" / "merged_registers.csv"


@dataclass(frozen=True)
class ScoreConfig:
    area_limit_pct: float = 20.0
    power_limit_pct: float = 50.0
    slack_min_ns: float = 0.0
    slack_ref_ns: float = 0.1
    ser_weight: float = 90.0
    area_margin_weight: float = 5.0
    power_margin_weight: float = 3.0
    slack_margin_weight: float = 2.0
    pareto_best_weight: float = 55.0
    pareto_hv_weight: float = 25.0
    pareto_diversity_weight: float = 10.0
    pareto_feasible_ratio_weight: float = 10.0
    overall_pareto_weight: float = 0.65
    overall_final_weight: float = 0.35
    diversity_ref_count: int = 50


STANDARD_COLUMNS = [
    "algorithm",
    "run",
    "source_file",
    "solution_index",
    "solution_id",
    "tmr_count",
    "residual_ser",
    "residual_ratio",
    "ser_reduction_pct",
    "area_total",
    "power_total",
    "worst_slack",
    "area_overhead",
    "power_overhead",
    "area_overhead_pct",
    "power_overhead_pct",
    "source_feasible",
    "genes",
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if pd.isna(value):
        return 0.0
    return max(low, min(high, float(value)))


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "ok"}


def count_input_registers(register_csv: Path | None) -> int | None:
    if not register_csv or not register_csv.exists():
        return None
    try:
        return int(pd.read_csv(register_csv, usecols=[0]).shape[0])
    except Exception:
        return None


def gene_key(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value))


def count_register_rows(path: str | Path, source: str) -> int | None:
    file_path = Path(path)
    if not path or not file_path.exists():
        return None
    try:
        if source == "sac_moea":
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            header_idx = None
            for idx, line in enumerate(lines):
                if line.strip().startswith("index,"):
                    header_idx = idx
                    break
            if header_idx is None:
                return None
            if header_idx + 1 >= len(lines):
                return 0
            return int(pd.read_csv(file_path, skiprows=header_idx).shape[0])
        return int(pd.read_csv(file_path).shape[0])
    except Exception:
        return None


def latest_sac_result_dir(root: Path) -> Path:
    result_root = root / "results"
    candidates = []
    if result_root.exists():
        for path in result_root.glob("sac_moea_*"):
            if (path / "sac_moea_pareto_front.csv").exists():
                candidates.append(path)
    if not candidates:
        log_root = root / "logs"
        if log_root.exists():
            for path in log_root.glob("sac_moea_*"):
                if (path / "sac_moea_pareto_front.csv").exists():
                    candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No SAC+MOEA result directory found below {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[STANDARD_COLUMNS].copy()


def load_sac_or_greedy(path: Path, algorithm: str, run: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, dtype=str, low_memory=False)
    df = pd.DataFrame(index=raw.index)
    df["algorithm"] = algorithm
    df["run"] = run
    df["source_file"] = str(path)
    df["solution_index"] = raw.get("eval_index", raw.get("solution_index", pd.Series(np.arange(len(raw)))))
    df["solution_id"] = raw.get("solution_id", df["solution_index"].astype(str))
    df["tmr_count"] = raw.get("tmr_count", np.nan)
    df["residual_ser"] = raw.get("residual_ser", np.nan)
    df["residual_ratio"] = raw.get("residual_ratio", np.nan)
    df["ser_reduction_pct"] = raw.get("ser_reduction_pct", np.nan)
    df["area_total"] = raw.get("area_total", np.nan)
    df["power_total"] = raw.get("power_total", np.nan)
    df["worst_slack"] = raw.get("worst_slack", np.nan)
    df["area_overhead"] = raw.get("area_overhead", np.nan)
    df["power_overhead"] = raw.get("power_overhead", np.nan)
    df["area_overhead_pct"] = raw.get("area_overhead_pct", np.nan)
    df["power_overhead_pct"] = raw.get("power_overhead_pct", np.nan)
    df["source_feasible"] = raw.get("feasible", np.nan)
    df["genes"] = raw.get("genes", np.nan)
    return ensure_standard_columns(df)


def load_pure_moea(output_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(output_dir.glob("run_*_seed_*/pareto_comparison.csv")):
        raw = pd.read_csv(path, dtype=str, low_memory=False)
        df = pd.DataFrame(index=raw.index)
        df["algorithm"] = "pure_moea"
        df["run"] = path.parent.name
        df["source_file"] = str(path)
        df["solution_index"] = raw.get("solution_index", pd.Series(np.arange(len(raw))))
        df["solution_id"] = df["run"].astype(str) + "_pareto_" + df["solution_index"].astype(str)
        df["tmr_count"] = raw.get("protected_count", np.nan)
        df["residual_ser"] = raw.get("SER", np.nan)
        df["ser_reduction_pct"] = raw.get("SER_reduction(%)", np.nan)
        df["residual_ratio"] = 1.0 - pd.to_numeric(df["ser_reduction_pct"], errors="coerce") / 100.0
        df["area_total"] = raw.get("area(um2)", np.nan)
        df["power_total"] = raw.get("power(mW)", np.nan)
        df["worst_slack"] = raw.get("WNS(ns)", np.nan)
        df["area_overhead"] = pd.to_numeric(raw.get("area_increase(%)", np.nan), errors="coerce") / 100.0
        df["power_overhead"] = pd.to_numeric(raw.get("power_increase(%)", np.nan), errors="coerce") / 100.0
        df["area_overhead_pct"] = raw.get("area_increase(%)", np.nan)
        df["power_overhead_pct"] = raw.get("power_increase(%)", np.nan)
        df["source_feasible"] = raw.get("WNS_satisfied", np.nan)
        df["genes"] = np.nan
        frames.append(ensure_standard_columns(df))
    if not frames:
        raise FileNotFoundError(f"No pure MOEA pareto_comparison.csv files found below {output_dir}")
    return pd.concat(frames, ignore_index=True)


def add_solution_scores(df: pd.DataFrame, config: ScoreConfig) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "solution_index",
        "tmr_count",
        "residual_ser",
        "residual_ratio",
        "ser_reduction_pct",
        "area_total",
        "power_total",
        "worst_slack",
        "area_overhead",
        "power_overhead",
        "area_overhead_pct",
        "power_overhead_pct",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["source_feasible"] = out["source_feasible"].map(normalize_bool)
    out["ppa_feasible"] = (
        (out["area_overhead_pct"] <= config.area_limit_pct)
        & (out["power_overhead_pct"] <= config.power_limit_pct)
        & (out["worst_slack"] >= config.slack_min_ns)
    )

    ser_norm = (out["ser_reduction_pct"] / 100.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    area_margin = ((config.area_limit_pct - out["area_overhead_pct"]) / config.area_limit_pct).clip(
        lower=0.0, upper=1.0
    ).fillna(0.0)
    power_margin = ((config.power_limit_pct - out["power_overhead_pct"]) / config.power_limit_pct).clip(
        lower=0.0, upper=1.0
    ).fillna(0.0)
    slack_margin = ((out["worst_slack"] - config.slack_min_ns) / config.slack_ref_ns).clip(
        lower=0.0, upper=1.0
    ).fillna(0.0)

    raw_score = (
        config.ser_weight * ser_norm
        + config.area_margin_weight * area_margin
        + config.power_margin_weight * power_margin
        + config.slack_margin_weight * slack_margin
    )
    out["ser_component"] = config.ser_weight * ser_norm
    out["area_margin_component"] = config.area_margin_weight * area_margin
    out["power_margin_component"] = config.power_margin_weight * power_margin
    out["slack_margin_component"] = config.slack_margin_weight * slack_margin
    out["solution_score"] = np.where(out["ppa_feasible"], raw_score, 0.0)
    out["constraint_violation_count"] = (
        (out["area_overhead_pct"] > config.area_limit_pct).astype(int)
        + (out["power_overhead_pct"] > config.power_limit_pct).astype(int)
        + (out["worst_slack"] < config.slack_min_ns).astype(int)
    )
    return out


def nondominated_min(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if not keep[i]:
            continue
        dominated_by_other = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        dominated_by_other[i] = False
        if np.any(dominated_by_other):
            keep[i] = False
    return points[keep]


def hypervolume_2d_min(points: np.ndarray, ref=(1.0, 1.0)) -> float:
    if len(points) == 0:
        return 0.0
    pts = points[np.all(np.isfinite(points), axis=1)]
    pts = pts[(pts[:, 0] <= ref[0]) & (pts[:, 1] <= ref[1]) & (pts[:, 0] >= 0.0) & (pts[:, 1] >= 0.0)]
    if len(pts) == 0:
        return 0.0
    pts = nondominated_min(pts)
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += max(ref[0] - x, 0.0) * max(prev_y - y, 0.0)
            prev_y = y
    return clamp(hv, 0.0, 1.0)


def find_best_register_file(algorithm: str, best_row: pd.Series, sac_dir: Path, greedy_dir: Path) -> tuple[str, int, str]:
    if best_row is None or best_row.empty:
        return "", 0, "no_best_row"

    if algorithm == "sac_moea":
        reg_dir = sac_dir / "final_solution_registers"
        if not reg_dir.exists():
            reg_dir = sac_dir / "final_solution_registers_from_pareto"
        files = sorted(p for p in reg_dir.glob("*.csv") if p.name.lower() != "summary.csv") if reg_dir.exists() else []
        summary = reg_dir / "summary.csv"
        best_genes = gene_key(best_row.get("genes"))
        best_count = safe_float(best_row.get("tmr_count"), default=np.nan)
        if summary.exists() and best_genes:
            try:
                summary_df = pd.read_csv(summary, dtype=str, low_memory=False)
                summary_df["_gene_key"] = summary_df.get("genes", "").map(gene_key)
                summary_df["_reinforced_count"] = pd.to_numeric(
                    summary_df.get("reinforced_count", np.nan), errors="coerce"
                )
                matches = summary_df[summary_df["_gene_key"] == best_genes]
                if np.isfinite(best_count):
                    count_matches = matches[matches["_reinforced_count"] == int(best_count)]
                    if not count_matches.empty:
                        file = reg_dir / str(count_matches.iloc[0].get("file", ""))
                        if file.exists():
                            return str(file), len(files), "matched_genes_and_count"
                if not matches.empty:
                    file = reg_dir / str(matches.iloc[0].get("file", ""))
                    if file.exists():
                        return str(file), len(files), "matched_genes"
                if np.isfinite(best_count):
                    count_matches = summary_df[summary_df["_reinforced_count"] == int(best_count)]
                    if not count_matches.empty:
                        file = reg_dir / str(count_matches.iloc[0].get("file", ""))
                        if file.exists():
                            return str(file), len(files), "matched_count_only"
            except Exception:
                pass
        return (str(files[0]) if files else "", len(files), "directory_available" if files else "missing")

    if algorithm == "greedy":
        candidates = [
            greedy_dir / "selected_registers.csv",
            greedy_dir / "selected_registers_best.csv",
        ]
        files = [p for p in candidates if p.exists()]
        return (str(files[0]) if files else "", len(files), "known_greedy_file" if files else "missing")

    if algorithm == "pure_moea":
        source = Path(str(best_row.get("source_file", "")))
        run_dir = source.parent
        idx = safe_float(best_row.get("solution_index"), default=np.nan)
        tmr = safe_float(best_row.get("tmr_count"), default=np.nan)
        files = []
        solutions_dir = run_dir / "solutions"
        if solutions_dir.exists():
            files = sorted(solutions_dir.glob("*.csv"))
            if np.isfinite(idx) and np.isfinite(tmr):
                exact = solutions_dir / f"pareto_{int(idx):03d}_protect_{int(tmr)}.csv"
                if exact.exists():
                    return str(exact), len(files), "matched_solution_index"
                for file in files:
                    if f"protect_{int(tmr)}" in file.name:
                        return str(file), len(files), "matched_tmr_count"
        return (str(files[0]) if files else "", len(files), "directory_available" if files else "missing")

    return "", 0, "unknown_algorithm"


def summarize_algorithm(
    scored: pd.DataFrame,
    algorithm: str,
    config: ScoreConfig,
    sac_dir: Path,
    greedy_dir: Path,
) -> dict:
    part = scored[scored["algorithm"] == algorithm].copy()
    feasible = part[part["ppa_feasible"]].copy()
    total_count = int(len(part))
    feasible_count = int(len(feasible))
    raw_feasible_ratio = feasible_count / total_count if total_count else 0.0
    # PPAR is treated as a hard gate for Pareto scoring. Infeasible candidates
    # are kept only for diagnostics and do not affect the constrained score.
    score_feasible_ratio = 1.0 if feasible_count else 0.0

    if feasible_count:
        best = feasible.sort_values(
            ["solution_score", "ser_reduction_pct", "area_overhead_pct", "power_overhead_pct"],
            ascending=[False, False, True, True],
        ).iloc[0]
        best_score = float(best["solution_score"])
    else:
        best = part.sort_values(["solution_score"], ascending=[False]).iloc[0] if total_count else pd.Series()
        best_score = 0.0

    hv_points = np.column_stack(
        [
            pd.to_numeric(feasible["area_overhead_pct"], errors="coerce").to_numpy() / config.area_limit_pct,
            pd.to_numeric(feasible["residual_ratio"], errors="coerce").to_numpy(),
        ]
    )
    hv2d = hypervolume_2d_min(hv_points, ref=(1.0, 1.0))

    nd_points = np.column_stack(
        [
            pd.to_numeric(feasible["residual_ratio"], errors="coerce").to_numpy(),
            pd.to_numeric(feasible["area_overhead_pct"], errors="coerce").to_numpy() / config.area_limit_pct,
            pd.to_numeric(feasible["power_overhead_pct"], errors="coerce").to_numpy() / config.power_limit_pct,
        ]
    )
    nd_count = int(len(nondominated_min(nd_points[np.all(np.isfinite(nd_points), axis=1)]))) if feasible_count else 0
    diversity_score = clamp(nd_count / config.diversity_ref_count)

    pareto_score = (
        config.pareto_best_weight * clamp(best_score / 100.0)
        + config.pareto_hv_weight * hv2d
        + config.pareto_diversity_weight * diversity_score
        + config.pareto_feasible_ratio_weight * score_feasible_ratio
    )
    final_solution_score = best_score
    overall_score = config.overall_pareto_weight * pareto_score + config.overall_final_weight * final_solution_score

    register_file, register_file_count, register_match = find_best_register_file(algorithm, best, sac_dir, greedy_dir)
    actual_register_count = count_register_rows(register_file, algorithm)
    expected_register_count = safe_float(best.get("tmr_count"), default=np.nan)
    if actual_register_count is not None and np.isfinite(expected_register_count):
        register_count_match = int(actual_register_count) == int(expected_register_count)
    else:
        register_count_match = False
    register_integrity_score = 100.0 if register_count_match else (
        70.0 if register_match.startswith("matched") else (50.0 if register_file_count > 0 else 0.0)
    )
    final_register_list_score = final_solution_score * (register_integrity_score / 100.0)
    overall_score = config.overall_pareto_weight * pareto_score + config.overall_final_weight * final_register_list_score

    return {
        "algorithm": algorithm,
        "overall_score": overall_score,
        "pareto_score": pareto_score,
        "final_solution_score": final_solution_score,
        "final_register_list_score": final_register_list_score,
        "register_integrity_score": register_integrity_score,
        "total_candidates": total_count,
        "ppa_feasible_candidates": feasible_count,
        "ppa_feasible_ratio": score_feasible_ratio,
        "raw_ppa_feasible_ratio": raw_feasible_ratio,
        "feasible_nondominated_count": nd_count,
        "hv2d_area_residual": hv2d,
        "best_solution_id": best.get("solution_id", ""),
        "best_run": best.get("run", ""),
        "best_tmr_count": best.get("tmr_count", np.nan),
        "best_ser_reduction_pct": best.get("ser_reduction_pct", np.nan),
        "best_residual_ratio": best.get("residual_ratio", np.nan),
        "best_area_overhead_pct": best.get("area_overhead_pct", np.nan),
        "best_power_overhead_pct": best.get("power_overhead_pct", np.nan),
        "best_worst_slack": best.get("worst_slack", np.nan),
        "best_source_file": best.get("source_file", ""),
        "best_register_file": register_file,
        "register_file_count": register_file_count,
        "register_match_status": register_match,
        "expected_register_count": expected_register_count,
        "actual_register_count": actual_register_count,
        "register_count_match": register_count_match,
    }


def build_best_solution_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm, part in scored.groupby("algorithm"):
        feasible = part[part["ppa_feasible"]].copy()
        if feasible.empty:
            best = part.sort_values("solution_score", ascending=False).iloc[0]
        else:
            best = feasible.sort_values(
                ["solution_score", "ser_reduction_pct", "area_overhead_pct", "power_overhead_pct"],
                ascending=[False, False, True, True],
            ).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).sort_values("solution_score", ascending=False)


def build_final_register_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "algorithm",
        "final_register_list_score",
        "final_solution_score",
        "register_integrity_score",
        "best_register_file",
        "register_match_status",
        "expected_register_count",
        "actual_register_count",
        "register_count_match",
        "best_ser_reduction_pct",
        "best_residual_ratio",
        "best_area_overhead_pct",
        "best_power_overhead_pct",
        "best_worst_slack",
        "best_tmr_count",
    ]
    return summary_df[cols].copy().sort_values("final_register_list_score", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score SAC+MOEA, greedy, and pure MOEA results under a common PPA-constrained SER objective."
    )
    parser.add_argument("--sac_root", type=Path, default=DEFAULT_SAC_ROOT)
    parser.add_argument("--sac_result_dir", type=Path, default=None)
    parser.add_argument("--greedy_dir", type=Path, default=DEFAULT_GREEDY_DIR)
    parser.add_argument("--pure_moea_output", type=Path, default=DEFAULT_PURE_MOEA_OUTPUT)
    parser.add_argument("--register_csv", type=Path, default=DEFAULT_REGISTER_CSV)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--area_limit_pct", type=float, default=20.0)
    parser.add_argument("--power_limit_pct", type=float, default=50.0)
    parser.add_argument("--slack_min_ns", type=float, default=0.0)
    parser.add_argument("--slack_ref_ns", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sac_dir = args.sac_result_dir or latest_sac_result_dir(args.sac_root)
    out_dir = args.out_dir or (
        args.sac_root / "results" / f"algorithm_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ScoreConfig(
        area_limit_pct=args.area_limit_pct,
        power_limit_pct=args.power_limit_pct,
        slack_min_ns=args.slack_min_ns,
        slack_ref_ns=args.slack_ref_ns,
    )

    sac_pareto = load_sac_or_greedy(sac_dir / "sac_moea_pareto_front.csv", "sac_moea", sac_dir.name)
    greedy_pareto = load_sac_or_greedy(args.greedy_dir / "pareto_front.csv", "greedy", args.greedy_dir.name)
    pure_moea = load_pure_moea(args.pure_moea_output)

    combined = pd.concat([sac_pareto, greedy_pareto, pure_moea], ignore_index=True)
    scored = add_solution_scores(combined, config)

    summaries = [
        summarize_algorithm(scored, "sac_moea", config, sac_dir, args.greedy_dir),
        summarize_algorithm(scored, "greedy", config, sac_dir, args.greedy_dir),
        summarize_algorithm(scored, "pure_moea", config, sac_dir, args.greedy_dir),
    ]
    summary_df = pd.DataFrame(summaries).sort_values("overall_score", ascending=False)
    best_df = build_best_solution_table(scored)
    final_register_df = build_final_register_table(summary_df)
    scored_feasible = scored[scored["ppa_feasible"]].copy()

    score_cols = [
        "algorithm",
        "run",
        "solution_id",
        "solution_index",
        "tmr_count",
        "ppa_feasible",
        "solution_score",
        "ser_component",
        "area_margin_component",
        "power_margin_component",
        "slack_margin_component",
        "ser_reduction_pct",
        "residual_ratio",
        "area_overhead_pct",
        "power_overhead_pct",
        "worst_slack",
        "constraint_violation_count",
        "source_file",
    ]

    summary_path = out_dir / "algorithm_scores.csv"
    best_path = out_dir / "best_solutions_comparison.csv"
    all_scores_path = out_dir / "pareto_solution_scores.csv"
    diagnostic_scores_path = out_dir / "all_candidate_solution_scores.csv"
    register_scores_path = out_dir / "final_register_list_scores.csv"
    config_path = out_dir / "scoring_config.json"

    summary_df.to_csv(summary_path, index=False)
    best_df[score_cols].to_csv(best_path, index=False)
    scored_feasible[score_cols].to_csv(all_scores_path, index=False)
    scored[score_cols].to_csv(diagnostic_scores_path, index=False)
    final_register_df.to_csv(register_scores_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "sac_result_dir": str(sac_dir),
                "greedy_dir": str(args.greedy_dir),
                "pure_moea_output": str(args.pure_moea_output),
                "register_csv": str(args.register_csv),
                "register_count": count_input_registers(args.register_csv),
                "score_notes": {
                    "solution_score": "PPA-feasible solutions only. 90% SER reduction, 5% area margin, 3% power margin, 2% slack margin.",
                    "pareto_score": "PPAR-feasible solutions only. 55% best feasible solution score, 25% feasible 2D hypervolume on area/residual, 10% feasible nondominated diversity, 10% feasibility gate.",
                    "final_register_list_score": "final_solution_score multiplied by register-list integrity.",
                    "overall_score": "65% pareto_score plus 35% final_register_list_score.",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    display_cols = [
        "algorithm",
        "overall_score",
        "pareto_score",
        "final_solution_score",
        "final_register_list_score",
        "best_ser_reduction_pct",
        "best_area_overhead_pct",
        "best_power_overhead_pct",
        "best_worst_slack",
        "best_tmr_count",
        "ppa_feasible_ratio",
        "hv2d_area_residual",
    ]
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {best_path}")
    print(f"Wrote: {all_scores_path}")
    print(f"Wrote: {diagnostic_scores_path}")
    print(f"Wrote: {register_scores_path}")
    print(f"Wrote: {config_path}")


if __name__ == "__main__":
    main()
