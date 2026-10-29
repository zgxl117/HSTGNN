from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SAC_ROOT = Path(r"D:\OneDrive\Desktop\SAC_parallel_gpu_bak")
DEFAULT_REGISTER_CSV = DEFAULT_SAC_ROOT / "data" / "merged_registers.csv"
DEFAULT_GREEDY_REGISTER_FILE = Path(
    r"D:\OneDrive\Desktop\greedy_hardening_package\runs\current_no_gnn_greedy\greedy\seed_deterministic\selected_registers.csv"
)
DEFAULT_GREEDY_PARETO_FILE = DEFAULT_GREEDY_REGISTER_FILE.parent / "pareto_front.csv"
DEFAULT_GREEDY_CANDIDATE_TABLE = DEFAULT_GREEDY_REGISTER_FILE.parents[2] / "candidate_topK.csv"
DEFAULT_PURE_MOEA_OUTPUT = Path(r"D:\OneDrive\Desktop\rv_data\output")


@dataclass(frozen=True)
class UnifiedConfig:
    voter_delay: float = 0.13
    voter_area: float = 0.5
    voter_power: float = 0.2
    area_limit_pct: float = 20.0
    power_limit_pct: float = 50.0
    slack_min_ns: float = 0.0
    residual_ratio_limit: float = 0.20
    slack_ref_ns: float = 0.10
    ser_weight: float = 90.0
    area_margin_weight: float = 5.0
    power_margin_weight: float = 3.0
    slack_margin_weight: float = 2.0


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if pd.isna(value):
        return 0.0
    return max(low, min(high, float(value)))


def read_merged_registers(path: Path) -> tuple[dict, pd.DataFrame]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        header = f.readline().strip().split(",")
        values = f.readline().strip().split(",")
    metrics = {key: float(value) for key, value in zip(header, values) if value != ""}

    df = pd.read_csv(path, skiprows=3, encoding="utf-8-sig")
    df.columns = [str(col).strip() for col in df.columns]
    numeric_cols = ["index", "area", "reg_power", "path_slack(ns)", "sensitivity"]
    for col in numeric_cols:
        if col not in df.columns:
            raise KeyError(f"{path} missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["index"]).copy()
    df["index"] = df["index"].astype(int)
    df["instance_name"] = df["instance_name"].astype(str)
    df = df.drop_duplicates("index", keep="first")
    return metrics, df


def latest_sac_result_dir(root: Path) -> Path:
    candidates = []
    result_root = root / "results"
    if result_root.exists():
        candidates.extend(
            path for path in result_root.glob("sac_moea_*") if (path / "final_solution_registers").exists()
        )
    if not candidates:
        raise FileNotFoundError(f"No SAC result directory with final_solution_registers found below {result_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_csv_header_line(path: Path, starts_with: str) -> int:
    lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith(starts_with):
            return idx
    raise ValueError(f"Could not find CSV header starting with {starts_with!r} in {path}")


def natural_key(path: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


def load_sac_register_candidates(sac_dir: Path) -> pd.DataFrame:
    reg_dir = sac_dir / "final_solution_registers"
    if not reg_dir.exists():
        raise FileNotFoundError(reg_dir)

    summary_path = reg_dir / "summary.csv"
    summary = pd.read_csv(summary_path, dtype=str, low_memory=False) if summary_path.exists() else pd.DataFrame()
    summary_by_file = {}
    if not summary.empty and "file" in summary.columns:
        for _, row in summary.iterrows():
            summary_by_file[str(row["file"])] = row.to_dict()

    rows = []
    for file in sorted((p for p in reg_dir.glob("*.csv") if p.name.lower() != "summary.csv"), key=natural_key):
        header_idx = find_csv_header_line(file, "index,")
        table = pd.read_csv(file, skiprows=header_idx, encoding="utf-8-sig")
        if "index" not in table.columns:
            continue
        indices = pd.to_numeric(table["index"], errors="coerce").dropna().astype(int).tolist()
        meta = summary_by_file.get(file.name, {})
        rows.append(
            {
                "algorithm": "sac_moea",
                "candidate_id": meta.get("solution_id", file.stem),
                "source_file": str(file),
                "source_group": meta.get("source", "final_solution_registers"),
                "selected_indices": indices,
                "selected_names": [],
                "reported_count": int(float(meta.get("reinforced_count", len(indices)))) if meta else len(indices),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No SAC register solution CSV files found in {reg_dir}")
    return pd.DataFrame(rows)


def load_greedy_register_candidate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, encoding="utf-8-sig")
    if "original_register_id" in table.columns:
        indices = pd.to_numeric(table["original_register_id"], errors="coerce").dropna().astype(int).tolist()
    elif "index" in table.columns:
        indices = pd.to_numeric(table["index"], errors="coerce").dropna().astype(int).tolist()
    else:
        raise KeyError(f"{path} does not contain original_register_id or index")
    return pd.DataFrame(
        [
            {
                "algorithm": "greedy",
                "candidate_id": "selected_registers",
                "source_file": str(path),
                "source_group": "final_solution",
                "selected_indices": indices,
                "selected_names": [],
                "reported_count": len(indices),
            }
        ]
    )


def load_greedy_gene_candidates(
    pareto_path: Path,
    candidate_table_path: Path,
) -> pd.DataFrame:
    if not pareto_path.exists():
        raise FileNotFoundError(pareto_path)
    if not candidate_table_path.exists():
        raise FileNotFoundError(candidate_table_path)

    pareto = pd.read_csv(pareto_path, dtype=str, low_memory=False)
    candidate_table = pd.read_csv(candidate_table_path, encoding="utf-8-sig")
    index_column = "original_register_id" if "original_register_id" in candidate_table.columns else "index"
    if index_column not in candidate_table.columns:
        raise KeyError(f"{candidate_table_path} does not contain original_register_id or index")
    candidate_indices = pd.to_numeric(candidate_table[index_column], errors="raise").astype(int).to_numpy()

    rows = []
    for row_index, row in pareto.iterrows():
        genes = str(row.get("genes", "")).strip()
        if len(genes) != len(candidate_indices) or set(genes) - {"0", "1"}:
            raise ValueError(
                f"Invalid greedy genes at row {row_index}: length={len(genes)}, expected={len(candidate_indices)}"
            )
        mask = np.fromiter((value == "1" for value in genes), dtype=bool, count=len(genes))
        selected_indices = candidate_indices[mask].tolist()
        rows.append(
            {
                "algorithm": "greedy",
                "candidate_id": row.get("solution_id", f"greedy_{row_index}"),
                "source_file": str(pareto_path),
                "source_group": row.get("source", "greedy_pareto"),
                "selected_indices": selected_indices,
                "selected_names": [],
                "reported_count": int(float(row.get("tmr_count", len(selected_indices)))),
            }
        )
    if not rows:
        raise ValueError(f"No greedy genes found in {pareto_path}")
    return pd.DataFrame(rows)


def load_pure_moea_register_candidates(output_dir: Path) -> pd.DataFrame:
    rows = []
    for file in sorted(output_dir.glob("run_*_seed_*/solutions/*.csv"), key=lambda p: (p.parent.parent.name, natural_key(p))):
        try:
            table = pd.read_csv(file, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()

        names = []
        if "register_path" in table.columns:
            names = table["register_path"].dropna().astype(str).tolist()

        count_match = re.search(r"protect_(\d+)", file.stem)
        reported_count = int(count_match.group(1)) if count_match else len(names)
        rows.append(
            {
                "algorithm": "pure_moea",
                "candidate_id": f"{file.parent.parent.name}_{file.stem}",
                "source_file": str(file),
                "source_group": file.parent.parent.name,
                "selected_indices": [],
                "selected_names": names,
                "reported_count": reported_count,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No pure MOEA solution CSV files found below {output_dir}")
    return pd.DataFrame(rows)


def resolve_selected_indices(
    row: pd.Series,
    register_df: pd.DataFrame,
    index_set: set[int],
    name_to_index: dict[str, int],
) -> tuple[list[int], int, int]:
    selected = []
    unknown = 0

    for idx in row.get("selected_indices", []) or []:
        idx = int(idx)
        if idx in index_set:
            selected.append(idx)
        else:
            unknown += 1

    for name in row.get("selected_names", []) or []:
        idx = name_to_index.get(str(name))
        if idx is None:
            unknown += 1
        else:
            selected.append(idx)

    unique_selected = sorted(set(selected))
    duplicate_count = len(selected) - len(unique_selected)
    return unique_selected, duplicate_count, unknown


def score_solution(metrics: dict, config: UnifiedConfig) -> tuple[bool, bool, float, float, float, float, float]:
    ppa_feasible = (
        metrics["area_overhead_pct"] <= config.area_limit_pct
        and metrics["power_overhead_pct"] <= config.power_limit_pct
        and metrics["worst_slack"] >= config.slack_min_ns
    )
    sens_feasible = metrics["residual_ratio"] <= config.residual_ratio_limit
    full_feasible = ppa_feasible and sens_feasible

    ser_norm = clamp(metrics["ser_reduction_pct"] / 100.0)
    area_margin = clamp((config.area_limit_pct - metrics["area_overhead_pct"]) / config.area_limit_pct)
    power_margin = clamp((config.power_limit_pct - metrics["power_overhead_pct"]) / config.power_limit_pct)
    slack_margin = clamp((metrics["worst_slack"] - config.slack_min_ns) / config.slack_ref_ns)
    score = (
        config.ser_weight * ser_norm
        + config.area_margin_weight * area_margin
        + config.power_margin_weight * power_margin
        + config.slack_margin_weight * slack_margin
    )
    if not ppa_feasible:
        score = 0.0
    return ppa_feasible, sens_feasible, full_feasible, score, area_margin, power_margin, slack_margin


def evaluate_candidates(
    candidates: pd.DataFrame,
    register_df: pd.DataFrame,
    design_metrics: dict,
    config: UnifiedConfig,
) -> pd.DataFrame:
    index_set = set(register_df["index"].astype(int).tolist())
    name_to_index = dict(zip(register_df["instance_name"].astype(str), register_df["index"].astype(int)))
    by_index = register_df.set_index("index", drop=False)

    base_area = float(design_metrics["total_area"])
    base_power = float(design_metrics["total_power"])
    base_slack = float(design_metrics["worst_slack(ns)"])
    base_ser = float(design_metrics["design_sensitivity"])
    total_sensitivity = float(register_df["sensitivity"].sum())
    total_registers = int(len(register_df))

    rows = []
    for _, row in candidates.iterrows():
        selected, duplicates, unknown = resolve_selected_indices(row, register_df, index_set, name_to_index)
        selected_df = by_index.loc[selected] if selected else by_index.iloc[0:0]

        tmr_count = int(len(selected))
        area_increase = float((2.0 * selected_df["area"] + config.voter_area).sum()) if tmr_count else 0.0
        power_increase = float((2.0 * selected_df["reg_power"] + config.voter_power).sum()) if tmr_count else 0.0
        area_total = base_area + area_increase
        power_total = base_power + power_increase

        slack_after = register_df["path_slack(ns)"].to_numpy(dtype=float).copy()
        if tmr_count:
            register_positions = by_index.loc[selected].index.to_series().map(
                {idx: pos for pos, idx in enumerate(register_df["index"].tolist())}
            )
            slack_after[register_positions.to_numpy(dtype=int)] -= config.voter_delay
        worst_slack = min(float(np.nanmin(slack_after)), base_slack)

        protected_sensitivity = float(selected_df["sensitivity"].sum()) if tmr_count else 0.0
        residual_ser = (total_sensitivity - protected_sensitivity) / (total_registers + 2.0 * tmr_count)
        residual_ratio = residual_ser / base_ser if base_ser > 0 else np.nan
        ser_reduction_pct = (1.0 - residual_ratio) * 100.0

        metrics = {
            "area_overhead_pct": (area_total / base_area - 1.0) * 100.0,
            "power_overhead_pct": (power_total / base_power - 1.0) * 100.0,
            "worst_slack": worst_slack,
            "residual_ratio": residual_ratio,
            "ser_reduction_pct": ser_reduction_pct,
        }
        ppa_feasible, sens_feasible, full_feasible, unified_score, area_margin, power_margin, slack_margin = score_solution(
            metrics, config
        )

        rows.append(
            {
                "algorithm": row["algorithm"],
                "candidate_id": row["candidate_id"],
                "source_group": row["source_group"],
                "source_file": row["source_file"],
                "reported_count": row["reported_count"],
                "tmr_count": tmr_count,
                "duplicate_count": duplicates,
                "unknown_count": unknown,
                "area_total": area_total,
                "power_total": power_total,
                "worst_slack": worst_slack,
                "residual_ser": residual_ser,
                "residual_ratio": residual_ratio,
                "ser_reduction_pct": ser_reduction_pct,
                "area_overhead_pct": metrics["area_overhead_pct"],
                "power_overhead_pct": metrics["power_overhead_pct"],
                "ppa_feasible": ppa_feasible,
                "sens_feasible": sens_feasible,
                "full_feasible": full_feasible,
                "unified_score": unified_score,
                "area_margin_norm": area_margin,
                "power_margin_norm": power_margin,
                "slack_margin_norm": slack_margin,
            }
        )
    return pd.DataFrame(rows)


def best_by_algorithm(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm, part in scored.groupby("algorithm"):
        valid = part[(part["unknown_count"] == 0) & (part["ppa_feasible"])].copy()
        if valid.empty:
            valid = part[part["unknown_count"] == 0].copy()
        if valid.empty:
            valid = part.copy()
        best = valid.sort_values(
            ["unified_score", "ser_reduction_pct", "area_overhead_pct", "power_overhead_pct"],
            ascending=[False, False, True, True],
        ).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).sort_values("unified_score", ascending=False)


def summarize(scored: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    best_by_name = {row["algorithm"]: row for _, row in best.iterrows()}
    for algorithm, part in scored.groupby("algorithm"):
        clean = part[part["unknown_count"] == 0]
        best_row = best_by_name[algorithm]
        rows.append(
            {
                "algorithm": algorithm,
                "candidate_count": int(len(part)),
                "clean_candidate_count": int(len(clean)),
                "ppa_feasible_count": int(clean["ppa_feasible"].sum()),
                "full_feasible_count": int(clean["full_feasible"].sum()),
                "ppa_feasible_ratio": float(clean["ppa_feasible"].mean()) if len(clean) else 0.0,
                "best_unified_score": float(best_row["unified_score"]),
                "best_candidate_id": best_row["candidate_id"],
                "best_tmr_count": int(best_row["tmr_count"]),
                "best_ser_reduction_pct": float(best_row["ser_reduction_pct"]),
                "best_residual_ratio": float(best_row["residual_ratio"]),
                "best_area_overhead_pct": float(best_row["area_overhead_pct"]),
                "best_power_overhead_pct": float(best_row["power_overhead_pct"]),
                "best_worst_slack": float(best_row["worst_slack"]),
                "best_ppa_feasible": bool(best_row["ppa_feasible"]),
                "best_full_feasible": bool(best_row["full_feasible"]),
                "best_source_file": best_row["source_file"],
            }
        )
    return pd.DataFrame(rows).sort_values("best_unified_score", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate SAC+MOEA, greedy, and pure MOEA with one unified PPAR model.")
    parser.add_argument("--sac_root", type=Path, default=DEFAULT_SAC_ROOT)
    parser.add_argument("--sac_result_dir", type=Path, default=None)
    parser.add_argument("--register_csv", type=Path, default=DEFAULT_REGISTER_CSV)
    parser.add_argument("--greedy_register_file", type=Path, default=DEFAULT_GREEDY_REGISTER_FILE)
    parser.add_argument("--greedy_pareto_file", type=Path, default=DEFAULT_GREEDY_PARETO_FILE)
    parser.add_argument("--greedy_candidate_table", type=Path, default=DEFAULT_GREEDY_CANDIDATE_TABLE)
    parser.add_argument("--pure_moea_output", type=Path, default=DEFAULT_PURE_MOEA_OUTPUT)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--voter_delay", type=float, default=0.13)
    parser.add_argument("--voter_area", type=float, default=0.5)
    parser.add_argument("--voter_power", type=float, default=0.2)
    parser.add_argument("--area_limit_pct", type=float, default=20.0)
    parser.add_argument("--power_limit_pct", type=float, default=50.0)
    parser.add_argument("--residual_ratio_limit", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sac_dir = args.sac_result_dir or latest_sac_result_dir(args.sac_root)
    out_dir = args.out_dir or (
        args.sac_root / "results" / f"unified_ppar_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = UnifiedConfig(
        voter_delay=args.voter_delay,
        voter_area=args.voter_area,
        voter_power=args.voter_power,
        area_limit_pct=args.area_limit_pct,
        power_limit_pct=args.power_limit_pct,
        residual_ratio_limit=args.residual_ratio_limit,
    )

    design_metrics, register_df = read_merged_registers(args.register_csv)
    if args.greedy_pareto_file.exists() and args.greedy_candidate_table.exists():
        greedy_candidates = load_greedy_gene_candidates(
            args.greedy_pareto_file,
            args.greedy_candidate_table,
        )
    else:
        greedy_candidates = load_greedy_register_candidate(args.greedy_register_file)
    candidates = pd.concat(
        [
            load_sac_register_candidates(sac_dir),
            greedy_candidates,
            load_pure_moea_register_candidates(args.pure_moea_output),
        ],
        ignore_index=True,
    )
    scored = evaluate_candidates(candidates, register_df, design_metrics, config)
    best = best_by_algorithm(scored)
    summary = summarize(scored, best)

    candidate_path = out_dir / "unified_candidate_scores.csv"
    best_path = out_dir / "unified_best_comparison.csv"
    summary_path = out_dir / "unified_summary.csv"
    config_path = out_dir / "unified_ppar_config.json"

    scored.to_csv(candidate_path, index=False)
    best.to_csv(best_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "sac_result_dir": str(sac_dir),
                "register_csv": str(args.register_csv),
                "greedy_register_file": str(args.greedy_register_file),
                "greedy_pareto_file": str(args.greedy_pareto_file),
                "greedy_candidate_table": str(args.greedy_candidate_table),
                "pure_moea_output": str(args.pure_moea_output),
                "design_metrics": design_metrics,
                "register_count": int(len(register_df)),
                "formula": {
                    "area_total": "base_area + sum(2 * selected_reg_area + voter_area)",
                    "power_total": "base_power + sum(2 * selected_reg_power + voter_power)",
                    "worst_slack": "min(base_worst_slack, min(path_slack - voter_delay for selected registers; original slack otherwise))",
                    "residual_ser": "(sum_all_sensitivity - sum_selected_sensitivity) / (num_registers + 2 * tmr_count)",
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    display_cols = [
        "algorithm",
        "best_unified_score",
        "best_ser_reduction_pct",
        "best_residual_ratio",
        "best_area_overhead_pct",
        "best_power_overhead_pct",
        "best_worst_slack",
        "best_tmr_count",
        "ppa_feasible_count",
        "candidate_count",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {best_path}")
    print(f"Wrote: {candidate_path}")
    print(f"Wrote: {config_path}")


if __name__ == "__main__":
    main()
