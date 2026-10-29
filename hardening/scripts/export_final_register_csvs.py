import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_register_table(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"empty register csv: {csv_path}")
    return df


def _normalise_gene(gene_text: str, k: int):
    gene = np.array([1 if ch == "1" else 0 for ch in str(gene_text).strip()], dtype=np.int8)
    if gene.size > k:
        gene = gene[:k]
    elif gene.size < k:
        gene = np.pad(gene, (0, k - gene.size), constant_values=0)
    return gene


def export_solution_registers(result_dir: Path, register_csv: Path, output_dir: Path, feasible_only: bool):
    pareto_path = result_dir / "sac_moea_pareto_front.csv"
    if not pareto_path.exists():
        raise FileNotFoundError(f"missing Pareto file: {pareto_path}")

    pareto_df = pd.read_csv(pareto_path)
    reg_df = _load_register_table(register_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "genes" not in pareto_df.columns:
        raise ValueError(f"{pareto_path} has no genes column")

    exported = []
    k = len(str(pareto_df["genes"].iloc[0]).strip())
    reg_view = reg_df.iloc[:k].copy()

    for _, row in pareto_df.iterrows():
        if feasible_only and str(row.get("feasible", "")).lower() not in ("true", "1"):
            continue

        gene = _normalise_gene(row["genes"], k)
        selected = np.where(gene == 1)[0]
        solution_id = str(row.get("solution_id", f"solution_{len(exported):04d}"))
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in solution_id)
        out_path = output_dir / f"{safe_id}_registers.csv"

        selected_df = reg_view.iloc[selected].copy()
        selected_df.insert(0, "prefilter_index", selected)
        selected_df.insert(1, "solution_id", solution_id)
        for col in [
            "tmr_count",
            "residual_ratio",
            "ser_reduction_pct",
            "area_overhead_pct",
            "power_overhead_pct",
            "worst_slack",
        ]:
            if col in row.index:
                selected_df[col] = row[col]

        selected_df.to_csv(out_path, index=False)
        exported.append(
            {
                "solution_id": solution_id,
                "file": out_path.name,
                "tmr_count": int(row.get("tmr_count", len(selected))),
                "feasible": row.get("feasible", ""),
                "residual_ratio": row.get("residual_ratio", ""),
                "ser_reduction_pct": row.get("ser_reduction_pct", ""),
            }
        )

    summary_path = output_dir / "final_solution_registers_summary.csv"
    pd.DataFrame(exported).to_csv(summary_path, index=False)
    print(f"exported {len(exported)} register CSV files to {output_dir}")
    print(f"summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Export final hardened-register CSVs from SAC+MOEA results.")
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--register_csv", default=Path("data/merged_registers.csv"), type=Path)
    parser.add_argument("--output_dir", default=None, type=Path)
    parser.add_argument("--all", action="store_true", help="export infeasible rows too")
    args = parser.parse_args()

    result_dir = args.result_dir
    output_dir = args.output_dir or (result_dir / "final_solution_registers_from_pareto")
    export_solution_registers(
        result_dir=result_dir,
        register_csv=args.register_csv,
        output_dir=output_dir,
        feasible_only=not args.all,
    )


if __name__ == "__main__":
    main()
