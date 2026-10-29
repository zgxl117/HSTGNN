"""Build reproducible narrow-range and fixed-operator ablation configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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


def _float_list(values) -> list[float]:
    return [float(value) for value in values]


def build_configs(
    base_config_path: Path,
    history_path: Path,
    output_dir: Path,
    tail_steps: int,
) -> tuple[Path, Path, pd.DataFrame]:
    with base_config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    history = pd.read_csv(history_path)
    missing = [column for column in ACTION_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f"History is missing action columns: {missing}")
    if len(history) < tail_steps:
        raise ValueError(
            f"History contains {len(history)} rows, fewer than tail_steps={tail_steps}"
        )

    late = history.loc[:, ACTION_COLUMNS].tail(tail_steps).astype(float)
    low = late.min(axis=0).to_numpy(dtype=float).copy()
    high = late.max(axis=0).to_numpy(dtype=float).copy()
    median = late.median(axis=0).to_numpy(dtype=float).copy()

    # Gym and the SAC scaling path require a non-zero interval. A dimension
    # that was exactly constant in the source run receives only numerical
    # padding; fixed mode still executes the exact median below.
    span = high - low
    zero_span = span <= np.finfo(float).eps
    padding = np.maximum(np.abs(median) * 1.0e-6, 1.0e-8)
    low[zero_span] = median[zero_span] - padding[zero_span]
    high[zero_span] = median[zero_span] + padding[zero_span]

    source = str(history_path.resolve())
    common = {
        "source_history": source,
        "tail_steps": int(tail_steps),
        "action_columns": ACTION_COLUMNS,
        "action_low": _float_list(low),
        "action_high": _float_list(high),
    }

    narrow_config = dict(base_config)
    narrow_config["operator_control"] = {"mode": "narrow", **common}

    fixed_config = dict(base_config)
    fixed_config["operator_control"] = {
        "mode": "fixed",
        **common,
        "fixed_action": _float_list(median),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    narrow_path = output_dir / "operator_ablation_late_narrow.yaml"
    fixed_path = output_dir / "operator_ablation_late_fixed.yaml"
    for path, config in ((narrow_path, narrow_config), (fixed_path, fixed_config)):
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    summary = pd.DataFrame(
        {
            "parameter": ACTION_COLUMNS,
            "late_min": low,
            "late_q1": late.quantile(0.25).to_numpy(dtype=float),
            "late_median_fixed": median,
            "late_q3": late.quantile(0.75).to_numpy(dtype=float),
            "late_max": high,
            "late_mean": late.mean(axis=0).to_numpy(dtype=float),
        }
    )
    summary.to_csv(output_dir / "operator_ablation_values.csv", index=False)
    return narrow_path, fixed_path, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-steps", type=int, default=24)
    args = parser.parse_args()

    narrow_path, fixed_path, summary = build_configs(
        args.base_config,
        args.history,
        args.output_dir,
        args.tail_steps,
    )
    print(summary.to_string(index=False))
    print(f"narrow_config={narrow_path}")
    print(f"fixed_config={fixed_path}")


if __name__ == "__main__":
    main()
