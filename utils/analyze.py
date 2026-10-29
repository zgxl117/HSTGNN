#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""analyze_4plots.py
指令: python analyze.py --csv /home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/stgnns/predictions/results.csv --exclude-all-zeros
      python analyze.py --csv /home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/stgnns/predictions/results.csv --include-all-zeros
把两个脚本合并成一个：
- 同一个 CSV 只读一次
- 默认输出 **4 个子图在同一个 Figure（2x2）**：
  1) Absolute Bias 直方图
  2) True ratio vs Predicted ratio 散点图（颜色为 absolute bias）
  3) Bias direction 饼图
  4) Accuracy(%) vs Bias threshold(%) 折线图

CSV 需要包含列：
  source_path, graph_id, node_id, predicted_label, true_label
若误写成 predicted_tabel，会自动改名。

“去零值/不去零值”说明：
- 这里的“零值”指 **true_label 全 0 的 node_id**。
- 默认：去零值（排除 true 全 0 的 node）
- 可选：不去零值（保留这些 node）

"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_chinese_font() -> None:
    """尽量设置一个可用的中文字体；找不到就退回默认字体（避免 findfont 警告刷屏）。"""
    import matplotlib
    from matplotlib import font_manager

    candidates = [
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break

    matplotlib.rcParams["axes.unicode_minus"] = False


def load_and_compute(csv_file: str, *, exclude_all_zeros: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 CSV 并按 node_id 统计 predicted/true 比例与 bias。"""

    csv_file = os.path.expanduser(csv_file)
    df = pd.read_csv(csv_file)

    # 兼容列名拼写错误
    if "predicted_label" not in df.columns and "predicted_tabel" in df.columns:
        df = df.rename(columns={"predicted_tabel": "predicted_label"})

    required = ["source_path", "graph_id", "node_id", "predicted_label", "true_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV 缺少必要列: {missing}\n"
            f"实际列: {list(df.columns)}"
        )

    # 数字化
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce")
    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")

    grouped = df.groupby("node_id", sort=False)

    results = []
    excluded_nodes = 0

    for node_id, g in grouped:
        total = len(g)
        if total == 0:
            continue

        true_ones = int((g["true_label"] == 1).sum())

        # “去零值”：排除 true_label 全 0 的 node
        if exclude_all_zeros and true_ones == 0:
            excluded_nodes += 1
            continue

        pred_ones = int((g["predicted_label"] == 1).sum())

        true_ratio = true_ones / total
        pred_ratio = pred_ones / total
        bias_abs = abs(pred_ratio - true_ratio)

        # 相对偏差（true_ratio=0 时避免除 0）
        bias_rel = (bias_abs / true_ratio * 100.0) if true_ratio > 0 else 0.0

        if pred_ratio > true_ratio:
            direction = "Over-prediction"
        elif pred_ratio < true_ratio:
            direction = "Under-prediction"
        else:
            direction = "Accurate"

        results.append(
            {
                "node_id": node_id,
                "predicted_1_count": pred_ones,
                "predicted_total": total,
                "predicted_1_ratio": round(pred_ratio, 4),
                "true_1_count": true_ones,
                "true_1_ratio": round(true_ratio, 4),
                "bias_absolute": round(bias_abs, 4),
                "bias_relative": round(bias_rel, 2),
                "bias_direction": direction,
                "predicted_true_diff": round(pred_ratio - true_ratio, 4),
            }
        )

    result_df = pd.DataFrame(results)

    if len(result_df) == 0:
        raise RuntimeError(
            "统计结果为空：可能所有 node 都被排除了（exclude_all_zeros=True），或者 CSV 内容异常。"
        )

    # 打印整体统计（便于写报告）
    total_records = len(df)
    total_pred_ones = int((df["predicted_label"] == 1).sum())
    total_true_ones = int((df["true_label"] == 1).sum())
    overall_pred_ratio = total_pred_ones / total_records if total_records else 0.0
    overall_true_ratio = total_true_ones / total_records if total_records else 0.0
    overall_bias = abs(overall_pred_ratio - overall_true_ratio)

    print("=" * 80)
    print("Statistical Analysis Results")
    print("=" * 80)
    print(f"CSV: {csv_file}")
    print(f"Total records: {total_records:,}")
    print(f"Total 1s in predicted labels: {total_pred_ones:,} (ratio: {overall_pred_ratio:.4f})")
    print(f"Total 1s in true labels: {total_true_ones:,} (ratio: {overall_true_ratio:.4f})")
    print(f"Overall bias (global ratio diff): {overall_bias:.4f}")
    print(f"Unique node_ids analyzed: {len(result_df):,}")
    if exclude_all_zeros:
        print(f"Excluded node_ids (true all zeros): {excluded_nodes:,}")
    print("=" * 80)

    return df, result_df


def _compute_accuracy_rates(result_df: pd.DataFrame, bias_thresholds: list[int]) -> tuple[list[int], list[float]]:
    """在不同 bias 阈值下，统计 Accuracy(%) = P(bias_abs <= threshold)。"""

    bias_abs = result_df["bias_absolute"].astype(float).to_numpy()

    pairs: list[tuple[int, float]] = []
    for t in bias_thresholds:
        thr = int(t)
        acc = float((bias_abs <= (thr / 100.0)).mean() * 100.0)
        pairs.append((thr, acc))

    pairs.sort(key=lambda x: x[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return xs, ys


def _set_accuracy_ylim(ax, xs: list[int], ys: list[float], y_min: float | None, y_max: float | None) -> None:
    """设置 accuracy 折线图 y 轴。

    - 如果用户给了 y_min/y_max：严格使用
    - 否则：用 **最小/最大阈值**（比如 0 和 30）对应的 accuracy 自动确定范围
    """

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
        return

    if not ys:
        return

    # 只用端点（min/max threshold 对应的 accuracy）来决定范围
    y0 = ys[0]
    y1 = ys[-1]
    lo = min(y0, y1)
    hi = max(y0, y1)
    pad = max(1.0, (hi - lo) * 0.15)

    y_min_auto = max(0.0, lo - pad)
    y_max_auto = min(100.0, hi + pad)

    if (y_max_auto - y_min_auto) < 0.5:
        y_min_auto = max(0.0, y_min_auto - 1.0)
        y_max_auto = min(100.0, y_max_auto + 1.0)

    ax.set_ylim(y_min_auto, y_max_auto)


def plot_4plots_in_one_figure(
    result_df: pd.DataFrame,
    bias_thresholds: list[int],
    *,
    exclude_all_zeros: bool = True,
    out_file: str = "all_4plots_in_one.png",
    show: bool = True,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """把 4 张图放在同一个 Figure(2x2) 里输出。"""

    setup_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    exclusion_text = " (Excluding nodes with all zeros in true_label)" if exclude_all_zeros else ""

    # (1) 直方图
    ax = axes[0, 0]
    ax.hist(
        result_df["bias_absolute"],
        bins=20,
        edgecolor="black",
        alpha=0.7,
        color="skyblue",
    )
    ax.set_xlabel("Absolute Bias")
    ax.set_ylabel("Number of Nodes")
    ax.set_title(f"Distribution of Absolute Bias{exclusion_text}")
    ax.grid(True, alpha=0.3, linestyle="--")
    mean_bias = float(result_df["bias_absolute"].mean())
    ax.axvline(x=mean_bias, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_bias:.3f}")
    ax.legend()

    # (2) 散点图
    ax = axes[0, 1]
    sc = ax.scatter(
        result_df["true_1_ratio"],
        result_df["predicted_1_ratio"],
        c=result_df["bias_absolute"],
        alpha=0.6,
        cmap="viridis",
        s=45,
    )
    max_val = float(max(result_df["true_1_ratio"].max(), result_df["predicted_1_ratio"].max()))
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.7, linewidth=2, label="Ideal Line (y=x)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Absolute Bias")
    ax.set_xlabel("True Label 1 Ratio")
    ax.set_ylabel("Predicted Label 1 Ratio")
    ax.set_title(f"Predicted Ratio vs True Ratio{exclusion_text}")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()

    # (3) 饼图
    ax = axes[1, 0]
    direction_counts = result_df["bias_direction"].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(direction_counts)))
    ax.pie(
        direction_counts.values,
        labels=direction_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax.set_title(f"Bias Direction Distribution{exclusion_text}")

    # (4) Accuracy vs Bias threshold 折线图
    ax = axes[1, 1]
    xs, ys = _compute_accuracy_rates(result_df, bias_thresholds)
    ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_xlabel("Bias threshold (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Bias Threshold")
    ax.grid(True, linestyle="--", alpha=0.3)

    _set_accuracy_ylim(ax, xs, ys, y_min, y_max)

    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10,
        )

    out_file = os.path.expanduser(out_file)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"[Saved] 4-plots in one figure -> {out_file}")

    if show:
        plt.show()
    plt.close(fig)


def plot_bias_distribution_3plots(
    result_df: pd.DataFrame,
    *,
    exclude_all_zeros: bool = True,
    out_file: str = "bias_distribution_3plots.png",
    show: bool = True,
) -> None:
    """生成 3 个偏差分布图（1 张图里 3 个子图）。"""

    setup_chinese_font()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    exclusion_text = " (Excluding nodes with all zeros in true_label)" if exclude_all_zeros else ""

    # 1) 直方图
    axes[0].hist(
        result_df["bias_absolute"],
        bins=20,
        edgecolor="black",
        alpha=0.7,
        color="skyblue",
    )
    axes[0].set_xlabel("Absolute Bias")
    axes[0].set_ylabel("Number of Nodes")
    axes[0].set_title(f"Distribution of Absolute Bias{exclusion_text}")
    axes[0].grid(True, alpha=0.3, linestyle="--")
    mean_bias = float(result_df["bias_absolute"].mean())
    axes[0].axvline(x=mean_bias, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_bias:.3f}")
    axes[0].legend()

    # 2) 散点图
    scatter = axes[1].scatter(
        result_df["true_1_ratio"],
        result_df["predicted_1_ratio"],
        c=result_df["bias_absolute"],
        alpha=0.6,
        cmap="viridis",
        s=50,
    )
    max_val = float(max(result_df["true_1_ratio"].max(), result_df["predicted_1_ratio"].max()))
    axes[1].plot([0, max_val], [0, max_val], "r--", alpha=0.7, linewidth=2, label="Ideal Line (y=x)")
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label("Absolute Bias")
    axes[1].set_xlabel("True Label 1 Ratio")
    axes[1].set_ylabel("Predicted Label 1 Ratio")
    axes[1].set_title(f"Predicted Ratio vs True Ratio{exclusion_text}")
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].legend()

    # 3) 饼图
    direction_counts = result_df["bias_direction"].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(direction_counts)))
    axes[2].pie(
        direction_counts.values,
        labels=direction_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    axes[2].set_title(f"Bias Direction Distribution{exclusion_text}")

    plt.tight_layout()
    out_file = os.path.expanduser(out_file)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"[Saved] 3-plots figure -> {out_file}")

    if show:
        plt.show()
    plt.close(fig)


def plot_accuracy_vs_bias_threshold(
    result_df: pd.DataFrame,
    bias_thresholds: list[int],
    *,
    out_file: str = "spfm_accuracy_with_bias_rate.png",
    show: bool = True,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """单独输出 Accuracy vs Bias Threshold 折线图。"""

    setup_chinese_font()

    xs, ys = _compute_accuracy_rates(result_df, bias_thresholds)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Bias threshold (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Bias Threshold")
    plt.grid(True, linestyle="--", alpha=0.3)

    ax = plt.gca()
    _set_accuracy_ylim(ax, xs, ys, y_min, y_max)

    for x, y in zip(xs, ys):
        plt.annotate(
            f"{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    out_file = os.path.expanduser(out_file)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"[Saved] accuracy curve -> {out_file}")

    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV bias analysis: generate plots in one run.")

    p.add_argument(
        "--csv",
        default="~/FaultSim/ml-method4-fault-sim/FsimNNs/stgnns/predictions/results.csv",
        help="CSV file path",
    )

    p.add_argument(
        "--bias-thresholds",
        nargs="+",
        type=int,
        default=[0, 8, 10, 15, 20, 30],
        help="Bias thresholds in percent (e.g. 0 8 10 15 20 30)",
    )

    # “去零值/不去零值”开关（互斥）
    zg = p.add_mutually_exclusive_group()
    zg.add_argument(
        "--exclude-all-zeros",
        action="store_true",
        help="Exclude node_ids whose true_label are all zeros (default behavior).",
    )
    zg.add_argument(
        "--include-all-zeros",
        action="store_true",
        help="Include node_ids whose true_label are all zeros (do NOT drop zeros).",
    )

    p.add_argument(
        "--separate-figures",
        action="store_true",
        help="Output two separate figures (1x3 distribution + 1 accuracy) instead of one 2x2 figure.",
    )

    p.add_argument("--out-dir", default=".", help="Output directory for figures/results.")
    p.add_argument("--prefix", default="", help="Filename prefix for outputs (optional).")

    p.add_argument(
        "--ymin",
        type=float,
        default=-1.0,
        help="y-axis min for accuracy plot (-1 means auto from endpoints)",
    )
    p.add_argument(
        "--ymax",
        type=float,
        default=-1.0,
        help="y-axis max for accuracy plot (-1 means auto from endpoints)",
    )

    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not pop up windows (save figures only).",
    )

    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()

    # 默认“去零值”（排除 true 全 0 的 node）
    # - 如果用户显式指定 --include-all-zeros：不去零值
    # - 如果用户显式指定 --exclude-all-zeros：去零值
    # - 两个都不指定：去零值
    exclude_all_zeros = True
    if args.include_all_zeros:
        exclude_all_zeros = False
    if args.exclude_all_zeros:
        exclude_all_zeros = True

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    prefix = (args.prefix + "_") if args.prefix else ""
    show = not args.no_show

    # 1) 读 CSV & 统计一次
    _, result_df = load_and_compute(args.csv, exclude_all_zeros=exclude_all_zeros)

    # 2) 导出统计 CSV
    result_csv = os.path.join(out_dir, f"{prefix}analysis_results_with_bias.csv")
    result_df.to_csv(result_csv, index=False)
    print(f"[Saved] per-node stats -> {result_csv}")

    y_min = None if args.ymin == -1 else args.ymin
    y_max = None if args.ymax == -1 else args.ymax

    if args.separate_figures:
        # 两张图文件：1x3 + 1x1
        bias_fig = os.path.join(
            out_dir,
            f"{prefix}bias_distribution_3plots_exclude_zeros.png" if exclude_all_zeros else f"{prefix}bias_distribution_3plots.png",
        )
        plot_bias_distribution_3plots(
            result_df,
            exclude_all_zeros=exclude_all_zeros,
            out_file=bias_fig,
            show=show,
        )

        acc_fig = os.path.join(out_dir, f"{prefix}spfm_accuracy_with_bias_rate.png")
        plot_accuracy_vs_bias_threshold(
            result_df,
            args.bias_thresholds,
            out_file=acc_fig,
            show=show,
            y_min=y_min,
            y_max= 100,
        )
    else:
        # 默认：四张图放在一个 2x2 Figure 里
        out_fig = os.path.join(out_dir, f"{prefix}all_4plots_in_one.png")
        plot_4plots_in_one_figure(
            result_df,
            args.bias_thresholds,
            exclude_all_zeros=exclude_all_zeros,
            out_file=out_fig,
            show=show,
            y_min=y_min,
            y_max=100,
        )


if __name__ == "__main__":
    main()
