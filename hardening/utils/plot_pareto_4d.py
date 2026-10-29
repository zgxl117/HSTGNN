# utils/plot_pareto_4d.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _ensure_arrays(pareto):
    """
    从 pareto 数据中解析出:
      - total_area
      - total_power
      - worst_slack
      - design_sensitivity
    优先使用显式字段；若不存在则回退到 'objs' 矩阵。
    """
    if isinstance(pareto, dict) and "total_area" in pareto:
        area = np.asarray(pareto["total_area"], dtype=float)
        power = np.asarray(pareto["total_power"], dtype=float)
        slack = np.asarray(pareto.get("worst_slack", pareto.get("slack")), dtype=float)
        sens = np.asarray(
            pareto.get("design_sensitivity", pareto.get("sensitivity")), dtype=float
        )
    else:
        # 兼容旧接口: pareto['objs'] 为 (N,4) 目标数组
        objs = np.asarray(pareto["objs"], dtype=float)
        area = objs[:, 0]
        power = objs[:, 1]
        slack = -objs[:, 2]
        sens = objs[:, 3]

    return area, power, slack, sens


def plot_pareto_4d(pareto, save_path=None):
    """
    绘制Pareto前沿的多视图:
      1) 面积-功耗 (颜色=敏感度)
      2) 面积-敏感度
      3) 功耗-敏感度
      4) 时序(最差slack)-敏感度

    参数:
    ----------
    pareto : dict
        可以是:
          - {'total_area', 'total_power', 'worst_slack', 'design_sensitivity'}
          - {'objs': np.ndarray (N,4)}  (兼容旧接口)
    save_path : str or Path, optional
        基础保存路径; 若为文件名, 则主视图保存为该文件,
        其它视图自动添加后缀。
    """
    area, power, slack, sens = _ensure_arrays(pareto)

    # 统一处理保存路径
    base_dir = None
    base_name = "pareto_front"
    if save_path is not None:
        save_path = Path(save_path)
        base_dir = save_path.parent
        base_name = save_path.stem
    else:
        base_dir = Path(".")

    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Area vs Power, color=Sensitivity
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(area, power, c=sens, s=40, cmap="viridis")
    plt.colorbar(sc, label="Sensitivity")
    plt.xlabel("Total Area")
    plt.ylabel("Total Power")
    plt.title("Pareto Front (Area vs Power, color=Sensitivity)")
    plt.tight_layout()
    plt.savefig(base_dir / f"{base_name}.png", dpi=150)
    plt.close()

    # 2. Area vs Sensitivity
    plt.figure(figsize=(7, 6))
    plt.scatter(area, sens, s=40)
    plt.xlabel("Total Area")
    plt.ylabel("Design Sensitivity")
    plt.title("Area vs Sensitivity")
    plt.tight_layout()
    plt.savefig(base_dir / f"{base_name}_area_sens.png", dpi=150)
    plt.close()

    # 3. Power vs Sensitivity
    plt.figure(figsize=(7, 6))
    plt.scatter(power, sens, s=40)
    plt.xlabel("Total Power")
    plt.ylabel("Design Sensitivity")
    plt.title("Power vs Sensitivity")
    plt.tight_layout()
    plt.savefig(base_dir / f"{base_name}_power_sens.png", dpi=150)
    plt.close()

    # 4. Worst Slack vs Sensitivity
    plt.figure(figsize=(7, 6))
    plt.scatter(slack, sens, s=40)
    plt.xlabel("Worst Slack")
    plt.ylabel("Design Sensitivity")
    plt.title("Timing (Slack) vs Sensitivity")
    plt.tight_layout()
    plt.savefig(base_dir / f"{base_name}_slack_sens.png", dpi=150)
    plt.close()
