# SAC-MOEA 三模冗余加固优化工程（精简版）

从 `E:\acl` 提取出的**可运行核心工程**，剔除了 Python 环境、缓存、废弃模块和实验中间产物。
原始目录 `E:\acl` 未做任何改动。

- 提取日期：2026-10-28
- 源码基线：2026-08-25（最后一次大改之后）
- 数据基线：`data/merged_registers.csv`，2131 个寄存器，K=1380 预筛

---

## 一、这是什么

用 **SAC（软演员-评论家）在线调节 MOEA 搜索算子**，让多目标进化算法更快收敛到
「哪些寄存器做 TMR 三模冗余」的 Pareto 前沿。

关键分工，别搞反：

- **MOEA 决定加固谁** —— 0/1 基因向量，长度 K=1380
- **SAC 不决定加固谁** —— 它每步输出 8 维动作，调 SPEA2/NSGA2 的算子旋钮

## 二、四个目标与四个约束

每个加固方案被评价为 8 个标量（`env/real_moea_wrapper.py` → `PPAModel.evaluate_with_details`）：

| 目标 | 公式 | 方向 |
|---|---|---|
| 面积 | `A₀ + Σ(2·aᵢ + voter_area)` | 越小越好 |
| 功耗 | `P₀ + Σ(2·pᵢ + voter_power)` | 越小越好 |
| 时序 | `min(slackᵢ − voter_delay·xᵢ)` | 越大越好（取负进目标） |
| 失效率 | `(Σs − Σs_hardened) / (n + 2k)` | 越小越好 |

**失效率的分母 `n + 2k` 把 TMR 复制出的新寄存器也计入**，用来抵消「分母变大」
带来的虚降。校验：`374.58 / 2131 = 0.1757766`，与 CSV 头部 `design_sensitivity=0.175777` 吻合。

约束（`configs/sac_config.yaml` → `constraint_thresholds`）：

- 面积增幅 ≤ 50%
- 功耗增幅 ≤ 80%
- 最差 slack ≥ 0（严格模式，按 100× 违例量重罚）
- 加固后失效率 ≤ 0.35 × 原始失效率

## 三、目录结构

```
main.py                    入口：读配置 → 建寄存器表 → 训练 SAC-MOEA
configs/sac_config.yaml    主配置（含 constraint_thresholds / reward_params / action 空间）
configs/operator_ablation_20260825/   消融实验配置（fixed / narrow 两档）
algorithms/                SPEA2、NSGA2 并行实现 + 公共数据结构
env/                       sac_moea_env.py（Gym 环境）+ real_moea_wrapper.py（TMR 模型）
mo_optimizer/              ICDesignSimulator（CSV 解析/预筛）+ 并行/GPU 评估器
sac/                       SACAgent（策略、回放、软更新）
utils/                     约束分析、启发式初始化、缓存、绘图
scripts/                   消融构建、统一评分、外部基线对比、冒烟测试
data/                      输入数据（见下）
results/                   精选实验结论
logs/                      一次完整运行的日志与参数轨迹
```

## 四、如何运行

⚠️ 原目录的 `.venv_sac` **已损坏**（python.exe 指向不存在的 Codex 运行时路径），
必须重建环境：

```bash
python -m venv .venv
.venv\Scripts\pip install torch numpy pandas gymnasium pyyaml matplotlib
```

运行主程序：

```bash
python main.py --csv_file data/merged_registers.csv --K 1380
```

冒烟测试（不跑完整训练，只验证数据管线）：

```bash
python scripts/smoke_test_real_data_pipeline.py
```

## 五、data/ 每个文件的性质

| 文件 | 性质 | 说明 |
|---|---|---|
| `merged_registers.csv` | **【核心输入】** | 2131 个寄存器全量明细，唯一不可再生 |
| `register_info_sim.csv` | **【仿真输入】** | 10000 条随机数据，仅做规模测试 |

`main.py` 运行时还会**自动生成** `data/register_info/{register_table.csv,
register_table.npz, design_info.csv}`。这三个是可再生的中间产物，未放入本精简版。

⚠️ **`register_table.csv` 不能直接当 `--csv_file` 用**：它缺少头部设计基准行，
`ICDesignSimulator._load_from_csv` 会解析失败并**静默回退到随机数据**。

## 六、结果与当前结论

`results/operator_ablation_comparison_20260826/operator_ablation_summary.csv`
是三次消融的汇总（adaptive / narrow / fixed）。

**核心发现：SAC 目前几乎没有贡献。**

| 变体 | overall_score | pareto_score | final_solution_score |
|---|---|---|---|
| adaptive（SAC 在线调参） | 86.7101 | 86.4662 | 87.1629 |
| narrow（动作收窄） | 86.7099 | 86.4660 | 87.1629 |
| fixed（动作冻结） | 86.7096 | 86.4655 | 87.1629 |

三次输出的最优解**完全相同**（781 个寄存器，面积 +14.04%，功耗 +49.99%，失效率 −92.97%）。
且 Pareto 前沿 80% 的点标记 `is_fixed_baseline=True`，真正由 SAC 策略步贡献的仅 12~24 个，
`search_hv_gain` 只有 1e-5 量级。

原因：`sac_config.yaml` 里 `operator_control.mode: fixed`，SAC 动作被固定到历史中位数，
闭环实际是关闭的。

**下一步建议**：把 `operator_control.mode` 改回自适应并重跑，验证 SAC 能否真正超过 87.163。

外部基线对比见 `results/unified_ppar_compare_20260826_001635_ablation/`。
⚠️ 其中 greedy 与 pure_moea 两行来自 `D:\OneDrive\Desktop\...`，**该路径已不存在**，无法复核。

---

## 七、已剔除内容及原因

| 剔除项 | 体积 | 原因 |
|---|---|---|
| `.venv_sac/` | 4.31 GB | 虚拟环境，且已损坏 |
| `.conda/` | 1.18 GB | 完整 conda 环境 |
| `results/experiences/` | 342 MB | 36 个 .pkl 热启动经验，可从零重跑再生 |
| 另外两次实验的 results/logs | ~30 MB | 与 fixed 档结论重复，仅保留最新一次 |
| 全部 `__pycache__/` | — | 字节码缓存 |
| `*.codex_backup_*`（9 个） | — | 2026-08-25 大改前的编辑器备份 |
| `_aborted_*`（2 个目录） | 7.4 MB | 中断的实验 |
| `algorithms/moead.py` | 36 KB | 死代码：`main.py` 未引用，wrapper 只用 SPEA2/NSGA2 |
| `algorithms/large_scale_optimizer.py` | 47 KB | 同上 |
| `env/hybrid_moea_wrapper.py` | 32 KB | 同上 |
| `env/large_scale_wrapper.py` | 21 KB | 同上 |
| `utils/plot_thresholds.py` | — | 源码不存在，只有 .pyc 残留 |
| `.pytest_cache/`、`.vscode/` | — | 工具缓存，非工程内容 |

> 注：`algorithms/__init__.py` 会 import moead 与 large_scale_optimizer，
> 但 `main.py` 从不 import `algorithms` 包（只做子模块导入），所以不受影响。
> 若将来需要混血/大规模算法做对比实验，可从 `E:\acl` 原目录取回。

## 八、保留但需要注意的

- `mo_optimizer/simulator.py` 的 `ICDesignSimulator` 只用于**建寄存器表**；
  TMR 评估实际由 `PPAModel` 完成。两处都实现了失效率计算，公式不同，别混用。
- `voter_delay` 在配置里是 `0.13`，`PPAModel` 默认 `0.2`，旧 simulator 里硬编码 `0.08`。
  以配置文件为准。
- `checkpoints/`（3 个 .pth，各 3.2 MB）未放入：属于某一次运行的可再生产物。
