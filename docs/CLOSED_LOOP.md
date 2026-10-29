# 预测 → 敏感性 → 强化学习 TMR 筛选

这条流程把 HSTGNN 与 acl_clean 中的 SAC/MOEA 核心集成到同一个工程。
保留原模型入口，强化学习代码位于 hardening/，通过独立子进程运行，避免同名 utils 等包冲突。

## Shell / Make 调用

```bash
cd /home/zgxl117/HSTGNN
make smoke
# 指定数据和唯一运行名：
make smoke DATASET=/path/to/dataset.yaml RUN_NAME=my_trial
# 复用已有预测，只运行敏感性汇聚和优化：
make reuse DATASET=/path/to/dataset.yaml \
  TRAIN_CONFIG=configs/task_train.yaml PREDICTIONS=/path/to/results.csv SMOKE=1
# Shell 等价形式：
bash run.sh smoke --dataset /path/to/dataset.yaml
bash run.sh reuse --dataset /path/to/dataset.yaml --smoke \
  --train-config configs/task_train.yaml --predictions /path/to/results.csv
```

make run 使用完整预算；make smoke 使用快速验证预算。默认 DATASET 是原项目 FsimNNs/data/data 下的 RISC-V test2。
make help 查看目标与环境变量；DRY_RUN=1 可在执行前打印准确命令。
解释器优先级：显式 PYTHON/HSTGNN_PYTHON、已激活非 base Conda、工程 .venv、本机 ckt、PATH。
显式解释器应为单个可执行文件路径，不能填入整条 conda run 命令。
启动器不需要 conda activate；若没有可用环境，会由已有依赖预检查给出缺项。
参数经 argv 数组传递，不使用 eval。Ctrl+C 和子任务失败沿用原 Python 流程的处理。

## 使用

```bash
cd /home/zgxl117/HSTGNN
conda run --no-capture-output -n ckt python terminal_ui.py
```

选 **9 预测与强化学习加固闭环**，选 RISC-V workload，然后选择快速冒烟或完整运行。
菜单只列出已经配置 hardening.profile 的数据。当前已配置 RISC-V test1、test2、test3。

直接执行（--smoke 只取前 3 张图、模型 2 轮、32 个候选、小 SAC/MOEA 预算）：

```bash
python pipeline.py --dataset /home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/riscy_test/test2/dataset.yaml --smoke
```

去掉 --smoke 使用全 campaign 和数据 YAML 的模型训练轮数，以及 profile 的候选数和优化预算。
当前数据 YAML 默认仅 2 轮，正式实验先在各 dataset.yaml 中调整 trainer.epochs。
完整优化配置继承 hardening/configs/sac_config.yaml，profile 可覆盖其中各部分；集成流程强制 adaptive 控制。
本版本整合 worker 使用 CPU 训练 SAC，MOEA 按配置且默认禁用 GPU。

### 使用已经完成的预测

保留预测结果与对应缓存，然后运行：

```bash
python pipeline.py --dataset /path/to/dataset.yaml --profile riscv \
  --train-config /path/to/task_train.yaml \
  --predictions /path/to/predictions/results.csv
```

两个参数必须配对；对应 cache/processed/test_processed_data.pt 必须仍存在且与该次预测一致。
此模式跳过模型训练和预测，从敏感性汇聚开始。--smoke 在此模式仅缩小优化预算，不重新截取已有预测。

## 数据集与 profile

数据 YAML 顶层添加：

```yaml
hardening:
  profile: riscv
```

profile 为 hardening/profiles/<名称>.yaml，声明：

```yaml
ppa_csv: ../data/merged_registers.csv
mapping_csv: ../data/riscv_register_mapping.csv
optimizer_config: ../configs/sac_config.yaml
candidate_count: 1380
```

profile 内路径相对于 profile 文件。可通过 --profile 指定自定义 profile 路径。
更换电路必须提供该电路自己的 PPA 和 node_id→instance_name 映射；不能把 FIFO/SPI/I2C 接到 RISC-V PPA。
默认 mapping 来源于原项目 regs_hier.csv：去掉 dut. 前缀、层次点换成斜线后，2131 个名称与 PPA 全部匹配。
详细来源及 SHA-256 在 hardening/data/mapping_provenance.json。

## 输出

每次创建全新 runs/loop_<时间戳>/，模型产物在对应的 *_model/。配置在 configs/ 下。

| 文件 | 含义 |
| --- | --- |
| inputs/ | 只复制四类 raw CSV；预处理仅写副本，不修改外部原数据 |
| inputs/provenance.json | 数据来源哈希和冒烟截取范围 |
| sensitivity/sensitivity.csv | 每个寄存器的混合敏感性与已测/预测样本数 |
| sensitivity/sample_outcomes.csv | 每个样本实际采用的标签及来源 |
| sensitivity/measured_reference.csv | 真值参考，只用于后续回评 |
| optimizer_input.csv | 保留设计 PPA 块、替换全部 sensitivity 的严格输入 |
| hardening/candidate_registers.csv | 真实预筛选顺序，gene 位与候选一一对应 |
| hardening/pareto_solutions.csv | PPA 可行方案、genes、目标值 |
| hardening/transitions.csv | SAC 交互、动作与网络更新计数 |
| hardening/sac_checkpoint.pth | 本次 SAC 权重 |
| selected_registers.csv | 最终选定方案的 node_id、物理寄存器名称及 PPA 属性 |
| hardening/result.json | 最佳方案及搜索来源归因 |
| feedback.json | 预测敏感性与测量参考上的同方案回评 |
| pipeline_report.json | 流程状态、日志、来源、最终 PPA 和步骤耗时 |

选取规则：在满足面积/功耗/slack 约束的方案中，选 backend 残余敏感性最小者；同值按面积、加固数排序。
SAC 控制 MOEA 搜索算子，MOEA 决定加固哪些寄存器；报告保留 baseline/搜索归因，不把基线解算成 SAC 独立贡献。

## 敏感性与回评口径

- 采用缓存的显式 train/val/test mask。仅 train/val 行可使用已知故障标签，test 行必须有预测。
- 缺预测、重复 ID、重叠分区、错误电路、缺映射、坏 PPA 都明确失败，不读取真值补洞、不回退随机寄存器。
- 类别 0 为无错误，非零类别计入错误。按同一 FF 的注入窗口汇聚错误比例，不附加物理时间单位。
- PPA CSV 中原来的 sensitivity 全部被当前结果覆盖；可选 failure_prob 不沿用，避免旧标签影响预筛选。
- 优化器保留上游公式：剩余敏感性总和 / (原寄存器数 + 2×加固数)。未预筛选的寄存器仍计入总量。
- feedback 同时给出原始分母 N 的结果，避免把扩充分母本身造成的下降解释成屏蔽效果。
- measured_reference 是原始注入标签上的理想 TMR 移除回评，不是加固后重新注入结果。
- 此处假定被选位的单点故障被理想 TMR 屏蔽，未计投票器故障、共模失效。PPA 为解析估算。
- 按本次需求，闭环截至强化学习筛选结果及离线回评。未执行网表改写、重新综合或加固后故障注入。

## 验证与溯源

```bash
python -m pytest tests -q
```

上游 acl_clean 的算法源码保存在 hardening/UPSTREAM_MANIFEST.json 对应清单中。
修复了其 algorithms/__init__.py 对已删除 moead/large_scale_optimizer 的导入，否则原精简工程无法导入运行。
新增 worker 绕过原 simulator 的随机数据回退，严格消费 bridge 输入，未改写原 acl_clean 工程。
HSTGNN 模型 Python 文件保持原实现；新增 pipeline.py、export_sensitivity.py 与 workflow/。
