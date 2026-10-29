# HSTGNN + SAC-MOEA：故障预测与寄存器加固筛选

将异构时空图神经网络 HSTGNN 与 SAC/MOEA 优化器集成，完成从故障结果预测到 TMR 寄存器选择的流程。

```text
电路拓扑 + FF 时序活动 + 模块层次 + 数据集 YAML
                      ↓
               HSTGNN 训练与预测
                      ↓
             按寄存器汇聚故障敏感性
                      ↓
       SAC 控制搜索算子 + MOEA 选择加固寄存器
                      ↓
      寄存器清单 + Pareto 方案 + PPA/敏感性离线回评
```

当前交付截止到加固寄存器筛选结果与解析回评；不包含加固网表生成、重新综合及加固后故障注入。

## 快速开始

```bash
git clone https://github.com/zgxl117/HSTGNN.git
cd HSTGNN

# 在隔离环境中安装与机器匹配的 PyTorch，再安装其余依赖
python -m pip install -r requirements.txt
make check PYTHON="$(command -v python)"
make test PYTHON="$(command -v python)"
make ui PYTHON="$(command -v python)"
```

需要 Python >= 3.9、Bash；使用 Make 入口还需 GNU Make。训练和优化核心仍为 Python/PyTorch。
依赖清单未锁定版本；已验证环境与兼容性说明见 [验证记录](docs/VALIDATION.md)。其中历史验证环境使用 torch 2.4.1，清单声明 torch>=2.5；不能将已有验证视为任意新环境均兼容。

本仓库包含代码、配置模板及内置 RISC-V PPA/寄存器映射参考文件。训练 CSV、缓存、检查点和运行日志需另行准备。

## 用 Shell 或 Make 运行整个流程

```bash
make ui          # 终端交互界面
make check       # 环境依赖检查
make datasets    # 列出已注册的数据 YAML
make test        # 回归测试
make help        # 参数说明

# 指定数据运行端到端冒烟或完整任务
make smoke DATASET=/path/to/dataset.yaml
make run DATASET=/path/to/dataset.yaml RUN_NAME=my_experiment
```

Shell 等价入口：

```bash
./run.sh ui
./run.sh smoke --dataset /path/to/dataset.yaml
./run.sh run --dataset /path/to/dataset.yaml --name my_experiment
```

`run.sh` 自动进入工程目录；相对路径以工程根目录解析。Make 可从其他目录通过 `make -C /path/to/HSTGNN ...` 调用。
指定解释器用 `PYTHON=/path/to/python`（Make）或 `HSTGNN_PYTHON=/path/to/python`（Shell）。默认优先选择已激活的非 base Conda 环境、工程 `.venv`、本机 `ckt` 环境，最后使用 PATH。

当前机器省略 DATASET 时默认读取同级 AI4FaultSim 下的 `fault-sim/FsimNNs/data/data/riscy_test/test2/dataset.yaml`；新机器请显式指定数据或修改 [数据源注册表](dataset_sources.yaml)。可用 `make smoke DRY_RUN=1` 预览调用。

冒烟使用前 3 张图、模型 2 轮及小优化预算，验证流程能运行。完整任务读取数据 YAML 的 `trainer.epochs` 与加固 profile 的优化预算；正式实验应先设置训练轮数。

## 每个数据集一个 YAML

```text
campaign/
├── dataset.yaml
└── raw/
    ├── nodes.csv
    ├── edges.csv
    ├── graphs.csv
    └── ff_module.csv
```

复制 [dataset.example.yaml](configs/dataset.example.yaml) 到数据目录并按真实数据设置节点数、时间窗口、边维度、划分方式及训练参数。数据 YAML 内的路径相对于该 YAML 所在目录解析。

终端菜单 **2** 从数据 YAML 自动生成配对配置，菜单 **9** 启动预测与加固流程。
完整字段见 [数据集 YAML 说明](docs/DATASET_YAML.md)，CSV 契约见 [数据格式](docs/DATA_FORMAT.md)。

运行加固流程还需在数据 YAML 顶层配置：

```yaml
hardening:
  profile: riscv
```

内置 `riscv` profile 对应 2131 个寄存器。其他电路必须提供匹配的 PPA、node_id→物理实例名映射和 profile，不能复用 RISC-V 映射。可用 `PROFILE=/path/to/profile.yaml` 覆盖。

## 复用已有预测结果

```bash
make reuse DATASET=/path/to/dataset.yaml \
  TRAIN_CONFIG=configs/task_train.yaml \
  PREDICTIONS=/path/to/predictions/results.csv \
  SMOKE=1
```

跳过模型训练与预测，从敏感性汇聚开始。训练配置、预测 CSV 及对应的 `processed/test_processed_data.pt` 缓存必须匹配并保留。去掉 `SMOKE=1` 使用完整优化预算。

## 输出结果

每次任务创建独立 `runs/<任务名>/`；模型产物放在对应的 `<任务名>_model/`。

| 文件 | 内容 |
| --- | --- |
| `selected_registers.csv` | 最终选择的 node_id、物理寄存器名与 PPA 属性 |
| `sensitivity/sensitivity.csv` | 每个寄存器的敏感性及样本来源计数 |
| `hardening/pareto_solutions.csv` | 满足约束的 Pareto 方案 |
| `hardening/transitions.csv` | SAC 交互与更新记录 |
| `hardening/result.json` | 最终方案及搜索来源归因 |
| `feedback.json` | 同方案的预测敏感性与测量参考回评 |
| `pipeline_report.json` | 步骤状态、日志位置、来源与耗时 |

完整输出契约和计算公式见 [闭环工程说明](docs/CLOSED_LOOP.md)。

## 计算口径

- 类别 0 表示无错误，非零类别计入错误；敏感性为各 FF 在样本窗口中的错误比例，不是带物理时间单位的 SER。
- 汇聚采用显式划分：train/val 使用已知标签，test 必须使用预测；测量真值参考只参与后续回评。
- SAC 选择 MOEA 搜索算子，MOEA 决定加固寄存器。报告记录基线与搜索归因；冒烟结果不证明 SAC 优于基线。
- 回评假设理想 TMR 屏蔽被选位的单点故障，PPA 为解析估算，尚未通过加固后重新注入验证。
- 当前未集成 SAT 联合筛选或以 RTL 仿真数据替代网表仿真输入。

## 工程导航与 Agent 调用

| 入口 | 用途 |
| --- | --- |
| `run.sh` / `Makefile` | Shell / Make 启动入口 |
| `terminal_ui.py` | 终端界面、环境检查与数据 YAML 配置 |
| `pipeline.py` | 整个流程编排 |
| `models/` / `mydatasets/` | HSTGNN 模型与数据处理 |
| `train.py` / `predict.py` | 独立训练与预测 |
| `workflow/` / `export_sensitivity.py` | 敏感性汇聚及输入桥接 |
| `hardening/worker.py` | 严格输入的 SAC/MOEA 优化入口 |
| `hardening/profiles/` | 电路 PPA 与映射配置 |
| `configs/*.example.yaml` | 可移植配置模板 |
| `tests/` | 模型、数据、界面及闭环回归测试 |

其他 Agent 先读 [AGENTS.md](AGENTS.md) 与 [调用说明](docs/AGENT_HANDOFF.md)，优先通过独立子进程执行启动器。
非交互检查可用 `./run.sh check`、`./run.sh datasets`，或 `python terminal_ui.py --check-config <yaml> --mode train`。

直接运行 `train.py` 会删除配置中已有的 checkpoint 目录；使用独立任务目录。根目录历史训练/预测 YAML 来自不同电路，请使用模板或数据 YAML 生成配对配置。

## 来源与验证

原 HSTGNN 模型源码来自 `zgxl117/AI4FaultSim` 提交 `95b8e7bd1db7f3672918e8ee018eba9eb4fd25c2` 的 `HSTGNN/`，来源清单见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。
强化学习后端来自本地 acl_clean，整合说明与来源清单见 [hardening/README.md](hardening/README.md) 和 [UPSTREAM_MANIFEST.json](hardening/UPSTREAM_MANIFEST.json)。

已有验证包含 41 项回归测试，以及 RISC-V test2 前 3 图的真实数据全流程冒烟；详细范围、环境和限制见 [验证记录](docs/VALIDATION.md)。

## 开源许可证

本项目采用 [MIT License](LICENSE)。源码来源记录见上文及对应 manifest；第三方依赖遵循各自的许可证。
