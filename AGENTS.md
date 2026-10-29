# Agent 操作约定

## 先读

读取 `README.md`、`docs/DATA_FORMAT.md` 与 `docs/AGENT_HANDOFF.md`。
本工程根目录就是包含 `train.py` 的目录，不需要访问原 AI4FaultSim。
工程支持 FF 故障预测及 SAC/MOEA 寄存器加固筛选。闭环入口为 pipeline.py，先读 docs/CLOSED_LOOP.md。

## 操作入口

从本工程根目录执行：

- 检查依赖：`python -c "import torch, torch_geometric, pytorch_lightning, torchmetrics, yaml"`
- 回归测试：`python -m pytest tests -q`
- 训练：`python train.py --config configs/<task>_train.yaml`
- 预测：`python predict.py --config configs/<task>_predict.yaml`

配置相对路径按进程当前工作目录解析，不按 YAML 位置解析。
其他工作目录中调用时，先将子进程 `cwd` 设为工程根目录，优先采用数据绝对路径。
先确认解释器和依赖；pytest 的 skip 不能当作执行成功的证据。
`requirements.txt` 未锁版本，切勿假定任意最新依赖都兼容旧训练代码。

## 执行前核对

- 用户提供数据目录、目标解释器、输出目录；没有真实数据时只运行已有合成数据测试，不宣称完成训练。
- 训练和预测 model 配置必须匹配 checkpoint。
- 使用新的任务配置，不直接使用根目录遗留的 SPI 训练/RISC-V 预测配置。
- 输入是正常活动窗口与拓扑；训练标签不能用于标签无关采样或作为预测输入。
- 确认 label 含义、节点与图 ID、窗口长度、节点数和 PCA 后边维度。
- 输入 CSV 改变或切换任务时使用新的 `dataset.root_dir`，现有缓存不会完整追踪输入文件内容变化。
- 训练会递归删除已有 `trainer.checkpoint_dir`；必须使用新的独立目录。
- 预处理会在 campaign 的 `raw/processed/` 写过滤后边文件，缓存目录和数据目录需要相应写权限。
- 不改写外部数据集；需更改时创建任务副本。不要从本工程自动提交或推送远端。

## 修改和汇报

只改任务所需代码；修改模型、数据处理或划分后运行相关已有测试。
汇报解释器、执行命令、通过/跳过/失败数量和产物位置，区分真实电路数据与合成测试。
原模型源码对应 `SOURCE_MANIFEST.json`；修改后说明哪些文件偏离来源版本。
不要把有限样本错误比例说成有物理单位的 SER；PPA 和加固回评是解析估算。未实现 SAT 联合筛选、RTL 输入替代或网表改写。

## 终端界面

交互入口：`python terminal_ui.py`，从任意目录调用均可；任务 cwd 固定为工程根目录。
`--python` 指定子任务解释器；默认继承界面解释器。
Agent 优先使用 `--diagnose` 或 `--check-config <path> --mode train|predict` 获取 JSON 检查结果。
终端层拒绝训练到已有 checkpoint 目录，但直接执行原 train.py 仍会删除已有目录。
新增界面源码为 terminal_ui.py；模型 Python 源码仍对应 SOURCE_MANIFEST.json。

## 数据集 YAML 自动配置

数据目录中的 `dataset.yaml` 是参数来源；路径相对于它所在目录解析。
终端菜单 2 只需选择数据，不再逐项输入维度。支持 `--dataset <yaml或目录>` 自动配置后进入菜单。
Agent 可用 `--dataset <path> --prepare-only --task-name <新任务名>` 生成配对配置并获取 JSON。
每次任务自动隔离 cache/checkpoints/predictions，数据 YAML 不指定这些输出路径。
格式及支持字段见 docs/DATASET_YAML.md；导入配置不会自动启动训练。

数据根目录注册在 dataset_sources.yaml。菜单 2 自动扫描每个 campaign 的 dataset.yaml，
或使用 `python terminal_ui.py --list-datasets` 获取路径列表；可以直接 --dataset 指定其中一个。
每个数据的维度独立，不复用 FIFO 的 num_nodes 等参数。缺少模块映射或特征时不得虚构数据补齐。

## 合并后的闭环约定

- 一键流程：`python pipeline.py --dataset <yaml> --smoke`，或终端菜单 9。
- hardening/ 后端只通过子进程调用，防止与模型侧的 utils/models 等包名冲突。
- 各电路必须配置匹配的 PPA 和 node_id→instance_name profile。当前内置 RISC-V 2131 位，严禁套用到其他电路。
- 只用显式已知 train/val 标签加 test 预测汇聚敏感性；缺预测不得用测试真值补齐。
- measured_reference.csv 仅用于加固后同方案的离线回评，不进入预筛选/奖励/方案选择。
- worker 严格验证输入，不调用上游 simulator 的随机数据回退。原 acl_clean 不改动。
- 报告 SAC 更新次数、策略交互和基线来源归因；不得声称最终方案的改进一定由 SAC 带来。
- 运行真实完整任务可能耗时，先用 --smoke 验证；冒烟仅使用前 3 图和小优化预算，不是收敛实验。

## Shell 与 Make 启动

推荐 `make ui`、`make smoke`、`make check`、`make test`；Shell 等价入口为 `bash run.sh <子命令>`。
默认数据为已配套的 RISC-V test2；DATASET 与 PYTHON 可覆盖。`make smoke DRY_RUN=1` 只查看命令。
复用预测：`make reuse TRAIN_CONFIG=<yaml> PREDICTIONS=<csv> SMOKE=1`。
run.sh 自动进入工程根目录，所有相对路径按该目录解析；不要把参数拼接成未转义的 shell 命令。
启动器复用原 Python 核心，不能将此描述为去除了 PyTorch/Python 依赖。
