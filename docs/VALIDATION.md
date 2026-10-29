# 独立工程验证记录

日期：2026-10-24。

## 已执行

- 全部 21 个 Python 源文件 AST 语法检查通过。
- 对照 GitHub 提交 `95b8e7bd1db7f3672918e8ee018eba9eb4fd25c2` 的本地上传快照，全部 Python 源文件 SHA-256 一致。
- `configs/train.example.yaml` 与 `configs/predict.example.yaml` 的 model/dataset 一致，checkpoint 目录契约一致。
- 命令：`conda run -n ckt python -m pytest tests -q`。
- 结果：**12 passed in 5.75s**，无跳过或失败。测试使用合成数据，覆盖图数据构建、划分、模型前向与反向传播。

## 测试环境

| 依赖 | 实际版本 |
| --- | --- |
| Python | 3.9.23 |
| torch | 2.4.1+cu121 |
| torch-geometric | 2.6.1 |
| pytorch-lightning | 2.6.0 |
| torchmetrics | 1.8.2 |
| numpy | 2.0.2 |
| pandas | 2.3.1 |
| scikit-learn | 1.6.1 |
| PyYAML | 6.0.2 |
| pytest | 8.4.1 |

原上传 requirements.txt 声明 torch>=2.5，保持未修改；本次使用已有 torch 2.4.1 环境测试通过，
说明仅上述已有测试在该环境可运行，不代表满足依赖清单或完整训练已验证。

## 未执行

未运行真实电路训练、完整 train.py 训练周期、检查点预测或新依赖环境安装。
没有包含真实数据、模型权重、SAT 原型，也没有执行新的 GitHub 提交或推送。

## 最终目录复验

在 `/home/zgxl117/HSTGNN` 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/zgxl117/anaconda3/envs/ckt/bin/python -m pytest tests -q -p no:cacheprovider
```

结果：**12 passed in 3.79s**。本地沙箱首次因无法创建日志被阻止，放行新目录日志写入后测试通过。

## 终端交互界面（2026-10-28）

新增 terminal_ui.py、8 项终端层测试，补充 README 和 AGENTS 操作说明。
在暂存独立工程使用 ckt 解释器执行 `python -m pytest tests -q`：20 passed in 5.46s。
真实 PTY 手动验证菜单启动、环境检查、配置选择、缺失 CSV 提示与正常退出。
已验证 `--diagnose` JSON 输出；未执行真实数据训练或预测。原模型 Python 文件 SHA-256 均保持不变。

## 数据集 YAML 自动配置（2026-10-28）

新增 dataset.yaml 导入、菜单发现、配置路径解析、配对配置生成与 CLI prepare-only。
暂存工程全量回归：26 passed in 3.99s；从任意工作目录对真实 FIFO 元数据执行 prepare-only 成功。
生成模型、优化器、loss、训练轮数与已通过的 FIFO 冒烟配置一致，输出目录按任务隔离。
本次配置功能验证未重新训练模型；模型源文件保持不变。

## 每数据独立 YAML 与目录注册

12 个原始 campaign 各有 dataset.yaml，未改动原 CSV。菜单新增 dataset_sources.yaml 外部根目录发现和 --list-datasets。
暂存工程回归 29 passed in 6.27s；逐个调用导入器：FIFO 可生成配置，其余 11 个明确拒绝缺失输入。
没有给缺少层次信息的旧数据虚构 ff_module.csv，没有运行其余数据的训练。

## 更正数据根目录

用户明确指定 AI4FaultSim/fault-sim/FsimNNs/data/data。此前 fault-sim/data/data 的盘点属于错误目录，其结论不适用于当前数据。
正确目录 15 个 campaign 各有 dataset.yaml；逐个导入验证 6 个输入完整、9 个因缺少必需字段被拒绝。
registry 已改为正确根目录，include_local: false 排除旧冒烟副本；未删除历史产物，未更改 CSV。

## HSTGNN + acl_clean 集成

新增 pipeline.py、export_sensitivity.py、workflow/ 与 hardening/；终端菜单 9 为一键闭环。
原 acl_clean 保持不变；合并副本修复缺失模块导入。模型源文件哈希保持原版本。
回归测试：41 passed in 3.69s。RISC-V test2 前 3 图实测全链路通过：2131 位敏感性，32 个候选，8 条交互，7 次 SAC 更新，6 条已学习策略交互。
该冒烟验证功能通路，不代表收敛或 SAC 优于基线；输出解析 PPA 与理想 TMR 回评，未执行网表生成。

正式目录回归：41 passed in 4.08s。正式目录 RISC-V 闭环冒烟结果位于 runs/closed_loop_riscv_smoke_20261028。
2131 个敏感性条目，32 个最终寄存器，7 次 SAC 更新，6 次策略交互；已核对清单无重复、原始 raw 文件和模型源码哈希未变化。

## 独立仓库发布前验证（2026-10-29）

完善 README 的克隆、安装、Shell/Make、数据 YAML、输出及计算口径说明。
命令：`make test`，解释器 `/home/zgxl117/anaconda3/envs/ckt/bin/python`（Python 3.9.23）。
结果：**41 passed in 6.35s**，无失败或跳过；本次未重复运行真实电路训练。
检查 README 本地链接全部存在；暂存文件排除了训练数据、权重、缓存、日志和自动生成任务配置。
保留 hardening/data/ 的 RISC-V PPA 与寄存器映射，供内置 profile 使用。
