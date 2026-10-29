# 从数据集 YAML 自动配置

## 使用方式

每个数据 campaign 与 raw/ 同级放置 dataset.yaml：

```text
fifo/test1/
  dataset.yaml
  raw/nodes.csv
  raw/edges.csv
  raw/graphs.csv
  raw/ff_module.csv
```

终端中选 **2 从数据 YAML 配置**，再选数据编号；也可以直接输入数据目录或 YAML 路径。
列表扫描工程 data/ 下的 dataset.yaml。导入后自动选中该次训练/预测配置，选 5 训练、选 6 预测。
不再手动填写任务名、节点数、窗口和轮数。默认任务名为数据名称加时间戳，每次使用新的输出目录。

也可以启动时直接提供数据：

```bash
python terminal_ui.py --dataset /path/to/fifo/test1/dataset.yaml
# 只生成配置，适合 agent 调用：
python terminal_ui.py --dataset /path/to/fifo/test1 --prepare-only --task-name fifo_trial_001
```

第二条输出 JSON，包含 train_config 与 predict_config；不会启动训练。
再次使用相同 task-name 会拒绝覆盖，省略则自动生成新名字。

## 最小示例

```yaml
schema_version: 1
name: fifo_test1
model:
  num_nodes: 60
  time_window_size: 60
  edge_vector_size: 8
  num_classes: 5
trainer:
  epochs: 2
  batch_size: 1
dataset:
  train_val_dirs: ['.']
  test_dirs: ['.']
```

四个 model 维度与 name/schema_version 必须声明；未指定的网络与优化参数继承 configs/train.example.yaml。
如果 num_classes 不是 5，需同时给出长度匹配的 loss.class_weights。
空白 dataset 默认 train_val_dirs/test_dirs 均为 ['.']。

## 路径与参数

- 数据路径相对于 **dataset.yaml 所在目录**，不是 shell 当前目录，也不是工程根目录。
- 可以把 dataset.yaml 放在电路父目录，train_val_dirs 为 ['test1']、test_dirs 为 ['test2']。
- 生成的任务 YAML 中数据路径转成绝对路径；cache/checkpoints/predictions 始终在工程 runs/<任务>/ 下。
- 不允许在数据 YAML 设置 root_dir、checkpoint_dir 或预测输出路径。
- 可选部分：manual_seed、description、model、trainer、optimizer、loss。
- model/trainer/optimizer/loss 的覆盖字段按 train.example.yaml 校验，拼错或未知字段报错；不要写执行命令。
- 当前数据 YAML 导入支持 feature_coverage 划分；dataset 可设置 train_val_dirs、test_dirs、split_mode、feature_coverage。
- FeatureCoverage 可设置 train_ratio、val_ratio、bins、min_per_group、group_feature、features。
- 训练和预测生成相同的 model、dataset 和 seed；配置中记录数据 YAML 的路径与 SHA-256。
- 导入检查 CSV 文件及表头，但不代替节点维度、标签语义与 PCA 可行性验证。
- 更改数据 YAML 后重新导入才能生成新配置；已经生成的任务配置不会随 YAML 变化。

参考模板：configs/dataset.example.yaml。该模板需复制到数据目录后使用。
FIFO 已配置好默认 2 轮，目的是快速跑通流程；正式训练请调整数据 YAML 中的 trainer.epochs。

## 多数据目录

每个 campaign 自带自己的 dataset.yaml，不共用一个 FIFO 配置。
工程根目录 `dataset_sources.yaml` 注册数据根目录，例如：

```yaml
roots:
  - ../AI4FaultSim/fault-sim/FsimNNs/data/data
```

相对路径以工程根目录解析；界面会递归查找这些目录和本地 data/ 下的 dataset.yaml，并去重。
目录不存在时跳过，便于工程搬迁。也可直接通过 --dataset 指定某个 YAML，无需注册。
`python terminal_ui.py --list-datasets` 输出所有 YAML 路径，供其他 agent 调用。
原始数据清单和缺项见 DATASET_CATALOG.md。目前正确目录中的 15 个 campaign 均有独立 YAML，6 个具备当前模型所需字段。

当前 registry 设置 include_local: false，不将旧 smoke 副本放入默认菜单；设为 true 可重新包括工程 data/。
当前实际数据目录为 AI4FaultSim/fault-sim/FsimNNs/data/data，避免与 fault-sim/data/data 混淆。
