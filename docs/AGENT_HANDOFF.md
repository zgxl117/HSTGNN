# 给其他 agent 的调用说明

## 推荐任务指令

可以给另一个 agent 如下指令，补齐尖括号中的参数：

> 在 `/home/zgxl117/HSTGNN` 工作，先读 AGENTS.md。使用解释器 `<python路径>`，
> 数据 campaign 为 `<绝对目录>`，任务名 `<run_id>`。先运行已有测试并检查四类 CSV。
> 从 configs 模板生成配对任务配置，核对模型维度及标签定义；缓存和检查点放在新的 runs/<run_id>/ 下。
> 训练后用对应 checkpoint 评估，汇报命令、指标、配置、环境版本和产物路径。

## 从任意位置启动

不依赖 shell 的调用方式：

```python
from pathlib import Path
import subprocess

project = Path('/home/zgxl117/HSTGNN')
python = '/path/to/environment/bin/python'
subprocess.run([python, 'train.py', '--config', 'configs/task_train.yaml'],
               cwd=project, check=True)
subprocess.run([python, 'predict.py', '--config', 'configs/task_predict.yaml'],
               cwd=project, check=True)
```

任务 YAML 应由该次任务单独创建；模板未包含真实数据。

## 模型接口

```python
from models.models import get_model
model = get_model(config)  # config['model']['name'] == 'ASPP_STGATs'
logits = model(hetero_data) # FF 节点的分类 logits，典型形状 [N_ff, num_classes]
```

从其他项目导入时，使用独立子进程并设置 cwd；此工程存在 `models`、`utils` 等通用顶层包名，
直接加入全局 sys.path 可能与调用工程冲突。当前没有 pip 可安装包接口或服务 API。

## 输出契约

- 训练：`trainer.checkpoint_dir/best_model.ckpt`；Lightning 日志默认在 `lightning_logs/`。
- 预测：`output_path/results.csv`，列包括 source_path、graph_id、node_id、predicted_label、true_label。
- 总体指标：`scores.csv`，包含 Accuracy、Macro_Precision、Macro_Recall、Macro_F1。
- 分类别指标：`per_class_scores.csv`。
- 预测默认 batch_size=1，只导出 test_mask 所选节点，不是全图全部节点。
- `train_hierarchical.py` 是可选层级训练入口；不能直接假定默认 predict.py 支持两阶段模型合并推理。

## 已知兼容性与排障

- 默认 base 环境可能没有 torch；先确认使用的 Python。
- CUDA 可用不代表版本匹配，保留实际 torch/PyG/Lightning 版本与报错信息。
- 旧训练代码中 ReduceLROnPlateau 使用 verbose 参数，较新 PyTorch 可能不兼容；不要把宽松依赖下的安装成功当成训练成功。
- 修改 CSV 或数据路径后使用新缓存目录。原缓存检查未完整追踪原文件内容、路径与随机种子。
- 调整窗口/节点数/模型结构后不能继续使用不匹配的 checkpoint。
- 模板是配对配置，根目录两个 YAML 是不同电路的历史配置。
- 工程不含真实数据和权重，因此交付验证主要依靠已有合成数据回归测试。

导入部分模块时日志器即会在当前工作目录创建或追加 `fault_injection.txt`；即使只做测试也需要工作目录可写。
