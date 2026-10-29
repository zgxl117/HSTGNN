# 数据格式与路径

## 目录结构

```text
data/circuit/
  campaign/
    raw/
      nodes.csv
      edges.csv
      graphs.csv
      ff_module.csv
  cache_run_001/       # dataset.root_dir，独立的可写缓存目录
```

`dataset.train_val_dirs` 和 `dataset.test_dirs` 指向 campaign 目录，程序会自动拼接 `raw/`。
它们不指向 raw 目录本身。`root_dir` 是 PyG 缓存工作目录，不是原始 CSV 的替代位置。
相对路径按启动命令的 cwd 解析，推荐其他 agent 配置绝对路径。
训练/验证与测试列表一致时，在同一 campaign 内创建划分；不一致时测试 campaign 全部节点进入测试。
多 campaign 使用时保持 graph_id 不冲突；代码部分划分以 graph_id 为键，未实现跨来源的复合键。

## 四类输入

| 文件 | 字段 | 说明 |
| --- | --- | --- |
| nodes.csv | graph_id, node_id, label, feat | 每行一个 FF 样本；同一 graph_id 内 node_id 唯一且为整数 |
| nodes.csv | toggle_rate, duty_cycle, fanin, fanout, hier_id | 静态/活动字段；FeatureCoverage 使用对应配置列，应提供 |
| edges.csv | graph_id, src_id, dst_id, feat | 有向 FF 关系，端点必须存在于同图 nodes |
| graphs.csv | graph_id | 需要构建的图 ID |
| ff_module.csv | graph_id, REG_ID, module_id | REG_ID 对应 node_id；module_id 是非负整数 |

`feat` 是一个 CSV 单元格里的逗号分隔数值序列，写 CSV 时需要双引号包裹。例如：

```csv
graph_id,node_id,label,feat,toggle_rate,duty_cycle,fanin,fanout,hier_id
1,1,0,"0,1,0,1",0.75,0.5,2,3,0
```

这只是字段示意，不是可用于默认 60 点窗口配置的完整数据。

## 标签与张量

默认 `num_classes=5`，label 为 0..4；0 为无错误，1..4 是异常、安全、结果、超时等故障类别。
具体 1..4 的语义必须核对数据生成协议；代码不会验证自然语言类别名。

- `data['ff'].x`：`[N_ff, T]`。
- `data['ff'].y`、`node_ids` 与三个 mask：`[N_ff]`。
- `data['module'].x`：`[N_module, T]`，由所属 FF 活动均值初始化。
- `('ff','connects','ff')`：`edge_index [2,E]` 与 `edge_attr [E,D]`。
- 另外两类关系：`('ff','belongs_to','module')`、`('module','contains','ff')`。

节点窗口会被截断到 `time_window_size`，不会自动补齐过短序列；所有行应具有一致且足够的长度。
`num_nodes` 应匹配图的 FF 数；不要假定当前结构能直接混合不同节点规模的电路。

## 边处理

`filter_edges()` 先检查原边 feat 从 `edge_vector_size` 起的尾部，含非零值的边会被过滤；
再截取前 `edge_vector_size` 个分量，对留下的边执行 PCA，输出维度为 `edge_vector_size - 2`。
因此每条原始边需有足够的特征长度，保留边数也需足够支撑该 PCA 维度。
所有 src_id/dst_id 必须有效，当前实现不是完整的数据校验器。

FeatureCoverage 只使用配置中的预注入特征和层次信息。不要使用未注入样本的故障标签设计采样。
预测入口仍读取 label 并计算评估指标；未知标签部署需要另行设计不依赖真值的推理入口。
