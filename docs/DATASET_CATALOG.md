# FsimNNs 数据集 YAML 清单

当前数据根目录：`/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data`。

每个 campaign 的 dataset.yaml 与 raw/ 同级。此前 fault-sim/data 的清单已被本清单替代。

| 数据集 | 每图 FF | 图数 | 标签值 | 当前缺项 |
| --- | ---: | ---: | --- | --- |
| [ethmac](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/ethmac/dataset.yaml) | 1233 | 32 | [0, 1] | 缺少 raw/ff_module.csv；nodes.csv 缺少 duty_cycle,fanin,fanout,hier_id,toggle_rate |
| [fifo_test1](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/fifo/test1/dataset.yaml) | 60 | 28 | [0, 2, 3, 4] | 无 |
| [i2c](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/i2c/dataset.yaml) | 153 | 50 | [0, 1] | 缺少 raw/ff_module.csv；nodes.csv 缺少 duty_cycle,fanin,fanout,hier_id,toggle_rate |
| [i2c_5class_test_test1](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/i2c_5class_test/test1/dataset.yaml) | 153 | 50 | [0, 1, 2, 3, 4] | 无 |
| [i2c_test_test1](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/i2c_test/test1/dataset.yaml) | 123 | 50 | [0, 1] | nodes.csv 缺少 duty_cycle,fanin,fanout,hier_id,toggle_rate |
| [ibex_test2](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/ibex/test2/dataset.yaml) | 2126 | 46 | [0, 1] | 缺少 raw/ff_module.csv；nodes.csv 缺少 duty_cycle,fanin,fanout,hier_id,toggle_rate |
| [ibex_test3](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/ibex/test3/dataset.yaml) | 2126 | 20 | [0, 1] | 缺少 raw/ff_module.csv；nodes.csv 缺少 duty_cycle,fanin,fanout,hier_id,toggle_rate |
| [ibex_test4](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/ibex/test4/dataset.yaml) | 2126 | 20 | [0, 1] | 缺少 raw/ff_module.csv；nodes.csv 缺少 duty_cycle,fanin,fanout,hier_id,toggle_rate |
| [riscy_test1](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/riscy/test1/dataset.yaml) | 3041 | 40 | [0, 1] | 缺少 raw/ff_module.csv；nodes.csv 缺少 duty_cycle,fanin,fanout,hier_id,toggle_rate |
| [riscy_test_test1](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/riscy_test/test1/dataset.yaml) | 2131 | 50 | [0, 1, 2, 3, 4] | 无 |
| [riscy_test_test2](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/riscy_test/test2/dataset.yaml) | 2131 | 50 | [0, 1, 2, 3, 4] | 无 |
| [riscy_test_test3](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/riscy_test/test3/dataset.yaml) | 2131 | 50 | [0, 1, 2, 3, 4] | 无 |
| [spi](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/spi/dataset.yaml) | 131 | 50 | [0, 1] | 缺少 raw/ff_module.csv；nodes.csv 缺少 duty_cycle,fanin,fanout,hier_id,toggle_rate |
| [spi_5class_test1](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/spi_5class/test1/dataset.yaml) | 229 | 50 | [0, 1, 3, 4] | 无 |
| [xgemac](/home/zgxl117/AI4FaultSim/fault-sim/FsimNNs/data/data/xgemac/dataset.yaml) | 1358 | 36 | [0, 1] | 缺少 raw/ff_module.csv；nodes.csv 缺少 duty_cycle,fanin,fanout,hier_id,toggle_rate |

输入完整的 6 个 campaign：FIFO、I2C 五分类、SPI 五分类、RISC-V test1/test2/test3。
其余 9 个含旧二分类数据或缺少模型必需特征/模块映射；未虚构数据来补齐。
每份 YAML 使用对应数据的真实维度；默认 2 轮用于冒烟，正式训练自行调整 trainer.epochs。
dataset_sources.yaml 使用 include_local: false，只显示正确根目录中的数据，保留旧冒烟副本作为历史产物。
