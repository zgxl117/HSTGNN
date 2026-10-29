# SAC/MOEA 加固模块

核心来自 /home/zgxl117/acl_clean，原工程保持不变。上游说明保留在 README.upstream.md（历史说明可能与当前配置不一致）。
推荐由根目录 pipeline.py 调用 worker.py。不要把这一目录直接作为 HSTGNN 的 Python 包导入；二者含同名 utils。

- profiles/riscv.yaml：PPA、映射、优化参数。
- worker.py：严格读取当前敏感性，运行真实 SAC/MOEA，导出统一结果。
- data/：现有 RISC-V PPA 和显式 node_id/instance_name 映射。
- UPSTREAM_MANIFEST.json：原始核心源码哈希；algorithms/__init__.py 为导入修复。

原 main.py 留作上游参照。它的数据解析存在随机回退，集成流程不调用该路径。
所有结果写入根工程指定的新 runs/ 目录，原 data/merged_registers.csv 不会被修改。
完整说明见 ../docs/CLOSED_LOOP.md。
