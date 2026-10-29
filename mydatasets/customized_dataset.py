import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd

from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import add_remaining_self_loops

from utils.utils import get_logger

from mydatasets.data_preprocessing import filter_edges
from mydatasets.data_splitting import (
    split_nodes_spatio,
    split_nodes_temporal,
    split_nodes_hybrid,
    split_nodes_stratified,
    split_nodes_feature_coverage,
    split_nodes_grouped_ff_feature_coverage,
    split_nodes_common_test,
)

logger = get_logger('FsimNNDataset')


class CustomGraphDataset(Dataset):
    def __init__(
        self,
        root,
        train_val_dirs,
        test_dirs,
        phase,
        split_mode="spatio",
        split_config=None,
        max_node_feat_len=60,
        max_edge_feat_len=10,
        transform=None,
        pre_transform=None
    ):
        self.root = root
        self.train_val_dirs = train_val_dirs
        self.test_dirs = test_dirs
        self.split_mode = split_mode
        self.split_config = split_config or {}
        self.phase = phase
        self.max_node_feat_len = max_node_feat_len
        self.max_edge_feat_len = max_edge_feat_len
        self.node_id_map = {}

        self.split_file = os.path.join(self.processed_dir, "splits.pkl")
        self.config_file = os.path.join(self.processed_dir, "config.pkl")
        self.train_val_data_file = os.path.join(self.processed_dir, "train_val_processed_data.pt")
        self.test_data_file = os.path.join(self.processed_dir, "test_processed_data.pt")

        super().__init__(root, transform, pre_transform)

        self._ensure_data_is_ready()
        self._load_phase_data()

    @property
    def raw_file_names(self):
        return ["nodes.csv", "edges.csv", "graphs.csv", "ff_module.csv"]

    @property
    def processed_file_names(self):
        return ["splits.pkl", "config.pkl", "train_val_processed_data.pt", "test_processed_data.pt"]

    def _ensure_data_is_ready(self):
        config = self._load_config()

        if self._needs_processing(config):
            logger.info('Need to reprocess the hetero dataset ...')
            self.process()
        else:
            self.train_val_data_list = torch.load(self.train_val_data_file, weights_only=False)
            self.test_data_list = torch.load(self.test_data_file, weights_only=False)

    def _load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "rb") as f:
                return pickle.load(f)
        return None

    def _needs_processing(self, config):
        if (not os.path.exists(self.train_val_data_file)) or (not os.path.exists(self.test_data_file)):
            return True
        if config is None:
            return True

        # HeteroData 版本号：避免复用旧 Data 缓存
        if config.get("dataset_version", None) != "hetero_v2":
            logger.info("Dataset version changed to hetero_v2, need reprocess.")
            return True

        if (
            config.get("max_node_feat_len") != self.max_node_feat_len
            or config.get("max_edge_feat_len") != self.max_edge_feat_len
            or config.get("split_mode") != self.split_mode
            or config.get("split_config") != self.split_config
        ):
            logger.info('The configuration of max_node_feat_len, max_edge_feat_len, split_mode or split_config is changed')
            return True

        return False

    def process(self):
        logger.info('Processing Start ...')
        train_val_data_list = []
        test_data_list = []

        for dir_path in self.train_val_dirs:
            raw_dir = os.path.join(dir_path, "raw")
            train_val_data_list.extend(self._process_graphs(raw_dir))

        if self.train_val_dirs != self.test_dirs:
            for dir_path in self.test_dirs:
                raw_dir = os.path.join(dir_path, "raw")
                test_data_list.extend(self._process_graphs(raw_dir))
        else:
            test_data_list = train_val_data_list

        self.train_val_data_list = train_val_data_list
        self.test_data_list = test_data_list

        splits = self._load_or_create_splits()
        self._apply_splits(splits)

        torch.save(self.train_val_data_list, self.train_val_data_file)
        torch.save(self.test_data_list, self.test_data_file)

        with open(self.config_file, "wb") as f:
            pickle.dump(
                {
                    "dataset_version": "hetero_v2",
                    "split_mode": self.split_mode,
                    "split_config": self.split_config,
                    "max_node_feat_len": self.max_node_feat_len,
                    "max_edge_feat_len": self.max_edge_feat_len,
                },
                f
            )

        logger.info('Processing complete.')

    @staticmethod
    def _safe_circuit_name(raw_dir):
        # 原工程使用 raw_dir.split('/')[2]，路径稍变时容易越界。
        # 这里优先取 raw 的上一级目录名作为 circuit 名。
        return os.path.basename(os.path.dirname(raw_dir)) or "unknown_circuit"

    def _build_module_features(self, ff_x, ff_index, ff2module, num_modules):
        """用模块包含的 FF 节点特征均值初始化 module 节点特征。"""
        module_x = torch.zeros(num_modules, ff_x.size(1), dtype=ff_x.dtype)
        if ff_index.numel() == 0:
            return module_x

        mapped_ff_x = ff_x[ff_index]
        module_x.index_add_(0, ff2module, mapped_ff_x)
        counts = torch.bincount(ff2module, minlength=num_modules).float().unsqueeze(-1)
        module_x = module_x / counts.clamp(min=1.0)
        return module_x

    @staticmethod
    def _node_column_to_tensor(series, dtype=torch.float):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            return torch.tensor(numeric.to_numpy(), dtype=dtype)

        codes, _ = pd.factorize(series.astype(str), sort=True)
        return torch.tensor(codes, dtype=torch.long)

    def _attach_static_node_features(self, data, graph_nodes):
        for column in ["toggle_rate", "duty_cycle", "fanin", "fanout", "hier_id"]:
            if column not in graph_nodes.columns:
                continue

            dtype = torch.long if column == "hier_id" else torch.float
            value = self._node_column_to_tensor(graph_nodes[column], dtype=dtype)
            data["ff"][column] = value
            setattr(data, column, value)

    def _process_graphs(self, raw_dir):
        circuit = self._safe_circuit_name(raw_dir)
        logger.info(f'Process data in {circuit} ...')

        nodes_df = pd.read_csv(os.path.join(raw_dir, "nodes.csv"))
        graphs_df = pd.read_csv(os.path.join(raw_dir, "graphs.csv"))
        module_df = pd.read_csv(os.path.join(raw_dir, "ff_module.csv"))

        orig_edge_file = os.path.join(raw_dir, "edges.csv")
        edge_file = os.path.join(raw_dir, "processed", "filtered_edges.csv")
        filter_edges(orig_edge_file, edge_file, self.max_edge_feat_len)
        edges_df = pd.read_csv(edge_file)

        data_list = []
        for graph_id in graphs_df["graph_id"].unique():
            graph_nodes = nodes_df[nodes_df["graph_id"] == graph_id]
            graph_edges = edges_df[edges_df["graph_id"] == graph_id]
            graph_module = module_df[module_df["graph_id"] == graph_id]

            node_ids = graph_nodes["node_id"].unique()
            node_num = len(node_ids)
            node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
            self.node_id_map[circuit] = node_id_map

            node_features = np.array([
                list(map(float, feat.split(',')))[:self.max_node_feat_len]
                for feat in graph_nodes["feat"]
            ])
            node_labels = graph_nodes["label"].values
            ff_x = torch.tensor(node_features, dtype=torch.float)

            src_nodes = graph_edges["src_id"].map(node_id_map).dropna().astype(int).tolist()
            dst_nodes = graph_edges["dst_id"].map(node_id_map).dropna().astype(int).tolist()
            ff_edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

            edge_features = np.array([
                list(map(float, feat.split(',')))
                for feat in graph_edges["feat"]
            ])
            ff_edge_attr = torch.tensor(edge_features, dtype=torch.float)

            # FF-FF 自环仍然属于同一种边类型：('ff', 'connects', 'ff')
            ff_edge_index, ff_edge_attr = add_remaining_self_loops(
                ff_edge_index,
                edge_attr=ff_edge_attr,
                num_nodes=node_num,
                fill_value=0.
            )

            # ===== 构造真正异构边：FF -> Module 和 Module -> FF =====
            valid_module = graph_module[graph_module["REG_ID"].isin(node_ids)].copy()
            valid_module["ff_idx"] = valid_module["REG_ID"].map(node_id_map)

            ff_index = valid_module["ff_idx"].dropna().astype(int).to_numpy()
            ff2module = valid_module.loc[valid_module["ff_idx"].notna(), "module_id"].astype(int).to_numpy()

            ff_index = torch.tensor(ff_index, dtype=torch.long)
            ff2module = torch.tensor(ff2module, dtype=torch.long)

            if ff2module.numel() == 0:
                num_modules = 1
            else:
                # 保持原始 module_id 编号空间；若 module_id 不连续，中间空 module 会得到零特征。
                num_modules = int(ff2module.max().item()) + 1

            module_x = self._build_module_features(ff_x, ff_index, ff2module, num_modules)

            if ff_index.numel() == 0:
                ff_to_module_edge_index = torch.empty((2, 0), dtype=torch.long)
                module_to_ff_edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                ff_to_module_edge_index = torch.stack([ff_index, ff2module], dim=0)
                module_to_ff_edge_index = torch.stack([ff2module, ff_index], dim=0)

            data = HeteroData()

            # 两类真实节点
            data["ff"].x = ff_x
            data["ff"].y = torch.tensor(node_labels, dtype=torch.long)
            data["ff"].node_ids = torch.tensor(node_ids, dtype=torch.long)
            data["ff"].train_mask = torch.zeros(node_num, dtype=torch.bool)
            data["ff"].val_mask = torch.zeros(node_num, dtype=torch.bool)
            data["ff"].test_mask = torch.zeros(node_num, dtype=torch.bool)
            self._attach_static_node_features(data, graph_nodes)

            data["module"].x = module_x

            # 三类边关系
            data["ff", "connects", "ff"].edge_index = ff_edge_index
            data["ff", "connects", "ff"].edge_attr = ff_edge_attr
            data["ff", "belongs_to", "module"].edge_index = ff_to_module_edge_index
            data["module", "contains", "ff"].edge_index = module_to_ff_edge_index

            # 全局信息
            data.graph_id = torch.tensor([graph_id], dtype=torch.long)
            data.circuit = circuit
            data.source_path = raw_dir

            # 兼容现有 split_nodes_* 函数的顶层别名。
            # 真正训练/验证/测试时仍然使用 data['ff'] 下的字段。
            data.x = data["ff"].x
            data.y = data["ff"].y
            data.node_ids = data["ff"].node_ids
            data.edge_index = data["ff", "connects", "ff"].edge_index
            data.edge_attr = data["ff", "connects", "ff"].edge_attr

            data_list.append(data)

        return data_list

    def _load_or_create_splits(self):
        config = self._load_config()
        if config is not None:
            if (
                config.get("dataset_version") != "hetero_v2"
                or config.get("split_mode") != self.split_mode
                or config.get("split_config") != self.split_config
            ):
                if os.path.exists(self.split_file):
                    os.remove(self.split_file)
                logger.info(
                    f'The split config is changed to mode={self.split_mode}, '
                    f'delete the old split_file: {self.split_file} and generate a new one.'
                )

        if os.path.exists(self.split_file):
            with open(self.split_file, "rb") as f:
                return pickle.load(f)
        else:
            splits = self._create_splits()
            with open(self.split_file, "wb") as f:
                pickle.dump(splits, f)
            return splits

    def _create_splits(self):
        if self.train_val_dirs == self.test_dirs:
            split_test = True
        else:
            split_test = False

        data_list = self.train_val_data_list
        if self.split_mode == 'spatio':
            splits = split_nodes_spatio(data_list, use_test=split_test, **self.split_config)
        elif self.split_mode == 'temporal':
            splits = split_nodes_temporal(data_list, use_test=split_test, **self.split_config)
        elif self.split_mode == 'hybrid':
            splits = split_nodes_hybrid(data_list, use_test=split_test, **self.split_config)
        elif self.split_mode == 'stratified':
            splits = split_nodes_stratified(data_list, use_test=split_test, **self.split_config)
        elif self.split_mode == 'feature_coverage':
            splits = split_nodes_feature_coverage(data_list, use_test=split_test, **self.split_config)
        elif self.split_mode == 'grouped_ff_feature_coverage':
            splits = split_nodes_grouped_ff_feature_coverage(
                data_list, use_test=split_test, **self.split_config
            )
        elif self.split_mode == 'common_test':
            splits = split_nodes_common_test(
                data_list, use_test=split_test, **self.split_config
            )
        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")
            sys.exit()

        if split_test is False:
            splits['test'] = {}
            for data in self.test_data_list:
                gid = int(data.graph_id.item())
                splits['test'][gid] = data["ff"].node_ids[:]

        return splits

    def _apply_splits(self, splits):
        self._apply_splits_phase("train", self.train_val_data_list, splits)
        self._apply_splits_phase("val", self.train_val_data_list, splits)
        self._apply_splits_phase("test", self.test_data_list, splits)

    def _apply_splits_phase(self, phase, data_list, splits):
        for data in data_list:
            gid = int(data.graph_id.item())
            if gid in splits[phase]:
                selected_node_ids = splits[phase][gid].to(data["ff"].node_ids.device)
                phase_mask = torch.isin(data["ff"].node_ids, selected_node_ids)

                if phase == "train":
                    data["ff"].train_mask |= phase_mask
                elif phase == "val":
                    data["ff"].val_mask |= phase_mask
                elif phase == "test":
                    data["ff"].test_mask |= phase_mask
                else:
                    raise ValueError(f"Unknown phase: {phase}")

    def _load_phase_data(self):
        if self.phase == "train":
            self.dataset = [d for d in self.train_val_data_list if d["ff"].train_mask.sum() > 0]
        elif self.phase == "val":
            self.dataset = [d for d in self.train_val_data_list if d["ff"].val_mask.sum() > 0]
        elif self.phase == "test":
            self.dataset = [d for d in self.test_data_list if d["ff"].test_mask.sum() > 0]
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


def DisplayDatasetInfo(dataset, phase_name):
    num_graphs = len(dataset)

    node_counts = []
    graph_ids = []
    module_counts = []
    label_distribution = GetLabelDistribution(dataset, phase_name)
    for data in dataset.dataset:
        if phase_name == 'train':
            node_cnt = data["ff"].train_mask.sum().item()
        elif phase_name == 'val':
            node_cnt = data["ff"].val_mask.sum().item()
        elif phase_name == 'test':
            node_cnt = data["ff"].test_mask.sum().item()
        else:
            node_cnt = data["ff"].num_nodes

        node_counts.append(node_cnt)
        graph_ids.append(int(data.graph_id.item()))
        module_counts.append(data["module"].num_nodes)

    logger.info(f"{phase_name} Dataset:")
    logger.info(f"    Number of Graphs: {num_graphs}")
    logger.info(f"    Graph IDs are: {graph_ids}")
    logger.info(f"    Number of FF Nodes in Each Graph: {node_counts}")
    logger.info(f"    Number of Module Nodes in Each Graph: {module_counts}")
    logger.info(f"    Total Number of FF Nodes: {sum(node_counts)}")
    logger.info(f"    Label Distribution: {label_distribution['counts']}")
    logger.info(f"    Label Ratio: {label_distribution['ratios']}")
    logger.info("-" * 50)


def _phase_mask(data, phase_name):
    if phase_name == 'train':
        return data["ff"].train_mask
    if phase_name == 'val':
        return data["ff"].val_mask
    if phase_name == 'test':
        return data["ff"].test_mask
    return torch.ones(data["ff"].y.size(0), dtype=torch.bool, device=data["ff"].y.device)


def GetLabelDistribution(dataset, phase_name):
    labels = []
    for data in dataset.dataset:
        mask = _phase_mask(data, phase_name)
        if mask.sum().item() > 0:
            labels.append(data["ff"].y[mask].detach().cpu())

    if not labels:
        return {"counts": {}, "ratios": {}}

    labels = torch.cat(labels)
    unique_labels, counts = torch.unique(labels, sorted=True, return_counts=True)
    total = counts.sum().item()

    count_dict = {
        int(label.item()): int(count.item())
        for label, count in zip(unique_labels, counts)
    }
    ratio_dict = {
        label: count / total
        for label, count in count_dict.items()
    }
    return {"counts": count_dict, "ratios": ratio_dict}
