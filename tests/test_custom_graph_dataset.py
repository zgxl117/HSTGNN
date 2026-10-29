import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from mydatasets.customized_dataset import CustomGraphDataset, GetLabelDistribution


def write_minimal_raw_dataset(base_dir):
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True)

    rows = []
    for node_id in range(15):
        label = node_id % 3
        feat = ",".join(str(float(node_id + offset)) for offset in range(6))
        rows.append(
            {
                "graph_id": 0,
                "node_id": node_id,
                "feat": feat,
                "label": label,
                "toggle_rate": node_id / 15.0,
                "duty_cycle": (node_id % 5) / 5.0,
                "fanin": node_id % 4,
                "fanout": (node_id + 1) % 4,
                "hier_id": node_id % 3,
            }
        )
    pd.DataFrame(rows).to_csv(raw_dir / "nodes.csv", index=False)

    edge_rows = []
    for src_id in range(15):
        dst_id = (src_id + 1) % 15
        feat = ",".join(
            [
                str(float(src_id % 3)),
                str(float(dst_id % 3)),
                "0.0",
                "0.0",
            ]
        )
        edge_rows.append(
            {
                "graph_id": 0,
                "src_id": src_id,
                "dst_id": dst_id,
                "feat": feat,
            }
        )
    pd.DataFrame(edge_rows).to_csv(raw_dir / "edges.csv", index=False)

    pd.DataFrame([{"graph_id": 0}]).to_csv(raw_dir / "graphs.csv", index=False)

    module_rows = []
    for node_id in range(15):
        module_rows.append(
            {
                "graph_id": 0,
                "REG_ID": node_id,
                "module_id": node_id % 3,
            }
        )
    pd.DataFrame(module_rows).to_csv(raw_dir / "ff_module.csv", index=False)


def test_custom_graph_dataset_builds_expected_heterodata(tmp_path):
    data_dir = tmp_path / "dataset1"
    write_minimal_raw_dataset(data_dir)

    dataset = CustomGraphDataset(
        root=str(tmp_path / "processed_root"),
        train_val_dirs=[str(data_dir)],
        test_dirs=[str(data_dir)],
        phase="train",
        split_mode="feature_coverage",
        max_node_feat_len=6,
        max_edge_feat_len=4,
    )

    assert len(dataset) == 1

    data = dataset[0]

    assert "ff" in data.node_types
    assert "module" in data.node_types
    assert ("ff", "connects", "ff") in data.edge_types
    assert ("ff", "belongs_to", "module") in data.edge_types
    assert ("module", "contains", "ff") in data.edge_types

    assert data["ff"].x.shape == (15, 6)
    assert data["ff"].y.shape == (15,)
    assert data["module"].x.shape == (3, 6)

    assert data["ff", "connects", "ff"].edge_attr.shape[1] == 2
    assert data["ff"].train_mask.shape == (15,)
    assert data["ff"].val_mask.shape == (15,)
    assert data["ff"].test_mask.shape == (15,)
    assert data["ff"].train_mask.sum().item() > 0
    assert data["ff"].toggle_rate.shape == (15,)
    assert data["ff"].duty_cycle.shape == (15,)
    assert data["ff"].fanin.shape == (15,)
    assert data["ff"].fanout.shape == (15,)
    assert data["ff"].hier_id.shape == (15,)


def test_custom_graph_dataset_supports_feature_coverage_split(tmp_path):
    data_dir = tmp_path / "dataset1"
    write_minimal_raw_dataset(data_dir)

    dataset = CustomGraphDataset(
        root=str(tmp_path / "processed_root"),
        train_val_dirs=[str(data_dir)],
        test_dirs=[str(data_dir)],
        phase="train",
        split_mode="feature_coverage",
        split_config={
            "train_ratio": 0.2,
            "val_ratio": 0.2,
            "bins": 3,
            "min_per_group": 1,
        },
        max_node_feat_len=6,
        max_edge_feat_len=4,
    )

    assert len(dataset) == 1
    data = dataset[0]
    assert data["ff"].train_mask.sum().item() == 3
    assert data["ff"].val_mask.sum().item() == 3
    assert data["ff"].test_mask.sum().item() == 9

    train_dist = GetLabelDistribution(dataset, "train")
    assert sum(train_dist["counts"].values()) == 3
    assert set(train_dist["counts"]).issubset({0, 1, 2})
    assert torch.isclose(
        torch.tensor(sum(train_dist["ratios"].values())),
        torch.tensor(1.0),
    )


def test_custom_graph_dataset_supports_grouped_ff_split(tmp_path):
    data_dir = tmp_path / "dataset1"
    write_minimal_raw_dataset(data_dir)

    dataset = CustomGraphDataset(
        root=str(tmp_path / "processed_root"),
        train_val_dirs=[str(data_dir)],
        test_dirs=[str(data_dir)],
        phase="train",
        split_mode="grouped_ff_feature_coverage",
        split_config={
            "train_ratio": 0.6,
            "val_ratio": 0.1,
            "split_seed": 3407,
        },
        max_node_feat_len=6,
        max_edge_feat_len=4,
    )

    data = dataset[0]
    train_ids = set(data["ff"].node_ids[data["ff"].train_mask].tolist())
    val_ids = set(data["ff"].node_ids[data["ff"].val_mask].tolist())
    test_ids = set(data["ff"].node_ids[data["ff"].test_mask].tolist())

    assert (len(train_ids), len(val_ids), len(test_ids)) == (9, 2, 4)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(range(15))


def test_custom_graph_dataset_reuses_processed_cache(tmp_path):
    data_dir = tmp_path / "dataset1"
    root_dir = tmp_path / "processed_root"
    write_minimal_raw_dataset(data_dir)

    first_dataset = CustomGraphDataset(
        root=str(root_dir),
        train_val_dirs=[str(data_dir)],
        test_dirs=[str(data_dir)],
        phase="train",
        split_mode="feature_coverage",
        max_node_feat_len=6,
        max_edge_feat_len=4,
    )
    second_dataset = CustomGraphDataset(
        root=str(root_dir),
        train_val_dirs=[str(data_dir)],
        test_dirs=[str(data_dir)],
        phase="train",
        split_mode="feature_coverage",
        max_node_feat_len=6,
        max_edge_feat_len=4,
    )

    assert len(first_dataset) == len(second_dataset) == 1
    assert (root_dir / "processed" / "train_val_processed_data.pt").exists()
    assert (root_dir / "processed" / "test_processed_data.pt").exists()
    assert (root_dir / "processed" / "splits.pkl").exists()
    assert (root_dir / "processed" / "config.pkl").exists()
