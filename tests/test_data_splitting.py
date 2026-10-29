import pytest

torch = pytest.importorskip("torch")

from mydatasets.data_splitting import (
    split_nodes_feature_coverage,
    split_nodes_grouped_ff_feature_coverage,
    split_nodes_common_test,
    split_nodes_spatio,
    split_nodes_stratified,
)


class DummyData:
    def __init__(self, graph_id, node_ids, labels):
        self.graph_id = torch.tensor([graph_id])
        self.node_ids = torch.tensor(node_ids, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)


class FeatureCoverageData(DummyData):
    def __init__(self):
        super().__init__(
            graph_id=3,
            node_ids=list(range(12)),
            labels=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        )
        self.hier_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        self.toggle_rate = torch.linspace(0.0, 1.0, steps=12)
        self.duty_cycle = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9])
        self.fanin = torch.tensor([1, 1, 2, 3, 1, 4, 2, 5, 1, 3, 6, 8], dtype=torch.float)
        self.fanout = torch.tensor([2, 3, 1, 1, 4, 1, 3, 2, 5, 1, 2, 7], dtype=torch.float)


class TimeVaryingFeatureCoverageData(FeatureCoverageData):
    def __init__(self, graph_id, order, activity_offset=0.0):
        super().__init__()
        order = torch.tensor(order, dtype=torch.long)
        self.graph_id = torch.tensor([graph_id])
        for name in (
            "node_ids",
            "y",
            "hier_id",
            "toggle_rate",
            "duty_cycle",
            "fanin",
            "fanout",
        ):
            setattr(self, name, getattr(self, name)[order])
        self.toggle_rate = self.toggle_rate + activity_offset
        self.duty_cycle = self.duty_cycle + activity_offset / 2


def _as_set(tensor):
    return set(tensor.tolist())


def test_spatio_split_covers_nodes_without_overlap():
    data = DummyData(graph_id=0, node_ids=list(range(10)), labels=[0] * 10)

    split = split_nodes_spatio([data], train_ratio=0.6, val_ratio=0.2, use_test=True)

    train = _as_set(split["train"][0])
    val = _as_set(split["val"][0])
    test = _as_set(split["test"][0])

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(10))
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2


def test_spatio_split_without_test_normalizes_train_val_ratios():
    data = DummyData(graph_id=1, node_ids=list(range(10)), labels=[0] * 10)

    split = split_nodes_spatio([data], train_ratio=0.6, val_ratio=0.2, use_test=False)

    assert "test" not in split

    train = _as_set(split["train"][1])
    val = _as_set(split["val"][1])

    assert train.isdisjoint(val)
    assert train | val == set(range(10))
    assert len(train) == 7
    assert len(val) == 3


def test_stratified_split_keeps_each_label_in_train():
    data = DummyData(
        graph_id=2,
        node_ids=list(range(12)),
        labels=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    )

    split = split_nodes_stratified([data], train_ratio=0.6, val_ratio=0.2, use_test=True)

    train_nodes = split["train"][2]
    val_nodes = split["val"][2]
    test_nodes = split["test"][2]

    assert _as_set(train_nodes).isdisjoint(_as_set(val_nodes))
    assert _as_set(train_nodes).isdisjoint(_as_set(test_nodes))
    assert _as_set(val_nodes).isdisjoint(_as_set(test_nodes))

    train_labels = data.y[train_nodes]
    assert set(train_labels.tolist()) == {0, 1, 2}


def test_feature_coverage_split_uses_static_features_without_label_stratification():
    data = FeatureCoverageData()

    split = split_nodes_feature_coverage(
        [data],
        train_ratio=0.25,
        val_ratio=0.25,
        use_test=True,
        bins=3,
    )

    train = _as_set(split["train"][3])
    val = _as_set(split["val"][3])
    test = _as_set(split["test"][3])

    assert len(train) == 3
    assert len(val) == 3
    assert len(test) == 6
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(12))

    sampled_hiers = set(data.hier_id[list(train)].tolist())
    assert sampled_hiers


def test_grouped_ff_split_keeps_physical_ff_exclusive_across_all_times():
    graph0 = TimeVaryingFeatureCoverageData(0, list(range(12)))
    graph1 = TimeVaryingFeatureCoverageData(
        1, list(reversed(range(12))), activity_offset=0.2
    )

    split = split_nodes_grouped_ff_feature_coverage(
        [graph0, graph1],
        train_ratio=0.5,
        val_ratio=0.25,
        use_test=True,
        bins=3,
        split_seed=17,
    )

    phase_sets = {}
    for phase, expected_size in (("train", 6), ("val", 3), ("test", 3)):
        phase_sets[phase] = _as_set(split[phase][0])
        assert phase_sets[phase] == _as_set(split[phase][1])
        assert len(phase_sets[phase]) == expected_size

    assert phase_sets["train"].isdisjoint(phase_sets["val"])
    assert phase_sets["train"].isdisjoint(phase_sets["test"])
    assert phase_sets["val"].isdisjoint(phase_sets["test"])
    assert set().union(*phase_sets.values()) == set(range(12))


def test_grouped_ff_split_is_reproducible_and_does_not_use_labels():
    graph0 = TimeVaryingFeatureCoverageData(0, list(range(12)))
    graph1 = TimeVaryingFeatureCoverageData(1, list(range(12)), activity_offset=0.1)

    first = split_nodes_grouped_ff_feature_coverage(
        [graph0, graph1], split_seed=73
    )
    graph0.y = torch.flip(graph0.y, dims=[0])
    graph1.y = torch.zeros_like(graph1.y)
    second = split_nodes_grouped_ff_feature_coverage(
        [graph0, graph1], split_seed=73
    )

    for phase in ("train", "val", "test"):
        assert _as_set(first[phase][0]) == _as_set(second[phase][0])


def test_common_test_is_identical_across_samplers_and_paired_seeds():
    graph0 = TimeVaryingFeatureCoverageData(0, list(range(12)))
    graph1 = TimeVaryingFeatureCoverageData(1, list(range(12)), activity_offset=0.1)
    data_list = [graph0, graph1]

    splits = {
        (mode, seed): split_nodes_common_test(
            data_list,
            base_mode=mode,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            fixed_test_seed=2718,
            split_seed=seed,
        )
        for mode in ("temporal", "spatio", "hybrid", "feature_coverage")
        for seed in (17, 73)
    }

    reference_test = {
        graph_id: _as_set(nodes)
        for graph_id, nodes in splits[("spatio", 17)]["test"].items()
    }
    for split in splits.values():
        assert {
            graph_id: _as_set(nodes) for graph_id, nodes in split["test"].items()
        } == reference_test
        phase_totals = {
            phase: sum(nodes.numel() for nodes in split[phase].values())
            for phase in ("train", "val", "test")
        }
        assert phase_totals == {"train": 12, "val": 6, "test": 6}
        for graph_id in (0, 1):
            train = _as_set(split["train"][graph_id])
            val = _as_set(split["val"][graph_id])
            test = _as_set(split["test"][graph_id])
            assert train.isdisjoint(val)
            assert train.isdisjoint(test)
            assert val.isdisjoint(test)
            assert train | val | test == set(range(12))

    test_labels = []
    for data in data_list:
        selected_ids = splits[("spatio", 17)]["test"][int(data.graph_id)]
        mask = torch.isin(data.node_ids, selected_ids)
        test_labels.extend(data.y[mask].tolist())
    assert set(test_labels) == {0, 1}
