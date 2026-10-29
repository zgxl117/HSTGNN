import random
from types import SimpleNamespace
from typing import List, Dict
import torch


DEFAULT_COVERAGE_FEATURES = ("toggle_rate", "duty_cycle", "fanin", "fanout")


def _graph_id(data):
    return int(data.graph_id)


def _empty_like_node_ids(node_ids):
    return torch.tensor([], dtype=node_ids.dtype, device=node_ids.device)


def _get_node_attr(data, name):
    if hasattr(data, "node_types") and "ff" in data.node_types:
        store = data["ff"]
        if name in store:
            return store[name]
    if hasattr(data, name):
        return getattr(data, name)
    raise ValueError(f"feature_coverage split requires node attribute '{name}'")


def _quantile_bucketize(values, bins):
    values = values.float()
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    if bins <= 1 or torch.all(values == values[0]):
        return torch.zeros(values.numel(), dtype=torch.long, device=values.device)

    quantiles = torch.linspace(0, 1, bins + 1, device=values.device)[1:-1]
    boundaries = torch.unique(torch.quantile(values, quantiles))
    if boundaries.numel() == 0:
        return torch.zeros(values.numel(), dtype=torch.long, device=values.device)
    return torch.bucketize(values, boundaries)


def _coverage_sample_indices(
    data,
    train_ratio,
    features,
    group_feature,
    bins,
    min_per_group,
    generator=None,
):
    node_ids = data.node_ids
    num_nodes = node_ids.numel()
    if num_nodes == 0:
        return torch.empty(0, dtype=torch.long, device=node_ids.device)

    target = int(round(train_ratio * num_nodes))
    target = max(1, min(target, num_nodes))

    group_values = _get_node_attr(data, group_feature).reshape(-1)
    feature_bins = [
        _quantile_bucketize(_get_node_attr(data, feature).reshape(-1), bins)
        for feature in features
    ]

    groups = {}
    for idx in range(num_nodes):
        key = [int(group_values[idx].item())]
        key.extend(int(feature_bin[idx].item()) for feature_bin in feature_bins)
        groups.setdefault(tuple(key), []).append(idx)

    group_items = list(groups.values())
    group_order = torch.randperm(len(group_items), generator=generator).tolist()

    selected = []
    selected_set = set()
    quota_per_group = max(1, int(min_per_group))

    for group_idx in group_order:
        if len(selected) >= target:
            break
        group = group_items[group_idx]
        order = torch.randperm(len(group), generator=generator).tolist()
        for local_idx in order[:quota_per_group]:
            if len(selected) >= target:
                break
            node_idx = group[local_idx]
            selected.append(node_idx)
            selected_set.add(node_idx)

    if len(selected) < target:
        remaining = [idx for idx in range(num_nodes) if idx not in selected_set]
        remaining_order = torch.randperm(
            len(remaining), generator=generator
        ).tolist()
        for local_idx in remaining_order:
            if len(selected) >= target:
                break
            selected.append(remaining[local_idx])

    return torch.tensor(selected, dtype=torch.long, device=node_ids.device)


def _aggregate_grouped_ff_features(data_list, features, group_feature):
    """Build one feature row per physical FF across all injection times.

    Activity and structural features are averaged over time.  The hierarchy
    group is required to be invariant because it identifies a physical design
    partition rather than an injection outcome.  Labels are deliberately not
    accessed here.
    """
    if not data_list:
        raise ValueError("grouped-by-FF split requires at least one graph")

    first_node_ids = data_list[0].node_ids.detach().cpu().reshape(-1)
    if torch.unique(first_node_ids).numel() != first_node_ids.numel():
        raise ValueError("node_ids must be unique within every graph")

    ordered_node_ids = torch.sort(first_node_ids).values
    reference_ids = set(ordered_node_ids.tolist())
    requested_attrs = tuple(dict.fromkeys((group_feature, *features)))
    aligned_values = {name: [] for name in requested_attrs}

    for data in data_list:
        node_ids = data.node_ids.detach().cpu().reshape(-1)
        if torch.unique(node_ids).numel() != node_ids.numel():
            raise ValueError(
                f"graph {_graph_id(data)} contains duplicate physical node_ids"
            )
        if set(node_ids.tolist()) != reference_ids:
            raise ValueError(
                "grouped-by-FF split requires the same physical FF set in "
                f"every graph; graph {_graph_id(data)} differs"
            )

        index_by_id = {
            int(node_id): index for index, node_id in enumerate(node_ids.tolist())
        }
        align_indices = torch.tensor(
            [index_by_id[int(node_id)] for node_id in ordered_node_ids.tolist()],
            dtype=torch.long,
        )

        for name in requested_attrs:
            values = _get_node_attr(data, name).detach().cpu().reshape(-1)
            if values.numel() != node_ids.numel():
                raise ValueError(
                    f"node attribute '{name}' has {values.numel()} values for "
                    f"{node_ids.numel()} FFs in graph {_graph_id(data)}"
                )
            aligned_values[name].append(values[align_indices])

    group_stack = torch.stack(aligned_values[group_feature])
    if not torch.all(group_stack == group_stack[0]):
        raise ValueError(
            f"group attribute '{group_feature}' changes across injection times"
        )

    aggregate = SimpleNamespace(node_ids=ordered_node_ids)
    setattr(aggregate, group_feature, group_stack[0])
    for name in features:
        feature_stack = torch.stack(aligned_values[name]).float()
        setattr(aggregate, name, feature_stack.mean(dim=0))
    return aggregate


def _node_ids_in_set(node_ids, selected_ids):
    selected = set(selected_ids.tolist())
    mask = torch.tensor(
        [int(node_id) in selected for node_id in node_ids.detach().cpu().tolist()],
        dtype=torch.bool,
        device=node_ids.device,
    )
    return node_ids[mask]


def _allocate_exact_counts(weights, total, capacities=None, minimums=None):
    """Allocate an integer total proportionally with deterministic tie breaking."""
    if total < 0:
        raise ValueError("allocation total must be non-negative")
    count = len(weights)
    capacities = list(capacities if capacities is not None else [total] * count)
    minimums = list(minimums if minimums is not None else [0] * count)
    if not (len(capacities) == len(minimums) == count):
        raise ValueError("weights, capacities, and minimums must have equal lengths")
    if sum(minimums) > total or sum(capacities) < total:
        raise ValueError("requested total is incompatible with allocation bounds")

    allocation = minimums[:]
    remaining = total - sum(allocation)
    residual_weights = [max(0.0, float(value)) for value in weights]
    while remaining:
        eligible = [i for i in range(count) if allocation[i] < capacities[i]]
        if not eligible:
            raise RuntimeError("allocation exhausted all capacities")
        denominator = sum(residual_weights[i] for i in eligible)
        if denominator <= 0:
            for index in eligible[:remaining]:
                allocation[index] += 1
            remaining = total - sum(allocation)
            continue

        quotas = {
            i: remaining * residual_weights[i] / denominator for i in eligible
        }
        added = 0
        for index in eligible:
            increment = min(
                int(quotas[index]), capacities[index] - allocation[index]
            )
            allocation[index] += increment
            added += increment
        remaining -= added
        if not remaining:
            break

        order = sorted(
            (i for i in eligible if allocation[i] < capacities[i]),
            key=lambda i: (-(quotas[i] - int(quotas[i])), i),
        )
        if not order:
            continue
        for index in order[:remaining]:
            allocation[index] += 1
        remaining = total - sum(allocation)
    return allocation


def _fixed_stratified_test_indices(data_list, test_ratio, generator):
    """Select one campaign-level, class-stratified test set for offline evaluation."""
    cases_by_class = {}
    total_cases = 0
    for data_index, data in enumerate(data_list):
        labels = data.y.detach().cpu().reshape(-1)
        total_cases += labels.numel()
        for node_index, label in enumerate(labels.tolist()):
            cases_by_class.setdefault(int(label), []).append((data_index, node_index))

    target_total = int(round(test_ratio * total_cases))
    class_ids = sorted(cases_by_class)
    class_sizes = [len(cases_by_class[class_id]) for class_id in class_ids]
    minimums = [1 if target_total >= len(class_ids) and size else 0 for size in class_sizes]
    class_targets = _allocate_exact_counts(
        weights=class_sizes,
        total=target_total,
        capacities=class_sizes,
        minimums=minimums,
    )

    selected = {index: [] for index in range(len(data_list))}
    for class_id, class_target in zip(class_ids, class_targets):
        cases = cases_by_class[class_id]
        order = torch.randperm(len(cases), generator=generator).tolist()
        for local_index in order[:class_target]:
            data_index, node_index = cases[local_index]
            selected[data_index].append(node_index)
    return selected


def _candidate_view(data, candidate_indices):
    candidate_indices = candidate_indices.detach().cpu()
    view = SimpleNamespace(node_ids=data.node_ids.detach().cpu()[candidate_indices])
    for name in ("hier_id", *DEFAULT_COVERAGE_FEATURES):
        setattr(view, name, _get_node_attr(data, name).detach().cpu()[candidate_indices])
    return view


def split_nodes_common_test(
    data_list: List[Dict],
    base_mode,
    train_ratio=0.6,
    val_ratio=0.1,
    test_ratio=0.3,
    use_test=True,
    fixed_test_seed=2718,
    split_seed=3407,
    features=DEFAULT_COVERAGE_FEATURES,
    group_feature="hier_id",
    bins=3,
    min_per_group=1,
):
    """Compare label-free samplers on one fixed, class-controlled test set.

    Test membership is selected once with outcome labels solely to construct a
    stable offline evaluation population.  The sampler receives only the
    remaining cases and never reads their outcome labels.  The fixed test seed
    is intentionally separate from the paired train/validation split seed.
    """
    supported_modes = {"temporal", "spatio", "hybrid", "feature_coverage"}
    if base_mode not in supported_modes:
        raise ValueError(
            f"common-test base_mode must be one of {sorted(supported_modes)}"
        )
    if not use_test:
        raise ValueError("common-test evaluation requires use_test=True")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("train, validation, and test ratios must be non-negative")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    split = {"train": {}, "val": {}, "test": {}}
    if not data_list:
        return split

    test_generator = torch.Generator(device="cpu")
    test_generator.manual_seed(int(fixed_test_seed))
    test_indices = _fixed_stratified_test_indices(
        data_list, test_ratio=test_ratio, generator=test_generator
    )

    candidate_indices = []
    graph_sizes = []
    candidate_capacities = []
    for data_index, data in enumerate(data_list):
        node_ids = data.node_ids.detach().cpu().reshape(-1)
        graph_sizes.append(node_ids.numel())
        test_index_tensor = torch.tensor(test_indices[data_index], dtype=torch.long)
        test_mask = torch.zeros(node_ids.numel(), dtype=torch.bool)
        test_mask[test_index_tensor] = True
        candidate = torch.where(~test_mask)[0]
        candidate_indices.append(candidate)
        candidate_capacities.append(candidate.numel())
        split["test"][_graph_id(data)] = node_ids[test_index_tensor]

    total_cases = sum(graph_sizes)
    train_total = int(round(train_ratio * total_cases))
    train_targets = _allocate_exact_counts(
        weights=graph_sizes,
        total=train_total,
        capacities=candidate_capacities,
    )

    split_generator = torch.Generator(device="cpu")
    split_generator.manual_seed(int(split_seed))
    python_rng = random.Random(int(split_seed))

    if base_mode == "temporal":
        graph_order = list(range(len(data_list)))
        python_rng.shuffle(graph_order)
        remaining_train = train_total
        for data_index in graph_order:
            candidate = candidate_indices[data_index]
            take = min(remaining_train, candidate.numel())
            if 0 < take < candidate.numel():
                order = torch.randperm(candidate.numel(), generator=split_generator)
                candidate = candidate[order]
                candidate_indices[data_index] = candidate
            train_targets[data_index] = take
            remaining_train -= take
        if remaining_train:
            raise RuntimeError("temporal common-test split did not reach train target")

    elif base_mode == "hybrid":
        provisional = []
        for data_index, candidate in enumerate(candidate_indices):
            if python_rng.choice(("temporal", "spatio")) == "temporal":
                priority = 0 if python_rng.random() < train_ratio / (train_ratio + val_ratio) else 2
                scores = torch.full((candidate.numel(),), float(priority))
                scores += torch.rand(candidate.numel(), generator=split_generator) * 0.1
            else:
                scores = 1.0 + torch.rand(candidate.numel(), generator=split_generator)
            provisional.extend(
                (float(score), data_index, local_index)
                for local_index, score in enumerate(scores.tolist())
            )
        provisional.sort(key=lambda item: (item[0], item[1], item[2]))
        chosen = provisional[:train_total]
        train_targets = [0] * len(data_list)
        chosen_local = {index: [] for index in range(len(data_list))}
        for _, data_index, local_index in chosen:
            chosen_local[data_index].append(local_index)
            train_targets[data_index] += 1
        for data_index, candidate in enumerate(candidate_indices):
            chosen_set = set(chosen_local[data_index])
            order = chosen_local[data_index] + [
                index for index in range(candidate.numel()) if index not in chosen_set
            ]
            candidate_indices[data_index] = candidate[torch.tensor(order, dtype=torch.long)]

    for data_index, data in enumerate(data_list):
        graph_id = _graph_id(data)
        candidate = candidate_indices[data_index]
        target = train_targets[data_index]

        if base_mode == "feature_coverage":
            view = _candidate_view(data, candidate)
            selected_local = _coverage_sample_indices(
                data=view,
                train_ratio=(target / candidate.numel()) if candidate.numel() else 0.0,
                features=features,
                group_feature=group_feature,
                bins=bins,
                min_per_group=min_per_group,
                generator=split_generator,
            )
            selected_set = set(selected_local.tolist())
            local_order = selected_local.tolist() + [
                index for index in range(candidate.numel()) if index not in selected_set
            ]
            candidate = candidate[torch.tensor(local_order, dtype=torch.long)]
        elif base_mode == "spatio":
            order = torch.randperm(candidate.numel(), generator=split_generator)
            candidate = candidate[order]
        elif base_mode in {"temporal", "hybrid"}:
            pass

        node_ids = data.node_ids.detach().cpu().reshape(-1)
        split["train"][graph_id] = node_ids[candidate[:target]]
        split["val"][graph_id] = node_ids[candidate[target:]]

    return split

def split_nodes_stratified(data_list: List[Dict], train_ratio=0.6, val_ratio=0.2, use_test=True):
    if not use_test:
        total = train_ratio + val_ratio
        train_ratio /= total
        val_ratio /= total

    split = {'train': {}, 'val': {}}
    if use_test:
        split['test'] = {}

    for data in data_list:
        graph_id = _graph_id(data)

        y = data.y
        node_ids = data.node_ids

        train_nodes = []
        val_nodes = []
        test_nodes = []

        unique_labels = torch.unique(y)

        for label in unique_labels:
            label_mask = (y == label)
            label_nodes = node_ids[label_mask].clone()

            if label_nodes.numel() == 0:
                continue

            indices = torch.randperm(label_nodes.size(0))
            label_nodes = label_nodes[indices]

            n = len(label_nodes)
            train_end = int(train_ratio * n)
            val_end = int((train_ratio + val_ratio) * n)

            # 保证小类不至于被切没
            if n >= 3:
                train_end = max(train_end, 1)
                if use_test:
                    val_end = max(val_end, train_end + 1)
                    val_end = min(val_end, n - 1)
            elif n == 2:
                train_end = 1
                val_end = 2 if not use_test else 1
            elif n == 1:
                train_end = 1
                val_end = 1

            train_nodes.append(label_nodes[:train_end])
            val_nodes.append(label_nodes[train_end:val_end])

            if use_test:
                test_nodes.append(label_nodes[val_end:])

        split['train'][graph_id] = torch.cat(train_nodes) if train_nodes else torch.tensor([], dtype=node_ids.dtype)
        split['val'][graph_id] = torch.cat(val_nodes) if val_nodes else torch.tensor([], dtype=node_ids.dtype)

        if use_test:
            split['test'][graph_id] = torch.cat(test_nodes) if test_nodes else torch.tensor([], dtype=node_ids.dtype)

    return split

def split_nodes_spatio(data_list: List[Dict], train_ratio=0.6, val_ratio=0.2, use_test=True):
    if not use_test:
        # Normalize ratios so train + val = 1.0
        total = train_ratio + val_ratio
        train_ratio /= total
        val_ratio /= total

    split = {'train': {}, 'val': {}}
    if use_test:
        split['test'] = {}

    for data in data_list:
        graph_id = data.graph_id
        # Here we can't use random.shuffle(nodes) because random.shuffle() only works on Python lists, not on PyTorch tensors. 
        # It may silently corrupt the data or do nothing meaningful.
        nodes = data.node_ids.clone()
        indices = torch.randperm(nodes.size(0))
        nodes = nodes[indices]
        n = len(nodes)
        train_end = int(train_ratio * n)
        val_end = n if not use_test else int((train_ratio + val_ratio) * n)

        # Each call to tensor() creates a new tensor object, and even if the values are the same, 
        # they are not the same key in Python dictionaries unless they are the same object or 
        # Python sees them as equal and hash-compatible. Therefore, here we convert graph_id.
        graph_id = _graph_id(data)
        split['train'][graph_id] = nodes[:train_end]
        split['val'][graph_id] = nodes[train_end:val_end]

        if use_test:
            split['test'][graph_id] = nodes[val_end:]

    return split


def split_nodes_temporal(data_list: List[Dict], train_ratio=0.6, val_ratio=0.2, use_test=True):
    if not use_test:
        total = train_ratio + val_ratio
        train_ratio /= total
        val_ratio /= total

    # creates a shallow copy of data_list, so when random.shuffle(graph_list) is called, 
    # it does not change the order of data_list
    graph_list = data_list[:]
    random.shuffle(graph_list)
    total_graphs = len(graph_list)

    train_end = int(train_ratio * total_graphs)
    val_end = int((train_ratio + val_ratio) * total_graphs)

    split = {'train': {}, 'val': {}}
    if use_test:
        split['test'] = {}

    for i, data in enumerate(graph_list):
        graph_id = data.graph_id
        nodes = data.node_ids

        graph_id = _graph_id(data)
        if i < train_end:
            split['train'][graph_id] = nodes
        elif i < val_end:
            split['val'][graph_id] = nodes
        elif use_test:
            split['test'][graph_id] = nodes

    return split


def split_nodes_hybrid(data_list: List[Dict], train_ratio=0.6, val_ratio=0.2, use_test=True):
    if not use_test:
        total = train_ratio + val_ratio
        train_ratio /= total
        val_ratio /= total

    split = {'train': {}, 'val': {}}
    if use_test:
        split['test'] = {}

    graph_list = data_list[:]
    random.shuffle(graph_list)

    for data in graph_list:
        graph_id = _graph_id(data)
        # Here we can't use random.shuffle(nodes) because random.shuffle() only works on Python lists, not on PyTorch tensors. 
        # It may silently corrupt the data or do nothing meaningful.
        nodes = data.node_ids.clone()
        indices = torch.randperm(nodes.size(0))
        nodes = nodes[indices]

        # Decide mode per graph
        mode = random.choice(['temporal', 'spatio'])

        if mode == 'temporal':
            choices = ['train', 'val']
            weights = [train_ratio, val_ratio]
            if use_test:
                choices.append('test')
                weights.append(1 - train_ratio - val_ratio)
            group = random.choices(choices, weights=weights)[0]
            split[group][graph_id] = nodes

        else:  # spatio mode
            n = len(nodes)
            train_end = int(train_ratio * n)
            val_end = n if not use_test else int((train_ratio + val_ratio) * n)
            
            split['train'][graph_id] = nodes[:train_end]
            split['val'][graph_id] = nodes[train_end:val_end]

            if use_test:
                split['test'][graph_id] = nodes[val_end:]

    return split


def split_nodes_feature_coverage(
    data_list: List[Dict],
    train_ratio=0.1,
    val_ratio=0.2,
    use_test=True,
    features=DEFAULT_COVERAGE_FEATURES,
    group_feature="hier_id",
    bins=3,
    min_per_group=1,
):
    """Split nodes by unlabeled static feature coverage.

    This mode is for reduced-fault-simulation workflows. It selects train
    nodes using static features available before full simulation, not labels.
    Labels can still be used later for offline validation and test metrics.
    """
    split = {'train': {}, 'val': {}}
    if use_test:
        split['test'] = {}

    for data in data_list:
        graph_id = _graph_id(data)
        node_ids = data.node_ids
        num_nodes = node_ids.numel()

        if num_nodes == 0:
            split['train'][graph_id] = _empty_like_node_ids(node_ids)
            split['val'][graph_id] = _empty_like_node_ids(node_ids)
            if use_test:
                split['test'][graph_id] = _empty_like_node_ids(node_ids)
            continue

        train_indices = _coverage_sample_indices(
            data=data,
            train_ratio=train_ratio,
            features=features,
            group_feature=group_feature,
            bins=bins,
            min_per_group=min_per_group,
        )

        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=node_ids.device)
        train_mask[train_indices] = True

        remaining_indices = torch.where(~train_mask)[0]
        remaining_indices = remaining_indices[torch.randperm(remaining_indices.numel())]

        if use_test:
            val_size = int(round(val_ratio * num_nodes))
            val_size = min(val_size, remaining_indices.numel())
            val_indices = remaining_indices[:val_size]
            test_indices = remaining_indices[val_size:]
        else:
            val_indices = remaining_indices
            test_indices = None

        split['train'][graph_id] = node_ids[train_indices]
        split['val'][graph_id] = node_ids[val_indices]

        if use_test:
            split['test'][graph_id] = node_ids[test_indices]

    return split


def split_nodes_grouped_ff_feature_coverage(
    data_list: List[Dict],
    train_ratio=0.6,
    val_ratio=0.1,
    use_test=True,
    features=DEFAULT_COVERAGE_FEATURES,
    group_feature="hier_id",
    bins=3,
    min_per_group=1,
    split_seed=3407,
):
    """Assign every fault-injection case of a physical FF to one phase.

    Feature Coverage is applied once to time-aggregated, fault-free attributes
    to select training FFs.  The remaining physical FFs are divided into
    validation and test sets.  Exactly the same three node-id sets are then
    used in every injection-time graph, preventing FF-identity reuse across
    training, validation, and testing.
    """
    split = {'train': {}, 'val': {}}
    if use_test:
        split['test'] = {}
    if not data_list:
        return split

    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("train_ratio and val_ratio must be non-negative")
    if use_test and train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio cannot exceed 1")
    if not use_test:
        total = train_ratio + val_ratio
        if total <= 0:
            raise ValueError("train_ratio + val_ratio must be positive")
        train_ratio /= total
        val_ratio /= total

    aggregate = _aggregate_grouped_ff_features(
        data_list=data_list,
        features=features,
        group_feature=group_feature,
    )
    node_ids = aggregate.node_ids
    num_nodes = node_ids.numel()
    if num_nodes == 0:
        for data in data_list:
            graph_id = _graph_id(data)
            split['train'][graph_id] = _empty_like_node_ids(data.node_ids)
            split['val'][graph_id] = _empty_like_node_ids(data.node_ids)
            if use_test:
                split['test'][graph_id] = _empty_like_node_ids(data.node_ids)
        return split

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(split_seed))
    train_indices = _coverage_sample_indices(
        data=aggregate,
        train_ratio=train_ratio,
        features=features,
        group_feature=group_feature,
        bins=bins,
        min_per_group=min_per_group,
        generator=generator,
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    remaining_indices = torch.where(~train_mask)[0]
    remaining_indices = remaining_indices[
        torch.randperm(remaining_indices.numel(), generator=generator)
    ]

    if use_test:
        val_size = int(round(val_ratio * num_nodes))
        val_size = min(val_size, remaining_indices.numel())
        val_indices = remaining_indices[:val_size]
        test_indices = remaining_indices[val_size:]
    else:
        val_indices = remaining_indices
        test_indices = None

    phase_node_ids = {
        'train': node_ids[train_indices],
        'val': node_ids[val_indices],
    }
    if use_test:
        phase_node_ids['test'] = node_ids[test_indices]

    phase_sets = {phase: set(ids.tolist()) for phase, ids in phase_node_ids.items()}
    phases = list(phase_sets)
    for index, phase in enumerate(phases):
        for other_phase in phases[index + 1:]:
            if not phase_sets[phase].isdisjoint(phase_sets[other_phase]):
                raise RuntimeError(
                    f"physical FF overlap between {phase} and {other_phase}"
                )
    if set().union(*phase_sets.values()) != set(node_ids.tolist()):
        raise RuntimeError("grouped-by-FF split does not cover all physical FFs")

    for data in data_list:
        graph_id = _graph_id(data)
        for phase, selected_ids in phase_node_ids.items():
            split[phase][graph_id] = _node_ids_in_set(data.node_ids, selected_ids)

    return split
