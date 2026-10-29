"""Strict, name-based contracts between fault prediction and TMR optimization."""
import csv
import hashlib
import math
from pathlib import Path

REGISTER_COLUMNS = ['index', 'instance_name', 'reg_type', 'area', 'reg_power',
                    'reg_dynamic_power(uW)', 'reg_leakage_power(uW)', 'path_slack(ns)', 'sensitivity']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value, name, lower=None, upper=None):
    number = float(value)
    if not math.isfinite(number) or (lower is not None and number < lower) or (upper is not None and number > upper):
        raise ValueError(f'Invalid {name}: {value}')
    return number


def read_ppa(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        rows = [row for row in csv.reader(handle) if row and not row[0].startswith('#')]
    if len(rows) < 4 or rows[0][0] != 'total_area' or rows[2] != REGISTER_COLUMNS:
        raise ValueError('PPA CSV needs design header/value and the exact nine register columns; no random fallback')
    if len(rows[0]) not in (4, 5) or len(rows[1]) != len(rows[0]):
        raise ValueError('Invalid design header/value lengths')
    for index, value in enumerate(rows[1]):
        finite(value, 'design metric', lower=0 if index in (0, 1, 3, 4) else None)
    if float(rows[1][0]) <= 0 or float(rows[1][1]) <= 0:
        raise ValueError('Design area and power must be positive')
    registers = []
    for row in rows[3:]:
        if len(row) != len(REGISTER_COLUMNS):
            raise ValueError('Malformed PPA register row')
        reg = dict(zip(REGISTER_COLUMNS, row))
        reg['index'] = int(reg['index'])
        if not reg['instance_name'] or any(c in reg['instance_name'] for c in ',\n\r'):
            raise ValueError('Invalid instance_name')
        for key in ('area', 'reg_power', 'reg_dynamic_power(uW)', 'reg_leakage_power(uW)'):
            finite(reg[key], key, lower=0)
        finite(reg['path_slack(ns)'], 'slack')
        finite(reg['sensitivity'], 'sensitivity', 0, 1)
        registers.append(reg)
    if not registers or len({r['index'] for r in registers}) != len(registers) or len({r['instance_name'] for r in registers}) != len(registers):
        raise ValueError('Empty PPA or duplicate register IDs/names')
    return rows[0], rows[1], registers


def join_sensitivity(ppa_path, mapping_path, sensitivity_path, output):
    header, design, registers = read_ppa(ppa_path)
    mapping = read_csv(mapping_path)
    by_node = {}
    for row in mapping:
        node = int(row['node_id'])
        if node in by_node:
            raise ValueError('Duplicate node in register mapping')
        by_node[node] = row['instance_name']
    if len(set(by_node.values())) != len(by_node):
        raise ValueError('Mapping is not one-to-one')
    if set(by_node.values()) != {r['instance_name'] for r in registers}:
        raise ValueError('Mapping must exactly cover PPA instance names; offset guessing is forbidden')
    sensitivity = read_csv(sensitivity_path)
    scores = {}
    for row in sensitivity:
        node = int(row['node_id'])
        if node in scores:
            raise ValueError('Duplicate sensitivity node')
        scores[node] = finite(row['sensitivity'], 'predicted sensitivity', 0, 1)
    if set(scores) != set(by_node):
        raise ValueError('Sensitivity node IDs must exactly cover the mapping; do not mix circuits or fill missing nodes')
    by_name = {by_node[node]: score for node, score in scores.items()}
    for reg in registers:
        reg['sensitivity'] = by_name[reg['instance_name']]
    design = list(design)
    design[3] = sum(scores.values()) / len(scores)
    output = Path(output)
    with output.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(design)
        writer.writerow([])
        writer.writerow(REGISTER_COLUMNS)
        writer.writerows([[reg[key] for key in REGISTER_COLUMNS] for reg in registers])
    # Confirm the generated input is parseable before handing it to a backend.
    read_ppa(output)
    return {'registers': len(registers), 'design_sensitivity': design[3],
            'ppa_sha256': digest(ppa_path), 'mapping_sha256': digest(mapping_path),
            'sensitivity_sha256': digest(sensitivity_path)}


def aggregate_samples(samples, predictions, error_labels):
    """Use known labels ONLY for explicitly marked train/validation samples."""
    errors = set(error_labels)
    if not errors or 0 in errors:
        raise ValueError('Explicit nonzero error_labels required')
    pred = {}
    for row in predictions:
        key = (row['source_path'], int(row['graph_id']), int(row['node_id']))
        if key in pred:
            raise ValueError('Duplicate prediction key')
        pred[key] = int(row['predicted_label'])
    totals = {}
    outcomes = []
    seen = set()
    expected = set()
    for row in samples:
        key = (row['source_path'], int(row['graph_id']), int(row['node_id']))
        if key in seen:
            raise ValueError('Duplicate sample key')
        seen.add(key)
        known = row['partition'] in ('train', 'val')
        if row['partition'] not in ('train', 'val', 'test'):
            raise ValueError('Unpartitioned sample; cannot substitute ground truth')
        if known:
            label = int(row['label'])
            if key in pred:
                raise ValueError('Prediction overlaps known training/validation samples')
        else:
            expected.add(key)
            if key not in pred:
                raise ValueError('Missing prediction for an unknown/test sample; no ground-truth fallback')
            label = pred[key]
        node = key[2]
        counts = totals.setdefault(node, {'n': 0, 'errors': 0, 'measured_errors': 0, 'known': 0, 'predicted': 0})
        counts['n'] += 1
        counts['errors'] += int(label in errors)
        counts['measured_errors'] += int(int(row['label']) in errors)
        counts['known' if known else 'predicted'] += 1
        outcomes.append({'source_path': key[0], 'graph_id': key[1], 'node_id': node,
                         'partition': row['partition'], 'label_source': 'measured_known' if known else 'model',
                         'used_label': label})
    if set(pred) != expected:
        raise ValueError('Unexpected prediction keys outside target test partition')
    if not totals:
        raise ValueError('No samples to aggregate')
    sensitivity = []
    reference = []
    for node, c in sorted(totals.items()):
        sensitivity.append({'node_id': node, 'sensitivity': c['errors'] / c['n'],
                            'num_samples': c['n'], 'num_known': c['known'], 'num_predicted': c['predicted']})
        reference.append({'node_id': node, 'measured_sensitivity': c['measured_errors'] / c['n']})
    return sensitivity, reference, outcomes


def rescore_selection(sensitivity, mapping, selected, column):
    names = {int(row['node_id']): row['instance_name'] for row in mapping}
    scores = {int(row['node_id']): finite(row[column], column, 0, 1) for row in sensitivity}
    if set(names) != set(scores) or len(names) != len(mapping):
        raise ValueError('Invalid reference mapping')
    selected_names = set(selected)
    if len(selected_names) != len(selected) or not selected_names <= set(names.values()):
        raise ValueError('Invalid selected register list')
    total = sum(scores.values())
    removed = sum(value for node, value in scores.items() if names[node] in selected_names)
    n = len(scores)
    return {'before': total / n, 'ideal_tmr_after_original_denominator': (total - removed) / n,
            'backend_after_expanded_denominator': (total - removed) / (n + 2 * len(selected)),
            'scope': 'ideal single-upset TMR proxy, excludes voter/common-mode faults; not post-hardening injection'}
