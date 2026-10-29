"""Export sensitivity from cached masks and held-out predictions, without label leakage."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from workflow.bridge import aggregate_samples, read_csv, write_csv


def export(config_path, predictions_path, output):
    config = yaml.safe_load(Path(config_path).read_text())
    root = Path(__file__).resolve().parent
    cache = Path(config['dataset']['root_dir']).expanduser()
    if not cache.is_absolute():
        cache = root / cache
    graphs = torch.load(cache / 'processed/test_processed_data.pt', weights_only=False)
    samples = []
    classes = config['model']['num_classes']
    for graph in graphs:
        ff = graph['ff']
        masks = [ff.train_mask, ff.val_mask, ff.test_mask]
        membership = sum(mask.long() for mask in masks)
        if not torch.all(membership == 1):
            raise ValueError('Every target sample needs exactly one explicit train/val/test membership')
        for index, node in enumerate(ff.node_ids.tolist()):
            label = int(ff.y[index])
            if not 0 <= label < classes:
                raise ValueError('Invalid ground-truth label')
            partition = next(name for name, mask in zip(('train', 'val', 'test'), masks) if mask[index])
            samples.append({'source_path': str(graph.source_path), 'graph_id': int(graph.graph_id.item()),
                            'node_id': int(node), 'partition': partition, 'label': label})
    predictions = read_csv(predictions_path)
    if any(not 0 <= int(row['predicted_label']) < classes for row in predictions):
        raise ValueError('Invalid predicted label')
    sensitivity, reference, outcomes = aggregate_samples(samples, predictions, list(range(1, classes)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'sensitivity.csv', sensitivity)
    # Evaluation-only file: never passed to the optimizer.
    write_csv(output / 'measured_reference.csv', reference)
    write_csv(output / 'sample_outcomes.csv', outcomes)
    summary = {'num_registers': len(sensitivity), 'num_samples': len(samples),
               'num_predictions': len(predictions), 'error_labels': list(range(1, classes)),
               'definition': 'known train/val labels + predicted test labels, aggregated per register',
               'measured_reference_usage': 'evaluation only, never optimization input'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export(args.config, args.predictions, args.output)
