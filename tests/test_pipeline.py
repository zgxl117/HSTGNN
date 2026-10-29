import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pipeline', ROOT / 'pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def test_smoke_copy_filters_graphs_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / 'campaign'
            raw = source / 'raw'
            raw.mkdir(parents=True)
            for filename in pipeline.ui.REQUIRED_CSV:
                with (raw / filename).open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['graph_id', 'node_id'])
                    writer.writeheader()
                    writer.writerows({'graph_id': i, 'node_id': 1} for i in range(1, 6))
            before = {p.name: p.read_bytes() for p in raw.iterdir()}
            spec = {'dataset': {'train_val_dirs': ['.'], 'test_dirs': ['.']}}
            result = pipeline.copy_campaigns(source / 'dataset.yaml', spec, base / 'copy', smoke=True)
            config = yaml.safe_load(result.read_text())
            self.assertEqual(config['dataset']['train_val_dirs'], config['dataset']['test_dirs'])
            copied = Path(config['dataset']['test_dirs'][0]) / 'raw'
            self.assertEqual(len(pipeline.read_csv(copied / 'nodes.csv')), 3)
            self.assertEqual(before, {p.name: p.read_bytes() for p in raw.iterdir()})
            self.assertEqual(config['trainer']['epochs'], 2)

    def test_profile_is_required_instead_of_using_wrong_circuit_ppa(self):
        with self.assertRaisesRegex(ValueError, 'needs a hardening profile'):
            pipeline.resolve_profile({'name': 'fifo'})

    def test_child_failure_is_reported(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                pipeline.execute([sys.executable, '-c', 'raise SystemExit(4)'], run, 'bad', run)
            self.assertTrue((run / 'bad.log').is_file())
