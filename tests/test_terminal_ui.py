import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('terminal_ui', ROOT / 'terminal_ui.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'configs').mkdir()
        shutil.copy(ROOT / 'configs/train.example.yaml', self.root / 'configs/train.example.yaml')
        self.campaign = self.root / 'campaign with spaces'
        raw = self.campaign / 'raw'
        raw.mkdir(parents=True)
        for name, columns in ui.REQUIRED_CSV.items():
            columns = sorted(columns | ({'hier_id', 'toggle_rate', 'duty_cycle', 'fanin', 'fanout'}
                                       if name == 'nodes.csv' else set()))
            (raw / name).write_text(','.join(columns) + '\n' + ','.join('0' for _ in columns) + '\n')

    def task(self):
        return ui.create_task(self.root, 'test_task', str(self.campaign), 16, 8, 2)

    def test_create_paired_configs(self):
        train, predict = map(ui.load_config, self.task())
        self.assertEqual(train['model'], predict['model'])
        self.assertEqual(train['dataset'], predict['dataset'])
        self.assertEqual(train['trainer']['checkpoint_dir'], predict['model_path'])
        self.assertEqual(train['model']['num_nodes'], 16)
        self.assertEqual(ui.check_config(train, 'train', self.root), [])

    def test_name_and_duplicate_protection(self):
        with self.assertRaises(ValueError):
            ui.create_task(self.root, '../escape', str(self.campaign), 1, 8, 1)
        self.task()
        with self.assertRaises(ValueError):
            self.task()

    def test_existing_checkpoint_never_launches_training(self):
        train, _ = self.task()
        checkpoint = self.root / 'runs/test_task/checkpoints'
        checkpoint.mkdir(parents=True)
        marker = checkpoint / 'best_model.ckpt'
        marker.write_text('keep me')
        with patch.object(ui, 'run_job') as run:
            with self.assertRaisesRegex(ValueError, '检查点目录已存在'):
                ui.launch('train', train, sys.executable, self.root)
            run.assert_not_called()
        self.assertEqual(marker.read_text(), 'keep me')

    def test_prediction_requires_checkpoint_and_fresh_outputs(self):
        _, path = self.task()
        config = ui.load_config(path)
        self.assertTrue(any('缺少模型权重' in x for x in ui.check_config(config, 'predict', self.root)))
        model = self.root / config['model_path']
        model.mkdir(parents=True)
        (model / 'best_model.ckpt').write_text('fake')
        self.assertEqual(ui.check_config(config, 'predict', self.root), [])
        output = self.root / config['output_path']
        output.mkdir()
        (output / 'results.csv').write_text('keep')
        self.assertTrue(any('预测结果已存在' in x for x in ui.check_config(config, 'predict', self.root)))

    def test_bad_yaml_and_missing_data(self):
        empty = self.root / 'empty.yaml'
        empty.write_text('[]')
        with self.assertRaises(ValueError):
            ui.load_config(empty)
        path, _ = self.task()
        config = ui.load_config(path)
        missing = copy.deepcopy(config)
        missing['dataset']['test_dirs'] = ['does_not_exist']
        self.assertTrue(any('缺少数据文件' in x for x in ui.check_config(missing, 'train', self.root)))

    def test_launch_arguments_use_project_cwd_contract(self):
        path, _ = self.task()
        with patch.object(ui, 'run_job', return_value=0) as run:
            self.assertEqual(ui.launch('train', path, '/python with spaces/bin/python', self.root), 0)
        command, root = run.call_args.args
        self.assertEqual(command, ['/python with spaces/bin/python', str(self.root / 'train.py'),
                                   '--config', str(path.resolve())])
        self.assertEqual(root, self.root)

    def test_real_child_output_failure_and_metadata(self):
        status = ui.run_job([sys.executable, '-c', 'print("toy child"); raise SystemExit(3)'], self.root)
        self.assertEqual(status, 3)
        log = next((self.root / 'runs/ui_logs').glob('*.log'))
        self.assertIn('toy child', log.read_text())
        metadata = json.loads(log.with_suffix('.json').read_text())
        self.assertEqual(metadata['status'], 'failed')
        self.assertEqual(metadata['returncode'], 3)

    def test_menu_exit_and_eof(self):
        with patch('builtins.input', return_value='0'):
            ui.menu(sys.executable, self.root)

    def manifest(self, **overrides):
        spec = {'schema_version': 1, 'name': 'fifo',
                'model': {'num_nodes': 16, 'time_window_size': 8,
                          'edge_vector_size': 4, 'num_classes': 5},
                'trainer': {'epochs': 2}}
        spec.update(overrides)
        path = self.campaign / 'dataset.yaml'
        path.write_text(ui.yaml.safe_dump(spec))
        return path

    def test_dataset_relative_paths_and_model_overrides(self):
        manifest = self.manifest(optimizer={'learning_rate': 0.002},
                                 dataset={'train_val_dirs': ['.'], 'test_dirs': ['.']})
        paths = ui.create_dataset_task(self.root, manifest, 'from_yaml')
        train, predict = map(ui.load_config, paths)
        self.assertEqual(train['dataset']['train_val_dirs'], [str(self.campaign)])
        self.assertEqual(train['model'], predict['model'])
        self.assertEqual(train['dataset'], predict['dataset'])
        self.assertEqual(train['model']['edge_vector_size'], 4)
        self.assertEqual(train['optimizer']['learning_rate'], 0.002)
        self.assertEqual(train['trainer']['epochs'], 2)
        self.assertEqual(len(train['dataset_manifest_sha256']), 64)

    def test_dataset_directory_and_automatic_unique_names(self):
        self.manifest()
        first = ui.create_dataset_task(self.root, self.campaign)
        second = ui.create_dataset_task(self.root, self.campaign)
        self.assertNotEqual(first, second)
        self.assertEqual(ui.load_config(first[0])['dataset']['train_val_dirs'], [str(self.campaign)])

    def test_manifest_rejects_unsafe_outputs_unknown_keys_and_bad_dimensions(self):
        cases = [dict(schema_version=2), dict(model={'num_nodes': 0}),
                 dict(trainer={'checkpoint_dir': '/old/weights'}),
                 dict(dataset={'root_dir': '/old/cache'}),
                 dict(trainer={'epohcs': 3}), dict(model={'num_nodes': '60', 'time_window_size': 8,
                     'edge_vector_size': 4, 'num_classes': 5})]
        for index, overrides in enumerate(cases):
            with self.subTest(overrides=overrides):
                path = self.manifest(**overrides)
                with self.assertRaises(ValueError):
                    ui.create_dataset_task(self.root, path, f'bad_{index}')
                self.assertFalse((self.root / 'configs' / f'bad_{index}_train.yaml').exists())

    def test_manifest_missing_data_does_not_create_configs(self):
        path = self.manifest(dataset={'train_val_dirs': ['missing'], 'test_dirs': ['.']})
        with self.assertRaisesRegex(ValueError, '缺少数据文件'):
            ui.create_dataset_task(self.root, path, 'missing_data')
        self.assertFalse((self.root / 'configs/missing_data_train.yaml').exists())

    def test_manifest_duplicate_task_keeps_previous_config(self):
        path = self.manifest()
        train, _ = ui.create_dataset_task(self.root, path, 'same')
        original = train.read_bytes()
        with self.assertRaises(ValueError):
            ui.create_dataset_task(self.root, path, 'same')
        self.assertEqual(train.read_bytes(), original)

    def test_menu_import_needs_only_dataset_path(self):
        path = self.manifest()
        with patch('builtins.input', side_effect=['2', str(path), '0']) as prompt:
            ui.menu(sys.executable, self.root)
        self.assertEqual(prompt.call_count, 3)
        self.assertEqual(len(list((self.root / 'configs').glob('fifo_*_train.yaml'))), 1)
        with patch('builtins.input', side_effect=EOFError):
            ui.menu(sys.executable, self.root)

    def test_external_dataset_discovery_and_selection(self):
        path = self.manifest()
        (self.root / 'dataset_sources.yaml').write_text(ui.yaml.safe_dump({'roots': ['campaign with spaces']}))
        self.assertEqual(ui.discover_datasets(self.root), [path])
        with patch('builtins.input', return_value='1'):
            self.assertEqual(ui.pick_dataset(self.root), path)

    def test_discovery_deduplicates_roots_and_tolerates_missing_sources(self):
        path = self.manifest()
        (self.root / 'dataset_sources.yaml').write_text(ui.yaml.safe_dump({
            'roots': [str(path.parent), str(path.parent), 'missing']}))
        self.assertEqual(ui.discover_datasets(self.root), [path])

    def test_invalid_dataset_source_configuration(self):
        (self.root / 'dataset_sources.yaml').write_text('roots: not-a-list\n')
        with self.assertRaisesRegex(ValueError, 'roots'):
            ui.discover_datasets(self.root)

    def test_registry_can_hide_historical_local_datasets(self):
        external = self.manifest()
        local = self.root / 'data/old_run'
        local.mkdir(parents=True)
        (local / 'dataset.yaml').write_text('schema_version: 1\n')
        (self.root / 'dataset_sources.yaml').write_text(ui.yaml.safe_dump({
            'include_local': False, 'roots': [str(external.parent)]}))
        self.assertEqual(ui.discover_datasets(self.root), [external])

    def test_dataset_yaml_can_carry_hardening_profile(self):
        path = self.manifest(hardening={'profile': 'riscv'})
        train, predict = ui.create_dataset_task(self.root, path, 'combined')
        self.assertEqual(ui.load_config(train)['model'], ui.load_config(predict)['model'])

    def test_closed_loop_menu_uses_isolated_pipeline_process(self):
        path = self.manifest(hardening={'profile': 'riscv'})
        with patch('builtins.input', side_effect=['9', str(path), 's', '0']), patch.object(ui, 'run_job') as run:
            ui.menu(sys.executable, self.root)
        command, root = run.call_args.args
        self.assertEqual(command, [sys.executable, str(self.root / 'pipeline.py'), '--dataset', str(path), '--smoke'])
        self.assertEqual(root, self.root)


if __name__ == '__main__':
    unittest.main()
