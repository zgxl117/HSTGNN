#!/usr/bin/env python3
"""HSTGNN terminal menu. Does not import torch or change model implementations."""
import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parent
REQUIRED_CSV = {
    'nodes.csv': {'graph_id', 'node_id', 'label', 'feat'},
    'edges.csv': {'graph_id', 'src_id', 'dst_id', 'feat'},
    'graphs.csv': {'graph_id'},
    'ff_module.csv': {'graph_id', 'REG_ID', 'module_id'},
}


def project_path(value, root=ROOT):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path):
    if yaml is None:
        raise ValueError('缺少 PyYAML，请使用已安装项目依赖的 Python 环境。')
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError('配置必须是 YAML 字典。')
    return config


def check_config(config, mode, root=ROOT):
    """Read-only launch preflight, not a full dataset/semantic validator."""
    issues = []
    try:
        model = config['model']
        dataset = config['dataset']
        if not isinstance(model, dict) or not isinstance(dataset, dict):
            raise ValueError('model 和 dataset 必须是字典')
        if model.get('name') != 'ASPP_STGATs':
            issues.append('终端入口当前只支持五类主模型 ASPP_STGATs。')
        for name in ('num_nodes', 'time_window_size', 'num_classes'):
            if not isinstance(model.get(name), int) or model[name] < 1:
                issues.append(f'model.{name} 必须是正整数。')
        if not isinstance(model.get('edge_vector_size'), int) or model['edge_vector_size'] < 3:
            issues.append('model.edge_vector_size 必须 >= 3。')
        project_path(dataset['root_dir'], root)
        campaigns = []
        for key in ('train_val_dirs', 'test_dirs'):
            values = dataset.get(key)
            if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values):
                issues.append(f'dataset.{key} 必须是非空路径列表。')
            else:
                campaigns.extend(values)
        for value in dict.fromkeys(campaigns):
            raw = project_path(value, root) / 'raw'
            for filename, columns in REQUIRED_CSV.items():
                path = raw / filename
                if not path.is_file():
                    issues.append(f'缺少数据文件：{path}')
                    continue
                with path.open(newline='') as handle:
                    reader = csv.DictReader(handle)
                    required = set(columns)
                    if filename == 'nodes.csv' and dataset.get('split_mode') in (
                            'feature_coverage', 'grouped_ff_feature_coverage'):
                        split = dataset.get(dataset['split_mode'], {})
                        required.update(split.get('features', []))
                        required.add(split.get('group_feature', 'hier_id'))
                    missing = required - set(reader.fieldnames or [])
                    if missing:
                        issues.append(f'{path.name} 缺少列：{", ".join(sorted(missing))}')
                    if next(reader, None) is None:
                        issues.append(f'数据文件为空：{path}')
        if mode == 'train':
            checkpoint = project_path(config['trainer']['checkpoint_dir'], root)
            if checkpoint.exists():
                issues.append(f'检查点目录已存在，拒绝启动以免旧代码清空它：{checkpoint}')
            if 'optimizer' not in config:
                issues.append('缺少 optimizer 配置。')
        else:
            checkpoint = project_path(config['model_path'], root) / 'best_model.ckpt'
            if not checkpoint.is_file():
                issues.append(f'缺少模型权重：{checkpoint}')
            output = project_path(config['output_path'], root)
            if any((output / name).exists() for name in (
                    'results.csv', 'scores.csv', 'per_class_scores.csv')):
                issues.append(f'预测结果已存在，请更换 output_path：{output}')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        issues.append(f'配置或数据无法读取：{exc}')
    return issues


def create_task(root, name, campaign, num_nodes, window, epochs):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('任务名只能含字母、数字、下划线、短横线，长度 1..64。')
    if min(num_nodes, window, epochs) < 1:
        raise ValueError('节点数、窗口和轮数必须大于零。')
    paths = [root / 'configs' / f'{name}_{mode}.yaml' for mode in ('train', 'predict')]
    if any(p.exists() for p in paths) or (root / 'runs' / name).exists():
        raise ValueError('任务名已使用，请创建新任务。')
    campaign = str(project_path(campaign, root))
    if not all((Path(campaign) / 'raw' / filename).is_file() for filename in REQUIRED_CSV):
        raise ValueError('campaign 目录必须包含 raw/ 下的四类 CSV；不要填写 raw 目录本身。')
    train = load_config(root / 'configs/train.example.yaml')
    train['dataset'].update(root_dir=f'./runs/{name}/cache',
                            train_val_dirs=[campaign], test_dirs=[campaign])
    train['model'].update(num_nodes=num_nodes, time_window_size=window)
    train['trainer'].update(epochs=epochs, checkpoint_dir=f'./runs/{name}/checkpoints')
    predict = {'manual_seed': train.get('manual_seed', 3407),
               'dataset': copy.deepcopy(train['dataset']), 'model': copy.deepcopy(train['model']),
               'model_path': train['trainer']['checkpoint_dir'],
               'output_path': f'./runs/{name}/predictions'}
    for path, config in zip(paths, (train, predict)):
        with path.open('x') as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return paths


def merge_settings(target, overrides, context):
    if not isinstance(overrides, dict):
        raise ValueError(f'{context} 必须是字典。')
    for key, value in overrides.items():
        if key not in target:
            raise ValueError(f'不支持的字段：{context}.{key}')
        if isinstance(target[key], dict):
            merge_settings(target[key], value, f'{context}.{key}')
        else:
            target[key] = copy.deepcopy(value)


def create_dataset_task(root, manifest, name=None):
    """Resolve dataset-relative settings into a new isolated train/predict pair."""
    manifest = project_path(manifest, root)
    if manifest.is_dir():
        manifest = manifest / 'dataset.yaml'
    spec = load_config(manifest)
    allowed = {'schema_version', 'name', 'description', 'manual_seed', 'dataset',
               'model', 'trainer', 'optimizer', 'loss', 'hardening'}
    if set(spec) - allowed:
        raise ValueError('数据 YAML 含不支持的字段：' + ', '.join(sorted(set(spec) - allowed)))
    if type(spec.get('schema_version')) is not int or spec['schema_version'] != 1:
        raise ValueError('dataset.yaml 必须包含 schema_version: 1。')
    dataset_name = spec.get('name')
    if not isinstance(dataset_name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', dataset_name):
        raise ValueError('数据 YAML 的 name 必须是 1..64 位字母/数字/_/-。')
    name = name or dataset_name[:32] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('任务名只能含字母、数字、下划线、短横线，长度 1..64。')
    paths = [root / 'configs' / f'{name}_{mode}.yaml' for mode in ('train', 'predict')]
    if any(p.exists() for p in paths) or (root / 'runs' / name).exists():
        raise ValueError('任务名已使用，请创建新任务。')
    train = load_config(root / 'configs/train.example.yaml')
    model = spec.get('model', {})
    if not isinstance(model, dict) or not all(key in model for key in (
            'num_nodes', 'time_window_size', 'edge_vector_size', 'num_classes')):
        raise ValueError('model 需声明 num_nodes、time_window_size、edge_vector_size、num_classes。')
    for section in ('model', 'trainer', 'optimizer', 'loss'):
        overrides = spec.get(section, {})
        if section == 'trainer' and isinstance(overrides, dict) and 'checkpoint_dir' in overrides:
            raise ValueError('数据 YAML 不设置 checkpoint_dir，由每次任务自动生成。')
        merge_settings(train[section], overrides, section)
    for key in ('num_nodes', 'time_window_size', 'num_classes', 'edge_vector_size'):
        if type(train['model'][key]) is not int or train['model'][key] < 1:
            raise ValueError(f'model.{key} 必须是正整数。')
    if type(train['trainer']['epochs']) is not int or train['trainer']['epochs'] < 1:
        raise ValueError('trainer.epochs 必须是正整数。')
    if not isinstance(train['loss'].get('class_weights'), list) or len(train['loss']['class_weights']) != train['model']['num_classes']:
        raise ValueError('loss.class_weights 数量必须匹配 model.num_classes。')
    settings = spec.get('dataset', {})
    allowed_dataset = {'train_val_dirs', 'test_dirs', 'split_mode', 'feature_coverage'}
    if not isinstance(settings, dict) or set(settings) - allowed_dataset:
        raise ValueError('dataset 仅支持 train_val_dirs、test_dirs、split_mode、feature_coverage。')
    if settings.get('split_mode', 'feature_coverage') != 'feature_coverage':
        raise ValueError('数据 YAML 当前支持 feature_coverage 划分。')
    merge_settings(train['dataset'], settings, 'dataset')
    for key in ('train_val_dirs', 'test_dirs'):
        values = settings.get(key, ['.'])
        if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values):
            raise ValueError(f'dataset.{key} 必须是非空路径列表。')
        train['dataset'][key] = [str(project_path(v, manifest.parent)) for v in values]
    if 'manual_seed' in spec:
        if type(spec['manual_seed']) is not int:
            raise ValueError('manual_seed 必须是整数。')
        train['manual_seed'] = spec['manual_seed']
    train['dataset']['root_dir'] = f'./runs/{name}/cache'
    train['trainer']['checkpoint_dir'] = f'./runs/{name}/checkpoints'
    train['dataset_manifest'] = str(manifest)
    train['dataset_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    issues = check_config(train, 'train', root)
    if issues:
        raise ValueError('数据配置检查未通过：\n- ' + '\n- '.join(issues))
    predict = {'manual_seed': train.get('manual_seed', 3407),
               'dataset': copy.deepcopy(train['dataset']), 'model': copy.deepcopy(train['model']),
               'model_path': train['trainer']['checkpoint_dir'],
               'output_path': f'./runs/{name}/predictions',
               'dataset_manifest': str(manifest),
               'dataset_manifest_sha256': train['dataset_manifest_sha256']}
    created = []
    try:
        for path, config in zip(paths, (train, predict)):
            with path.open('x') as handle:
                created.append(path)
                yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return paths


def discover_datasets(root):
    roots = [root / 'data']
    source_file = root / 'dataset_sources.yaml'
    if source_file.is_file():
        registry = load_config(source_file)
        include_local = registry.get('include_local', True)
        if type(include_local) is not bool:
            raise ValueError('dataset_sources.yaml 的 include_local 必须是布尔值。')
        roots = [root / 'data'] if include_local else []
        sources = registry.get('roots')
        if not isinstance(sources, list) or not all(isinstance(p, str) and p for p in sources):
            raise ValueError('dataset_sources.yaml 的 roots 必须是路径列表。')
        roots.extend(project_path(value, root) for value in sources)
    paths = set()
    for directory in roots:
        if directory.is_dir():
            paths.update(p.resolve() for p in directory.rglob('dataset.yaml') if p.is_file())
    return sorted(paths)


def pick_dataset(root, hardening_only=False):
    paths = discover_datasets(root)
    if hardening_only:
        paths = [path for path in paths if load_config(path).get('hardening', {}).get('profile')]
        print('已筛选配置了 PPA 与寄存器映射的闭环数据集。')
    print('\n选择数据集（参数从 dataset.yaml 自动读取）：')
    for index, path in enumerate(paths, 1):
        label = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        print(f'  {index}. {label}')
    if not paths:
        print('  data/ 下暂无 dataset.yaml，也可以输入工程外的数据目录。')
    answer = input('数据集编号或数据 YAML/目录路径（留空取消）：').strip()
    if not answer:
        return None
    if answer.isdigit():
        if not 1 <= int(answer) <= len(paths):
            raise ValueError('数据集编号超出范围。')
        return paths[int(answer) - 1]
    return project_path(answer, root)


def diagnose(python):
    code = '''import importlib.util,json,sys
names=['torch','torch_geometric','pytorch_lightning','torchmetrics','numpy','pandas','sklearn','yaml','pytest','gymnasium','scipy','psutil']
print(json.dumps({'python':sys.executable,'version':sys.version.split()[0],
 'dependencies':{name:importlib.util.find_spec(name) is not None for name in names}}))
'''
    result = subprocess.run([python, '-c', code], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError(result.stderr.strip() or '解释器检查失败')
    return json.loads(result.stdout)


def run_job(command, root=ROOT):
    """Stream child logs and forward Ctrl+C to the entire launched job."""
    logs = root / 'runs' / 'ui_logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_path = logs / f'{stamp}.log'
    meta_path = logs / f'{stamp}.json'
    print('\n执行：' + shlex.join(command))
    print('工作目录：' + str(root))
    print('日志：' + str(log_path) + '\n', flush=True)
    start = time.monotonic()
    interrupted = False
    metadata = {'command': command, 'cwd': str(root), 'started_at': datetime.now().isoformat(),
                'status': 'running', 'returncode': None}
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    try:
        with log_path.open('w', buffering=1) as log:
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors='replace',
                                       bufsize=1, start_new_session=True,
                                       env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                returncode = process.wait()
            except KeyboardInterrupt:
                interrupted = True
                print('\n正在停止任务……', flush=True)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                    if process.poll() is not None:
                        break
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        process.wait(timeout=3)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                returncode = process.wait()
            finally:
                process.stdout.close()
        metadata.update(returncode=returncode, status='interrupted' if interrupted else (
            'completed' if returncode == 0 else 'failed'))
    except OSError as exc:
        metadata.update(status='failed', error=str(exc), returncode=127)
        print(f'启动失败：{exc}')
    finally:
        metadata['seconds'] = round(time.monotonic() - start, 3)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f'\n任务结束：{metadata["status"]}；退出码 {metadata["returncode"]}')
    return metadata['returncode']


def launch(mode, path, python, root=ROOT):
    config = load_config(path)
    issues = check_config(config, mode, root)
    if issues:
        raise ValueError('启动前检查未通过：\n- ' + '\n- '.join(issues))
    return run_job([python, str(root / f'{mode}.py'), '--config', str(Path(path).resolve())], root)


def pick_config(root, mode):
    configs = []
    for path in sorted((root / 'configs').glob('*.yaml')):
        try:
            config = load_config(path)
            if ('trainer' if mode == 'train' else 'model_path') in config:
                configs.append(path)
        except (ValueError, OSError, yaml.YAMLError):
            continue
    for index, path in enumerate(configs, 1):
        print(f'  {index}. {path.relative_to(root)}')
    answer = input('选择编号，或输入 YAML 路径（留空取消）：').strip()
    if not answer:
        return None
    if answer.isdigit() and 1 <= int(answer) <= len(configs):
        return configs[int(answer) - 1]
    path = project_path(answer, root)
    load_config(path)
    return path


def positive(prompt, default):
    value = input(f'{prompt} [{default}]：').strip()
    number = int(value or default)
    if number < 1:
        raise ValueError('请输入正整数。')
    return number


def show_results(root):
    files = sorted((root / 'runs').rglob('scores.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print('尚无评估结果。训练预测后在 runs/<任务>/predictions/ 查看。')
        return
    for path in files[:10]:
        print(f'\n{path.relative_to(root)}')
        with path.open(newline='') as handle:
            for row in list(csv.DictReader(handle))[:3]:
                for key, value in row.items():
                    print(f'  {key}: {value}')


def menu(python, root=ROOT, selected=None):
    selected = selected or {'train': None, 'predict': None}
    while True:
        print('\n' + '═' * 58)
        print('  HSTGNN + SAC-MOEA · 故障预测与加固控制台')
        print('═' * 58)
        print(f'  工程：{root}\n  Python：{python}')
        print(f'  训练配置：{selected["train"] or "未选择"}')
        print(f'  预测配置：{selected["predict"] or "未选择"}')
        print('\n  1 环境检查      2 从数据 YAML 配置\n  3 选择/查看配置  4 检查配置和数据'
              '\n  5 启动训练      6 启动预测\n  7 运行测试      8 查看评估结果'
              '\n  9 预测与强化学习加固闭环\n  0 退出')
        try:
            answer = input('\n请选择 > ').strip()
            if answer == '0':
                return
            if answer == '1':
                info = diagnose(python)
                print(f'\n解释器：{info["python"]}  Python {info["version"]}')
                for name, found in info['dependencies'].items():
                    print(f'  {"✓" if found else "缺失"} {name}')
                print('这是模块可发现性检查；实际兼容性请运行测试。')
            elif answer == '2':
                manifest = pick_dataset(root)
                if manifest:
                    train, predict = create_dataset_task(root, manifest)
                    selected.update(train=train, predict=predict)
                    config = load_config(train)
                    print(f'\n已自动生成并选择配对配置：\n  {train}\n  {predict}')
                    print(f'FF 数 {config["model"]["num_nodes"]}；窗口 {config["model"]["time_window_size"]}；'
                          f'训练 {config["trainer"]["epochs"]} 轮。')
            elif answer in ('3', '4'):
                mode = 'predict' if input('训练配置 t / 预测配置 p [t]：').strip().lower() == 'p' else 'train'
                path = pick_config(root, mode)
                if path:
                    selected[mode] = path
                    config = load_config(path)
                    if answer == '3':
                        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
                    else:
                        issues = check_config(config, mode, root)
                        print('\n'.join('- ' + x for x in issues) if issues else '基础检查通过。运行前仍需核对数据语义与维度。')
            elif answer in ('5', '6'):
                mode = 'train' if answer == '5' else 'predict'
                path = selected[mode] or pick_config(root, mode)
                if path:
                    selected[mode] = path
                    launch(mode, path, python, root)
            elif answer == '7':
                run_job([python, '-m', 'pytest', 'tests', '-q'], root)
            elif answer == '8':
                show_results(root)
                for path in sorted((root / 'runs').glob('*/pipeline_report.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                    report = json.loads(path.read_text())
                    print(f'闭环 {path.parent.name}: {report["status"]}, 加固数 {report.get("selected_register_count", "待生成")}')
            elif answer == '9':
                manifest = pick_dataset(root, hardening_only=True)
                if manifest:
                    smoke = input('快速冒烟 s / 完整运行 f [s]：').strip().lower() != 'f'
                    command = [python, str(root / 'pipeline.py'), '--dataset', str(manifest)]
                    if smoke:
                        command.append('--smoke')
                    run_job(command, root)
            else:
                print('请输入 0..9。')
        except EOFError:
            return
        except KeyboardInterrupt:
            print('\n已取消当前操作，输入 0 退出。')
        except (OSError, ValueError, subprocess.SubprocessError, yaml.YAMLError) as exc:
            print(f'\n操作未完成：{exc}')


def main():
    parser = argparse.ArgumentParser(description='HSTGNN 终端交互界面')
    parser.add_argument('--python', default=sys.executable, help='训练、预测、测试使用的解释器')
    parser.add_argument('--diagnose', action='store_true', help='输出环境检查 JSON 后退出')
    parser.add_argument('--check-config', help='只读检查指定配置，适合 agent 调用')
    parser.add_argument('--mode', choices=('train', 'predict'), default='train')
    parser.add_argument('--dataset', help='数据 YAML 或包含 dataset.yaml 的目录，自动完成配置')
    parser.add_argument('--task-name', help='指定新任务名，默认从数据名与时间戳生成')
    parser.add_argument('--prepare-only', action='store_true', help='配合 --dataset，仅生成配对配置并输出 JSON')
    parser.add_argument('--list-datasets', action='store_true', help='列出本地数据和 dataset_sources.yaml 中的数据 YAML')
    args = parser.parse_args()
    if (args.prepare_only or args.task_name) and not args.dataset:
        parser.error('--prepare-only / --task-name 需要 --dataset')
    if sum(bool(value) for value in (args.diagnose, args.check_config, args.dataset, args.list_datasets)) > 1:
        parser.error('--diagnose、--check-config、--dataset、--list-datasets 不能同时使用')
    if yaml is None:
        parser.exit(2, '缺少 PyYAML，请切换至项目 Python 环境或安装 PyYAML。\n')
    try:
        if args.list_datasets:
            print(json.dumps({'datasets': [str(p) for p in discover_datasets(ROOT)]}, ensure_ascii=False, indent=2))
            return 0
        if args.diagnose:
            info = diagnose(args.python)
            print(json.dumps(info, ensure_ascii=False, indent=2))
            return 0 if all(info['dependencies'].values()) else 1
        if args.check_config:
            path = project_path(args.check_config)
            issues = check_config(load_config(path), args.mode)
            print(json.dumps({'config': str(path), 'mode': args.mode, 'issues': issues}, ensure_ascii=False, indent=2))
            return 1 if issues else 0
        selected = None
        if args.dataset:
            train, predict = create_dataset_task(ROOT, args.dataset, args.task_name)
            if args.prepare_only:
                print(json.dumps({'train_config': str(train), 'predict_config': str(predict)}, ensure_ascii=False, indent=2))
                return 0
            selected = {'train': train, 'predict': predict}
        menu(args.python, selected=selected)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
