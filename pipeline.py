"""Prediction -> sensitivity -> SAC/MOEA TMR selection -> analytic feedback."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import yaml
import terminal_ui as ui
from workflow.bridge import digest, join_sensitivity, read_csv, rescore_selection, write_csv

ROOT = Path(__file__).resolve().parent


def resolve_profile(spec, override=None):
    profile = override or spec.get('hardening', {}).get('profile')
    if not profile:
        raise ValueError('This dataset needs a hardening profile with its own PPA and register mapping; see hardening/profiles/riscv.yaml')
    path = Path(profile).expanduser()
    if not path.suffix:
        path = ROOT / 'hardening/profiles' / (str(profile) + '.yaml')
    elif not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    config = ui.load_config(path)
    for key in ('ppa_csv', 'mapping_csv', 'optimizer_config'):
        config[key] = str(ui.project_path(config[key], path.parent))
        if not Path(config[key]).is_file():
            raise ValueError(f'Missing {key}: {config[key]}')
    return path, config


def copy_campaigns(manifest, spec, output, smoke=False):
    """Make private raw copies because legacy preprocessing writes raw/processed/."""
    working = json.loads(json.dumps(spec))
    settings = working.setdefault('dataset', {})
    copies = {}
    provenance = []
    for key in ('train_val_dirs', 'test_dirs'):
        resolved = []
        for value in settings.get(key, ['.']):
            source = ui.project_path(value, manifest.parent)
            if source not in copies:
                target = output / f'campaign_{len(copies):02d}'
                raw = target / 'raw'
                raw.mkdir(parents=True)
                graph_ids = None
                if smoke:
                    graph_ids = {row['graph_id'] for row in read_csv(source / 'raw/graphs.csv')[:3]}
                    if not graph_ids:
                        raise ValueError('Empty campaign')
                for filename in ui.REQUIRED_CSV:
                    origin = source / 'raw' / filename
                    if graph_ids is None:
                        shutil.copy2(origin, raw / filename)
                    else:
                        with origin.open(newline='') as handle, (raw / filename).open('w', newline='') as dest:
                            reader = csv.DictReader(handle)
                            writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
                            writer.writeheader()
                            writer.writerows(row for row in reader if row['graph_id'] in graph_ids)
                    provenance.append({'source': str(origin), 'sha256': digest(origin),
                                       'copy': str(raw / filename), 'smoke_graphs': sorted(graph_ids) if graph_ids else None})
                copies[source] = target
            resolved.append(str(copies[source]))
        settings[key] = resolved
    if smoke:
        working.setdefault('trainer', {})['epochs'] = 2
        working['trainer']['batch_size'] = 1
    path = output / 'dataset.yaml'
    path.write_text(yaml.safe_dump(working, allow_unicode=True, sort_keys=False))
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    return path


def optimizer_config(profile, output, smoke, seed):
    config = ui.load_config(profile['optimizer_config'])
    config['operator_control'] = {'mode': 'adaptive'}
    config['cpu_threads'] = 2
    config.setdefault('evaluator', {}).update(use_gpu=False, num_processes=1)
    config['moea'].update(num_processes=1, use_gpu_acceleration=False)
    # The profile may override the upstream defaults, but not register input/output paths.
    for key, values in profile.get('optimizer_overrides', {}).items():
        if key not in config or not isinstance(values, dict):
            raise ValueError('Optimizer overrides must target existing config sections')
        config[key].update(values)
    if config.get('operator_control', {}).get('mode') != 'adaptive':
        raise ValueError('The integrated workflow requires adaptive SAC control')
    if smoke:
        config['sac'].update(batch_size=2, min_replay_size=2, updates_per_step=1,
                             hidden_dim=32, buffer_size=100, pure_exploration_steps=2,
                             max_steps_per_episode=4, min_sac_steps_before_convergence_stop=4)
        config['training']['max_episodes'] = 2
        config['moea'].update(pop_size=12, archive_size=12, max_generations=2, generations_per_step=1,
                              initial_heuristic_seeds=8, greedy_prefix_seed_count=8,
                              enable_milp_archive=False, greedy_archive_size=24,
                              greedy_archive_max_pool=32, greedy_archive_injection_count=4,
                              greedy_trajectory_refine_count=0, archive_unlock_updates=1,
                              elite_pool_save_limit=32, memetic_refine_count=2,
                              swap_local_search_max_swaps=2, pair_swap_local_search_enabled=False,
                              allow_early_stop_on_convergence=False)
    config['training']['checkpoint_dir'] = str(output / 'sac_checkpoints')
    (output / 'optimizer_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    return output / 'optimizer_config.yaml'


def execute(command, run, name, cwd=ROOT):
    print(f'\n[{name}] ' + ' '.join(map(str, command)), flush=True)
    path = run / (name + '.log')
    started = time.monotonic()
    # Inherit the terminal: Ctrl+C reaches both runner and current child.
    with path.open('w') as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors='replace', env={**os.environ, 'PYTHONUNBUFFERED': '1',
                                   'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f'{name} failed (exit {code}); see {path}')
    return {'stage': name, 'returncode': code, 'seconds': round(time.monotonic() - started, 3), 'log': str(path)}


def run_pipeline(args):
    manifest = ui.project_path(args.dataset)
    if manifest.is_dir():
        manifest = manifest / 'dataset.yaml'
    spec = ui.load_config(manifest)
    profile_path, profile = resolve_profile(spec, args.profile)
    environment = ui.diagnose(sys.executable)
    missing = [key for key, available in environment['dependencies'].items() if not available and key != 'pytest']
    if missing:
        raise ValueError('Missing pipeline dependencies: ' + ', '.join(missing))
    name = args.name or 'loop_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,49}', name):
        raise ValueError('Run name must contain 1..50 letters/digits/_/-')
    run = ROOT / 'runs' / name
    run.mkdir(parents=True, exist_ok=False)
    report_path = run / 'pipeline_report.json'
    report = {'status': 'running', 'source_dataset': str(manifest), 'dataset_sha256': digest(manifest),
              'profile': str(profile_path), 'profile_sha256': digest(profile_path), 'smoke': args.smoke,
              'python': sys.executable, 'stages': [], 'run': str(run)}
    report['environment'] = environment
    report_path.write_text(json.dumps(report, indent=2))
    try:
        if args.train_config:
            train = ui.project_path(args.train_config)
            predictions = ui.project_path(args.predictions)
            report['model_mode'] = 'reuse_existing_predictions'
            report.update(train_config=str(train), predictions=str(predictions),
                          predictions_sha256=digest(predictions), train_config_sha256=digest(train))
        else:
            source = run / 'inputs'
            source.mkdir()
            working = copy_campaigns(manifest, spec, source, args.smoke)
            train, predict = ui.create_dataset_task(ROOT, working, name + '_model')
            report['model_mode'] = 'train_and_predict'
            report['train_config'] = str(train)
            report['predict_config'] = str(predict)
            for mode, path in [('train', train), ('predict', predict)]:
                issues = ui.check_config(ui.load_config(path), mode)
                if issues:
                    raise ValueError('\n'.join(issues))
                report['stages'].append(execute([sys.executable, str(ROOT / f'{mode}.py'), '--config', str(path)], run, mode))
                report_path.write_text(json.dumps(report, indent=2))
            config = ui.load_config(predict)
            predictions = ui.project_path(config['output_path']) / 'results.csv'
        report['stages'].append(execute([sys.executable, str(ROOT / 'export_sensitivity.py'),
                                          '--config', str(train), '--predictions', str(predictions),
                                          '--output', str(run / 'sensitivity')], run, 'sensitivity'))
        table = run / 'optimizer_input.csv'
        report['bridge'] = join_sensitivity(profile['ppa_csv'], profile['mapping_csv'],
                                             run / 'sensitivity/sensitivity.csv', table)
        shutil.copy2(profile['mapping_csv'], run / 'register_mapping.csv')
        config = optimizer_config(profile, run, args.smoke, spec.get('manual_seed', 3407))
        k = min(32, report['bridge']['registers']) if args.smoke else int(profile['candidate_count'])
        worker_cwd = run / 'optimizer_work'
        worker_cwd.mkdir()
        report['stages'].append(execute([sys.executable, str(ROOT / 'hardening/worker.py'),
                                          '--input', str(table), '--config', str(config), '--K', str(k),
                                          '--seed', str(spec.get('manual_seed', 3407)),
                                          '--output', str(run / 'hardening')], run, 'hardening', worker_cwd))
        result = json.loads((run / 'hardening/result.json').read_text())
        selected = result['selected_instance_names']
        mapping = read_csv(run / 'register_mapping.csv')
        feedback = {'prediction_based': rescore_selection(read_csv(run / 'sensitivity/sensitivity.csv'),
                                                          mapping, selected, 'sensitivity'),
                    'measured_reference_based': rescore_selection(read_csv(run / 'sensitivity/measured_reference.csv'),
                                                                  mapping, selected, 'measured_sensitivity'),
                    'scope': 'offline rescore of ideal TMR selection; excludes netlist/EDA and physical post-injection',
                    'measured_reference_used_for_selection': False}
        (run / 'feedback.json').write_text(json.dumps(feedback, indent=2))
        by_name = {row['instance_name']: row['node_id'] for row in mapping}
        rows = read_csv(run / 'hardening/selected_registers.csv')
        write_csv(run / 'selected_registers.csv', [{'node_id': by_name[row['instance_name']], **row} for row in rows],
                  ['node_id', 'candidate_id', 'index', 'instance_name', 'reg_type', 'area', 'reg_power',
                   'reg_dynamic_power(uW)', 'reg_leakage_power(uW)', 'path_slack(ns)', 'sensitivity'])
        report.update(status='completed', selected_register_count=len(selected),
                      sac_updates=result['sac_updates'], policy_transitions=result['policy_transitions'],
                      search_attribution=result.get('search_attribution', {}),
                      optimization_scope=result['evaluation_scope'], feedback=feedback,
                      chosen_solution=result['chosen_solution'])
        (run / 'README.md').write_text('# HSTGNN → SAC-MOEA 闭环结果\n\n状态：completed。\n\n'
            'selected_registers.csv 是最终选择清单，hardening/pareto_solutions.csv 是可行方案。\n'
            'pipeline_report.json 记录来源、步骤、PPA 和敏感性；feedback.json 对同一选择给出预测与测量参考的回评。\n'
            '测量参考只用于回评，不进入优化。PPA 与 TMR 敏感性均为解析模型结果。\n')
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2))
    print(f'\nPIPELINE_REPORT={report_path}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--profile', help='Built-in profile name or path to a custom profile YAML')
    parser.add_argument('--name')
    parser.add_argument('--smoke', action='store_true', help='First three graphs, two model epochs, small SAC/MOEA budget')
    parser.add_argument('--train-config', help='Reuse completed prediction run; must pair with --predictions')
    parser.add_argument('--predictions')
    args = parser.parse_args()
    if bool(args.train_config) != bool(args.predictions):
        parser.error('--train-config and --predictions must be supplied together')
    try:
        run_pipeline(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'Pipeline failed: {exc}\n')
