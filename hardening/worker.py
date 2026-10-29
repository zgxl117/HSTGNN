"""Run the bundled SAC/MOEA core with strict inputs and machine-readable outputs."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml

# Keep hardening/ first, so its legacy utils/env/algorithms packages stay isolated.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from workflow.bridge import read_ppa, write_csv
from env.real_moea_wrapper import RealMoeaWrapper
from env.sac_moea_env import SACMOEAEnv
from sac.sac_agent import SACAgent


def table_from_ppa(path, k, config):
    _, design, registers = read_ppa(path)
    if not 1 <= k <= len(registers):
        raise ValueError('K must be within the real register count')
    scores = np.array([float(row['sensitivity']) for row in registers])
    indices = np.argsort(-scores, kind='stable')[:k]
    slacks = np.array([float(row['path_slack(ns)']) for row in registers])
    table = {key: np.array([float(registers[i][column]) for i in indices])
             for key, column in [('area', 'area'), ('power', 'reg_power'),
                                 ('slack', 'path_slack(ns)'), ('sensitivity', 'sensitivity')]}
    table.update(K=k, design_area=float(design[0]), design_power=float(design[1]),
                 design_slack=float(design[2]), design_sensitivity=float(scores.mean()),
                 sens_orig=float(scores.mean()), total_num_registers=len(registers),
                 total_sensitivity_sum=float(scores.sum()),
                 non_prefiltered_sensitivity_sum=float(scores.sum() - scores[indices].sum()),
                 is_critical=(slacks[indices] < slacks.mean() * 0.6).astype(int),
                 original_indices=indices, instance_names=np.array([registers[i]['instance_name'] for i in indices]),
                 reg_types=np.array([registers[i]['reg_type'] for i in indices]), config=config)
    return table, indices, registers


def run(input_csv, config_path, output, k, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get('operator_control', {}).get('mode', 'adaptive') != 'adaptive':
        raise ValueError('Integrated RL workflow requires adaptive operator control')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    table, indices, registers = table_from_ppa(input_csv, k, config)
    write_csv(output / 'candidate_registers.csv', [
        {'candidate_id': i, **registers[int(index)]} for i, index in enumerate(indices)])
    moea = config['moea']
    wrapper = RealMoeaWrapper(table, pop_size=moea['pop_size'], max_gen=moea['max_generations'],
                              archive_size=moea['archive_size'], num_processes=moea['num_processes'],
                              config=config, use_gpu=config.get('evaluator', {}).get('use_gpu', False))
    env = SACMOEAEnv(wrapper, config)
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space,
                     config['sac'], device='cpu')
    episodes = int(config['training']['max_episodes'])
    steps = int(config['sac']['max_steps_per_episode'])
    if episodes < 1 or steps < 1:
        raise ValueError('Positive episode/step budget required')
    records = []
    started = time.monotonic()
    global_step = 0
    for episode in range(episodes):
        if hasattr(env, 'set_episode_index'):
            env.set_episode_index(episode + 1)
        state, _ = env.reset(seed=seed + episode)
        for step in range(steps):
            policy = global_step >= int(config['sac'].get('pure_exploration_steps', 2)) and agent.update_step > 0
            action = agent.select_action(state) if policy else env.action_space.sample()
            if hasattr(env, 'set_sac_training_context'):
                env.set_sac_training_context(update_step=agent.update_step,
                                             action_mode='stochastic_sample' if policy else 'random_exploration')
            next_state, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(reward) or not np.isfinite(next_state).all():
                raise ValueError('Nonfinite SAC transition')
            agent.store_transition(state, action, reward, next_state, terminated)
            if len(agent.replay_buffer) >= max(agent.batch_size, int(config['sac'].get('min_replay_size', agent.batch_size))):
                for _ in range(int(config['sac'].get('updates_per_step', 1))):
                    agent.update()
            records.append({'episode': episode + 1, 'step': step + 1, 'reward': float(reward),
                            'sac_updates': agent.update_step, 'action_mode': 'learned_policy' if policy else 'exploration',
                            'action': json.dumps(np.asarray(action).tolist())})
            global_step += 1
            state = next_state
            if terminated or truncated:
                break
    agent.save_checkpoint(str(output / 'sac_checkpoint.pth'))
    write_csv(output / 'transitions.csv', records)
    solutions = []
    population = wrapper.get_last_pareto(include_history=True)
    threshold = config['constraint_thresholds']
    for individual in population:
        gene = np.asarray(individual.gene, dtype=int).reshape(-1)
        if len(gene) != k or not np.isin(gene, [0, 1]).all():
            raise ValueError('Invalid backend gene')
        area, power, slack, sensitivity = map(float, wrapper.model.evaluate_with_details(gene)[:4])
        # A candidate subset cannot improve the untouched design's worst timing.
        slack = min(slack, table['design_slack'])
        area_overhead = area / table['design_area'] - 1
        power_overhead = power / table['design_power'] - 1
        if not (area_overhead <= threshold['delta_area_max'] + 1e-9
                and power_overhead <= threshold['delta_power_max'] + 1e-9
                and slack >= threshold['slack_min'] - 1e-9):
            continue
        if not config['moea'].get('ser_as_objective', True) and sensitivity > threshold['delta_sens_min'] * table['design_sensitivity'] + 1e-12:
            continue
        solutions.append({'solution_id': f'solution_{len(solutions):04d}',
                          'genes': ''.join(str(int(v)) for v in gene), 'tmr_count': int(gene.sum()),
                          'area_total': area, 'power_total': power, 'worst_slack': slack,
                          'area_overhead': area_overhead, 'power_overhead': power_overhead,
                          'residual_sensitivity': sensitivity,
                          'residual_ratio': sensitivity / max(table['design_sensitivity'], 1e-20)})
    if not solutions:
        raise RuntimeError('Backend produced no feasible solution; no success report will be issued')
    write_csv(output / 'pareto_solutions.csv', solutions)
    best = min(solutions, key=lambda row: (row['residual_sensitivity'], row['area_total'], row['tmr_count']))
    gene = np.array([int(v) for v in best['genes']])
    chosen = [{'candidate_id': i, **registers[int(indices[i])]} for i in range(k) if gene[i]]
    write_csv(output / 'selected_registers.csv', chosen, ['candidate_id'] + list(registers[0]))
    report = {'status': 'completed', 'chosen_solution': best, 'K': k, 'total_registers': len(registers),
              'sac_updates': agent.update_step, 'transitions': len(records),
              'policy_transitions': sum(row['action_mode'] == 'learned_policy' for row in records),
              'operator_control': env.operator_control_mode,
              'selected_instance_names': [row['instance_name'] for row in chosen],
              'seconds': round(time.monotonic() - started, 3),
              'evaluation_scope': 'analytic PPA and ideal TMR proxy; no netlist synthesis or post-injection'}
    if hasattr(wrapper, 'get_search_attribution'):
        report['search_attribution'] = wrapper.get_search_attribution(population)
    (output / 'result.json').write_text(json.dumps(report, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    print(json.dumps({key: report[key] for key in ('status', 'sac_updates', 'policy_transitions', 'K')}, indent=2))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    run(args.input, args.config, args.output, args.K, args.seed)
