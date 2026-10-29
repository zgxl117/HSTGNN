"""Small real-data smoke test for the score-aligned SAC+MOEA pipeline."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.real_moea_wrapper import RealMoeaWrapper
from env.sac_moea_env import SACMOEAEnv
from main import create_reg_table
from mo_optimizer.simulator import ICDesignSimulator


def load_smoke_config() -> dict:
    with (ROOT / "configs" / "sac_config.yaml").open("r", encoding="utf-8") as handle:
        config = copy.deepcopy(yaml.safe_load(handle))

    config["moea"].update(
        {
            "pop_size": 24,
            "max_generations": 2,
            "generations_per_step": 1,
            "archive_size": 24,
            "num_processes": 1,
            "enable_milp_archive": False,
            "greedy_archive_size": 64,
            "greedy_archive_max_pool": 256,
            "greedy_trajectory_refine_count": 8,
            "greedy_archive_injection_count": 8,
            "elite_pool_max_size": 256,
            "elite_pool_save_limit": 256,
        }
    )
    config["evaluator"]["use_gpu"] = False
    return config


def assert_strict_feasible_front(wrapper: RealMoeaWrapper, front: list) -> None:
    assert front, "final Pareto front is empty"
    for individual in front:
        metrics = wrapper._gene_metrics_fast(individual.gene)
        assert metrics["feasible"], "infeasible solution entered the final Pareto front"

    points = [wrapper._score_aligned_point(individual) for individual in front]
    points = [np.asarray(point, dtype=np.float64) for point in points if point is not None]
    assert len(points) == len(front)
    for i, point_i in enumerate(points):
        for j, point_j in enumerate(points):
            if i == j:
                continue
            dominates = np.all(point_j <= point_i + 1e-12) and np.any(point_j < point_i - 1e-12)
            assert not dominates, "dominated solution entered the final score-aligned front"


def main() -> None:
    config = load_smoke_config()
    simulator = ICDesignSimulator(
        csv_file_path=str(ROOT / "data" / "merged_registers.csv"),
        K=96,
        config=config,
    )
    reg_table = create_reg_table(simulator, config=config)
    wrapper = RealMoeaWrapper(
        reg_table=reg_table,
        pop_size=config["moea"]["pop_size"],
        max_gen=config["moea"]["max_generations"],
        archive_size=config["moea"]["archive_size"],
        num_processes=1,
        config=config,
        use_gpu=False,
    )
    env = SACMOEAEnv(moea_wrapper=wrapper, config=config)

    state, reset_info = env.reset(seed=7)
    assert state.shape == (24,)
    assert np.all(np.isfinite(state))
    baseline_hv = float(wrapper.get_feasible_archive_hv())
    assert baseline_hv > 0.0, "greedy baseline archive was not initialized"

    action = np.asarray(config["initial_action"], dtype=np.float32)
    next_state, reward, terminated, truncated, step_info = env.step(action)
    assert next_state.shape == (24,)
    assert np.all(np.isfinite(next_state))
    assert int(step_info.get("episode_step", 0)) == 1

    final_front = wrapper.get_last_pareto(include_history=True)
    assert_strict_feasible_front(wrapper, final_front)
    final_hv = float(wrapper._calculate_scoring_hv_2d(final_front))
    assert final_hv + 1e-12 >= baseline_hv
    attribution = wrapper.get_search_attribution(final_front)
    assert attribution["final_hv"] + 1e-12 >= attribution["baseline_hv"]

    result = {
        "K": wrapper.K,
        "state_dim": int(state.shape[0]),
        "action_dim": int(action.shape[0]),
        "initial_population": int(reset_info.get("initial_population_size", 0)),
        "baseline_hv": baseline_hv,
        "final_hv": final_hv,
        "front_size": len(final_front),
        "search_hv_gain": float(attribution["search_hv_gain"]),
        "nonbaseline_front_points": int(attribution["nonbaseline_front_points"]),
        "exploration_step_front_points": int(attribution["exploration_step_front_points"]),
        "policy_step_front_points": int(attribution["policy_step_front_points"]),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    print("REAL_DATA_SMOKE=" + json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
