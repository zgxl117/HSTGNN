"""Lightweight checks for search_control_v2. This does not run a circuit simulation."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
import gymnasium as gym


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.sac_moea_env import SACMOEAEnv
from env.real_moea_wrapper import Individual, RealMoeaWrapper


class _Algorithm:
    def __init__(self):
        self.param_bounds = {}
        self.last_params = None
        self.pm = 0.001

    def update_parameters(self, params):
        self.last_params = dict(params)
        self.pm = float(params.get("pm", self.pm))


class _MoeaStub:
    def __init__(self, K=10000):
        self.K = K
        self.spea2 = _Algorithm()
        self.nsga2 = _Algorithm()
        self.algorithms = {"spea2": self.spea2, "nsga2": self.nsga2}
        self.search_controls = None

    def update_parameters(self, params):
        self.spea2.update_parameters(params["spea2"])
        self.nsga2.update_parameters(params["nsga2"])
        self.search_controls = dict(params["search_controls"])


def _load_config():
    with (ROOT / "configs" / "sac_config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_action_decode_and_dispatch():
    config = _load_config()
    moea = _MoeaStub(K=10000)
    env = SACMOEAEnv.__new__(SACMOEAEnv)
    env.config = config
    env.moea = moea
    env.action_schema = config["action_schema"]
    env._fixed_shared_moea_params = {
        "epsilon_dom": 0.00525,
        "accept_threshold": 0.0525,
        "delta_div": 0.0525,
        "beta_penalty": 65.0,
    }

    action = np.array([40.0, 0.90, 80.0, 0.85, 0.70, 0.12, 0.60, 0.08])
    decoded = env._decode_action_params(action)
    assert np.isclose(decoded["spea2"]["pm"], 0.004)
    assert np.isclose(decoded["nsga2"]["pm"], 0.008)
    assert decoded["controls"]["spea2_budget_share"] == 0.70

    env._update_moea_parameters(action)
    assert np.isclose(moea.spea2.last_params["pm"], 0.004)
    assert np.isclose(moea.nsga2.last_params["pm"], 0.008)
    assert moea.search_controls["elite_injection_ratio"] == 0.12


def test_adaptive_bounds_and_budget_schedule():
    config = _load_config()
    moea = _MoeaStub(K=10000)
    env = SACMOEAEnv.__new__(SACMOEAEnv)
    env.config = config
    env.moea = moea
    env.action_schema = config["action_schema"]
    env._init_adaptive_parameters()
    assert env._adaptive_action_low.shape == (8,)
    assert env._adaptive_action_high.shape == (8,)
    assert env._adaptive_action_low[0] == 4.0
    assert env._adaptive_action_high[0] == 128.0
    assert np.allclose(moea.spea2.param_bounds["pm"], (0.0004, 0.0128))

    wrapper = RealMoeaWrapper.__new__(RealMoeaWrapper)
    wrapper.pop_size = 100
    wrapper.search_controls = {
        "spea2_budget_share": 0.5,
        "elite_injection_ratio": 0.1,
        "local_search_strength": 0.5,
        "restart_ratio": 0.0,
    }
    wrapper._base_memetic_refine_count = 16
    wrapper._base_swap_local_search_max_swaps = 24
    wrapper._base_pair_swap_local_search_max_swaps = 3
    wrapper.configure_search_controls({
        "spea2_budget_share": 0.75,
        "elite_injection_ratio": 0.10,
        "local_search_strength": 0.50,
        "restart_ratio": 0.05,
    })
    spea2_updates = sum(wrapper._algorithm_active_for_generation("spea2", i, 8) for i in range(8))
    nsga2_updates = sum(wrapper._algorithm_active_for_generation("nsga2", i, 8) for i in range(8))
    assert spea2_updates == 8
    assert nsga2_updates == 4
    assert wrapper.greedy_archive_injection_count == 10
    assert wrapper.memetic_refine_count == 8

    wrapper.defer_archive_until_policy_ready = True
    wrapper.archive_unlock_updates = 8
    wrapper.set_sac_training_context(update_step=7)
    assert wrapper.sac_policy_ready is False
    wrapper.set_sac_training_context(update_step=8)
    assert wrapper.sac_policy_ready is True

    reward_env = SACMOEAEnv.__new__(SACMOEAEnv)
    reward_env.config = config
    small_positive = reward_env._normalize_action_credit("population_hv", 2.0e-5, 80.0)
    small_negative = reward_env._normalize_action_credit("population_hv", -2.0e-5, 80.0)
    assert small_positive > 0.5
    assert small_negative < -0.5
    assert reward_env._action_credit_scales["population_hv"] >= 1.0e-6


def test_operator_ablation_bounds_and_fixed_action():
    config = _load_config()
    low = [28.0, 0.83, 25.0, 0.83, 0.57, 0.09, 0.50, 0.11]
    high = [32.0, 0.85, 30.0, 0.85, 0.67, 0.13, 0.63, 0.15]
    fixed = [30.0, 0.84, 27.0, 0.84, 0.63, 0.11, 0.56, 0.13]
    config["operator_control"] = {
        "mode": "fixed",
        "action_low": low,
        "action_high": high,
        "fixed_action": fixed,
    }

    env = SACMOEAEnv.__new__(SACMOEAEnv)
    env.config = config
    env.moea = _MoeaStub(K=1380)
    env.action_schema = config["action_schema"]
    env.operator_control = config["operator_control"]
    env.operator_control_mode = "fixed"
    env._fixed_execution_action = None
    env._init_adaptive_parameters()

    assert np.allclose(env._adaptive_action_low, low)
    assert np.allclose(env._adaptive_action_high, high)
    assert np.allclose(env._fixed_execution_action, fixed)


def test_elite_injection_does_not_precredit_archive():
    wrapper = RealMoeaWrapper.__new__(RealMoeaWrapper)
    wrapper.pop_size = 4
    wrapper.current_population = []
    wrapper.spea2 = _Algorithm()
    wrapper.nsga2 = _Algorithm()
    wrapper.spea2.population = []
    wrapper.nsga2.population = []
    wrapper.greedy_archive_max_pool = 32
    wrapper._budget_aware_seed_genes = (
        lambda _count, record_archive=True: [np.array([1, 0, 1])]
    )
    wrapper._make_individual_from_gene = lambda gene: Individual(
        np.array(gene, copy=True), np.array([1.0, 1.0, -0.1, 0.2])
    )
    wrapper._should_memetic_refine = lambda _context: False
    wrapper._gene_key = lambda gene: tuple(np.asarray(gene, dtype=int).tolist())

    archive_calls = []
    wrapper._append_to_cumulative_elite_pool = (
        lambda *args, **kwargs: archive_calls.append((args, kwargs)) or 1
    )
    wrapper._install_budget_aware_elites(
        max_new=1,
        context="credit-test",
        record_archive=False,
    )

    assert len(wrapper.current_population) == 1
    assert archive_calls == []

    snapshot = wrapper._snapshot_population_for_credit(wrapper.current_population)
    wrapper.current_population[0].gene[0] = 0
    assert snapshot[0].gene[0] == 1

    # 局部搜索强度不能依赖精英注入比例；即使max_new=0，也应允许精修。
    wrapper._budget_aware_seed_genes = lambda _count, record_archive=True: []
    wrapper.memetic_refine_count = 1
    wrapper._should_memetic_refine = lambda _context: True
    wrapper._memetic_refine_population = lambda *_args, **_kwargs: [
        Individual(np.array([0, 1, 0]), np.array([1.0, 1.0, -0.1, 0.1]))
    ]
    wrapper._install_budget_aware_elites(
        max_new=0,
        context="generation-0",
        record_archive=False,
    )
    assert any(np.array_equal(ind.gene, [0, 1, 0]) for ind in wrapper.current_population)

    gated = RealMoeaWrapper.__new__(RealMoeaWrapper)
    gated.defer_archive_until_policy_ready = True
    gated.sac_policy_ready = False
    assert gated._budget_aware_seed_genes(max_genes=8, record_archive=False) == []


def test_compact_state_is_24d_and_finite():
    class _CompactMoea:
        def __init__(self):
            self.no_improvement_count = 1
            self.spea2 = _Algorithm()
            self.nsga2 = _Algorithm()
            self.spea2.population = [Individual([1, 0])]
            self.nsga2.population = [Individual([0, 1])]

        @staticmethod
        def _gene_metrics_fast(gene):
            selected = int(np.sum(gene))
            return {
                "feasible": True,
                "area_increase": 10.0 * selected,
                "power_increase": 5.0 * selected,
                "worst_slack": 0.2,
                "residual_ratio": 1.0 - 0.2 * selected,
            }

        @staticmethod
        def _calculate_scoring_hv_2d(population):
            return 0.4 if population else 0.0

        @staticmethod
        def get_feasible_archive_hv():
            return 0.5

        @staticmethod
        def _extract_ppa_feasible_pareto(population):
            return list(population or [])

    env = SACMOEAEnv.__new__(SACMOEAEnv)
    env.moea = _CompactMoea()
    env.config = {
        "sac": {"max_steps_per_episode": 6},
        "convergence": {"patience": 5},
    }
    env.state_schema = "compact_score_aligned_v1"
    env.state_dim = 24
    env.action_space = gym.spaces.Box(
        low=np.zeros(8, dtype=np.float32),
        high=np.ones(8, dtype=np.float32),
    )
    env.delta_area_max = 0.2
    env.delta_power_max = 0.5
    env.area_orig = 100.0
    env.power_orig = 100.0
    env.slack_orig = 0.3
    env.min_slack = 0.0
    env.current_episode_step = 2
    population = [Individual([1, 0]), Individual([0, 1])]
    state = env._extract_state(population, np.full(8, 0.5, dtype=np.float32))
    assert state.shape == (24,)
    assert np.all(np.isfinite(state))
    assert np.allclose(state[-8:], 0.5)


def test_score_aligned_selection_and_fresh_export():
    wrapper = RealMoeaWrapper.__new__(RealMoeaWrapper)
    wrapper.K = 3
    wrapper.pop_size = 2
    wrapper.area_orig = 100.0
    wrapper.power_orig = 100.0
    wrapper.score_aligned_environmental_selection = True
    wrapper.score_aligned_budget_bins = 8
    wrapper._ppa_limits = lambda: (0.2, 0.5, 0.0)
    wrapper._normalize_gene = lambda gene: np.asarray(gene, dtype=np.int32).flatten()
    wrapper._gene_key = lambda gene: tuple(np.asarray(gene, dtype=int).tolist())

    values = {
        (1, 0, 0): (0.10, 0.80),
        (0, 1, 0): (0.50, 0.40),
        (0, 0, 1): (0.90, 0.10),
    }

    def metrics(gene):
        area_ratio, residual = values[tuple(np.asarray(gene, dtype=int).tolist())]
        return {
            "feasible": True,
            "area_increase": 20.0 * area_ratio,
            "power_increase": 10.0,
            "worst_slack": 0.1,
            "residual_ratio": residual,
        }

    wrapper._gene_metrics_fast = metrics
    population = []
    for gene, (area_ratio, residual) in values.items():
        ind = Individual(gene, [100.0 + 20.0 * area_ratio, 110.0, -0.1, residual])
        ind.original_objectives = ind.objectives.copy()
        population.append(ind)
    selected = wrapper._environmental_selection(population)
    assert len(selected) == 2
    selected_keys = {tuple(ind.gene.tolist()) for ind in selected}
    assert (1, 0, 0) in selected_keys
    assert (0, 0, 1) in selected_keys

    exported = RealMoeaWrapper.__new__(RealMoeaWrapper)
    exported.K = 2
    exported.current_population = []
    exported.greedy_archive = []
    exported.cumulative_elite_pool = [{
        "gene": np.array([1, 0]),
        "objectives": np.array([999.0, 999.0, 999.0, 999.0]),
        "source": "stale",
    }]
    exported.ppa_pareto_only = False
    exported._normalize_gene = lambda gene: np.asarray(gene, dtype=np.int32).flatten()
    exported._gene_key = lambda gene: tuple(np.asarray(gene, dtype=int).tolist())
    exported._make_individual_from_gene = lambda gene: Individual(gene, [1.0, 2.0, -0.1, 0.25])
    exported._extract_pareto_front = lambda candidates: candidates
    result = exported.get_last_pareto(include_history=True)
    assert len(result) == 1
    assert np.isclose(result[0].objectives[3], 0.25)


def test_score_aligned_selection_restores_fixed_population_size():
    wrapper = RealMoeaWrapper.__new__(RealMoeaWrapper)
    wrapper.K = 8
    wrapper.pop_size = 6
    wrapper.area_orig = 100.0
    wrapper.power_orig = 100.0
    wrapper.score_aligned_environmental_selection = True
    wrapper.score_aligned_budget_bins = 4
    wrapper._ppa_limits = lambda: (1.0, 1.0, -1.0)
    wrapper._normalize_gene = lambda gene: np.asarray(gene, dtype=np.int32).flatten()
    wrapper._gene_key = lambda gene: tuple(np.asarray(gene, dtype=int).tolist())
    wrapper._repair_gene_to_ppa = lambda gene: np.asarray(gene, dtype=np.int32).flatten()

    def metrics(gene):
        gene = np.asarray(gene, dtype=np.int32)
        selected = float(np.sum(gene))
        weighted = float(np.dot(gene, np.arange(1, wrapper.K + 1)))
        return {
            "feasible": True,
            "area_increase": selected,
            "power_increase": selected,
            "worst_slack": 0.1,
            "residual_ratio": max(0.0, 1.0 - 0.01 * weighted),
        }

    wrapper._gene_metrics_fast = metrics
    wrapper._make_individual_from_gene = lambda gene: Individual(
        gene,
        [100.0 + np.sum(gene), 100.0 + np.sum(gene), -0.1, metrics(gene)["residual_ratio"]],
    )
    duplicate = Individual(np.array([1, 1, 1, 0, 0, 0, 0, 0]), [103.0, 103.0, -0.1, 0.94])

    np.random.seed(7)
    selected = wrapper._environmental_selection([duplicate] * 8)
    assert len(selected) == wrapper.pop_size
    assert len({tuple(ind.gene.tolist()) for ind in selected}) == wrapper.pop_size


def test_new_log_plots():
    import matplotlib
    matplotlib.use("Agg")
    from utils.plot_parameters import (
        plot_parameter_history,
        plot_parameter_statistics,
        plot_parameter_vs_performance,
    )

    rows = []
    for global_step in range(1, 17):
        rows.append({
            "episode": 1 + (global_step - 1) // 4,
            "sac_step": 1 + (global_step - 1) % 4,
            "global_step": global_step,
            "mutation_flips_spea2": 8.0 + global_step,
            "pc_spea2": 0.80 + 0.005 * global_step,
            "mutation_flips_nsga2": 12.0 + 0.5 * global_step,
            "pc_nsga2": 0.90 - 0.003 * global_step,
            "spea2_budget_share": 0.35 + 0.01 * global_step,
            "elite_injection_ratio": 0.04 + 0.002 * global_step,
            "local_search_strength": 0.25 + 0.01 * global_step,
            "restart_ratio": 0.08 - 0.002 * global_step,
            "action_quality": 0.01 * np.sin(global_step),
        })

    frame = pd.DataFrame(rows)
    with patch("pathlib.Path.exists", return_value=True), patch(
        "pandas.read_csv", side_effect=lambda *_args, **_kwargs: frame.copy()
    ):
        plot_parameter_history(Path("memory-log"), save_fig=False)
        plot_parameter_vs_performance(Path("memory-log"), save_fig=False)
        plot_parameter_statistics(Path("memory-log"), save_fig=False)

    import matplotlib.pyplot as plt
    assert plt.get_fignums()
    plt.close("all")


if __name__ == "__main__":
    test_action_decode_and_dispatch()
    test_adaptive_bounds_and_budget_schedule()
    test_elite_injection_does_not_precredit_archive()
    test_compact_state_is_24d_and_finite()
    test_score_aligned_selection_and_fresh_export()
    test_score_aligned_selection_restores_fixed_population_size()
    test_new_log_plots()
    print("search_control_v2 checks passed")
