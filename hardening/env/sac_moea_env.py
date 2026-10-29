# env/sac_moea_env.py
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import time
from pathlib import Path
from algorithms.core_data import Individual
from utils.constraint_analyzer import ConstraintAnalyzer
from utils.plot_pareto_4d import plot_pareto_4d

class SACMOEAEnv(gym.Env):
    """
    SAC环境，用于动态控制混合MOEA搜索过程。
    状态包含Pareto前沿、约束、算法状态与跨episode上下文；
    search_control_v2使用8维动作分别控制两算法及混合搜索资源。
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, moea_wrapper, config: Dict[str, Any]):
        super().__init__()
        self.moea = moea_wrapper
        self.config = config
        # 当前Episode编号（由外部训练循环设置，用于结果保存路径）
        self.current_episode_index: int = 1
        
        # 收敛检测相关参数
        self.convergence_threshold = config.get('convergence_threshold', 1e-4)  # 收敛阈值
        self.patience = config.get('convergence_patience', 5)  # 连续多少代没有改进则收敛
        self.no_improvement_count = 0  # 没有改进的代数计数器

        # HV历史记录，用于收敛检测
        self.hv_history = []
        self.current_episode_step = 0

        # ============ 跨Episode长期学习组件 ============
        # 调参历史：记录每个Episode的调参决策和效果，用于跨Episode学习
        self.parameter_tuning_history: List[Dict] = []

        # 经验回放缓冲区：存储历史最优状态-动作对
        self.expert_buffer: List[Dict] = []
        self.max_expert_buffer_size = config.get('sac', {}).get('max_expert_buffer_size', 100)

        # 全局最优参数记录
        self.global_best_params: Optional[Dict] = None
        self.global_best_hv: float = 0.0

        # Episode间的性能追踪
        self.episode_hv_record: List[float] = []  # 记录每个Episode的最终HV
        
        # ============ 计算缓存和增量Pareto维护 ============
        from utils.compute_cache import ComputationCache, IncrementalParetoMaintainer, PerformanceProfiler
        self.compute_cache = ComputationCache()  # HV/IGD/多样性缓存
        self.pareto_maintainer = IncrementalParetoMaintainer()  # 增量式Pareto维护
        self.profiler = PerformanceProfiler()  # 性能分析
        self.enable_incremental_pareto = config.get('optimization', {}).get('enable_incremental_pareto', True)
        
        # 加载约束阈值
        self.constraint_thresholds = config.get('constraint_thresholds', {})
        self.delta_area_max = self.constraint_thresholds.get('delta_area_max', 0.2)
        self.delta_power_max = self.constraint_thresholds.get('delta_power_max', 0.15)
        self.delta_sens_min = self.constraint_thresholds.get('delta_sens_min', 0.5)
        self.min_slack = self.constraint_thresholds.get('min_slack', 0.0)
        self.ser_as_objective = bool(config.get('moea', {}).get('ser_as_objective', True))
        self.action_schema = str(config.get('action_schema', 'legacy_hyperparameters'))
        # Ablation controls are deliberately kept outside the SAC algorithm:
        # adaptive is the normal baseline, narrow constrains each action
        # dimension to a late-training interval, and fixed removes action
        # control from the search while keeping the same MOEA budget.
        self.operator_control = config.get('operator_control', {}) or {}
        self.operator_control_mode = str(self.operator_control.get('mode', 'adaptive')).lower()
        if self.operator_control_mode not in {'adaptive', 'narrow', 'fixed'}:
            raise ValueError(
                "operator_control.mode must be one of adaptive, narrow, fixed"
            )
        self._fixed_execution_action = None
        
        # 违约权重
        self.area_weight = self.constraint_thresholds.get('area_weight', 0.1)
        self.power_weight = self.constraint_thresholds.get('power_weight', 0.1)
        self.slack_weight = self.constraint_thresholds.get('slack_weight', 1000.0)
        self.sens_weight = self.constraint_thresholds.get('sens_weight', 0.5)
        
        # 时序约束强化参数
        self.timing_violation_multiplier = self.constraint_thresholds.get('timing_violation_multiplier', 100.0)
        self.strict_timing_mode = self.constraint_thresholds.get('strict_timing_mode', True)
        
        # 从模拟器获取原始值
        self._init_original_values()

        # ============ 自适应参数初始化 ============
        self._init_adaptive_parameters()

        ### 归一化范围（基于实际电路参数动态计算）
        self._setup_normalization_ranges()

    def _update_hv_igd_normalization(self, current_hv: float, current_igd: float):
        """根据实际HV/IGD值动态更新归一化范围"""
        # 收集足够的历史数据后再更新（避免初始值波动影响）
        min_history = 10

        # 更新 HV 范围
        if len(self.hv_history) >= min_history:
            max_hv = max(self.hv_history[-50:]) * 1.2  # 留20%余量
            min_hv = 0
            self.normalization_ranges['hv'] = (min_hv, max(max_hv, 0.001))

        # 更新 IGD 范围
        if current_igd >= 0 and len(self.igd_history) >= min_history:
            max_igd = max(self.igd_history[-50:]) * 1.2
            self.normalization_ranges['igd'] = (0, max(max_igd, 0.001))

    def _setup_normalization_ranges(self):
        """
        根据实际电路参数动态设置归一化范围

        归一化范围 = [原始值, 原始值 * (1 + 最大允许增量)]
        这样即使面积超过200000，也能正确归一化
        """
        # 从之前获取的原始值计算归一化范围
        area_orig = getattr(self, 'area_orig', 100000.0)
        power_orig = getattr(self, 'power_orig', 50.0)
        slack_orig = getattr(self, 'slack_orig', 0.05)

        # 获取最大允许增量（从约束阈值）
        delta_area_max = self.delta_area_max  # 如 0.5 表示允许增加50%
        delta_power_max = self.delta_power_max

        # 归一化范围 = [0, 原始值 * (1 + 最大允许增量)]
        # slack越小越危险，所以范围是 [0, slack_orig * 1.5]
        self.normalization_ranges = {
            'area': (0, area_orig * (1 + delta_area_max) * 1.2),  # 留20%余量
            'power': (0, power_orig * (1 + delta_power_max) * 1.2),
            'slack': (0, slack_orig * 1.5 if slack_orig > 0 else 0.2),  # slack为正时用1.5倍，否则用默认值
            'sens': (0, 1.0),         # 敏感度范围固定
            'hv': (0, 1000),          # 超体积范围（动态）
            'igd': (0, 10),           # IGD范围（动态）
        }

        print(f"[归一化范围] 基于实际电路参数:")
        print(f"  area: {self.normalization_ranges['area']}")
        print(f"  power: {self.normalization_ranges['power']}")
        print(f"  slack: {self.normalization_ranges['slack']}")
        
        # 自适应范围在 _init_adaptive_parameters() 中按电路规模生成。
        # 这里不能再次用全局配置覆盖，否则大规模电路仍会采样过宽的动作。
        action_low = getattr(self, '_adaptive_action_low', self.config['action_low'])
        action_high = getattr(self, '_adaptive_action_high', self.config['action_high'])
        self.action_space = gym.spaces.Box(
            low=np.array(action_low, dtype=np.float32),
            high=np.array(action_high, dtype=np.float32),
            dtype=np.float32
        )

        self.state_schema = str(self.config.get('state_schema', 'legacy_64d'))

        # 紧凑状态只保留与最终评分、约束和两算法分工直接相关的特征。
        if self.state_schema == 'compact_score_aligned_v1':
            self.use_algorithm_specific_states = False
            self.expert_context_dim = 0
            self.state_dim = int(self.config.get('state_dim', 24))
        else:
            # 状态空间维度处理：支持两种配置习惯
            base_state_dim = self.config.get('state_dim', 24)
            self.use_algorithm_specific_states = self.config.get('use_algorithm_specific_states', True)

            # 如果配置看起来已经是完整维度（大于等于48），直接采用，避免重复累加
            if self.use_algorithm_specific_states and isinstance(base_state_dim, int) and base_state_dim >= 48:
                self.state_dim = base_state_dim
            else:
                # 否则把 base_state_dim 视为“合并种群”的基础维度
                self.state_dim = base_state_dim + (24 if self.use_algorithm_specific_states else 0)

            # 跨Episode专家上下文仅用于旧状态结构。
            self.expert_context_dim = 10
            self.state_dim += self.expert_context_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # 状态和动作历史
        self.state_history = []
        self.action_history = []
        
        # 性能指标历史
        self.hv_history = []
        self.igd_history = []
        self.diversity_history = []
        
        # 当前状态
        self.current_state = None
        self.prev_metrics = {
            'hv': 0.0,
            'igd': float('inf'),
            'diversity': 0.0,
            'violation': 0.0
        }
        
        # 前一次的Pareto前沿中心（用于计算前沿振荡项）
        self.prev_pareto_center = None
        self.prev_population = None
        
        # 算法独立的状态历史（用于区分两个算法的贡献）
        self.spea2_metrics = {
            'hv': 0.0,
            'diversity': 0.0,
            'violation': 0.0,
            'prev_hv': 0.0
        }
        self.nsga2_metrics = {
            'hv': 0.0,
            'diversity': 0.0,
            'violation': 0.0,
            'prev_hv': 0.0
        }
        
        # 算法HV历史（用于状态空间）
        self.spea2_hv_history = []
        self.nsga2_hv_history = []
        
        # Episode收敛标志（用于在第一次收敛时结束episode）
        self.episode_converged = False
        self._min_steps_before_convergence_stop = max(
            1,
            int(self.config.get('sac', {}).get('min_sac_steps_before_convergence_stop', 4)),
        )

        # 每步MOEA预算模式：固定代数，尽量不提前结束
        self.fixed_budget_per_sac_step = bool(self.config.get('moea', {}).get('fixed_budget_per_sac_step', True))
        self.generations_per_sac_step = int(self.config.get('moea', {}).get('generations_per_step', 20))
        self.allow_early_stop = bool(self.config.get('moea', {}).get('allow_early_stop_on_convergence', False))
        self.strict_early_stop = bool(self.config.get('moea', {}).get('strict_early_stop', False))
        self.action_schema = str(self.config.get('action_schema', 'legacy_hyperparameters'))

        # 收敛历史记录
        self.convergence_history = {
            'hv': [],  # 超体积历史
            'igd': [],  # IGD历史
            'generation': 0,  # 当前代数
        }
        
        # 参考点（用于HV计算）
        self.reference_point = None

    def close(self):
        """关闭环境并释放并行资源（进程池/GPU worker/Manager 等）。"""
        try:
            if hasattr(self, 'moea') and self.moea is not None and hasattr(self.moea, 'close'):
                self.moea.close()
        except Exception as e:
            try:
                print(f"[WARNING] SACMOEAEnv.close() 失败: {e}")
            except Exception:
                pass
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境（开始新的Episode）
        每次Episode开始时重置MOEA
        """
        super().reset(seed=seed)

        # 重置计数器
        self.current_episode_step = 0

        # 重置MOEA（新Episode开始），若存在上一episode经验则传递以进行热启动
        transfer_experience = getattr(self, 'last_experience', None)
        if hasattr(self.moea, 'reset'):
            print("[ENV]: 重置MOEA，开始新Episode")
            # 传递经验（如果存在）以便热启动
            try:
                self.moea.reset(transfer_experience=transfer_experience)
            except TypeError:
                # 向后兼容：如果 reset 不接受参数则直接调用
                self.moea.reset()
        else:
            # 重置内部状态
            self.moea.current_population = None
            self.moea.current_generation = 0

        # 重置历史记录
        self.state_history = []
        self.action_history = []
        self.hv_history = []

        # 重置性能指标
        self.prev_metrics = {
            'hv': 0.0,
            'igd': 0.0,
            'diversity': 0.0,
            'violation': 0.0
        }

        # ── 跨Episode最优解追踪（不清零，保留历史最佳）───────────────────────
        if not hasattr(self, 'global_best_sensitivity'):
            self.global_best_sensitivity = 1.0   # 越低越好
        if not hasattr(self, 'global_best_residual_ratio'):
            self.global_best_residual_ratio = 1.0
        if not hasattr(self, 'global_best_feasible_ser'):
            self.global_best_feasible_ser = 1.0
        self.prev_step_best_feasible_ser = None
        if not hasattr(self, 'episode_best_sensitivity'):
            self.episode_best_sensitivity = []   # 每 episode 最佳 sensitivity 列表
        
        # 重置Pareto前沿中心历史
        self.prev_pareto_center = None
        self.prev_population = None
        
        # 重置算法独立状态
        self.spea2_metrics = {
            'hv': 0.0,
            'diversity': 0.0,
            'violation': 0.0,
            'prev_hv': 0.0
        }
        self.nsga2_metrics = {
            'hv': 0.0,
            'diversity': 0.0,
            'violation': 0.0,
            'prev_hv': 0.0
        }
        self.spea2_hv_history = []
        self.nsga2_hv_history = []
        
        # 重置Episode收敛标志
        self.episode_converged = False

        # 重置增量Pareto维护器（允许episode间出现相同解，不再跨episode累积）
        if hasattr(self, 'pareto_maintainer') and self.pareto_maintainer is not None:
            self.pareto_maintainer.current_pareto = []
            self.pareto_maintainer.pareto_objectives = None
            self.pareto_maintainer.last_update_generation = 0

        # 使用初始动作运行一代MOEA作为初始化
        # 支持12维和6维动作空间
        default_initial_action = getattr(
            self,
            '_adaptive_initial_action',
            self.config.get('initial_action', [0.05, 0.1, 0.05, 10.0, 0.01, 0.8]),
        )
        initial_action = np.array(default_initial_action)
        
        # 如果存在上一episode经验，尝试从中构建初始动作
        if hasattr(self, 'last_experience') and self.last_experience:
            exp = self.last_experience
            spea2_p = exp.get('spea2_params', {})
            nsga2_p = exp.get('nsga2_params', {})
            try:
                # 构建12维初始动作（优先使用历史参数，否则使用默认）
                def pick(p, k, default):
                    return float(p.get(k, default)) if p else float(default)

                sa = [pick(spea2_p, 'epsilon_dom', 0.05), pick(spea2_p, 'accept_threshold', 0.1),
                      pick(spea2_p, 'delta_div', 0.05), pick(spea2_p, 'beta_penalty', 10.0),
                      pick(spea2_p, 'pm', 0.01), pick(spea2_p, 'pc', 0.8)]

                na = [pick(nsga2_p, 'epsilon_dom', 0.05), pick(nsga2_p, 'accept_threshold', 0.1),
                      pick(nsga2_p, 'delta_div', 0.05), pick(nsga2_p, 'beta_penalty', 10.0),
                      pick(nsga2_p, 'pm', 0.01), pick(nsga2_p, 'pc', 0.8)]

                if self.action_space.shape[0] == 17:
                    # 使用配置文件中定义的默认 MOEA/D 参数 [0.5, 0.5, 0.5, 0.1, 0.05]
                    moead_defaults = [0.5, 0.5, 0.5, 0.1, 0.05]
                    initial_action = np.array(sa + na + moead_defaults, dtype=np.float32)
                    print("[ENV]: 使用上一Episode经验初始化 SPEA2+NSGA2，并补齐 MOEA/D 默认参数")
                elif self.action_space.shape[0] == 8 and self.action_schema == 'search_control_v2':
                    controls = exp.get('search_controls', {})
                    defaults = np.asarray(default_initial_action, dtype=np.float32)
                    K = max(1, int(getattr(self.moea, 'K', 1)))
                    initial_action = np.array(
                        [
                            max(1.0, sa[4] * K),
                            sa[5],
                            max(1.0, na[4] * K),
                            na[5],
                            float(controls.get('spea2_budget_share', defaults[4])),
                            float(controls.get('elite_injection_ratio', defaults[5])),
                            float(controls.get('local_search_strength', defaults[6])),
                            float(controls.get('restart_ratio', defaults[7])),
                        ],
                        dtype=np.float32,
                    )
                    print("[ENV]: 使用上一Episode的新搜索控制经验初始化动作 (search_control_v2)")
                elif self.action_space.shape[0] == 8:
                    initial_action = np.array(
                        [
                            (sa[0] + na[0]) / 2.0,
                            (sa[1] + na[1]) / 2.0,
                            (sa[2] + na[2]) / 2.0,
                            (sa[3] + na[3]) / 2.0,
                            sa[4],
                            sa[5],
                            na[4],
                            na[5],
                        ],
                        dtype=np.float32,
                    )
                    print("[ENV]: 使用上一Episode经验初始化半共享动作 (8维)")
                elif self.action_space.shape[0] == 6:
                    initial_action = np.array([(a + b) / 2.0 for a, b in zip(sa, na)], dtype=np.float32)
                    print("[ENV]: 使用上一Episode经验初始化共享动作 (6维)")
                else:
                    initial_action = np.array(sa + na, dtype=np.float32)
                    print("[ENV]: 使用上一Episode经验初始化动作 (12维)")
            except Exception as e:
                print(f"[ENV WARNING] 热启动动作构建失败: {e}，回退到默认初始动作")
                pass

        # 安全维度检查和补齐
        if len(initial_action) != self.action_space.shape[0]:
            print(f"[ENV WARNING] 警告: 热启动动作维度 ({len(initial_action)}) 与当前动作空间 ({self.action_space.shape[0]}) 不匹配！")
            
            if len(initial_action) < self.action_space.shape[0]:
                print("[ENV INFO] 正在自动扩展动作维度以匹配动作空间...")
                padded_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
                padded_action[:len(initial_action)] = initial_action
                initial_action = padded_action
            else:
                initial_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

        initial_action = np.clip(
            np.asarray(initial_action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )

        print("[ENV]: Episode开始，运行一代MOEA进行初始化")

        # 只运行一代MOEA，而不是一个完整的SAC步
        self._update_moea_parameters(initial_action)

        # 【关键修复】检查热启动是否成功，如果成功则跳过启发式初始化
        # 热启动设置的是 spea2.population 和 nsga2.population，需要检查这两个
        has_spea2_pop = hasattr(self.moea, 'spea2') and hasattr(self.moea.spea2, 'population') and self.moea.spea2.population
        has_nsga2_pop = hasattr(self.moea, 'nsga2') and hasattr(self.moea.nsga2, 'population') and self.moea.nsga2.population
        
        warm_start_success = has_spea2_pop and has_nsga2_pop
        
        if warm_start_success:
            print("[ENV]: 热启动成功，跳过启发式初始化（保留精英解）")
            # 将 spea2/nsga2 的种群复制到 current_population
            self.moea.current_population = list(self.moea.spea2.population)
            # 仅运行1代MOEA进化改进
            try:
                if hasattr(self.moea, 'nsga2') and self.moea.nsga2 is not None:
                    self.moea.current_population = self.moea.nsga2.run_one_generation(
                        self.moea.current_population, generation=0)
                    print("[ENV]: 初始化进行了1代NSGA2进化改进")
            except Exception as e:
                print(f"[ENV]: 初始化进化失败（非致命）: {e}")
        else:
            # 热启动失败，执行启发式初始化
            print("[ENV]: 热启动失败，执行启发式初始化")
            self._initialize_moea_population()

        # 贪婪/MILP档案是所有方法共享的固定基线，已由MOEA reset构造。
        # SAC只因后续搜索超越该基线而获得档案增量奖励。
        try:
            archive_best = None
            for entry in getattr(self.moea, 'cumulative_elite_pool', []) or []:
                gene = self.moea._get_elite_entry_gene(entry)
                if gene is None:
                    continue
                metrics = self.moea._gene_metrics_fast(gene)
                if not metrics.get('feasible', False):
                    continue
                value = float(metrics['residual_ratio'])
                archive_best = value if archive_best is None else min(archive_best, value)
            if archive_best is not None:
                self.global_best_feasible_ser = min(float(self.global_best_feasible_ser), archive_best)
                self.global_best_sensitivity = min(float(self.global_best_sensitivity), archive_best)
                self.global_best_residual_ratio = min(float(self.global_best_residual_ratio), archive_best)
                self.prev_step_best_feasible_ser = archive_best
                print(f"[ARCHIVE] reward baseline synced: best feasible SER={archive_best:.6f}")
        except Exception as exc:
            print(f"[ARCHIVE] reward baseline sync failed: {exc}")

        # 提取初始状态
        initial_state = self._extract_initial_state()

        # ============ 跨Episode学习：将专家上下文融入状态 ============
        # 必须始终拼接专家上下文，保证状态维度始终等于 observation_space
        initial_state = self.get_state_with_expert_context(initial_state)

        self.current_state = initial_state

        info = {
            'episode_step': 0,
            'episode_started': True,
            'initial_action': initial_action.tolist(),
            'initial_population_size': len(self.moea.current_population) if hasattr(self.moea, 'current_population') and self.moea.current_population else 0
        }

        return self.current_state, info

    def set_sac_training_context(self, update_step=0, action_mode="stochastic_sample"):
        """向MOEA传递策略训练阶段，用于控制强档案的解锁时机。"""
        if hasattr(self.moea, "set_sac_training_context"):
            self.moea.set_sac_training_context(
                update_step=update_step,
                action_mode=action_mode,
            )
    
    def _initialize_moea_population(self):
        """初始化MOEA种群（使用优化后的启发式初始化 + 约束感知评估）"""

        # 【关键修复】：如果热启动已完成（设置了 warm_start_completed 标志），
        # 则跳过启发式初始化，保留热启动种群
        if getattr(self.moea, 'warm_start_completed', False):
            print(f"[ENV]: 检测到热启动已完成，保留精英种群 "
                  f"(种群大小={len(self.moea.current_population) if self.moea.current_population else 0})")
            if self.moea.current_population:
                self.moea.current_generation = 1
            return

        try:
            from utils.heuristic_init import HeuristicInitializer
            from algorithms.core_data import Individual
            
            # 获取K值（基因长度）
            K = getattr(self.moea, 'K', 30)  # 默认30
            pop_size = self.config['moea']['pop_size']
            
            # 获取模型信息用于约束感知初始化
            model = getattr(self.moea, 'model', None)
            
            # 构建配置字典传递给启发式初始化器
            init_config = {
                'delta_area_max': self.delta_area_max,
                'delta_power_max': self.delta_power_max,
                'delta_sens_min': self.delta_sens_min,
                'greedy_prefix_seed_count': self.config.get('moea', {}).get('greedy_prefix_seed_count', 96),
                'base_area': self.area_orig,
                'base_power': self.power_orig,
                'base_slack': self.slack_orig,
            }
            
            # 如果模型存在，尝试获取寄存器属性
            if model is not None:
                init_config['reg_area'] = getattr(model, 'reg_area', None)
                init_config['reg_power'] = getattr(model, 'reg_power', None)
            
            # 步骤1：使用优化后的启发式初始化生成高质量初始解
            initializer = HeuristicInitializer(
                K=K,
                error_rates=getattr(self.moea, 'error_rates', np.zeros(K)),
                slack=getattr(self.moea, 'slack', np.ones(K) * 0.1),
                is_critical=getattr(self.moea, 'is_critical', np.zeros(K)),
                model=model,
                config=init_config
            )
            
            # 生成启发式初始种群（数量由配置控制）
            heuristic_seeds = self.config.get('moea', {}).get('initial_heuristic_seeds', min(10, pop_size))
            heuristic_pop = initializer.generate_heuristic_population(
                pop_size=min(int(heuristic_seeds), pop_size),
                seed=None
            )
            
            print(f"[ENV]: 启发式初始化生成了 {len(heuristic_pop)} 个初始解")
            
            # 步骤2：使用真实模型评估启发式初始种群
            heuristic_genes = [ind.gene for ind in heuristic_pop]
            if model is not None and hasattr(model, 'evaluate_population'):
                print(f"[ENV]: 使用真实PPA模型评估 {len(heuristic_genes)} 个启发式解...")
                heuristic_objectives = model.evaluate_population(heuristic_genes)
                
                # 分析初始解的可行性
                feasible_count = 0
                for i, obj in enumerate(heuristic_objectives):
                    if len(obj) >= 3:
                        # obj[2] = -slack，时序可行需要 slack >= 0，即 -slack <= 0
                        neg_slack = obj[2]
                        if neg_slack <= 0:  # -slack <= 0 等价于 slack >= 0
                            feasible_count += 1
                print(f"[ENV]: PPA评估 - {feasible_count}/{len(heuristic_objectives)} 个启发式解时序可行")
            else:
                # 简化的后备评估
                heuristic_objectives = []
                for gene in heuristic_genes:
                    area = np.sum(gene) * 10
                    power = np.sum(gene) * 0.5
                    slack = 0.1 - np.sum(gene * getattr(self.moea, 'is_critical', np.zeros(K))) * 0.01
                    sens = np.sum(getattr(self.moea, 'error_rates', np.zeros(K)) * (1 - gene))
                    heuristic_objectives.append([area, power, -slack, sens])
                heuristic_objectives = np.array(heuristic_objectives)
            
            # 步骤3：如果需要更多种群，使用约束感知的随机解补充
            if len(heuristic_pop) < pop_size:
                random_count = pop_size - len(heuristic_pop)
                print(f"[ENV]: 补充 {random_count} 个约束感知的随机解...")
                
                # 使用启发式初始化器生成约束感知随机解
                random_genes = []
                for _ in range(random_count):
                    gene = initializer._generate_constrained_random()
                    if gene is not None:
                        random_genes.append(gene)
                
                # 如果启发式方法失败，使用简单的随机生成
                while len(random_genes) < random_count:
                    gene = np.random.randint(0, 2, K)
                    # 确保至少有1个被加固
                    if np.sum(gene) == 0:
                        gene[np.random.randint(0, K)] = 1
                    random_genes.append(gene)
                
                if model is not None and hasattr(model, 'evaluate_population'):
                    random_objectives = model.evaluate_population(random_genes)
                else:
                    random_objectives = []
                    for gene in random_genes:
                        area = np.sum(gene) * 10
                        power = np.sum(gene) * 0.5
                        slack = 0.1 - np.sum(gene * getattr(self.moea, 'is_critical', np.zeros(K))) * 0.01
                        sens = np.sum(getattr(self.moea, 'error_rates', np.zeros(K)) * (1 - gene))
                        random_objectives.append([area, power, -slack, sens])
                    random_objectives = np.array(random_objectives)
                
                # 合并启发式和随机解
                all_genes = np.vstack([heuristic_genes, random_genes])
                all_objectives = np.vstack([heuristic_objectives, random_objectives])
            else:
                all_genes = heuristic_genes  # list of np.ndarray
                all_objectives = heuristic_objectives  # np.ndarray
            
            # 步骤4：创建初始种群对象
            self.moea.current_population = [
                Individual(gene, objectives) 
                for gene, objectives in zip(all_genes, all_objectives)
            ]
            
            # 步骤5：运行1代MOEA进化（可选，加快收敛）
            use_initial_evolution = self.config.get('moea', {}).get('use_initial_evolution', True)
            if use_initial_evolution and len(self.moea.current_population) > 0:
                try:
                    if hasattr(self.moea, 'nsga2') and self.moea.nsga2 is not None:
                        self.moea.current_population = self.moea.nsga2.run_one_generation(
                            self.moea.current_population, generation=0)
                        print("[ENV]: 初始化进行了1代NSGA2进化改进")
                except Exception as e:
                    print(f"[ENV]: 初始化进化失败（非致命）: {e}")
            
            # 设置当前代数为1
            self.moea.current_generation = 1
            
            heuristic_count = len(heuristic_pop)
            print(f"[ENV]: 优化后的启发式初始化完成 - 启发式解={heuristic_count}，总种群={len(self.moea.current_population)}，第1代")

        except Exception as e:
            print(f"ERROR[ENV]: 启发式初始化失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到随机初始化
            try:
                K = getattr(self.moea, 'K', 30)
                pop_size = self.config['moea']['pop_size']
                initial_genes = [np.random.randint(0, 2, K) for _ in range(pop_size)]
                
                if hasattr(self.moea, 'model') and hasattr(self.moea.model, 'evaluate_population'):
                    initial_objectives = self.moea.model.evaluate_population(initial_genes)
                else:
                    initial_objectives = np.ones((len(initial_genes), 4)) * 1e6
                
                from algorithms.core_data import Individual
                self.moea.current_population = [
                    Individual(gene, objectives) 
                    for gene, objectives in zip(initial_genes, initial_objectives)
                ]
                self.moea.current_generation = 1
                print(f"[ENV]: 已回退到随机初始化")
            except Exception as e2:
                print(f"ERROR[ENV]: 回退初始化也失败: {e2}")
                self.moea.current_population = []
                self.moea.current_generation = 0

    def _extract_initial_state(self):
        """从初始化种群中提取状态"""
        if not hasattr(self.moea, 'current_population') or not self.moea.current_population:
            # 如果种群为空，返回零状态
            return np.zeros(self.state_dim, dtype=np.float32)

        # 使用与_extract_state相同的逻辑，但使用None作为动作
        # 因为我们只是初始化，还没有SAC动作
        # 根据动作空间维度创建零动作
        action_dim = self.action_space.shape[0]
        return self._extract_state(self.moea.current_population, np.zeros(action_dim))

    def _init_adaptive_parameters(self):
        """
        根据电路规模K自适应初始化参数范围

        不同规模的电路需要不同的参数设置：
        - 小规模 (K < 50): 快速收敛，探索激进
        - 中等规模 (50 < K < 500): 平衡探索和收敛
        - 大规模 (500 < K < 2000): 保守探索，需要更多代数
        - 超大规模 (K > 2000): 极度保守，高精度
        """
        # 获取电路规模K
        K = getattr(self.moea, 'K', 100)

        # 获取自适应参数配置
        adaptive_cfg = self.config.get('adaptive_parameters', {})
        if not adaptive_cfg.get('enable', True):
            # 如果禁用自适应，使用配置文件中的默认值
            return

        # 根据K选择参数范围
        if K <= 50:
            scale = 'small_scale'
        elif K <= 500:
            scale = 'medium_scale'
        elif K <= 2000:
            scale = 'large_scale'
        else:
            scale = 'very_large_scale'

        params = adaptive_cfg.get(scale, {})

        if not params:
            print(f"[WARN] 未找到规模 {K} 对应的自适应参数，使用默认值")
            return

        # SPEA2参数
        epsilon_spea2 = params.get('epsilon', [0.01, 0.1])
        tau_spea2 = params.get('accept_threshold', [0.05, 0.3])
        delta_spea2 = params.get('delta_div', [0.05, 0.3])
        beta_spea2 = params.get('beta_penalty', [5.0, 30.0])
        pm_spea2 = params.get('pm', [0.05, 0.3])
        pc_spea2 = params.get('pc', [0.6, 0.9])

        # NSGA2参数（使用相同范围）
        epsilon_nsga2 = params.get('epsilon', [0.01, 0.1])
        tau_nsga2 = params.get('accept_threshold', [0.05, 0.3])
        delta_nsga2 = params.get('delta_div', [0.05, 0.3])
        beta_nsga2 = params.get('beta_penalty', [5.0, 30.0])
        pm_nsga2 = params.get('pm', [0.05, 0.3])
        pc_nsga2 = params.get('pc', [0.6, 0.9])

        action_dim = len(self.config.get('action_low', []))
        if action_dim == 8 and self.action_schema == 'search_control_v2':
            mutation_flips = params.get('mutation_flips')
            if not mutation_flips:
                pm_bounds = params.get('pm', [1.0 / max(K, 1), min(0.1, 16.0 / max(K, 1))])
                mutation_flips = [
                    max(1.0, float(pm_bounds[0]) * K),
                    max(2.0, float(pm_bounds[1]) * K),
                ]
            mutation_flips = [
                max(1.0, float(mutation_flips[0])),
                max(1.0, min(float(K), float(mutation_flips[1]))),
            ]
            if mutation_flips[1] < mutation_flips[0]:
                mutation_flips[1] = mutation_flips[0]

            control_bounds = self.config.get('search_control_bounds', {})
            budget_share = control_bounds.get('spea2_budget_share', [0.2, 0.8])
            elite_ratio = control_bounds.get('elite_injection_ratio', [0.0, 0.25])
            local_strength = control_bounds.get('local_search_strength', [0.0, 1.0])
            restart_ratio = control_bounds.get('restart_ratio', [0.0, 0.2])
            low = [
                mutation_flips[0], pc_spea2[0], mutation_flips[0], pc_nsga2[0],
                budget_share[0], elite_ratio[0], local_strength[0], restart_ratio[0],
            ]
            high = [
                mutation_flips[1], pc_spea2[1], mutation_flips[1], pc_nsga2[1],
                budget_share[1], elite_ratio[1], local_strength[1], restart_ratio[1],
            ]
            self._fixed_shared_moea_params = {
                'epsilon_dom': float(np.mean(epsilon_spea2)),
                'accept_threshold': float(np.mean(tau_spea2)),
                'delta_div': float(np.mean(delta_spea2)),
                'beta_penalty': float(np.mean(beta_spea2)),
            }
            params = dict(params)
            params['pm'] = [mutation_flips[0] / max(K, 1), mutation_flips[1] / max(K, 1)]
        elif action_dim == 8:
            # 半共享动作：[共享 eps/tau/delta/beta, SPEA2 pm/pc, NSGA-II pm/pc]
            low = [
                epsilon_spea2[0], tau_spea2[0], delta_spea2[0], beta_spea2[0],
                pm_spea2[0], pc_spea2[0], pm_nsga2[0], pc_nsga2[0],
            ]
            high = [
                epsilon_spea2[1], tau_spea2[1], delta_spea2[1], beta_spea2[1],
                pm_spea2[1], pc_spea2[1], pm_nsga2[1], pc_nsga2[1],
            ]
        elif action_dim == 6:
            low = [epsilon_spea2[0], tau_spea2[0], delta_spea2[0], beta_spea2[0], pm_spea2[0], pc_spea2[0]]
            high = [epsilon_spea2[1], tau_spea2[1], delta_spea2[1], beta_spea2[1], pm_spea2[1], pc_spea2[1]]
        elif action_dim == 12:
            low = [
                epsilon_spea2[0], tau_spea2[0], delta_spea2[0], beta_spea2[0], pm_spea2[0], pc_spea2[0],
                epsilon_nsga2[0], tau_nsga2[0], delta_nsga2[0], beta_nsga2[0], pm_nsga2[0], pc_nsga2[0],
            ]
            high = [
                epsilon_spea2[1], tau_spea2[1], delta_spea2[1], beta_spea2[1], pm_spea2[1], pc_spea2[1],
                epsilon_nsga2[1], tau_nsga2[1], delta_nsga2[1], beta_nsga2[1], pm_nsga2[1], pc_nsga2[1],
            ]
        else:
            # 兼容17维旧配置：前12维为两个算法参数，后5维为MOEA/D参数。
            low = [
                epsilon_spea2[0], tau_spea2[0], delta_spea2[0], beta_spea2[0], pm_spea2[0], pc_spea2[0],
                epsilon_nsga2[0], tau_nsga2[0], delta_nsga2[0], beta_nsga2[0], pm_nsga2[0], pc_nsga2[0],
                0.0, 0.0, 0.0, 0.01, 0.001,
            ]
            high = [
                epsilon_spea2[1], tau_spea2[1], delta_spea2[1], beta_spea2[1], pm_spea2[1], pc_spea2[1],
                epsilon_nsga2[1], tau_nsga2[1], delta_nsga2[1], beta_nsga2[1], pm_nsga2[1], pc_nsga2[1],
                1.0, 1.0, 1.0, 0.5, 0.2,
            ]

        # Experiment-level action bounds override the K-scale defaults. This
        # makes the narrow-range ablation faithful to the measured late basin
        # and also permits per-dimension bounds for SPEA2 and NSGA-II.
        operator_control = getattr(
            self,
            'operator_control',
            self.config.get('operator_control', {}) or {},
        )
        control_low = operator_control.get('action_low')
        control_high = operator_control.get('action_high')
        if control_low is not None or control_high is not None:
            if control_low is None or control_high is None:
                raise ValueError(
                    'operator_control.action_low and action_high must be provided together'
                )
            if len(control_low) != 8 or len(control_high) != 8:
                raise ValueError('operator_control action bounds must be 8-dimensional')
            low = [float(x) for x in control_low]
            high = [float(x) for x in control_high]
            if any(lo >= hi for lo, hi in zip(low, high)):
                raise ValueError('operator_control action_low must be strictly below action_high')

        fixed_action = operator_control.get('fixed_action')
        operator_control_mode = getattr(
            self,
            'operator_control_mode',
            str(operator_control.get('mode', 'adaptive')).lower(),
        )
        if operator_control_mode == 'fixed':
            if fixed_action is None or len(fixed_action) != 8:
                raise ValueError('fixed mode requires an 8-dimensional fixed_action')
            self._fixed_execution_action = np.asarray(fixed_action, dtype=np.float32)
            if np.any(self._fixed_execution_action < np.asarray(low)) or np.any(
                self._fixed_execution_action > np.asarray(high)
            ):
                raise ValueError('fixed_action must lie inside operator_control action bounds')

        self._adaptive_action_low = np.asarray(low, dtype=np.float32)
        self._adaptive_action_high = np.asarray(high, dtype=np.float32)
        self._adaptive_initial_action = (self._adaptive_action_low + self._adaptive_action_high) / 2.0

        # 更新MOEA算法的参数边界
        self._update_moea_param_bounds(params)

        print(f"[自适应参数] 电路规模K={K}, 使用{scale}参数范围")
        print(f"  ε范围: [{epsilon_spea2[0]:.4f}, {epsilon_spea2[1]:.4f}]")
        print(f"  β范围: [{beta_spea2[0]:.1f}, {beta_spea2[1]:.1f}]")
        if action_dim == 8 and self.action_schema == 'search_control_v2':
            print(f"  期望变异位数范围: [{low[0]:.1f}, {high[0]:.1f}]")
            print(f"  精英注入比例范围: [{low[5]:.2f}, {high[5]:.2f}]")
        else:
            print(f"  pm范围: [{pm_spea2[0]:.4f}, {pm_spea2[1]:.4f}]")

    def _update_moea_param_bounds(self, params):
        """更新MOEA算法的参数边界"""
        algorithms = getattr(self.moea, 'algorithms', {})

        for algo_name, algo in algorithms.items():
            if algo is None or not hasattr(algo, 'param_bounds'):
                continue

            algo.param_bounds['epsilon_dom'] = tuple(params.get('epsilon', [0.01, 0.1]))
            algo.param_bounds['accept_threshold'] = tuple(params.get('accept_threshold', [0.05, 0.3]))
            algo.param_bounds['delta_div'] = tuple(params.get('delta_div', [0.05, 0.3]))
            algo.param_bounds['beta_penalty'] = tuple(params.get('beta_penalty', [5.0, 30.0]))
            algo.param_bounds['pm'] = tuple(params.get('pm', [0.05, 0.3]))
            algo.param_bounds['pc'] = tuple(params.get('pc', [0.6, 0.9]))

    def _init_original_values(self):
        """从模拟器获取原始PPA值"""
        try:
            # 尝试从模拟器获取原始值
            if hasattr(self.moea, 'simulator'):
                simulator = self.moea.simulator
                # 这些值现在应该从CSV文件中读取
                self.area_orig = getattr(simulator, 'Area_orig', 100000.0)
                self.power_orig = getattr(simulator, 'Power_orig', 50.0)
                self.slack_orig = getattr(simulator, 'WNS_orig', 0.05)
                self.sens_orig = getattr(simulator, 'SER_orig', 1e-5)

                # 记录原始值
                print(f"从模拟器获取的原始PPA值:")
                print(f"  原始面积: {self.area_orig}")
                print(f"  原始功耗: {self.power_orig}")
                print(f"  原始最差slack: {self.slack_orig}")
                print(f"  原始敏感度: {self.sens_orig}")
            elif hasattr(self.moea, 'model'):
                # 如果使用PPAModel
                model = self.moea.model
                self.area_orig = getattr(model, 'base_area', 100000.0)
                self.power_orig = getattr(model, 'base_power', 50.0)
                self.slack_orig = getattr(model, 'base_slack', 0.05)
                self.sens_orig = getattr(model, 'base_sens', 1e-5)
            else:
                # 设置默认值
                self.area_orig = 100000.0
                self.power_orig = 50.0
                self.slack_orig = 0.05
                self.sens_orig = 1e-5

        except Exception as e:
            print(f"警告：无法获取原始PPA值，使用默认值: {e}")
            self.area_orig = 100000.0
            self.power_orig = 50.0
            self.slack_orig = 0.05
            self.sens_orig = 1e-5

    def _decode_action_params(self, action: np.ndarray) -> Dict[str, Any]:
        """将策略动作解码为可记录、可直接下发的搜索控制参数。"""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if len(action) == 8 and self.action_schema == 'search_control_v2':
            (
                spea2_flips, spea2_pc, nsga2_flips, nsga2_pc,
                spea2_budget_share, elite_ratio, local_strength, restart_ratio,
            ) = action
            K = max(1, int(getattr(self.moea, 'K', 1)))
            fixed = getattr(self, '_fixed_shared_moea_params', {
                'epsilon_dom': 0.01,
                'accept_threshold': 0.05,
                'delta_div': 0.05,
                'beta_penalty': 30.0,
            })

            def algorithm_params(expected_flips, pc):
                return {
                    'epsilon': float(fixed['epsilon_dom']),
                    'tau': float(fixed['accept_threshold']),
                    'delta': float(fixed['delta_div']),
                    'beta': float(fixed['beta_penalty']),
                    'mutation_flips': float(expected_flips),
                    'pm': float(np.clip(expected_flips / K, 1.0 / K, 1.0)),
                    'pc': float(pc),
                }

            return {
                'schema': self.action_schema,
                'spea2': algorithm_params(spea2_flips, spea2_pc),
                'nsga2': algorithm_params(nsga2_flips, nsga2_pc),
                'controls': {
                    'spea2_budget_share': float(spea2_budget_share),
                    'elite_injection_ratio': float(elite_ratio),
                    'local_search_strength': float(local_strength),
                    'restart_ratio': float(restart_ratio),
                },
            }

        if len(action) == 12:
            return {
                'spea2': {
                    'epsilon': float(action[0]), 'tau': float(action[1]),
                    'delta': float(action[2]), 'beta': float(action[3]),
                    'pm': float(action[4]), 'pc': float(action[5]),
                },
                'nsga2': {
                    'epsilon': float(action[6]), 'tau': float(action[7]),
                    'delta': float(action[8]), 'beta': float(action[9]),
                    'pm': float(action[10]), 'pc': float(action[11]),
                },
            }
        if len(action) == 8:
            return {
                'spea2': {
                    'epsilon': float(action[0]), 'tau': float(action[1]),
                    'delta': float(action[2]), 'beta': float(action[3]),
                    'pm': float(action[4]), 'pc': float(action[5]),
                },
                'nsga2': {
                    'epsilon': float(action[0]), 'tau': float(action[1]),
                    'delta': float(action[2]), 'beta': float(action[3]),
                    'pm': float(action[6]), 'pc': float(action[7]),
                },
            }
        return {
            'epsilon': float(action[0]), 'tau': float(action[1]),
            'delta': float(action[2]), 'beta': float(action[3]),
            'pm': float(action[4]), 'pc': float(action[5]),
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步环境交互
        每个SAC步运行一次MOEA（不重置，继续从当前种群运行）
        """
        # Fixed ablation keeps SAC training and logging alive, but removes its
        # ability to change the executed MOEA operators.
        if self.operator_control_mode == 'fixed' and self._fixed_execution_action is not None:
            action = self._fixed_execution_action.copy()

        # 确保动作在合理范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1. 解码并下发动作。记录与执行共用同一个解码结果，避免日志和实际参数不一致。
        action_params = self._decode_action_params(action)
        self._update_moea_parameters(action)

        # 打印动作参数
        sac_step_number = self.current_episode_step + 1
        current_gen = getattr(self.moea, 'current_generation', 0)
        generations_per_step = self.generations_per_sac_step
        print(f"[ENV]: SAC步 {sac_step_number}, 运行MOEA {self.generations_per_sac_step} 代"
              f" | fixed_budget={self.fixed_budget_per_sac_step}, allow_early_stop={self.allow_early_stop}, strict_early_stop={self.strict_early_stop}")
        if action_params.get('schema') == 'search_control_v2':
            controls = action_params['controls']
            print(
                f"[ENV]: SPEA2 lambda={action_params['spea2']['mutation_flips']:.1f}, "
                f"pm={action_params['spea2']['pm']:.6f}, pc={action_params['spea2']['pc']:.3f}; "
                f"NSGA2 lambda={action_params['nsga2']['mutation_flips']:.1f}, "
                f"pm={action_params['nsga2']['pm']:.6f}, pc={action_params['nsga2']['pc']:.3f}"
            )
            print(
                f"[ENV]: budget(SPEA2)={controls['spea2_budget_share']:.2f}, "
                f"elite={controls['elite_injection_ratio']:.2f}, "
                f"local={controls['local_search_strength']:.2f}, restart={controls['restart_ratio']:.2f}"
            )
        elif isinstance(action_params, dict) and 'spea2' in action_params and 'nsga2' in action_params:
            print(f"[ENV]: SPEA2 params: eps={action_params['spea2']['epsilon']:.4f}, tau={action_params['spea2']['tau']:.4f}, "
                  f"delta={action_params['spea2']['delta']:.4f}, beta={action_params['spea2']['beta']:.2f}, "
                  f"pm={action_params['spea2']['pm']:.4f}, pc={action_params['spea2']['pc']:.4f}")
            print(f"[ENV]: NSGA2 params: eps={action_params['nsga2']['epsilon']:.4f}, tau={action_params['nsga2']['tau']:.4f}, "
                  f"delta={action_params['nsga2']['delta']:.4f}, beta={action_params['nsga2']['beta']:.2f}, "
                  f"pm={action_params['nsga2']['pm']:.4f}, pc={action_params['nsga2']['pc']:.4f}")
        else:
            print(f"[ENV]: action params: eps={action_params['epsilon']:.4f}, tau={action_params['tau']:.4f}, "
                  f"delta={action_params['delta']:.4f}, beta={action_params['beta']:.2f}, "
                  f"pm={action_params['pm']:.4f}, pc={action_params['pc']:.4f}")

        print(f"[ENV]: SAC步 {sac_step_number}，从第{current_gen+1}代开始运行{generations_per_step}代MOEA")

        # 诊断快照（运行MOEA前）: 输出收敛历史的长度与尾部、以及当前Pareto前沿大小
        try:
            before_hv = list(self.moea.convergence_history.get('hv', [])) if hasattr(self.moea, 'convergence_history') else []
        except Exception:
            before_hv = []
        before_hv_tail = before_hv[-5:] if before_hv else []
        try:
            before_pareto = self.moea._extract_pareto_front(self.moea.current_population) if hasattr(self.moea, '_extract_pareto_front') else []
            before_pareto_size = len(before_pareto)
        except Exception:
            before_pareto_size = -1
        print(f"[ENV-MOEA-SNAP BEFORE] gen={current_gen}, hv_len={len(before_hv)}, hv_tail={before_hv_tail}, pareto_size={before_pareto_size}")

        # 3. 运行MOEA（不重置，使用当前种群继续）
        print(f"[ENV]: SAC步 {self.current_episode_step + 1}")
        archive_hv_before = 0.0
        if hasattr(self.moea, 'get_feasible_archive_hv'):
            try:
                archive_hv_before = float(self.moea.get_feasible_archive_hv())
            except Exception:
                archive_hv_before = 0.0

        # 记录动作执行前的当前种群质量。累计档案给出严格的全局质量，
        # 当前种群增量则为每个SAC动作提供更稠密、可归因的学习信号。
        population_before = list(getattr(self.moea, 'current_population', []) or [])
        population_hv_before = 0.0
        population_ser_before = None
        try:
            if hasattr(self.moea, '_calculate_scoring_hv_2d'):
                population_hv_before = float(self.moea._calculate_scoring_hv_2d(population_before))
            population_ser_before = self._get_best_ppa_feasible_ser(population_before).get('best_residual_ratio')
            if population_ser_before is not None:
                population_ser_before = float(population_ser_before)
        except Exception:
            population_hv_before = 0.0
            population_ser_before = None

        # run_with_convergence会先执行受控重启、档案精英注入和repair-fill，
        # 这些准备操作不应直接计入SAC动作归因。优先使用包装器提供的
        # 环境选择前后快照，避免把确定性质量恢复算作策略贡献。
        population_hv_raw_before = population_hv_before
        population_ser_raw_before = population_ser_before

        population, convergence_info = self._run_moea_continue(action, generations_per_step)

        credit_before = list(getattr(self.moea, 'last_pre_evolution_population', []) or [])
        credit_after = list(getattr(self.moea, 'last_evolution_population', []) or [])
        credit_source = 'moea_action_window'
        if credit_before and credit_after:
            try:
                population_hv_before = float(self.moea._calculate_scoring_hv_2d(credit_before))
                population_ser_before = self._get_best_ppa_feasible_ser(credit_before).get(
                    'best_residual_ratio'
                )
                if population_ser_before is not None:
                    population_ser_before = float(population_ser_before)
            except Exception:
                population_hv_before = population_hv_raw_before
                population_ser_before = population_ser_raw_before
                credit_source = 'raw_population_fallback'
        else:
            credit_source = 'raw_population_fallback'
        population_after_for_credit = credit_after if credit_after else population

        population_hv_after = 0.0
        population_ser_after = None
        try:
            if hasattr(self.moea, '_calculate_scoring_hv_2d'):
                population_hv_after = float(
                    self.moea._calculate_scoring_hv_2d(population_after_for_credit)
                )
            population_ser_after = self._get_best_ppa_feasible_ser(
                population_after_for_credit
            ).get('best_residual_ratio')
            if population_ser_after is not None:
                population_ser_after = float(population_ser_after)
        except Exception:
            population_hv_after = population_hv_before
            population_ser_after = None

        population_hv_gain = population_hv_after - population_hv_before
        population_ser_gain = 0.0
        if population_ser_before is not None and population_ser_after is not None:
            population_ser_gain = (
                population_ser_before - population_ser_after
            ) / max(population_ser_before, 1e-12)
        convergence_info.update({
            'population_hv_before': population_hv_before,
            'population_hv_after': population_hv_after,
            'population_hv_gain': population_hv_gain,
            'population_ser_before': population_ser_before,
            'population_ser_after': population_ser_after,
            'population_ser_gain': float(np.clip(population_ser_gain, -1.0, 1.0)),
            # search_control_v2的首步注入、局部搜索和重启同样由动作控制，
            # 动作前快照已经包含reset阶段的固定热启动，因此首步增量可以归因给SAC。
            'population_credit_valid': bool(credit_before and credit_after),
            'population_hv_raw_before': population_hv_raw_before,
            'population_ser_raw_before': population_ser_raw_before,
            'population_credit_source': credit_source,
            'preparation_quality_excluded': credit_source == 'moea_action_window',
            'archive_policy_ready': bool(getattr(self.moea, 'sac_policy_ready', True)),
            'sac_update_step_context': int(getattr(self.moea, 'sac_update_step', 0)),
            'sac_action_mode_context': str(getattr(self.moea, 'sac_action_mode', 'unknown')),
        })

        # 奖励档案只接收本次MOEA环境选择得到的解。最终精英注入仍会在本步
        # 结束后写入持久档案，但不会被错误归因给当前SAC动作。
        reward_population = population_after_for_credit
        archive_hv_after = archive_hv_before
        try:
            current_front = self.moea._extract_ppa_feasible_pareto(reward_population)
            if hasattr(self.moea, '_append_to_cumulative_elite_pool'):
                source_phase = (
                    "sac_policy_step"
                    if bool(getattr(self.moea, 'sac_policy_ready', False))
                    else "sac_exploration_step"
                )
                self.moea._append_to_cumulative_elite_pool(
                    current_front,
                    source=f"{source_phase}:{self.current_episode_index}:{self.current_episode_step + 1}",
                )
            if hasattr(self.moea, 'get_feasible_archive_hv'):
                archive_hv_after = float(self.moea.get_feasible_archive_hv())
        except Exception as exc:
            print(f"[ARCHIVE] feasible archive update failed: {exc}")
        convergence_info['archive_hv_before'] = archive_hv_before
        convergence_info['archive_hv_after'] = archive_hv_after
        convergence_info['archive_hv_gain'] = archive_hv_after - archive_hv_before

        # 诊断快照（运行MOEA后）: 输出收敛历史的长度与尾部、以及返回信息中的初始/最终HV
        try:
            after_hv = list(self.moea.convergence_history.get('hv', [])) if hasattr(self.moea, 'convergence_history') else []
        except Exception:
            after_hv = []
        after_hv_tail = after_hv[-5:] if after_hv else []
        try:
            after_pareto = self.moea._extract_pareto_front(self.moea.current_population) if hasattr(self.moea, '_extract_pareto_front') else []
            after_pareto_size = len(after_pareto)
        except Exception:
            after_pareto_size = -1
        init_hv = convergence_info.get('initial_hv', None)
        final_hv = convergence_info.get('final_hv', None)
        print(f"[ENV-MOEA-SNAP AFTER] hv_len={len(after_hv)}, hv_tail={after_hv_tail}, pareto_size={after_pareto_size}, initial_hv={init_hv}, final_hv={final_hv}")

        # 4. 检查是否收敛
        is_converged = bool(convergence_info.get('is_converged', False))

        if self.fixed_budget_per_sac_step and not self.allow_early_stop:
            # 固定预算模式：不把MOEA收敛映射为episode终止，仅用于日志/统计
            if is_converged and not self.episode_converged:
                self.episode_converged = True
                print(f"INFO[ENV]: MOEA在SAC步 {self.current_episode_step + 1} 内被判定为收敛，"
                      f"但因 fixed_budget_per_sac_step=True 不提前结束Episode")
            if is_converged:
                convergence_info['budget_ignored_convergence'] = True
        else:
            # MOEA收敛后可提前终止episode，但先跑完最少步数以积累经验
            min_steps = getattr(self, '_min_steps_before_convergence_stop', 2)
            if is_converged and not self.episode_converged and (self.current_episode_step + 1) >= min_steps:
                self.episode_converged = True
                print(f"INFO[ENV]: Episode在第{self.current_episode_step + 1}步SAC时首次收敛（已积累{min_steps}步经验），将结束当前Episode")
            elif is_converged and not self.episode_converged:
                print(f"INFO[ENV]: MOEA在第{self.current_episode_step + 1}步SAC收敛，仍需积累至{min_steps}步再结束Episode")
        
        if is_converged and population:
            # 提取Pareto前沿
            pareto_front = self._extract_pareto_front(population)

            if pareto_front:
                # ========== 时序可行性检查 (仅用于终端诊断打印，不再在此处保存CSV和绘图) ==========
                timing_violations = 0
                timing_violation_details = []
                for idx, ind in enumerate(pareto_front):
                    if ind.objectives is not None and len(ind.objectives) >= 3:
                        # 目标[2] 是 -WNS，所以 wns = -obj[2]
                        wns_value = -ind.objectives[2]
                        if wns_value < 0:
                            timing_violations += 1
                            timing_violation_details.append((idx, wns_value))
                
                if timing_violations > 0:
                    print(f"[WARNING] 内部收敛检测: Pareto前沿中发现 {timing_violations}/{len(pareto_front)} 个时序违例")
                else:
                    print(f"[OK] 内部收敛检测: Pareto前沿中所有 {len(pareto_front)} 个解都满足时序约束 (WNS >= 0)")

            # 记录收敛信息 (传递给外界)
            if hasattr(self.moea, 'convergence_history') and 'hv' in self.moea.convergence_history:
                current_hv = self.moea.convergence_history['hv'][-1] if self.moea.convergence_history['hv'] else 0.0
                convergence_info['final_hv'] = current_hv
        # 5. 计算奖励
        reward = self._calculate_reward_from_moea(reward_population, convergence_info)

        # 6. 提取新状态
        next_state = self._extract_state(population, action)

        # ============ 跨Episode学习：将专家上下文融入状态 ============
        # 必须始终拼接专家上下文，保证状态维度始终等于 observation_space
        next_state = self.get_state_with_expert_context(next_state)

        # 7. 更新计数器
        self.current_episode_step += 1

        # 8. 确定是否终止
        # 检查是否达到最大SAC步数
        max_sac_steps = self.config.get('sac', {}).get('max_steps_per_episode', 10)
        terminated_by_max_steps = (self.current_episode_step >= max_sac_steps)
        
        if self.fixed_budget_per_sac_step and not self.allow_early_stop:
            terminated_by_convergence = False
        else:
            terminated_by_convergence = self.episode_converged
        
        if self.fixed_budget_per_sac_step and self.strict_early_stop and terminated_by_convergence:
            terminated = True
        else:
            terminated = terminated_by_max_steps or terminated_by_convergence

        # 9. 检查截断条件
        truncated = False  # 这里不使用截断，只在达到最大步数或收敛时终止
        
        # 如果因收敛而终止，打印信息
        if terminated_by_convergence:
            print(f"INFO[ENV]: Episode因MOEA收敛而终止于第{self.current_episode_step}步SAC")

        # 10. 更新历史
        self.action_history.append(action.copy())
        self.state_history.append(next_state.copy())
        self.current_state = next_state

        # 11. 更新HV历史用于奖励计算
        final_hv = (
             self.moea.convergence_history['hv'][-1]
             if hasattr(self.moea, 'convergence_history')
             and self.moea.convergence_history.get('hv')
             else 0.0
        )
        if 'hv_history' not in self.__dict__:
            self.hv_history = []
        self.hv_history.append(final_hv)
        
        # 档案已在动作归因快照之后通过统一入口写入。这里只同步热启动经验，
        # 避免旧逻辑再次追加裸基因并用固定1000条均匀截断破坏预算覆盖。
        if hasattr(self, 'last_experience') and self.last_experience:
            self.last_experience['elite_genes'] = list(
                getattr(self.moea, 'cumulative_elite_pool', []) or []
            )

        # 12. 返回信息（包含动作参数）
        info = {
            'action': action,
            'action_params': action_params,
            'population_size': len(population) if population else 0,
            'hv': final_hv,
            'generations': convergence_info.get('final_generation', 0),
            'is_converged': convergence_info.get('is_converged', False),
            'episode_converged': self.episode_converged,  # Episode是否已收敛
            'terminated_by_convergence': terminated_by_convergence,  # 是否因收敛而终止
            'hv_improvement': convergence_info.get('hv_improvement', 0.0),
            'convergence_info': convergence_info,
            'episode_step': self.current_episode_step,
            'terminated': terminated,
            'truncated': truncated,
            'total_moea_generations': self.moea.current_generation if hasattr(self.moea, 'current_generation') else 0,
            'best_feasible_ser': getattr(self, 'last_reward_best_feasible_ser', None),
            'best_feasible_global_gain': getattr(self, 'last_reward_global_gain', 0.0),
            'best_feasible_step_gain': getattr(self, 'last_reward_step_gain', 0.0),
            'archive_hv': archive_hv_after,
            'archive_hv_gain': archive_hv_after - archive_hv_before,
            'population_hv_before': population_hv_before,
            'population_hv_after': population_hv_after,
            'population_hv_gain': population_hv_gain,
            'population_ser_before': population_ser_before,
            'population_ser_after': population_ser_after,
            'population_ser_gain': float(np.clip(population_ser_gain, -1.0, 1.0)),
            'population_hv_raw_before': convergence_info.get('population_hv_raw_before'),
            'population_ser_raw_before': convergence_info.get('population_ser_raw_before'),
            'population_credit_source': convergence_info.get('population_credit_source'),
            'preparation_quality_excluded': convergence_info.get('preparation_quality_excluded', False),
            'archive_policy_ready': convergence_info.get('archive_policy_ready', True),
            'sac_update_step_context': convergence_info.get('sac_update_step_context', 0),
            'sac_action_mode_context': convergence_info.get('sac_action_mode_context', 'unknown'),
            'population_hv_credit': convergence_info.get('population_hv_credit', 0.0),
            'population_ser_credit': convergence_info.get('population_ser_credit', 0.0),
            'population_hv_credit_scale': convergence_info.get('population_hv_credit_scale'),
            'population_ser_credit_scale': convergence_info.get('population_ser_credit_scale'),
            'action_quality': convergence_info.get('action_quality', 0.0),
            'global_quality_reward': convergence_info.get('global_quality_reward', 0.0),
            'spea2_updates': convergence_info.get('spea2_updates', 0),
            'nsga2_updates': convergence_info.get('nsga2_updates', 0),
            'algorithm_update_ratio': convergence_info.get('algorithm_update_ratio', 1.0),
            'restart_count': convergence_info.get('restart_count', 0),
        }

        # 如果Episode终止（收敛或达到最大步数），保存经验以便下个Episode使用
        if terminated:
            try:
                experience = self.moea.save_experience() if hasattr(self.moea, 'save_experience') else None
                if experience:
                    self.last_experience = experience
                    print(f"[ENV]: 在Episode终止时保存经验 (精英解数={len(experience.get('elite_genes', []))})")
            except Exception as e:
                print(f"[WARNING] 终止时保存经验失败: {e}")

            # ============ 跨Episode长期学习：记录调参经验 ============
            # 1. 更新全局最优参数
            if final_hv > self.global_best_hv:
                self.global_best_hv = final_hv
                self.global_best_params = {
                    'action': action.copy() if isinstance(action, np.ndarray) else action,
                    'action_params': action_params,
                    'hv': final_hv,
                    'episode': self.current_episode_index
                }
                print(f"[ENV]: 发现新的全局最优 HV={final_hv:.4f}")

            # 2. 将本次Episode的关键经验存入expert_buffer
            episode_experience = {
                'episode': self.current_episode_index,
                'final_hv': final_hv,
                'hv_improvement': convergence_info.get('hv_improvement', 0.0),
                'best_action': self._get_best_action_from_history(),
                'converged': self.episode_converged,
                'steps_to_converge': self.current_episode_step
            }
            self.expert_buffer.append(episode_experience)

            # 限制buffer大小
            if len(self.expert_buffer) > self.max_expert_buffer_size:
                self.expert_buffer.pop(0)

            # 3. 记录Episode HV用于追踪
            self.episode_hv_record.append(final_hv)

            # ── 跨Episode解质量追踪 ───────────────────────────────────────────
            # 提取本 episode 最终种群中的最佳 sensitivity
            if hasattr(self.moea, 'current_population') and self.moea.current_population:
                pareto = self._extract_pareto_front(self.moea.current_population)
                if pareto:
                    # sensitivity ratio = objectives[3]（越小越好）
                    # 防御性处理: objectives[3] 可能是数组，取其标量值
                    sens_vals = []
                    for ind in pareto:
                        try:
                            obj = ind.objectives
                            if obj is None:
                                continue
                            obj_len = int(np.asarray(obj).shape[0]) if hasattr(obj, 'shape') else len(obj)
                            if obj_len >= 4:
                                obj3 = obj[3]
                                if hasattr(obj3, '__len__') and not np.isscalar(obj3):
                                    val = float(np.asarray(obj3).flatten()[0])
                                else:
                                    val = float(obj3)
                                sens_vals.append(val)
                        except Exception:
                            continue
                    if sens_vals:
                        best_sens_ratio = min(sens_vals)
                        self.episode_best_sensitivity.append(best_sens_ratio)
                        if best_sens_ratio < self.global_best_sensitivity:
                            self.global_best_sensitivity = best_sens_ratio
                            self.global_best_residual_ratio = best_sens_ratio
                            print(f"[ENV] 新的全局最佳 SER: residual_ratio={best_sens_ratio:.4f}")
                        print(f"[ENV] Episode {self.current_episode_index} 最佳 SER: {best_sens_ratio:.4f} "
                              f"(全局最佳: {self.global_best_sensitivity:.4f})")

            print(f"[ENV]: Episode {self.current_episode_index} 经验已保存 (Buffer大小={len(self.expert_buffer)})")

        return next_state, reward, terminated, truncated, info
   
    def _run_moea_continue(self, action, generations_to_run):
        """从当前种群继续运行指定代数的MOEA"""
        # 更新MOEA参数
        self._update_moea_parameters(action)

        # 定义评估函数
        def eval_fn(genes):
            try:
                if hasattr(self.moea, 'model') and hasattr(self.moea.model, 'evaluate_population'):
                    return self.moea.model.evaluate_population(genes)
                else:
                    # 简化的后备评估
                    results = []
                    for gene in genes:
                        area = np.sum(gene) * 10
                        power = np.sum(gene) * 0.5
                        slack = 0.1 - np.sum(gene * self.moea.is_critical) * 0.01
                        sens = np.sum(self.moea.error_rates * (1 - gene))
                        results.append([area, power, -slack, sens])
                    return np.array(results)
            except Exception as e:
                print(f"评估函数出错: {e}")
                return np.ones((len(genes), 4)) * 1e6

        # 获取收敛参数
        convergence_config = self.config.get('convergence', {})
        convergence_threshold = convergence_config.get('hv_rel_threshold', 0.02)
        patience = convergence_config.get('patience', 6)
        min_feasible_ratio = convergence_config.get('min_feasible_ratio', 0.50)
        min_diversity = convergence_config.get('min_diversity', 0.10)
        min_generations = convergence_config.get('min_generations_before_check', 5)
        force_continue = convergence_config.get('force_continue_on_violation', True)
        max_force_continue_gen = convergence_config.get('max_force_continue_gen', 15)
        ignore_early_convergence = bool(self.fixed_budget_per_sac_step and not self.allow_early_stop)

        # 记录初始状态
        initial_generation = getattr(self.moea, 'current_generation', 0)
        initial_population = getattr(self.moea, 'current_population', [])

        # 计算初始HV
        # initial_hv = self._calculate_hypervolume(initial_population) if initial_population else 0.0
        if hasattr(self.moea, 'convergence_history') and self.moea.convergence_history.get('hv'):
            initial_hv = self.moea.convergence_history['hv'][-1]
        else:
            initial_hv = 0.0
        
        print(f"[MOEA_DEBUG] SAC步开始前状态:")
        print(f"  当前代数: {initial_generation}")
        print(f"  初始HV (来自历史): {initial_hv:.4f}")
        print(f"  收敛历史长度: {len(self.moea.convergence_history.get('hv', []))}")
        
        # 运行MOEA - 固定代数预算模式下，尽量跑完预算
        # 注意：RealMoeaWrapper.run_with_convergence 只支持以下参数
        population, _, stats, convergence_info = self.moea.run_with_convergence(
            eval_fn=eval_fn,
            max_generations=generations_to_run,
            convergence_threshold=convergence_threshold,
            patience=patience,
            generations_per_batch=generations_to_run,
            reset=False,
            ignore_early_convergence=ignore_early_convergence,
            min_generations_before_check=min_generations
        )

        # 计算最终HV
        # final_hv = self._calculate_hypervolume(population) if population else 0.0
        if hasattr(self.moea, 'convergence_history') and self.moea.convergence_history.get('hv'):
            final_hv = self.moea.convergence_history['hv'][-1]
        else:
            final_hv = initial_hv
        
        print(f"[MOEA_DEBUG] SAC步结束后状态:")
        print(f"  最终代数: {getattr(self.moea, 'current_generation', 0)}")
        print(f"  最终HV (来自历史): {final_hv:.4f}")
        print(f"  收敛历史长度: {len(self.moea.convergence_history.get('hv', []))}")
        print(f"  HV改进: {final_hv - initial_hv:.4f}")

        # 以实际执行代数计入效率奖励，避免提前收敛时仍按请求预算计费。
        actual_generations = int(convergence_info.get('final_generation', generations_to_run))

        # 更新收敛信息
        convergence_info.update({
            'initial_generation': initial_generation,
            'generations_run': actual_generations,
            'initial_hv': initial_hv,
            'final_hv': final_hv,
            # 改进的计算：即使 initial_hv=0 也能获得改进值
            'hv_improvement': final_hv - initial_hv  # 直接计算，不依赖 initial_hv > 0
        })

        return population, convergence_info
    
    def _run_moea_step(self, action):
        """运行MOEA一步（不重置，继续从当前种群运行）"""
        # 更新MOEA参数
        self._update_moea_parameters(action)

        # 定义评估函数
        def eval_fn(genes):
            try:
                if hasattr(self.moea, 'model') and hasattr(self.moea.model, 'evaluate_population'):
                    return self.moea.model.evaluate_population(genes)
                else:
                    # 简化的后备评估
                    results = []
                    for gene in genes:
                        area = np.sum(gene) * 10
                        power = np.sum(gene) * 0.5
                        slack = 0.1 - np.sum(gene * self.moea.is_critical) * 0.01
                        sens = np.sum(self.moea.error_rates * (1 - gene))
                        results.append([area, power, -slack, sens])
                    return np.array(results)
            except Exception as e:
                print(f"评估函数出错: {e}")
                return np.ones((len(genes), 4)) * 1e6

        # 获取MOEA配置参数
        generations_per_sac_step = self.generations_per_sac_step
        convergence_config = self.config.get('convergence', {})
        convergence_threshold = convergence_config.get('hv_rel_threshold', 0.02)
        patience = convergence_config.get('patience', 6)
        min_feasible_ratio = convergence_config.get('min_feasible_ratio', 0.50)
        min_diversity = convergence_config.get('min_diversity', 0.10)
        min_generations = convergence_config.get('min_generations_before_check', 5)
        force_continue = convergence_config.get('force_continue_on_violation', True)
        max_force_continue_gen = convergence_config.get('max_force_continue_gen', 15)
        ignore_early_convergence = bool(self.fixed_budget_per_sac_step and not self.allow_early_stop)

        # 运行MOEA（不重置，使用当前种群继续）
        population, _, stats, convergence_info = self.moea.run_with_convergence(
            eval_fn=eval_fn,
            max_generations=generations_per_sac_step,
            convergence_threshold=convergence_threshold,
            patience=patience,
            reset=False,  # 关键：不重置，使用当前种群继续
            ignore_early_convergence=ignore_early_convergence,
            min_generations_before_check=min_generations
        )

        if ignore_early_convergence and bool(convergence_info.get('is_converged', False)):
            convergence_info['budget_ignored_convergence'] = True

        return population, convergence_info

    def _run_complete_moea(self, action):
        """运行完整的MOEA，每次都从随机种群开始"""
        # 更新MOEA参数（使用辅助函数）
        self._update_moea_parameters(action)

        # 定义评估函数
        def eval_fn(genes):
            try:
                if hasattr(self.moea, 'model') and hasattr(self.moea.model, 'evaluate_population'):
                    return self.moea.model.evaluate_population(genes)
                else:
                    # 简化的后备评估
                    results = []
                    for gene in genes:
                        area = np.sum(gene) * 10
                        power = np.sum(gene) * 0.5
                        slack = 0.1 - np.sum(gene * self.moea.is_critical) * 0.01
                        sens = np.sum(self.moea.error_rates * (1 - gene))
                        results.append([area, power, -slack, sens])
                    return np.array(results)
            except Exception as e:
                print(f"评估函数出错: {e}")
                return np.ones((len(genes), 4)) * 1e6

        # 获取MOEA配置参数
        max_generations = self.generations_per_sac_step  # 使用每步代数
        convergence_config = self.config.get('convergence', {})
        convergence_threshold = convergence_config.get('hv_rel_threshold', 0.02)
        patience = convergence_config.get('patience', 6)
        min_feasible_ratio = convergence_config.get('min_feasible_ratio', 0.50)
        min_diversity = convergence_config.get('min_diversity', 0.10)
        min_generations = convergence_config.get('min_generations_before_check', 5)
        force_continue = convergence_config.get('force_continue_on_violation', True)
        max_force_continue_gen = convergence_config.get('max_force_continue_gen', 15)
        ignore_early_convergence = bool(self.fixed_budget_per_sac_step and not self.allow_early_stop)

        # 运行MOEA（每次都从新种群开始）
        population, _, stats, convergence_info = self.moea.run_with_convergence(
            eval_fn=eval_fn,
            max_generations=max_generations,  # 使用较小的代数，比如5代
            convergence_threshold=convergence_threshold,
            patience=patience,
            #reset=True  # 关键：每次都从随机种群开始
            reset=False,
            ignore_early_convergence=ignore_early_convergence,
            min_generations_before_check=min_generations
        )

        if ignore_early_convergence and bool(convergence_info.get('is_converged', False)):
            convergence_info['budget_ignored_convergence'] = True

        return population, convergence_info

    def _calculate_hypervolume_accurate(self, pareto_front, ref_point=None):
        """
        更准确的HV计算方法
        使用改进的算法，避免近似误差
        """
        if not pareto_front:
            return 0.0
        
        # 提取目标值
        objectives = []
        for ind in pareto_front:
            if ind.objectives is not None and not np.any(np.isnan(ind.objectives)):
                objectives.append(ind.objectives)
        
        if not objectives:
            return 0.0
        
        objectives_array = np.array(objectives)
        n_objs = objectives_array.shape[1]
        
        # 归一化目标值
        min_vals = np.min(objectives_array, axis=0)
        max_vals = np.max(objectives_array, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0
        
        normalized = (objectives_array - min_vals) / ranges
        
        # 设置参考点
        if ref_point is None:
            ref_point = np.ones(n_objs) * 1.2
        
        # 使用更准确的HV计算方法
        # 对于4维问题，使用改进的算法
        if n_objs == 4:
            return self._calculate_hypervolume_4d(normalized, ref_point)
        else:
            # 对于其他维度，使用通用方法
            return self._calculate_hypervolume_general(normalized, ref_point)
    
    def _calculate_hypervolume_4d(self, objectives, ref_point):
        """4维HV的精确计算方法"""
        if len(objectives) == 0:
            return 0.0
        
        # 按第一个目标排序
        sorted_idx = np.argsort(objectives[:, 0])
        sorted_objs = objectives[sorted_idx]
        
        hv = 0.0
        
        # 使用改进的算法：逐个计算每个解的贡献
        for i in range(len(sorted_objs)):
            obj = sorted_objs[i]
            
            # 计算到参考点的体积
            volume = 1.0
            for d in range(len(obj)):
                if obj[d] < ref_point[d]:
                    volume *= (ref_point[d] - obj[d])
            
            # 减去与前面解的重叠部分
            if i > 0:
                # 计算重叠体积（更精确的方法）
                overlap_volume = self._calculate_overlap_volume_4d(
                    sorted_objs[:i], obj, ref_point
                )
                volume = max(0.0, volume - overlap_volume)
            
            hv += volume
        
        return float(hv)
    
    def _calculate_overlap_volume_4d(self, prev_objs, current_obj, ref_point):
        """计算4维重叠体积"""
        if len(prev_objs) == 0:
            return 0.0
        
        # 找到支配current_obj的所有前面解
        dominated_by = []
        for prev_obj in prev_objs:
            if np.all(prev_obj <= current_obj) and np.any(prev_obj < current_obj):
                dominated_by.append(prev_obj)
        
        if not dominated_by:
            return 0.0
        
        # 计算重叠体积：取所有支配解的并集
        # 简化：使用最大重叠（保守估计）
        max_overlap = 0.0
        for dom_obj in dominated_by:
            overlap = 1.0
            for d in range(len(current_obj)):
                if dom_obj[d] < current_obj[d]:
                    overlap *= (ref_point[d] - current_obj[d])
                else:
                    overlap *= (ref_point[d] - dom_obj[d])
            max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def _calculate_hypervolume_general(self, objectives, ref_point):
        """通用HV计算方法（用于非4维情况）"""
        if len(objectives) == 0:
            return 0.0
        
        # 按第一个目标排序
        sorted_idx = np.argsort(objectives[:, 0])
        sorted_objs = objectives[sorted_idx]
        
        hv = 0.0
        prev_volume = 0.0
        
        for i, obj in enumerate(sorted_objs):
            volume = 1.0
            for d in range(len(obj)):
                if obj[d] < ref_point[d]:
                    volume *= (ref_point[d] - obj[d])
            
            if i == 0:
                hv += volume
            else:
                # 减去重叠部分（改进的近似）
                overlap = min(prev_volume, volume) * 0.6  # 改进的重叠系数
                hv += max(0.0, volume - overlap)
            
            prev_volume = volume
        
        return float(hv)
    
    def _calculate_pareto_center(self, population):
        """
        计算Pareto前沿中心（基于归一化目标值）
        返回: 4维归一化目标值的均值向量
        """
        pareto_front = self._extract_pareto_front(population)
        if not pareto_front:
            return None

        objectives = []
        for ind in pareto_front:
            if ind.objectives is not None and not np.any(np.isnan(ind.objectives)):
                objectives.append(ind.objectives)

        if not objectives:
            return None

        objectives_array = np.array(objectives, dtype=np.float64)
        # 用当前种群的 min/max 做行内归一化（与 HV 计算保持一致）
        min_vals = np.min(objectives_array, axis=0)
        max_vals = np.max(objectives_array, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0
        normalized = (objectives_array - min_vals) / ranges
        return np.mean(normalized, axis=0).astype(np.float32)
    
    def _calculate_algorithm_metrics(self, population, algorithm_name=''):
        """计算单个算法的指标"""
        if not population:
            return {'hv': 0.0, 'diversity': 0.0, 'violation': 0.0}
        
        # 计算HV
        pareto_front = self._extract_pareto_front(population)
        hv = self._calculate_hypervolume_accurate(pareto_front) if pareto_front else 0.0
        
        # 计算多样性
        diversity = self._calculate_diversity(population)
        
        # 计算约束违例
        violations, _ = self._calculate_constraint_violations(population)
        violation = np.mean(violations) if len(violations) > 0 else 0.0
        
        return {'hv': hv, 'diversity': diversity, 'violation': violation}
    
    def _get_best_ppa_feasible_ser(self, population):
        """Return best residual SER ratio among PPA-feasible candidates."""
        stats = {
            'candidate_count': 0,
            'feasible_count': 0,
            'feasible_ratio': 0.0,
            'best_residual_ratio': None,
            'best_residual_ser': None,
            'best_tmr_count': 0,
            'mean_topk_residual_ratio': None,
        }
        if population is None or len(population) == 0:
            return stats

        try:
            candidates = self._extract_pareto_front(population)
        except Exception:
            candidates = []
        if not candidates:
            candidates = list(population)

        reward_params = self.config.get('reward_params', {})
        candidate_limit = int(reward_params.get('feasible_ser_candidate_limit', 160) or 0)
        if candidate_limit > 0 and len(candidates) > candidate_limit:
            def _rough_residual_ratio(ind):
                obj = getattr(ind, 'objectives', None)
                if obj is None or len(obj) < 4:
                    return float('inf')
                try:
                    return float(np.asarray(obj[3]).flatten()[0])
                except Exception:
                    return float('inf')

            candidates = sorted(candidates, key=_rough_residual_ratio)[:candidate_limit]

        model = getattr(self.moea, 'model', None)
        thresholds = self.config.get('constraint_thresholds', {})
        area_limit = float(thresholds.get('delta_area_max', 0.2))
        power_limit = float(thresholds.get('delta_power_max', 0.5))
        slack_min = float(thresholds.get('slack_min', 0.0))

        area_orig = float(getattr(model, 'area_orig', getattr(self, 'area_orig', 1.0))) if model is not None else float(getattr(self, 'area_orig', 1.0))
        power_orig = float(getattr(model, 'power_orig', getattr(self, 'power_orig', 1.0))) if model is not None else float(getattr(self, 'power_orig', 1.0))
        sens_orig = float(getattr(model, 'sens_orig', getattr(self, 'sens_orig', 1.0))) if model is not None else float(getattr(self, 'sens_orig', 1.0))
        area_orig = max(area_orig, 1e-20)
        power_orig = max(power_orig, 1e-20)
        sens_orig = max(sens_orig, 1e-20)

        feasible_ratios = []
        for ind in candidates:
            gene = getattr(ind, 'gene', None)
            if gene is None:
                continue
            stats['candidate_count'] += 1
            try:
                gene = np.asarray(gene).flatten()
                tmr_count = int(np.sum(gene))

                if model is not None and hasattr(model, 'evaluate_with_details'):
                    details = model.evaluate_with_details(gene)
                    if len(details) < 7:
                        continue
                    area_total, power_total, min_slack, residual_ser = details[:4]
                    area_increase = details[5]
                    power_increase = details[6]
                    area_overhead = float(area_increase) / area_orig
                    power_overhead = float(power_increase) / power_orig
                    residual_ratio = float(residual_ser) / sens_orig
                    ppa_feasible = (
                        area_overhead <= area_limit + 1e-12 and
                        power_overhead <= power_limit + 1e-12 and
                        float(min_slack) >= slack_min - 1e-12
                    )
                else:
                    obj = getattr(ind, 'objectives', None)
                    if obj is None or len(obj) < 4:
                        continue
                    residual_ratio = float(np.asarray(obj[3]).flatten()[0])
                    residual_ser = residual_ratio * sens_orig
                    ppa_feasible = (
                        bool(getattr(ind, 'area_power_feasible', False)) and
                        float(getattr(ind, 'violation', 0.0)) <= 1e-12
                    )

                if not ppa_feasible:
                    continue

                stats['feasible_count'] += 1
                feasible_ratios.append(float(residual_ratio))
                best = stats['best_residual_ratio']
                if best is None or residual_ratio < best:
                    stats['best_residual_ratio'] = float(residual_ratio)
                    stats['best_residual_ser'] = float(residual_ser)
                    stats['best_tmr_count'] = tmr_count
            except Exception:
                continue

        if stats['candidate_count'] > 0:
            stats['feasible_ratio'] = float(stats['feasible_count']) / float(stats['candidate_count'])
        if feasible_ratios:
            reward_params = self.config.get('reward_params', {})
            topk = int(reward_params.get('feasible_ser_topk', 10) or 10)
            topk = max(1, min(topk, len(feasible_ratios)))
            stats['mean_topk_residual_ratio'] = float(np.mean(sorted(feasible_ratios)[:topk]))

        return stats

    def _get_feasible_resource_coverage(self, population, bins=5):
        """Estimate coverage of feasible candidates over resource/SER tradeoff bins."""
        try:
            bins = max(2, int(bins))
        except Exception:
            bins = 5
        if population is None:
            return 0.0
        try:
            if len(population) == 0:
                return 0.0
        except Exception:
            pass

        constraints = self.config.get('constraint_thresholds', {})
        delta_area_max = max(1e-12, float(constraints.get('delta_area_max', 0.2)))
        slack_min = float(constraints.get('slack_min', 0.0))

        cells = set()
        for ind in population:
            gene = getattr(ind, 'gene', None)
            if gene is None:
                continue
            try:
                if hasattr(self.moea, '_gene_metrics_fast'):
                    metrics = self.moea._gene_metrics_fast(gene)
                    if not metrics.get('feasible', False):
                        continue
                    area_over = float(metrics['area_increase']) / max(float(self.area_orig), 1e-20)
                    residual_ratio = float(metrics['residual_ratio'])
                    wns = float(metrics['worst_slack'])
                else:
                    obj = np.asarray(getattr(ind, 'objectives', None), dtype=float).flatten()
                    if obj.size < 4:
                        continue
                    area_over = float(obj[0]) / max(float(self.area_orig), 1e-20) - 1.0
                    residual_ratio = float(obj[3])
                    wns = -float(obj[2])
            except Exception:
                continue
            if not np.all(np.isfinite([area_over, residual_ratio, wns])) or wns < slack_min:
                continue
            area_norm = np.clip(area_over / delta_area_max, 0.0, 0.999999)
            ser_norm = np.clip(residual_ratio, 0.0, 0.999999)
            cells.add((int(area_norm * bins), int(ser_norm * bins)))

        return float(len(cells)) / float(bins * bins)

    def _normalize_action_credit(self, key: str, value: float, fallback_gain_scale: float) -> float:
        """将微小的动作质量增量按跨episode运行尺度归一化为稳定的有符号反馈。"""
        reward_params = self.config.get('reward_params', {})
        value = float(value) if np.isfinite(value) else 0.0
        if not bool(reward_params.get('adaptive_action_credit', True)):
            return float(np.clip(value * fallback_gain_scale, -1.0, 1.0))

        if not hasattr(self, '_action_credit_scales'):
            self._action_credit_scales = {}
        decay = float(np.clip(reward_params.get('action_credit_ema_decay', 0.90), 0.0, 0.999))
        floor = max(1e-12, float(reward_params.get(f'action_credit_{key}_floor', 1e-6)))
        observed = abs(value)
        previous = self._action_credit_scales.get(key)
        if observed > floor:
            scale = observed if previous is None else decay * float(previous) + (1.0 - decay) * observed
        else:
            scale = floor if previous is None else float(previous)
        scale = max(floor, scale)
        self._action_credit_scales[key] = scale

        ratio_clip = max(1.0, float(reward_params.get('action_credit_ratio_clip', 4.0)))
        normalized = np.clip(value / scale, -ratio_clip, ratio_clip)
        return float(np.tanh(normalized))

    def _calculate_reward_from_moea(self, population, convergence_info):
        """
        基于MOEA运行结果计算奖励（完整版：多目标指标综合评估）
        使用设计文档中的公式：
        r_t = α·(ΔHV/(HV_{t-1}+ε)) - β·viol - γ·div_drop - η·oscillation + ζ·IGD_improvement + balance_reward
        
        奖励分量：
        1. HV改进（超体积）：最重要的多目标指标，直接衡量Pareto前沿质量
        2. IGD改进（反向生成距离）：衡量前沿的覆盖性和均匀性
        3. 约束满足：惩罚约束违反，确保解的可行性
        4. 多样性维持：鼓励保持种群多样性，避免早熟收敛
        5. 前沿稳定性：惩罚振荡，鼓励渐进式改进
        6. 算法平衡：鼓励两个算法都表现好
        """
        try:
            # 获取MOEA运行结果
            final_hv = float(convergence_info.get('archive_hv_after', convergence_info.get('final_hv', 0.0)))
            initial_hv = float(convergence_info.get('archive_hv_before', convergence_info.get('initial_hv', 0.0)))
            is_converged = convergence_info.get('is_converged', False)

            # 计算HV改进
            hv_improvement = final_hv - initial_hv

            # 获取奖励参数
            reward_params = self.config['reward_params']

            def safe_float(value, default=0.0):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return float(default)

            alpha = safe_float(reward_params.get('alpha', 1.0))
            beta = safe_float(reward_params.get('beta', 1.0))
            gamma = safe_float(reward_params.get('gamma', 0.5))
            eta = safe_float(reward_params.get('eta', 0.1))
            zeta = safe_float(reward_params.get('zeta', 0.5))      # 新增：IGD改进权重
            epsilon = safe_float(reward_params.get('epsilon', 1e-6))
            balance_weight = safe_float(reward_params.get('balance_weight', 0.1))
            igd_norm = safe_float(reward_params.get('igd_normalization', 1.0))

            # ============ 1. HV改进项（主奖励）============
            # r_hv = α·(ΔHV/(HV_{t-1}+ε))
            # 处理初始 HV=0 的边界情况：使用绝对值而不是相对增益
            if initial_hv <= epsilon:
                # 初始化阶段：直接使用绝对 HV 值，归一化到 [0,1]
                # 假设合理的 HV 范围是 [0, 1000]，超过则裁剪
                hv_gain = float(np.clip(final_hv, 0.0, 1.0))
            else:
                # 进行中阶段：使用相对改进
                # Fixed-reference archive HV is in [0, 1]. Normalize by the
                # remaining improvable volume so late improvements stay visible.
                hv_denominator = max(1.0 - initial_hv, 0.05)
                hv_gain = hv_improvement / hv_denominator
                hv_gain = np.clip(hv_gain, -1.0, 1.0)
            
            hv_reward = alpha * hv_gain

            # ============ 当前种群的动作归因增益 ============
            # 全局累计档案只在真正发现新解时增长，信号非常稀疏。这里同时使用
            # 动作前后同一种群的固定口径HV与最佳可行SER变化，为SAC提供稠密反馈。
            population_hv_before = safe_float(convergence_info.get('population_hv_before', 0.0))
            population_hv_after = safe_float(convergence_info.get('population_hv_after', population_hv_before))
            population_hv_gain = population_hv_after - population_hv_before
            population_hv_gain_scale = safe_float(reward_params.get('population_hv_gain_scale', 40.0), 40.0)
            population_hv_weight = safe_float(reward_params.get('population_hv_weight', 1.0), 1.0)
            local_hv_gain = population_hv_gain / max(1.0 - population_hv_before, 0.05)

            population_ser_gain = safe_float(convergence_info.get('population_ser_gain', 0.0))
            population_ser_gain_scale = safe_float(reward_params.get('population_ser_gain_scale', 20.0), 20.0)
            population_ser_weight = safe_float(reward_params.get('population_ser_weight', 1.0), 1.0)
            credit_valid = bool(convergence_info.get('population_credit_valid', True))
            if credit_valid:
                population_hv_credit = self._normalize_action_credit(
                    'population_hv', local_hv_gain, population_hv_gain_scale
                )
                population_ser_credit = self._normalize_action_credit(
                    'population_ser', population_ser_gain, population_ser_gain_scale
                )
                population_hv_reward = population_hv_weight * population_hv_credit
                population_ser_reward = population_ser_weight * population_ser_credit
            else:
                population_hv_credit = 0.0
                population_ser_credit = 0.0
                population_hv_reward = 0.0
                population_ser_reward = 0.0
            convergence_info['population_hv_credit'] = population_hv_credit
            convergence_info['population_ser_credit'] = population_ser_credit
            credit_scales = getattr(self, '_action_credit_scales', {})
            convergence_info['population_hv_credit_scale'] = credit_scales.get('population_hv')
            convergence_info['population_ser_credit_scale'] = credit_scales.get('population_ser')

            # ============ 2. IGD改进项（新增）============
            # r_igd = ζ·(-ΔIGD/(IGD_{t-1}+ε))，IGD越小越好，所以改进是负数
            igd_reward = 0.0
            igd_gain = 0.0  # 初始化
            current_igd = 0.0  # 初始化
            if hasattr(self, 'reference_set') and self.reference_set is not None:
                try:
                    current_igd = self._calculate_igd(population, self.reference_set)
                    prev_igd = self.prev_metrics.get('igd', current_igd)
                    
                    igd_improvement = -(current_igd - prev_igd)  # 负数表示改进（IGD减少）
                    
                    igd_denominator = max(prev_igd, epsilon)
                    if igd_denominator > 0:
                        igd_gain = igd_improvement / igd_denominator
                        igd_gain = np.clip(igd_gain, -1.0, 1.0)
                    else:
                        igd_gain = igd_improvement / max(1.0, current_igd) if current_igd > 0 else 0.0
                    
                    igd_reward = zeta * igd_gain
                    
                except Exception as e:
                    print(f"WARNING: 计算IGD改进时出错: {e}")
                    igd_reward = 0.0
            
            # ============ 3. 约束违例项============
            # r_viol = -β·viol，约束满足时viol=0
            violations, violation_details = self._calculate_constraint_violations(population)
            current_violation = np.mean(violations) if len(violations) > 0 else 0.0
            
            # 约束违例惩罚：使用合理的缩放因子避免过度裁剪
            # 当前违例值范围可达数千到数万，除以 20000 可将典型值映射到 [0,1] 区间
            current_violation_clipped = min(current_violation / 20000.0, 1.0)
            violation_penalty = beta * current_violation_clipped

            # ============ PPA-feasible best SER reward ============
            # This is the main task reward: only PPA-feasible candidates can
            # improve the best residual SER ratio.
            feasible_ser_stats = self._get_best_ppa_feasible_ser(population)
            current_best_feasible_ser = feasible_ser_stats.get('best_residual_ratio')
            feasible_ser_reward = 0.0
            feasible_ser_quality_reward = 0.0
            feasible_set_quality_reward = 0.0
            feasible_ratio_reward = 0.0
            resource_coverage_reward = 0.0
            no_feasible_penalty = 0.0

            feasible_ser_weight = safe_float(reward_params.get('feasible_ser_weight', 4.0))
            feasible_ser_quality_weight = safe_float(reward_params.get('feasible_ser_quality_weight', 1.0))
            feasible_set_quality_weight = safe_float(reward_params.get('feasible_set_quality_weight', 0.6))
            feasible_ratio_weight = safe_float(reward_params.get('feasible_ratio_weight', 0.2))
            resource_coverage_weight = safe_float(reward_params.get('resource_coverage_weight', 0.0))
            infeasible_ser_penalty = safe_float(reward_params.get('infeasible_ser_penalty', 1.0))
            ser_quality_target = safe_float(reward_params.get('ser_quality_target', 0.072))
            if ser_quality_target <= epsilon:
                ser_quality_target = safe_float(
                    reward_params.get('fallback_ser_quality_target', 0.072), 0.072
                )

            global_gain = 0.0
            step_gain = 0.0
            self.last_reward_best_feasible_ser = None
            self.last_reward_global_gain = 0.0
            self.last_reward_step_gain = 0.0
            if current_best_feasible_ser is not None:
                current_best_feasible_ser = float(current_best_feasible_ser)
                self.last_reward_best_feasible_ser = current_best_feasible_ser
                prev_step_best = getattr(self, 'prev_step_best_feasible_ser', None)
                if prev_step_best is None:
                    step_gain = 0.0
                else:
                    step_gain = (float(prev_step_best) - current_best_feasible_ser) / max(float(prev_step_best), epsilon)
                    step_gain = float(np.clip(step_gain, -1.0, 1.0))

                global_best = float(getattr(self, 'global_best_feasible_ser', getattr(self, 'global_best_sensitivity', 1.0)))
                if current_best_feasible_ser < global_best:
                    global_gain = (global_best - current_best_feasible_ser) / max(global_best, epsilon)
                    global_gain = float(np.clip(global_gain, 0.0, 1.0))
                    self.global_best_feasible_ser = current_best_feasible_ser
                    self.global_best_sensitivity = min(float(getattr(self, 'global_best_sensitivity', 1.0)), current_best_feasible_ser)
                    self.global_best_residual_ratio = current_best_feasible_ser
                    print(
                        f"[REWARD] New best PPA-feasible SER ratio: "
                        f"{current_best_feasible_ser:.6f}, gain={global_gain:.6f}"
                    )
                else:
                    global_gain = 0.0

                self.last_reward_global_gain = global_gain
                self.last_reward_step_gain = step_gain
                feasible_ser_reward = feasible_ser_weight * (0.85 * global_gain + 0.15 * step_gain)
                quality_score = (ser_quality_target - current_best_feasible_ser) / max(ser_quality_target, epsilon)
                feasible_ser_quality_reward = feasible_ser_quality_weight * float(np.clip(quality_score, -1.0, 1.0))
                self.prev_step_best_feasible_ser = current_best_feasible_ser
            else:
                no_feasible_penalty = infeasible_ser_penalty

            mean_topk_ser = feasible_ser_stats.get('mean_topk_residual_ratio')
            if mean_topk_ser is not None:
                topk_quality = (ser_quality_target - float(mean_topk_ser)) / max(ser_quality_target, epsilon)
                feasible_set_quality_reward = feasible_set_quality_weight * float(np.clip(topk_quality, -1.0, 1.0))
            feasible_ratio_reward = feasible_ratio_weight * float(np.clip(feasible_ser_stats.get('feasible_ratio', 0.0), 0.0, 1.0))
            if resource_coverage_weight > 0.0:
                resource_coverage_bins = int(reward_params.get('resource_coverage_bins', 5))
                resource_coverage_reward = resource_coverage_weight * self._get_feasible_resource_coverage(
                    population, bins=resource_coverage_bins
                )

            best_ser_stagnation_penalty = 0.0
            if (
                current_best_feasible_ser is not None
                and global_gain <= 0.0
                and step_gain <= 0.0
                and hv_improvement <= epsilon
            ):
                no_improvement_scale = safe_float(reward_params.get('no_improvement_quality_scale', 0.15), 0.15)
                feasible_ser_quality_reward *= no_improvement_scale
                feasible_set_quality_reward *= no_improvement_scale
                resource_coverage_reward *= no_improvement_scale
                best_ser_stagnation_penalty = safe_float(
                    reward_params.get('best_ser_stagnation_penalty_weight', 0.25), 0.25
                )

            # ============ 额外奖励/惩罚：效率与时序违例============
            efficiency_weight = float(reward_params.get('efficiency_weight', 0.1))
            timing_penalty_weight = float(reward_params.get('timing_penalty_weight', 0.5))

            # 代数/效率惩罚：使用本次SAC步实际运行的代数比例（越少越好）
            generations_used = float(convergence_info.get('generations_run', convergence_info.get('final_generation', 0)))
            max_gen = float(self.config.get('moea', {}).get('max_generations', 1))
            algorithm_update_ratio = convergence_info.get('algorithm_update_ratio')
            if algorithm_update_ratio is None:
                cost_ratio = generations_used / max(1.0, max_gen)
            else:
                # search_control_v2会跳过弱侧算法的部分代数，按实际算法更新量计费。
                cost_ratio = float(algorithm_update_ratio) * generations_used / max(1.0, max_gen)
            efficiency_penalty = efficiency_weight * min(1.0, max(0.0, cost_ratio))

            # 时序违例惩罚：检查Pareto前沿中实际负WNS的数量（如果已计算）
            timing_violations = 0
            try:
                pareto_front = self._extract_pareto_front(population)
                for ind in pareto_front:
                    if ind.objectives is not None and len(ind.objectives) >= 3:
                        wns = -ind.objectives[2]
                        if wns < 0:
                            timing_violations += 1
            except Exception:
                timing_violations = 0
            timing_penalty = timing_penalty_weight * min(1.0, timing_violations / max(1, len(pareto_front) if pareto_front else 1))

            # ============ 4. 多样性下降项============
            # r_div = -γ·div_drop，多样性维持时div_drop=0
            current_diversity = self._calculate_diversity(population)

            # 跨 episode 重置后第一步不应惩罚
            if self.prev_metrics['diversity'] > 0 and self.current_episode_step > 0:
                delta_div = current_diversity - self.prev_metrics['diversity']
                div_drop = max(0.0, -delta_div)
                # 归一化到 [0,1]：归一化 diversity 通常在 [0, 1] 范围，取 max(1.0, prev) 作为分母
                div_drop_norm = div_drop / max(1.0, self.prev_metrics['diversity'])
                diversity_penalty = gamma * min(div_drop_norm, 1.0)
            else:
                div_drop = 0.0
                diversity_penalty = 0.0

            # ============ 5. 前沿振荡项============
            # r_osc = -η·oscillation，前沿稳定时oscillation≈0
            # 修复：
            #   1. 跨 episode 时 prev_pareto_center 已由 reset() 置 None，无累积问题
            #   2. 用归一化目标计算中心差，避免原始目标尺度不一致
            #   3. 除以归一化参考距离（diagonal of unit hypercube = sqrt(4) ≈ 2）获得稳定缩放
            current_center = self._calculate_pareto_center(population)
            oscillation = 0.0
            oscillation_norm = 0.0

            if self.prev_pareto_center is not None and current_center is not None:
                center_diff = current_center - self.prev_pareto_center
                norm_diff = float(np.linalg.norm(center_diff))
                oscillation_norm = norm_diff / np.sqrt(4.0)   # 归一化到 [0,1] 区间
                oscillation = eta * float(np.clip(oscillation_norm, 0.0, 1.0))
            
            # ============ 6. 算法平衡奖励============
            # r_balance：鼓励两个算法都表现好，避免一个算法主导
            balance_reward = 0.0
            if hasattr(self.moea, 'spea2') and hasattr(self.moea, 'nsga2'):
                try:
                    spea2_pop = getattr(self.moea.spea2, 'population', [])
                    nsga2_pop = getattr(self.moea.nsga2, 'population', [])
                    
                    if spea2_pop and nsga2_pop:
                        # 计算两个算法的指标
                        spea2_metrics = self._calculate_algorithm_metrics(spea2_pop, 'SPEA2')
                        nsga2_metrics = self._calculate_algorithm_metrics(nsga2_pop, 'NSGA2')
                        
                        self.spea2_metrics = spea2_metrics
                        self.nsga2_metrics = nsga2_metrics
                        
                        # 平衡度奖励：两个算法HV接近
                        spea2_hv = spea2_metrics['hv']
                        nsga2_hv = nsga2_metrics['hv']
                        total_alg_hv = spea2_hv + nsga2_hv
                        
                        if total_alg_hv > 0:
                            balance_ratio = 1.0 - abs(spea2_hv - nsga2_hv) / total_alg_hv
                            balance_reward = balance_weight * balance_ratio
                        
                        # 算法贡献奖励：两个算法都有改进
                        spea2_hv_improvement = spea2_hv - self.spea2_metrics.get('prev_hv', spea2_hv)
                        nsga2_hv_improvement = nsga2_hv - self.nsga2_metrics.get('prev_hv', nsga2_hv)
                        
                        if spea2_hv_improvement > 0 and nsga2_hv_improvement > 0:
                            balance_reward += balance_weight * 0.5
                        
                        self.spea2_metrics['prev_hv'] = spea2_hv
                        self.nsga2_metrics['prev_hv'] = nsga2_hv
                        
                except Exception as e:
                    print(f"WARNING: 计算算法平衡奖励时出错: {e}")
            
            # 更新Pareto前沿中心和种群
            self.prev_pareto_center = current_center.copy()
            self.prev_population = population

            # ============ 6. 直接敏感性改善奖励（新增，权重=2.0）============
            direct_sens_reward = 0.0
            try:
                sens_weight = float(reward_params.get('sens_weight', 2.0))
                
                pareto_front = self._extract_pareto_front(population)
                if pareto_front:
                    sens_vals = []
                    for ind in pareto_front:
                        try:
                            obj = ind.objectives
                            if obj is None:
                                continue
                            obj_len = int(np.asarray(obj).shape[0]) if hasattr(obj, 'shape') else len(obj)
                            if obj_len >= 4:
                                obj3 = obj[3]
                                if hasattr(obj3, '__len__') and not np.isscalar(obj3):
                                    val = float(np.asarray(obj3).flatten()[0])
                                else:
                                    val = float(obj3)
                                sens_vals.append(val)
                        except Exception:
                            continue

                    if sens_vals:
                        best_sens = min(sens_vals)
                        if hasattr(self, 'prev_pareto_sensitivity'):
                            sens_improvement = self.prev_pareto_sensitivity - best_sens
                            sens_denom = max(self.prev_pareto_sensitivity, 1e-6)
                            sens_gain = np.clip(sens_improvement / sens_denom, -1.0, 1.0)
                            direct_sens_reward = sens_weight * sens_gain
                            self.prev_pareto_sensitivity = best_sens
                        else:
                            self.prev_pareto_sensitivity = best_sens
            except Exception:
                direct_sens_reward = 0.0

            # ============ 总奖励计算============
            # r_t = HV + IGD + 敏感性改善 - viol - div_drop - oscillation + balance_reward
            #       + cross_ep_improvement + absolute_ser_quality
            total_reward = (
                feasible_ser_reward
                + feasible_ser_quality_reward
                + feasible_set_quality_reward
                + feasible_ratio_reward
                + resource_coverage_reward
                + hv_reward
                + population_hv_reward
                + population_ser_reward
                + igd_reward
                + direct_sens_reward
                - violation_penalty
                - no_feasible_penalty
                - diversity_penalty
                - oscillation
                - efficiency_penalty
                - timing_penalty
                - best_ser_stagnation_penalty
                + balance_reward
            )

            # ── FIX #2: 跨Episode改善奖励 ────────────────────────────────────
            # 当当前 episode 的最佳 SER 优于全局历史最佳时，给出大幅正奖励
            # 这解决了 "每个 episode 独立，reward 与绝对质量脱钩" 的根本问题
            cross_ep_reward = 0.0
            try:
                pareto_for_cross = self._extract_pareto_front(population)
                if pareto_for_cross and hasattr(self, 'global_best_sensitivity'):
                    sens_vals = []
                    for ind in pareto_for_cross:
                        try:
                            obj = ind.objectives
                            if obj is None:
                                continue
                            obj_len = int(np.asarray(obj).shape[0]) if hasattr(obj, 'shape') else len(obj)
                            if obj_len >= 4:
                                obj3 = obj[3]
                                if hasattr(obj3, '__len__') and not np.isscalar(obj3):
                                    val = float(np.asarray(obj3).flatten()[0])
                                else:
                                    val = float(obj3)
                                sens_vals.append(val)
                        except Exception:
                            continue
                    if sens_vals:
                        current_best_sens = min(sens_vals)
                        cross_ep_weight = float(reward_params.get('cross_ep_improvement_weight', 2.0))
                        if current_best_sens < self.global_best_sensitivity:
                            improvement = (self.global_best_sensitivity - current_best_sens) / max(self.global_best_sensitivity, 1e-6)
                            cross_ep_reward = cross_ep_weight * min(improvement * 10.0, 1.0)
                            self.global_best_sensitivity = current_best_sens
                            self.global_best_residual_ratio = current_best_sens
                            print(f"[REWARD] 跨Episode改善! current={current_best_sens:.4f}, reward={cross_ep_reward:.4f}")
            except Exception:
                cross_ep_reward = 0.0

            # ── FIX #3: 绝对 SER 质量奖励 ────────────────────────────────────
            abs_ser_reward = 0.0
            try:
                pareto_for_abs = self._extract_pareto_front(population)
                if pareto_for_abs:
                    sens_vals = []
                    for ind in pareto_for_abs:
                        try:
                            obj = ind.objectives
                            if obj is None:
                                continue
                            obj_len = int(np.asarray(obj).shape[0]) if hasattr(obj, 'shape') else len(obj)
                            if obj_len >= 4:
                                obj3 = obj[3]
                                if hasattr(obj3, '__len__') and not np.isscalar(obj3):
                                    val = float(np.asarray(obj3).flatten()[0])
                                else:
                                    val = float(obj3)
                                sens_vals.append(val)
                        except Exception:
                            continue
                    if sens_vals:
                        current_best_sens = min(sens_vals)
                        abs_ser_weight = float(reward_params.get('absolute_ser_quality_weight', 0.5))
                        quality_score = max(0.0, 1.0 - current_best_sens / 0.5)
                        abs_ser_reward = abs_ser_weight * (quality_score * 2.0 - 1.0)
            except Exception:
                abs_ser_reward = 0.0

            total_reward += cross_ep_reward + abs_ser_reward

            # 该值只汇总与本次动作直接相关的质量增量，用于参数诊断图；
            # 不包含绝对质量、可行比例等状态相关常数项。
            # 参数诊断只显示当前动作直接造成的种群质量增量。档案和全局最优
            # 奖励单独记录，避免把跨步积累误解为某个参数的即时因果效果。
            action_quality = float(hv_reward + population_hv_reward + population_ser_reward)
            convergence_info['action_quality'] = action_quality
            convergence_info['global_quality_reward'] = float(
                feasible_ser_reward
            )

            # 奖励缩放/裁剪（可配置）
            clip_reward = bool(reward_params.get('force_clip', True))
            use_tanh = bool(reward_params.get('use_tanh_scaling', False))
            original_reward = float(total_reward)
            if clip_reward:
                # 放宽 clip 范围以容纳新的跨Episode和绝对SER改善信号
                reward_clip_min = safe_float(reward_params.get('reward_clip_min', -2.0), -2.0)
                reward_clip_max = safe_float(reward_params.get('reward_clip_max', 2.0), 2.0)
                if reward_clip_max <= reward_clip_min:
                    reward_clip_min, reward_clip_max = -2.0, 2.0
                total_reward = np.clip(total_reward, reward_clip_min, reward_clip_max)
            else:
                if use_tanh:
                    soft_bound = max(
                        1e-6,
                        safe_float(reward_params.get('reward_soft_bound', 5.0), 5.0),
                    )
                    total_reward = float(soft_bound * np.tanh(total_reward / soft_bound))
                else:
                    total_reward = float(total_reward)

            if abs(original_reward - float(total_reward)) > 1e-9:
                print(f"[REWARD] reward scaled/clipped {original_reward:.6f} -> {total_reward:.6f}")

            # 详细的奖励组件日志（便于诊断）
            print(f"[REWARD_DEBUG] 奖励组件分解:")
            print(f"  HV_gain={hv_gain:.6f}, hv_reward={hv_reward:.6f}")
            print(
                f"  population_hv={population_hv_before:.6f}->{population_hv_after:.6f}, "
                f"gain={population_hv_gain:.6f}, reward={population_hv_reward:.6f}"
            )
            print(
                f"  population_ser_gain={population_ser_gain:.6f}, "
                f"reward={population_ser_reward:.6f}, action_quality={action_quality:.6f}"
            )
            print(f"  igd_gain={igd_gain:.6f}, igd_reward={igd_reward:.6f}")
            print(
                f"  feasible_best_ser={current_best_feasible_ser}, "
                f"feasible_count={feasible_ser_stats.get('feasible_count', 0)}/"
                f"{feasible_ser_stats.get('candidate_count', 0)}, "
                f"feasible_ser_reward={feasible_ser_reward:.6f}, "
                f"feasible_ser_quality={feasible_ser_quality_reward:.6f}, "
                f"feasible_set_quality={feasible_set_quality_reward:.6f}, "
                f"feasible_ratio_reward={feasible_ratio_reward:.6f}, "
                f"resource_coverage_reward={resource_coverage_reward:.6f}, "
                f"no_feasible_penalty={no_feasible_penalty:.6f}"
            )
            print(f"  直接敏感性奖励={direct_sens_reward:.6f}")
            print(f"  跨Episode改善奖励={cross_ep_reward:.6f}")
            print(f"  绝对SER质量奖励={abs_ser_reward:.6f}")
            print(f"  violation={current_violation:.6f} -> clipped={current_violation_clipped:.6f}, penalty={violation_penalty:.6f}")
            print(f"  div_drop={div_drop:.6f}, diversity_penalty={diversity_penalty:.6f}")
            print(f"  oscillation_norm={oscillation_norm:.6f}, oscillation={oscillation:.6f}")
            print(f"  balance_reward={balance_reward:.6f}")
            print(f"  总和前裁剪: {original_reward:.6f}, 裁剪后: {total_reward:.6f}")
            
            # 详细日志
            print(f"[REWARD] HV={final_hv:.4f}, ΔHV={hv_improvement:.4f}, "
                  f"HV_gain={hv_gain:.6f}(α×={hv_reward:.4f}) | "
                  f"IGD_gain={igd_gain:.6f}(ζ×={igd_reward:.4f}) | "
                  f"敏感性改善(×={direct_sens_reward:.4f}) | "
                  f"违例={current_violation:.4f}(β×={violation_penalty:.4f}) | "
                  f"多样性↓={div_drop:.4f}(γ×={diversity_penalty:.4f}) | "
                  f"振荡={oscillation_norm:.6f}(η×={oscillation:.4f}) | "
                  f"平衡={balance_reward:.4f} | "
                  f"跨Episode={cross_ep_reward:.4f} | "
                  f"绝对SER={abs_ser_reward:.4f} | 总={total_reward:.4f}")

            # 更新历史指标
            self.prev_metrics = {
                'hv': final_hv,
                'igd': current_igd,
                'diversity': current_diversity,
                'violation': current_violation
            }

            return float(total_reward)

        except Exception as e:
            print(f"ERROR: 计算MOEA奖励时出错: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _update_moea_parameters(self, action: np.ndarray):
        """
        更新MOEA参数
        支持的动作空间维度:
        - 6维:  兼容旧版本 [ε, τ, δ, β, pm, pc] (两个算法共用)
        - 8维:  search_control_v2直接控制两个算法和混合搜索资源
        - 12维: NSGA2 + SPEA2 [SPEA2(6), NSGA2(6)]
        - 17维: NSGA2 + SPEA2 + MOEA/D [SPEA2(6), NSGA2(6), MOEA/D(5)]
        """
        if len(action) == 8 and self.action_schema == 'search_control_v2':
            decoded = self._decode_action_params(action)
            params = {
                'spea2': {
                    'epsilon_dom': decoded['spea2']['epsilon'],
                    'accept_threshold': decoded['spea2']['tau'],
                    'delta_div': decoded['spea2']['delta'],
                    'beta_penalty': decoded['spea2']['beta'],
                    'pm': decoded['spea2']['pm'],
                    'pc': decoded['spea2']['pc'],
                },
                'nsga2': {
                    'epsilon_dom': decoded['nsga2']['epsilon'],
                    'accept_threshold': decoded['nsga2']['tau'],
                    'delta_div': decoded['nsga2']['delta'],
                    'beta_penalty': decoded['nsga2']['beta'],
                    'pm': decoded['nsga2']['pm'],
                    'pc': decoded['nsga2']['pc'],
                },
                'search_controls': dict(decoded['controls']),
            }
            if hasattr(self.moea, 'update_parameters'):
                self.moea.update_parameters(params)
            return

        # 处理不同维度的动作空间
        if len(action) == 17:
            # 17维动作空间：SPEA2(6) + NSGA2(6) + MOEA/D(5)
            # [SPEA2(6), NSGA2(6), MOEA/D(5)]
            eps_spea2, tau_spea2, delta_spea2, beta_spea2, pm_spea2, pc_spea2 = action[:6]
            eps_nsga2, tau_nsga2, delta_nsga2, beta_nsga2, pm_nsga2, pc_nsga2 = action[6:12]
            moead_ns, moead_us, moead_dec = action[12:15]
            moead_pc, moead_pm = action[15:17]
        elif len(action) == 12:
            # 12维动作空间：分别为两个算法设置独立参数
            eps_spea2, tau_spea2, delta_spea2, beta_spea2, pm_spea2, pc_spea2 = action[:6]
            eps_nsga2, tau_nsga2, delta_nsga2, beta_nsga2, pm_nsga2, pc_nsga2 = action[6:]
            # MOEA/D参数使用默认值
            moead_ns = moead_us = moead_dec = moead_pc = moead_pm = None
        elif len(action) == 8:
            # 8维半共享动作空间：
            # [epsilon, accept_threshold, delta_div, beta_penalty, spea2_pm, spea2_pc, nsga2_pm, nsga2_pc]
            eps, tau, delta, beta, pm_spea2, pc_spea2, pm_nsga2, pc_nsga2 = action
            eps_spea2 = eps_nsga2 = eps
            tau_spea2 = tau_nsga2 = tau
            delta_spea2 = delta_nsga2 = delta
            beta_spea2 = beta_nsga2 = beta
            moead_ns = moead_us = moead_dec = moead_pc = moead_pm = None
        elif len(action) == 6:
            # 兼容旧的6维动作空间（向后兼容）
            print("警告: 使用6维动作空间，所有算法将使用相同参数")
            eps_spea2, tau_spea2, delta_spea2, beta_spea2, pm_spea2, pc_spea2 = action
            eps_nsga2, tau_nsga2, delta_nsga2, beta_nsga2, pm_nsga2, pc_nsga2 = action
            moead_ns = moead_us = moead_dec = moead_pc = moead_pm = None
        else:
            raise ValueError(f"动作空间维度错误: 期望6/8/12/17维，得到{len(action)}维")

        if hasattr(self.moea, 'update_parameters'):
            # 构建参数字典
            params = {
                'spea2': {
                    'epsilon_dom': float(eps_spea2),
                    'accept_threshold': float(tau_spea2),
                    'delta_div': float(delta_spea2),
                    'beta_penalty': float(beta_spea2),
                    'pm': float(pm_spea2),
                    'pc': float(pc_spea2)
                },
                'nsga2': {
                    'epsilon_dom': float(eps_nsga2),
                    'accept_threshold': float(tau_nsga2),
                    'delta_div': float(delta_nsga2),
                    'beta_penalty': float(beta_nsga2),
                    'pm': float(pm_nsga2),
                    'pc': float(pc_nsga2)
                }
            }

            # 添加MOEA/D参数
            if moead_ns is not None:
                params['moead_neighborhood_size'] = int(moead_ns * 50)  # 缩放到 0-50
            if moead_us is not None:
                strategies = ['delta', 'probability', 'uniform']
                idx = int(moead_us * 2) % 3
                params['moead_update_strategy'] = strategies[idx]
            if moead_dec is not None:
                decompositions = ['weighted', 'tchebycheff', 'boundary_intersection']
                idx = int(moead_dec * 2) % 3
                params['moead_decomposition'] = decompositions[idx]
            if moead_pc is not None:
                params['moead_pc'] = float(moead_pc)
            if moead_pm is not None:
                params['moead_pm'] = float(moead_pm)

            self.moea.update_parameters(params)

    def _calculate_final_reward(self, population, convergence_info):
        """
        基于完整MOEA运行计算最终奖励
        """
        try:
            # 获取MOEA运行结果
            final_hv = convergence_info.get('final_hv', 0.0)
            initial_hv = convergence_info.get('initial_hv', 0.0)
            is_converged = convergence_info.get('is_converged', False)
            generations_used = convergence_info.get('final_generation', 0)

            # 奖励参数
            reward_params = self.config['reward_params']
            alpha = float(reward_params.get('alpha', 1.0))
            beta_weight = float(reward_params.get('beta', 0.1))
            gamma = float(reward_params.get('gamma', 0.1))

            # 1. 基础奖励：HV改进
            hv_improvement = final_hv - initial_hv if initial_hv > 0 else final_hv
            hv_reward = alpha * hv_improvement

            # 2. 收敛奖励
            convergence_reward = 0.0
            if is_converged:
                convergence_reward = 0.5

            # 3. 效率奖励（使用代数越少越好）
            max_generations = self.config['moea']['max_generations']
            efficiency_reward = gamma * (1.0 - generations_used / max_generations)

            # 4. 计算约束违例惩罚（如果有的话）
            violation_penalty = 0.0
            if population:
                violations, _ = self._calculate_constraint_violations(population)
                if len(violations) > 0:
                    avg_violation = np.mean(violations)
                    violation_penalty = beta_weight * avg_violation

            # 5. 最终奖励
            total_reward = hv_reward + convergence_reward + efficiency_reward - violation_penalty

            # 奖励缩放/裁剪（可配置）
            clip_reward = bool(reward_params.get('force_clip', True))
            use_tanh = bool(reward_params.get('use_tanh_scaling', False))
            original_reward = float(total_reward)
            if clip_reward:
                total_reward = np.clip(total_reward, -1.0, 1.0)
            else:
                if use_tanh:
                    total_reward = float(np.tanh(total_reward))
                else:
                    total_reward = float(total_reward)

            if abs(original_reward - float(total_reward)) > 1e-9:
                print(f"[REWARD] reward scaled/clipped {original_reward:.6f} -> {total_reward:.6f}")

            print(f"[REWARD]: HV={final_hv:.4f}, 改进={hv_improvement:.4f}, "
                  f"收敛={is_converged}, 代数={generations_used}, 奖励={total_reward:.4f}")

            return float(total_reward)

        except Exception as e:
            print(f"ERROR: 计算最终奖励时出错: {e}")
            return 0.0
        
    def _run_single_generation(self, eval_fn):
        """运行单代MOEA（后备方法）"""
        # 简化的MOEA实现
        pop_size = self.config['moea']['pop_size']
        
        if not hasattr(self, '_population') or self._population is None:
            # 初始化种群
            self._population = [
                Individual(np.random.randint(0, 2, self.moea.K), None)
                for _ in range(pop_size)
            ]
        
        # 评估
        genes = np.array([ind.gene for ind in self._population])
        objectives = eval_fn(genes)
        for i, ind in enumerate(self._population):
            ind.objectives = objectives[i]
        
        # 选择、交叉、变异（简化）
        # 这里应该实现完整的MOEA流程，但为简化只返回当前种群
        return self._population
    
    def _extract_compact_state(self, population: List[Individual], action: np.ndarray) -> np.ndarray:
        """Build a low-noise state aligned with feasible area/SER front quality."""
        population = list(population or [])
        feasible_metrics = []
        for ind in population:
            gene = getattr(ind, 'gene', None)
            if gene is None or not hasattr(self.moea, '_gene_metrics_fast'):
                continue
            try:
                metrics = self.moea._gene_metrics_fast(gene)
                if metrics.get('feasible', False):
                    feasible_metrics.append(metrics)
            except Exception:
                continue

        current_hv = 0.0
        archive_hv = 0.0
        front = []
        try:
            current_hv = float(self.moea._calculate_scoring_hv_2d(population))
            archive_hv = float(self.moea.get_feasible_archive_hv())
            front = self.moea._extract_ppa_feasible_pareto(population)
        except Exception:
            pass

        residuals = sorted(float(m['residual_ratio']) for m in feasible_metrics)
        best_ser_quality = 1.0 - residuals[0] if residuals else 0.0
        topk_ser_quality = 1.0 - float(np.mean(residuals[:min(10, len(residuals))])) if residuals else 0.0
        feasible_ratio = len(feasible_metrics) / max(1, len(population))
        front_ratio = len(front) / max(1, len(population))

        area_limit = max(float(self.delta_area_max), 1e-20)
        power_limit = max(float(self.delta_power_max), 1e-20)
        area_budget = max(float(self.area_orig) * area_limit, 1e-20)
        power_budget = max(float(self.power_orig) * power_limit, 1e-20)
        area_use = [float(m['area_increase']) / area_budget for m in feasible_metrics]
        power_use = [float(m['power_increase']) / power_budget for m in feasible_metrics]
        resource_use = [max(a, p) for a, p in zip(area_use, power_use)]
        occupied_bins = {
            min(7, int(np.clip(value, 0.0, 0.999999) * 8))
            for value in resource_use
        }
        coverage = len(occupied_bins) / 8.0

        slack_scale = max(abs(float(getattr(self, 'slack_orig', 0.0))), 0.13, 1e-6)
        min_wns = min((float(m['worst_slack']) for m in feasible_metrics), default=float(self.min_slack))
        slack_margin = float(np.clip((min_wns - float(self.min_slack)) / slack_scale, -1.0, 1.0))

        def algorithm_hv(algorithm_name):
            try:
                algorithm = getattr(self.moea, algorithm_name)
                return float(self.moea._calculate_scoring_hv_2d(getattr(algorithm, 'population', []) or []))
            except Exception:
                return 0.0

        episode_progress = self.current_episode_step / max(
            1, int(self.config.get('sac', {}).get('max_steps_per_episode', 1))
        )
        stagnation = float(getattr(self.moea, 'no_improvement_count', 0)) / max(
            1, int(self.config.get('convergence', {}).get('patience', 1))
        )
        base_features = np.array(
            [
                current_hv,
                archive_hv,
                best_ser_quality,
                topk_ser_quality,
                feasible_ratio,
                front_ratio,
                coverage,
                float(np.mean(area_use)) if area_use else 0.0,
                float(np.max(area_use)) if area_use else 0.0,
                float(np.mean(power_use)) if power_use else 0.0,
                float(np.max(power_use)) if power_use else 0.0,
                slack_margin,
                algorithm_hv('spea2'),
                algorithm_hv('nsga2'),
                float(np.clip(episode_progress, 0.0, 1.0)),
                float(np.clip(stagnation, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

        action = np.asarray(action, dtype=np.float32).flatten()
        low = np.asarray(self.action_space.low, dtype=np.float32).flatten()
        high = np.asarray(self.action_space.high, dtype=np.float32).flatten()
        if action.size != low.size:
            action = np.resize(action, low.size)
        normalized_action = (np.clip(action, low, high) - low) / np.maximum(high - low, 1e-12)
        state = np.concatenate([base_features, normalized_action.astype(np.float32)])
        if state.size < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.size))
        return np.nan_to_num(state[:self.state_dim], nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def _extract_state(self, population: List[Individual], action: np.ndarray) -> np.ndarray:
        """
        构建状态向量（24维合并种群 + 24维算法独立状态 = 48维）
        """
        if self.state_schema == 'compact_score_aligned_v1':
            return self._extract_compact_state(population, action)

        state = []
        
        # 1. 规模特征（2维）
        if not population or len(population) == 0:
            state.extend([0.0, 0.0])
        else:
            pareto_front = self._extract_pareto_front(population)
            front_size = len(pareto_front) if pareto_front else 0
            front_size_ratio = front_size / len(population) if len(population) > 0 else 0.0
            state.append(front_size_ratio)
            
            # 计算前沿密度
            if front_size > 1 and pareto_front:
                # 简单的密度计算：个体数量/目标空间范围
                objectives = np.array([ind.objectives for ind in pareto_front])
                if len(objectives) > 0 and objectives.shape[1] > 0:
                    # 计算目标值的范围
                    ranges = np.max(objectives, axis=0) - np.min(objectives, axis=0)
                    ranges[ranges == 0] = 1.0  # 避免除零
                    # 简单密度估计
                    density = front_size / np.prod(ranges + 1e-10)
                    state.append(min(1.0, density))
                else:
                    state.append(0.0)
            else:
                state.append(0.0)
        
        # 2. 收敛性特征（4维）
        # 2.1 HV比率
        if len(self.hv_history) > 0:
            current_hv = self.hv_history[-1]
            max_hv = max(self.hv_history) if self.hv_history else 1.0
            hv_ratio = current_hv / max(1.0, max_hv)
        else:
            hv_ratio = 0.0
        state.append(hv_ratio)
        
        # 2.2 IGD得分
        igd_score = 0.0
        if hasattr(self, 'reference_set') and self.reference_set is not None:
            pareto_front = self._extract_pareto_front(population)
            if pareto_front:
                try:
                    igd_value = self._calculate_igd(pareto_front, self.reference_set)
                    # 归一化IGD得分（假设IGD范围在0-10之间，值越小越好）
                    # 转换为0-1范围，1表示最好（IGD=0），0表示最差（IGD>=10）
                    igd_score = max(0.0, 1.0 - min(igd_value / 10.0, 1.0))
                except Exception as e:
                    # 如果计算失败，使用默认值
                    igd_score = 0.0
            else:
                igd_score = 0.0  # 如果没有前沿，IGD为最差值
        else:
            # 如果没有参考集，无法计算IGD，使用默认值
            igd_score = 0.0
        state.append(igd_score)
        
        # 2.3 收敛率
        convergence_rate = self._calculate_convergence_rate()
        state.append(convergence_rate)
        
        # 2.4 代数比例
        if hasattr(self, 'current_episode_step'):
            generation_ratio = self.current_episode_step / 50.0  # 假设最大步数为50
        else:
            generation_ratio = 0.0
        state.append(generation_ratio)
        
        # 3. 多样性特征（3维）
        # 3.1 平均拥挤距离
        avg_crowding = 0.0
        if population:
            crowding_distances = []
            for ind in population:
                if hasattr(ind, 'crowding_distance') and not np.isinf(ind.crowding_distance):
                    crowding_distances.append(ind.crowding_distance)
            if crowding_distances:
                avg_crowding = np.mean(crowding_distances)
        state.append(avg_crowding)
        
        # 3.2 间距指标
        spacing_metric = 0.0
        if population and len(population) > 1:
            pareto_front = self._extract_pareto_front(population)
            if pareto_front and len(pareto_front) > 1:
                objectives = np.array([ind.objectives for ind in pareto_front])
                # 简单间距计算
                distances = []
                for i in range(len(objectives)):
                    for j in range(i+1, len(objectives)):
                        dist = np.linalg.norm(objectives[i] - objectives[j])
                        distances.append(dist)
                if distances:
                    spacing_metric = np.mean(distances)
        state.append(min(1.0, spacing_metric))
        
        # 3.3 极值点覆盖比例
        extreme_coverage = 0.0
        if population and len(population) > 0:
            pareto_front = self._extract_pareto_front(population)
            if pareto_front and len(pareto_front) >= 2:
                objectives = []
                for ind in pareto_front:
                    if ind.objectives is not None and not np.any(np.isnan(ind.objectives)):
                        objectives.append(ind.objectives)
                
                if objectives:
                    objectives_array = np.array(objectives)
                    n_objs = objectives_array.shape[1]
                    
                    # 计算每个目标维度的极值点
                    extreme_indices = set()
                    for obj_idx in range(n_objs):
                        # 找到最小值和最大值对应的个体索引
                        min_idx = np.argmin(objectives_array[:, obj_idx])
                        max_idx = np.argmax(objectives_array[:, obj_idx])
                        extreme_indices.add(min_idx)
                        extreme_indices.add(max_idx)
                    
                    # 归一化：极值点数量 / 前沿大小
                    # 理想情况下，每个目标维度至少有一个极值点
                    extreme_coverage = len(extreme_indices) / len(pareto_front)
                    # 限制在0-1之间
                    extreme_coverage = min(1.0, extreme_coverage)
        state.append(extreme_coverage)
        
        # 4. 各目标统计特征（12维）
        if population and len(population) > 0:
            # 提取所有目标值
            all_objectives = []
            for ind in population:
                if ind.objectives is not None and not np.any(np.isnan(ind.objectives)):
                    all_objectives.append(ind.objectives)
            
            if all_objectives:
                objectives_array = np.array(all_objectives)
                
                # 归一化处理
                min_vals = np.min(objectives_array, axis=0)
                max_vals = np.max(objectives_array, axis=0)
                ranges = max_vals - min_vals
                ranges[ranges == 0] = 1.0
                
                normalized = (objectives_array - min_vals) / ranges
                
                # 4.1 面积统计（目标0）
                if normalized.shape[1] > 0:
                    area_values = normalized[:, 0]
                    state.append(np.mean(area_values))
                    state.append(np.std(area_values))
                    state.append(np.min(area_values))
                else:
                    state.extend([0.0, 0.0, 0.0])
                
                # 4.2 功耗统计（目标1）
                if normalized.shape[1] > 1:
                    power_values = normalized[:, 1]
                    state.append(np.mean(power_values))
                    state.append(np.std(power_values))
                    state.append(np.min(power_values))
                else:
                    state.extend([0.0, 0.0, 0.0])
                
                # 4.3 时序统计（目标2，注意是负值）
                if normalized.shape[1] > 2:
                    # 注意：目标2是 -WNS，所以需要转换
                    slack_values = -normalized[:, 2]  # 转换为正的WNS
                    state.append(np.mean(slack_values))
                    state.append(np.std(slack_values))
                    state.append(np.max(slack_values))  # 最差slack是最大值
                else:
                    state.extend([0.0, 0.0, 0.0])
                
                # 4.4 敏感性统计（目标3）
                if normalized.shape[1] > 3:
                    sens_values = normalized[:, 3]
                    state.append(np.mean(sens_values))
                    state.append(np.std(sens_values))
                    state.append(np.max(sens_values))
                else:
                    state.extend([0.0, 0.0, 0.0])
            else:
                state.extend([0.0] * 12)
        else:
            state.extend([0.0] * 12)
        
        # 5. 约束违例特征（3维）
        violation_ratio = 0.0
        slack_viol_ratio = 0.0
        worst_violation = 0.0
        
        if population:
            # 计算约束违例
            violations, violation_details = self._calculate_constraint_violations(population)
            if len(violations) > 0:
                # 违约解比例
                has_violation = violation_details['has_violation']
                violation_ratio = np.sum(has_violation) / len(population)
                
                # 时序违例解比例
                slack_violations = violation_details['slack'] > 0
                slack_viol_ratio = np.sum(slack_violations) / len(population)
                
                # 最严重违约程度
                worst_violation = np.max(violations) if len(violations) > 0 else 0.0
                worst_violation = min(1.0, worst_violation / 10000.0)  # 归一化
        
        state.append(violation_ratio)
        state.append(slack_viol_ratio)
        state.append(worst_violation)
        
        # 扩展状态空间：添加两个算法的独立状态（28维，SPEA2和NSGA2各14维）
        if self.use_algorithm_specific_states and hasattr(self.moea, 'spea2') and hasattr(self.moea, 'nsga2'):
            try:
                spea2_pop = getattr(self.moea.spea2, 'population', [])
                nsga2_pop = getattr(self.moea.nsga2, 'population', [])
                spea2_state = self._extract_algorithm_state(spea2_pop, 'spea2')
                state.extend(spea2_state)
                spea2_pareto = self._extract_pareto_front(spea2_pop)
                spea2_hv = self._calculate_hypervolume_accurate(spea2_pareto) if spea2_pareto else 0.0
                self.spea2_hv_history.append(spea2_hv)
                if len(self.spea2_hv_history) > 100:
                    self.spea2_hv_history = self.spea2_hv_history[-100:]
                nsga2_state = self._extract_algorithm_state(nsga2_pop, 'nsga2')
                state.extend(nsga2_state)
                nsga2_pareto = self._extract_pareto_front(nsga2_pop)
                nsga2_hv = self._calculate_hypervolume_accurate(nsga2_pareto) if nsga2_pareto else 0.0
                self.nsga2_hv_history.append(nsga2_hv)
                if len(self.nsga2_hv_history) > 100:
                    self.nsga2_hv_history = self.nsga2_hv_history[-100:]
            except Exception as e:
                print(f"WARNING: 提取算法独立状态失败: {e}")
                state.extend([0.0] * 24)
        else:
            if self.use_algorithm_specific_states:
                state.extend([0.0] * 24)
        
        # 确保状态维度正确
        # 基础状态 = 2+4+3+12+3+14+14 = 52维（规模2 + 收敛4 + 多样性3 + 目标统计12 + 约束违例3 + SPEA2算法14 + NSGA2算法14）
        # 专家上下文（10维）由 get_state_with_expert_context 单独追加，总计 62 维
        state_array = np.array(state, dtype=np.float32)
        # 基础状态维度是52维，padding会补到52维
        base_state_dim = 52
        expected_dim = base_state_dim
        if len(state_array) != expected_dim:
            print(f"警告：状态维度为{len(state_array)}，调整为{expected_dim}维")
            if len(state_array) < expected_dim:
                padding = np.zeros(expected_dim - len(state_array), dtype=np.float32)
                state_array = np.concatenate([state_array, padding])
            else:
                state_array = state_array[:expected_dim]
        
        return state_array
    
    def _extract_algorithm_state(self, population, algorithm_name=''):
        """
        提取单个算法的状态（14维）
        包含：规模特征(2) + 收敛特征(2) + 多样性特征(2) + 目标统计(8,含敏感性)
        """
        state = []
        
        if not population or len(population) == 0:
            return [0.0] * 14
        
        # 1. 规模特征（2维）
        pareto_front = self._extract_pareto_front(population)
        front_size = len(pareto_front) if pareto_front else 0
        front_size_ratio = front_size / len(population) if len(population) > 0 else 0.0
        state.append(front_size_ratio)
        
        # 前沿密度
        if front_size > 1 and pareto_front:
            objectives = np.array([ind.objectives for ind in pareto_front])
            if len(objectives) > 0:
                ranges = np.max(objectives, axis=0) - np.min(objectives, axis=0)
                ranges[ranges == 0] = 1.0
                density = front_size / np.prod(ranges + 1e-10)
                state.append(min(1.0, density))
            else:
                state.append(0.0)
        else:
            state.append(0.0)
        
        # 2. 收敛特征（2维）
        # HV比率（使用算法特定的HV）
        alg_hv = self._calculate_hypervolume_accurate(pareto_front) if pareto_front else 0.0
        hv_history_name = f'{algorithm_name}_hv_history'
        if hasattr(self, hv_history_name):
            hv_history = getattr(self, hv_history_name)
            if len(hv_history) > 0:
                max_hv = max(hv_history) if hv_history else 1.0
                hv_ratio = alg_hv / max(1.0, max_hv)
            else:
                hv_ratio = alg_hv / max(1.0, alg_hv) if alg_hv > 0 else 0.0
        else:
            hv_ratio = alg_hv / max(1.0, alg_hv) if alg_hv > 0 else 0.0
        
        state.append(hv_ratio)
        
        # 收敛率（最近5代的HV改进率）
        convergence_rate = 0.0
        if hasattr(self, hv_history_name):
            hv_history = getattr(self, hv_history_name)
            if len(hv_history) >= 6:
                recent_hv = hv_history[-5:]
                improvements = []
                for i in range(1, len(recent_hv)):
                    if recent_hv[i-1] > 0:
                        improvement = (recent_hv[i] - recent_hv[i-1]) / recent_hv[i-1]
                        improvements.append(improvement)
                if improvements:
                    convergence_rate = np.mean(improvements)
        state.append(convergence_rate)
        
        # 3. 多样性特征（2维）
        # 平均拥挤距离
        avg_crowding = 0.0
        if population:
            crowding_distances = []
            for ind in population:
                if hasattr(ind, 'crowding_distance') and not np.isinf(ind.crowding_distance):
                    crowding_distances.append(ind.crowding_distance)
            if crowding_distances:
                avg_crowding = np.mean(crowding_distances)
        state.append(avg_crowding)
        
        # 间距指标
        spacing_metric = 0.0
        if pareto_front and len(pareto_front) > 1:
            objectives = np.array([ind.objectives for ind in pareto_front])
            distances = []
            for i in range(len(objectives)):
                for j in range(i+1, len(objectives)):
                    dist = np.linalg.norm(objectives[i] - objectives[j])
                    distances.append(dist)
            if distances:
                spacing_metric = np.mean(distances)
        state.append(min(1.0, spacing_metric))
        
        # 4. 目标统计特征（6维：每个目标2维：mean, std）
        if population:
            all_objectives = []
            for ind in population:
                if ind.objectives is not None and not np.any(np.isnan(ind.objectives)):
                    all_objectives.append(ind.objectives)
            
            if all_objectives:
                objectives_array = np.array(all_objectives)
                min_vals = np.min(objectives_array, axis=0)
                max_vals = np.max(objectives_array, axis=0)
                ranges = max_vals - min_vals
                ranges[ranges == 0] = 1.0
                normalized = (objectives_array - min_vals) / ranges
                
                # 提取全部4个目标的统计（每个目标2维：mean, std，共8维）
                for obj_idx in range(min(4, normalized.shape[1])):
                    values = normalized[:, obj_idx]
                    state.append(np.mean(values))
                    state.append(np.std(values))
                
                # 填充零值至14维（4目标×2维 = 8维）
                while len(state) < 14:
                    state.append(0.0)
            else:
                state.extend([0.0] * 8)
        else:
            state.extend([0.0] * 8)
        
        return state[:14]  # 确保正好14维
    
    def _calculate_constraint_violations(self, population: List[Individual]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算约束违例
        """
        if not population:
            return np.array([]), {}

        n_pop = len(population)
        violations = np.zeros(n_pop)
        violation_details = {
            'area': np.zeros(n_pop),
            'power': np.zeros(n_pop),
            'slack': np.zeros(n_pop),
            'sens': np.zeros(n_pop),
            'has_violation': np.zeros(n_pop, dtype=bool),
            'timing_violation_count': 0,
        }

        # 优先从环境成员变量取，否则从配置字典取，默认 0.0
        slack_threshold = getattr(self, 'min_slack', 0.0)
        if hasattr(self, 'config'):
            slack_threshold = self.config.get('constraint_thresholds', {}).get('min_slack', slack_threshold)

        for i, ind in enumerate(population):
            if ind.objectives is None or len(ind.objectives) < 4:
                continue

            # Penalty-based MOEA branches may keep sentinel values in
            # original_objectives. Re-evaluate genes with the same fast PPA
            # model used by the feasible archive and final export.
            fresh_metrics = None
            gene = getattr(ind, 'gene', None)
            if gene is not None and hasattr(self.moea, '_gene_metrics_fast'):
                try:
                    fresh_metrics = self.moea._gene_metrics_fast(gene)
                except Exception:
                    fresh_metrics = None

            if fresh_metrics is not None:
                area_current = self.area_orig + float(fresh_metrics['area_increase'])
                power_current = self.power_orig + float(fresh_metrics['power_increase'])
                wns_current = float(fresh_metrics['worst_slack'])
                sens_ratio = float(fresh_metrics['residual_ratio'])
                obj = None
            else:
                obj = getattr(ind, 'original_objectives', None)
                if obj is None:
                    obj = ind.objectives
                area_current = float(obj[0])
                power_current = float(obj[1])
                wns_current = -float(obj[2])
                sens_ratio = float(obj[3])
            
            # 计算增量百分比
            delta_area = (area_current - self.area_orig) / self.area_orig
            delta_power = (power_current - self.power_orig) / self.power_orig

            # 2. 判定违例
            area_viol = max(0, delta_area - self.delta_area_max)
            power_viol = max(0, delta_power - self.delta_power_max)
            
            # 时序违例判断
            # 只要当前 slack 小于设定的最低门限，即视为违例
            if wns_current < slack_threshold:
                slack_shortage = slack_threshold - wns_current
                # 违例分 = 差值 * 权重
                slack_viol = slack_shortage * self.timing_violation_multiplier
                is_timing_viol = True
                violation_details['timing_violation_count'] += 1
            else:
                # 满足绝对阈值，违例分为0
                slack_viol = 0.0
                is_timing_viol = False

            # 【修复】：敏感性违例逻辑 - 优先使用原始敏感性属性
            raw_sens = getattr(ind, 'raw_sensitivity', None)
            if fresh_metrics is not None:
                sens_ratio = float(fresh_metrics['residual_ratio'])
            elif raw_sens is not None:
                if hasattr(raw_sens, '__len__') and not isinstance(raw_sens, (int, float, np.number)):
                    raw_sens = float(np.asarray(raw_sens).flatten()[0])
                else:
                    raw_sens = float(raw_sens)
                # 判断是归一化值(0~1)还是绝对值
                if raw_sens > 1.0:
                    sens_ratio = raw_sens / (getattr(self, 'sens_orig', 1.0) + 1e-10)
                else:
                    sens_ratio = raw_sens
            else:
                sens_ratio = float(obj[3])
            sens_viol = 0.0 if self.ser_as_objective else max(0, sens_ratio - self.delta_sens_min)

            # 3. 计算加权总分
            total_viol = (self.area_weight * area_viol +
                         self.power_weight * power_viol +
                         self.slack_weight * slack_viol +
                         self.sens_weight * sens_viol)

            violations[i] = total_viol

            # 4. 存储详细信息
            violation_details['area'][i] = area_viol
            violation_details['power'][i] = power_viol
            violation_details['slack'][i] = slack_viol
            violation_details['sens'][i] = sens_viol
            violation_details['has_violation'][i] = total_viol > 1e-7
            
            # 5. 打印警告
            if is_timing_viol and self.current_episode_step % 5 == 0:
                print(f"[WARNING] 时序违例: 个体{i} WNS={wns_current:.4f}ns < 阈值{slack_threshold:.4f}ns (扣分={slack_viol:.2f})")

        return violations, violation_details
    
    def _calculate_front_density(self, pareto_front: List[Individual]) -> float:
        """计算前沿密度"""
        if not pareto_front or len(pareto_front) < 2:
            return 0.0
        
        # 计算解之间的平均距离
        objectives = np.array([ind.objectives for ind in pareto_front])
        distances = []
        
        for i in range(len(objectives)):
            min_dist = float('inf')
            for j in range(len(objectives)):
                if i != j:
                    dist = np.linalg.norm(objectives[i] - objectives[j])
                    if dist < min_dist:
                        min_dist = dist
            if min_dist < float('inf'):
                distances.append(min_dist)
        
        if distances:
            avg_distance = np.mean(distances)
            # 归一化到0-1，假设距离在0-10之间
            density = min(1.0, avg_distance / 10.0)
            return 1.0 - density  # 距离越小，密度越大
        else:
            return 0.0
        
    def _calculate_convergence_rate(self) -> float:
        """计算最近5代的HV改进率"""
        if len(self.hv_history) < 6:
            return 0.0
        
        recent_hv = self.hv_history[-5:]
        improvements = []
        
        for i in range(1, len(recent_hv)):
            if recent_hv[i-1] > 0:
                improvement = (recent_hv[i] - recent_hv[i-1]) / recent_hv[i-1]
                improvements.append(improvement)
        
        if improvements:
            return np.mean(improvements)
        else:
            return 0.0
        
    def _calculate_avg_crowding(self, pareto_front: List[Individual]) -> float:
        """计算平均拥挤距离（归一化）"""
        if not pareto_front:
            return 0.0
        
        crowding_distances = []
        for ind in pareto_front:
            if hasattr(ind, 'crowding_distance'):
                crowding_distances.append(ind.crowding_distance)
        
        if crowding_distances:
            avg = np.mean(crowding_distances)
            # 归一化到0-1，假设拥挤距离在0-100之间
            return min(1.0, avg / 100.0)
        else:
            return 0.0
    
    def _calculate_spacing_metric(self, pareto_front: List[Individual]) -> float:
        """计算前沿间距指标"""
        if not pareto_front or len(pareto_front) < 2:
            return 0.0
        
        objectives = np.array([ind.objectives for ind in pareto_front])
        
        # 计算每个解到最近邻居的距离
        min_distances = []
        for i, point_i in enumerate(objectives):
            min_dist = float('inf')
            for j, point_j in enumerate(objectives):
                if i != j:
                    dist = np.linalg.norm(point_i - point_j)
                    if dist < min_dist:
                        min_dist = dist
            if min_dist < float('inf'):
                min_distances.append(min_dist)
        
        if min_distances:
            avg_min_dist = np.mean(min_distances)
            # 归一化到0-1
            return min(1.0, avg_min_dist / 10.0)
        else:
            return 0.0
    
    def _calculate_extreme_coverage(self, pareto_front: List[Individual]) -> float:
        """计算极值点覆盖比例"""
        if not pareto_front or len(pareto_front) < 3:
            return 0.0
        
        # 这里简化处理，计算在目标空间的四个角落是否有解
        objectives = np.array([ind.objectives for ind in pareto_front])
        
        # 找到每个维度的最小值和最大值
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)
        
        # 检查是否覆盖了边界
        coverage = 0.0
        for i in range(objectives.shape[1]):
            # 检查最小值边界
            min_idx = np.argmin(objectives[:, i])
            min_point = objectives[min_idx]
            # 检查最大值边界
            max_idx = np.argmax(objectives[:, i])
            max_point = objectives[max_idx]
            
            # 如果这些边界点在其他维度上也是极值，则增加覆盖率
            coverage += 0.5 / objectives.shape[1]
        
        return min(1.0, coverage)
    
    def _normalize_value(self, value: float, value_type: str) -> float:
        """归一化不同类型的值到0-1范围"""
        # 这些阈值应该根据实际数据范围调整
        if value_type in self.normalization_ranges:
            min_val, max_val = self.normalization_ranges[value_type]
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
                return np.clip(normalized, 0.0, 1.0)
        
        # 默认归一化
        return np.clip(value, 0.0, 1.0)
    
    def _normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """
        批量归一化目标值
        """
        if len(objectives) == 0:
            return objectives

        normalized = objectives.copy()
        n_objs = objectives.shape[1]

        # 对每个目标分别归一化
        for i in range(n_objs):
            if i == 0:  # 面积
                normalized[:, i] = self._normalize_value(objectives[:, i], 'area')
            elif i == 1:  # 功耗
                normalized[:, i] = self._normalize_value(objectives[:, i], 'power')
            elif i == 2:  # WNS（注意：已经是负值）
                # 先转换为正的WNS
                wns_pos = -objectives[:, i]
                normalized[:, i] = self._normalize_value(wns_pos, 'slack')
            elif i == 3:  # 敏感度
                normalized[:, i] = self._normalize_value(objectives[:, i], 'sens')

        return normalized

    def dominates_epsilon(self, obj_p, obj_q, epsilon=0.0):
        """
        ε-dominance: p ε-支配 q 如果:
            对所有维度：p[i] <= q[i] + epsilon
            且至少一维：p[i] < q[i] - epsilon
        """
        if epsilon < 1e-12:
            return np.all(obj_p <= obj_q) and np.any(obj_p < obj_q)

        cond1 = np.all(obj_p <= obj_q + epsilon)
        cond2 = np.any(obj_p < obj_q - epsilon)
        return cond1 and cond2

    def set_episode_index(self, episode_index: int) -> None:
        """由外部训练循环设置当前Episode编号，用于结果保存目录。"""
        try:
            self.current_episode_index = int(episode_index)
        except Exception:
            self.current_episode_index = 1

    def _extract_pareto_front(self, population, epsilon=0.0):
        """
        从种群中提取Pareto前沿
        支持增量更新（如果启用）以避免每次都重新计算
        """
        if not population:
            return []
        if (
            self.config.get('moea', {}).get('ppa_pareto_only', True)
            and hasattr(self.moea, '_extract_ppa_feasible_pareto')
        ):
            return self.moea._extract_ppa_feasible_pareto(population)
        
        # 检查是否启用增量式Pareto维护
        use_incremental = self.enable_incremental_pareto and hasattr(self.moea, 'current_generation')
        
        if use_incremental:
            # 增量更新Pareto前沿
            try:
                generation = getattr(self.moea, 'current_generation', 0)
                pareto_front = self.pareto_maintainer.update_incremental(population, generation)
                return pareto_front
            except Exception as e:
                print(f"[WARNING] 增量Pareto更新失败，回退到完整计算: {e}")
                # 回退到标准计算
        
        # 获取约束过滤阈值
        thresholds = self.config.get('constraint_thresholds', {})
        pareto_filter_area_viol_max = thresholds.get('pareto_filter_area_viol_max', 0.5)
        pareto_filter_power_viol_max = thresholds.get('pareto_filter_power_viol_max', 0.5)
        pareto_filter_timing_viol_max = thresholds.get('pareto_filter_timing_viol_max', 0.0)
        
        def _is_acceptable_solution(ind):
            """检查解是否可接受（违例在容忍范围内）"""
            if ind.objectives is None or len(ind.objectives) < 4:
                return False
            
            try:
                # 计算面积/功耗增量
                area_ratio = ind.objectives[0]
                power_ratio = ind.objectives[1]
                wns_obj = ind.objectives[2]  # 可能是 -WNS 或 1e6
                
                delta_area = area_ratio - 1.0
                delta_power = power_ratio - 1.0
                
                # 检查面积违例
                if delta_area > pareto_filter_area_viol_max:
                    return False
                
                # 检查功耗违例
                if delta_power > pareto_filter_power_viol_max:
                    return False
                
                # 检查时序违例 (wns_obj > 0 表示 wns < 0)
                if wns_obj > pareto_filter_timing_viol_max:
                    return False
                
                return True
            except Exception:
                return False
        
        # 标准Pareto前沿提取
        pareto_front = []
        for i, ind_i in enumerate(population):
            if ind_i.objectives is None:
                continue
            
            # 【关键修复】：跳过严重违例的解
            if not _is_acceptable_solution(ind_i):
                continue
            
            dominated = False
            for j, ind_j in enumerate(population):
                if i == j or ind_j.objectives is None:
                    continue
                
                # 使用ε-dominance检查是否被支配
                if self.dominates_epsilon(ind_j.objectives, ind_i.objectives, epsilon):
                    dominated = True
                    break
            
            if not dominated:
                pareto_front.append(ind_i)
        
        # 更新缓存
        if use_incremental and len(pareto_front) > 0:
            try:
                generation = getattr(self.moea, 'current_generation', 0)
                self.pareto_maintainer.current_pareto = pareto_front
                self.pareto_maintainer.pareto_objectives = np.array([ind.objectives for ind in pareto_front])
                self.pareto_maintainer.last_update_generation = generation
            except Exception:
                pass
        
        return pareto_front
    
    def _calculate_metrics(self, population, pareto_front):
        """计算各种指标"""
        metrics = {
            'hv': 0.0,
            'igd': float('inf'),
            'spacing': 0.0,
            'coverage': 0.0,
            'feasible_rate': 0.0,
            'max_violation': 0.0,
            'avg_violation': 0.0,
            'crowding_mean': 0.0,
            'crowding_var': 0.0,
            'diversity': 0.0
        }
        
        if not population or len(population) == 0:
            return metrics
        
        # 计算可行率和约束违约
        timing_violations = []
        area_violations = []
        power_violations = []
        sens_violations = []

        # 获取原始值（从模拟器或包装器）
        area_orig = getattr(self.moea, 'area_orig', 100000.0)
        power_orig = getattr(self.moea, 'power_orig', 50.0)
        sens_orig = getattr(self.moea, 'sens_orig', 1e-5)

        # 约束阈值
        delta_area_max = self.constraint_thresholds.get('delta_area_max', 0.2)
        delta_power_max = self.constraint_thresholds.get('delta_power_max', 0.15)
        sens_max = self.constraint_thresholds.get('delta_sens_min', 0.5) * sens_orig  # 注意逻辑

        # 计算可行率和约束违约
        violations = []
        for ind in population:
            if ind.objectives is not None and len(ind.objectives) >= 4:
                # 1. 检查时序违例（slack < 0）
                # ind.objectives[2] = -slack，所以如果>0表示slack<0
                if ind.objectives[2] > 0:
                    timing_violations.append(ind.objectives[2])

                # 2. 检查面积增加是否超过阈值
                # 注意：目标值是相对值还是绝对值？这里假设是绝对值
                area_current = ind.objectives[0]
                delta_area = (area_current - area_orig) / area_orig if area_orig > 0 else 0
                if delta_area > delta_area_max:
                    area_violations.append(delta_area - delta_area_max)

                # 3. 检查功耗增加是否超过阈值
                power_current = ind.objectives[1]
                delta_power = (power_current - power_orig) / power_orig if power_orig > 0 else 0
                if delta_power > delta_power_max:
                    power_violations.append(delta_power - delta_power_max)

                # 4. 检查敏感性是否超过阈值
                sens_current = ind.objectives[3]
                if sens_current > sens_max:
                    sens_violations.append(sens_current - sens_max)
        
        # 计算总违例数（加权和）
        total_violations = []
        total_violations.extend(timing_violations)  # 时序违例
        total_violations.extend([v * 0.1 for v in area_violations])  # 面积违例权重0.1
        total_violations.extend([v * 0.1 for v in power_violations])  # 功耗违例权重0.1
        total_violations.extend([v * 0.5 for v in sens_violations])  # 敏感性违例权重0.5

        # 计算指标
        metrics['feasible_rate'] = 1.0 - len(total_violations) / len(population) if population else 0.0
        metrics['max_violation'] = max(total_violations) if total_violations else 0.0
        metrics['avg_violation'] = np.mean(total_violations) if total_violations else 0.0

        # 记录各类违例的统计
        metrics['timing_violation_rate'] = len(timing_violations) / len(population) if population else 0.0
        metrics['area_violation_rate'] = len(area_violations) / len(population) if population else 0.0
        metrics['power_violation_rate'] = len(power_violations) / len(population) if population else 0.0
        metrics['sens_violation_rate'] = len(sens_violations) / len(population) if population else 0.0
        
        if not pareto_front:
            return metrics
        
        # 提取目标值
        objectives = np.array([ind.objectives for ind in pareto_front])
        
        # 计算超体积
        metrics['hv'] = self._calculate_hypervolume(pareto_front)  # 确保 pareto_front 是 Individual 列表
        
        # 计算反向世代距离（如果有参考集）
        if hasattr(self, 'reference_set') and self.reference_set is not None:
            metrics['igd'] = self._calculate_igd(objectives, self.reference_set)
        
        # 计算间距
        metrics['spacing'] = self._calculate_spacing(objectives)
        
        # 计算覆盖率
        metrics['coverage'] = self._calculate_coverage(objectives)
        
        # 计算拥挤距离统计
        crowding_distances = []
        for ind in pareto_front:
            if hasattr(ind, 'crowding_distance'):
                crowding_distances.append(ind.crowding_distance)
        
        if crowding_distances:
            metrics['crowding_mean'] = np.mean(crowding_distances)
            metrics['crowding_var'] = np.var(crowding_distances)
            metrics['diversity'] = metrics['crowding_mean']  # 用拥挤距离均值作为多样性指标
        
        # 计算PPA统计
        if objectives.shape[1] >= 4:
            metrics['ppa_stats'] = {
                'area_mean': np.mean(objectives[:, 0]),
                'area_std': np.std(objectives[:, 0]),
                'power_mean': np.mean(objectives[:, 1]),
                'power_std': np.std(objectives[:, 1]),
                'slack_mean': np.mean(objectives[:, 2]),
                'slack_std': np.std(objectives[:, 2]),
                'sens_mean': np.mean(objectives[:, 3]),
                'sens_std': np.std(objectives[:, 3])
            }
        
        return metrics
    
    def _calculate_hypervolume(self, pareto_front) -> float:
        """计算超体积"""
        if (
            self.config.get('moea', {}).get('scoring_aligned_hv', True)
            and hasattr(self.moea, '_calculate_scoring_hv_2d')
            and isinstance(pareto_front, list)
        ):
            return self.moea._calculate_scoring_hv_2d(pareto_front)
        # 处理 numpy 数组情况
        if isinstance(pareto_front, np.ndarray):
            if pareto_front.size == 0:
                return 0.0
            objectives = pareto_front
        # 处理列表情况
        elif isinstance(pareto_front, list):
            if not pareto_front:
                return 0.0
            objectives = np.array([ind.objectives for ind in pareto_front])
        else:
            return 0.0

        # 设置参考点（比最大值大10%）
        if len(objectives) == 0:
            return 0.0

        max_vals = np.max(objectives, axis=0) * 1.1

        # 简化计算：计算到参考点的体积
        hv = 0.0
        for point in objectives:
            volume = 1.0
            for i in range(len(point)):
                if point[i] < max_vals[i]:
                    volume *= (max_vals[i] - point[i])
            hv += volume

        return float(hv)
    
    def _calculate_normalized_hypervolume(self, pareto_front):
        """计算归一化的超体积"""
        if not pareto_front:
            return 0.0

        # 提取目标值
        objectives = []
        for ind in pareto_front:
            if ind.objectives is not None and not np.any(np.isnan(ind.objectives)):
                objectives.append(ind.objectives)

        if not objectives:
            return 0.0

        objectives = np.array(objectives)

        # 归一化目标值到[0,1]范围
        # 获取每个目标的最小值和最大值
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)
        ranges = max_vals - min_vals

        # 避免除零
        for i in range(len(ranges)):
            if ranges[i] < 1e-10:
                ranges[i] = 1.0
                max_vals[i] = min_vals[i] + 1.0

        # 归一化
        norm_objectives = (objectives - min_vals) / ranges

        # 设置参考点为[1.1, 1.1, 1.1, 1.1]
        reference_point = np.ones(objectives.shape[1]) * 1.1

        # 计算超体积
        hv = 0.0
        for obj in norm_objectives:
            volume = 1.0
            for i in range(len(obj)):
                if obj[i] < reference_point[i]:
                    volume *= (reference_point[i] - obj[i])
            hv += volume / len(norm_objectives)  # 平均贡献

        return hv

    def _calculate_igd(self, pareto_front: List[Individual], reference_set: np.ndarray) -> float:
        """计算反向世代距离"""
        if not pareto_front or not reference_set:
            return float('inf')
        
        front_objs = np.array([ind.objectives for ind in pareto_front])
        
        # 归一化
        all_objs = np.vstack([front_objs, reference_set])
        min_vals = np.min(all_objs, axis=0)
        max_vals = np.max(all_objs, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        norm_front = (front_objs - min_vals) / range_vals
        norm_ref = (reference_set - min_vals) / range_vals
        
        # 计算IGD
        distances = []
        for ref_point in norm_ref:
            min_dist = np.min(np.linalg.norm(norm_front - ref_point, axis=1))
            distances.append(min_dist)
        
        return np.mean(distances) if distances else float('inf')
    
    def _calculate_spacing(self, pareto_front):
        """计算间距"""
        if len(pareto_front) < 2:
            return 0.0
        
        distances = []
        for i, point_i in enumerate(pareto_front):
            min_dist = float('inf')
            for j, point_j in enumerate(pareto_front):
                if i != j:
                    dist = np.linalg.norm(point_i - point_j)
                    if dist < min_dist:
                        min_dist = dist
            distances.append(min_dist)
        
        d_mean = np.mean(distances)
        spacing = np.sqrt(np.sum((np.array(distances) - d_mean) ** 2) / (len(pareto_front) - 1))
        
        return spacing
    
    def _calculate_coverage(self, pareto_front, grid_size=10):
        """计算网格覆盖率"""
        if len(pareto_front) == 0:
            return 0.0
        
        # 归一化
        min_vals = np.min(pareto_front, axis=0)
        max_vals = np.max(pareto_front, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized = (pareto_front - min_vals) / range_vals
        
        # 创建网格
        grid_cells = set()
        for point in normalized:
            cell = tuple((point * grid_size).astype(int).tolist())
            grid_cells.add(cell)
        
        total_cells = grid_size ** pareto_front.shape[1]
        return len(grid_cells) / total_cells if total_cells > 0 else 0.0
    
    def _calculate_reward(self, population):
        """计算奖励"""
        # 处理空种群
        if population is None or len(population) == 0:
            print("WARNING: 种群为空，返回最小奖励")
            return -1.0
        
        # 处理数组情况
        if isinstance(population, np.ndarray):
            if len(population) == 0:
                return -1.0
            # 将数组转换为 Individual 对象列表
            population = self._convert_array_to_population(population)
        
        # 处理空列表
        if not population or len(population) == 0:
            return -1.0
        
        try:
            reward_params = self.config['reward_params']
            
            def safe_float(value, default=0.0):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return float(default)
    
            # 将字符串转换为浮点数
            alpha = safe_float(reward_params.get('alpha', 1.0))
            beta = safe_float(reward_params.get('beta', 0.1))  # 减小权重
            gamma = safe_float(reward_params.get('gamma', 0.1))  # 减小权重
            eta = safe_float(reward_params.get('eta', 0.01))  # 减小权重
            epsilon = safe_float(reward_params.get('epsilon', 1e-6))
            
            # 计算当前指标
            pareto_front = self._extract_pareto_front(population)
            # 计算当前HV
            current_hv = self._calculate_normalized_hypervolume(pareto_front)

            print(f"[REWARD]: 当前HV={current_hv:.6f}, 历史HV={self.prev_metrics['hv']:.6f}")
            
            # 计算HV增量
            delta_hv = current_hv - self.prev_metrics['hv']
            
            # 计算约束违例
            violations, _ = self._calculate_constraint_violations(population)
            current_violation = np.mean(violations) if len(violations) > 0 else 0.0
            delta_viol = self.prev_metrics['violation'] - current_violation
            
            # 计算多样性指标
            current_diversity = self._calculate_diversity(population)
            delta_div = current_diversity - self.prev_metrics['diversity']
            
            # 避免除零
            hv_denominator = max(self.prev_metrics['hv'], epsilon)
            
            # 计算HV奖励项（归一化）
            if hv_denominator > 0:
                hv_gain = delta_hv / hv_denominator
                # 限制范围
                hv_gain = np.clip(hv_gain, -1.0, 1.0)
            else:
                hv_gain = 0.0
            
            # 计算可行率
            feasible_count = sum(1 for ind in population if hasattr(ind, 'violation') and ind.violation <= 0)
            feasible_rate = feasible_count / len(population) if len(population) > 0 else 0.0
            
            # 计算振荡惩罚（如果动作历史足够长）
            oscillation = 0.0
            if len(self.action_history) >= 2:
                last_action = self.action_history[-1]
                prev_action = self.action_history[-2]
                action_change = np.linalg.norm(last_action - prev_action)
                oscillation = eta * min(action_change, 1.0)  # 限制在0-1之间
            
            # 计算最终奖励（按设计文档公式）
            reward = (
                alpha * hv_gain +                    # HV改进
                beta * feasible_rate -               # 可行率奖励
                gamma * current_violation -          # 约束违例惩罚
                eta * max(0, -delta_div) -           # 多样性下降惩罚
                oscillation                          # 振荡惩罚
            )
            
            # 限制奖励范围在[-1, 1]之间
            reward = np.clip(reward, -1.0, 1.0)
            
            # 更新历史指标
            self.prev_metrics = {
                'hv': current_hv,
                'igd': 0.0,  # 暂时设为0
                'diversity': current_diversity,
                'violation': current_violation
            }
            
            # 记录历史
            self.hv_history.append(current_hv)
            if 'igd' in self.prev_metrics:
                self.igd_history.append(self.prev_metrics.get('igd', 0.0))

            # 动态更新归一化范围
            self._update_hv_igd_normalization(current_hv, self.prev_metrics.get('igd', 0.0))

            print(f"INFO: 奖励计算: hv={current_hv:.4f}, delta_hv={delta_hv:.4f}, "
                  f"feasible={feasible_rate:.2f}, violation={current_violation:.4f}, "
                  f"reward={reward:.4f}")
            
            return float(reward)
            
        except Exception as e:
            print(f"ERROR: 计算奖励时出错: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        
    def _calculate_diversity(self, population):
        """计算种群多样性"""
        if not population or len(population) < 2:
            return 0.0

        # 提取有效目标值
        objectives = []
        for ind in population:
            if ind.objectives is not None and not np.any(np.isnan(ind.objectives)):
                objectives.append(ind.objectives)

        if len(objectives) < 2:
            return 0.0

        objectives = np.array(objectives)

        # 归一化
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0

        norm_objectives = (objectives - min_vals) / ranges

        # 计算平均距离
        total_distance = 0.0
        count = 0

        for i in range(len(norm_objectives)):
            for j in range(i+1, len(norm_objectives)):
                dist = np.linalg.norm(norm_objectives[i] - norm_objectives[j])
                total_distance += dist
                count += 1

        if count > 0:
            return total_distance / count
        else:
            return 0.0

    # ============ 跨Episode长期学习辅助方法 ============
    def _get_best_action_from_history(self) -> Optional[Dict]:
        """从历史中获取最优动作参数"""
        if not hasattr(self, 'parameter_tuning_history') or not self.parameter_tuning_history:
            return None
        return self.global_best_params

    def _get_expert_context(self) -> Dict:
        """
        获取专家上下文信息，用于增强状态表示

        返回包含以下信息的字典：
        - 全局最优HV
        - Episode历史表现
        - 收敛趋势
        """
        context = {
            'global_best_hv': self.global_best_hv,
            'recent_hv_trend': self.episode_hv_record[-10:] if self.episode_hv_record else [],
            'total_episodes': len(self.episode_hv_record),
            'best_params': self.global_best_params
        }
        return context

    def get_state_with_expert_context(self, base_state: np.ndarray) -> np.ndarray:
        """
        将专家上下文信息融入状态表示

        这使得SAC能够"看到"跨Episode的学习进度
        """
        if self.state_schema == 'compact_score_aligned_v1':
            state = np.asarray(base_state, dtype=np.float32).flatten()
            if state.size < self.state_dim:
                state = np.pad(state, (0, self.state_dim - state.size))
            return state[:self.state_dim]

        context = self._get_expert_context()

        # 构建额外的状态特征
        expert_features = np.zeros(10, dtype=np.float32)

        # 1-3: 全局最优HV相关
        expert_features[0] = np.clip(self.global_best_hv / 100, 0, 1)  # 归一化HV
        expert_features[1] = np.clip(context['total_episodes'] / 50, 0, 1)  # 训练进度

        # 4-6: 最近HV趋势（相对于全局最优）
        if context['recent_hv_trend']:
            recent_avg = np.mean(context['recent_hv_trend'])
            expert_features[2] = np.clip(recent_avg / max(self.global_best_hv, 0.001), 0, 1)
        else:
            expert_features[2] = 0.0

        # 7-10: 全局最优参数的编码（简化表示）
        if self.global_best_params is not None:
            best_action = self.global_best_params.get('action', [])
            if isinstance(best_action, np.ndarray) and len(best_action) >= 4:
                expert_features[3] = best_action[0]  # epsilon
                expert_features[4] = best_action[2]  # delta
                expert_features[5] = best_action[4]  # pm
                expert_features[6] = best_action[5]  # pc
                # 添加动作历史的变化趋势
                if len(self.action_history) >= 2:
                    expert_features[7] = np.clip(
                        np.mean(self.action_history[-1] - self.action_history[-2]) / 0.1, -1, 1
                    )

        # 拼接基础状态和专家上下文
        enhanced_state = np.concatenate([base_state, expert_features])

        # 确保维度正确
        expected_dim = self.state_dim  # = 64
        if len(enhanced_state) != expected_dim:
            if len(enhanced_state) < expected_dim:
                padding = np.zeros(expected_dim - len(enhanced_state), dtype=np.float32)
                enhanced_state = np.concatenate([enhanced_state, padding])
            else:
                enhanced_state = enhanced_state[:expected_dim]

        return enhanced_state
    
    def _convert_array_to_population(self, array):
        """将基因数组转换为 Individual 对象列表"""
        if isinstance(array, np.ndarray) and array.ndim == 2:
            population = []
            for i, gene in enumerate(array):
                ind = Individual(gene, None)
                # 如果需要评估，可以在这里评估
                population.append(ind)
            return population
        elif isinstance(array, list):
            return array
        else:
            return []
    
    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            print(f"Step: {len(self.action_history)}")
            if self.action_history:
                last_action = self.action_history[-1]
                if len(last_action) == 12:
                    print(f"SPEA2 Action: ε={last_action[0]:.4f}, τ={last_action[1]:.4f}, "
                          f"δ={last_action[2]:.4f}, β={last_action[3]:.2f}, "
                          f"pm={last_action[4]:.4f}, pc={last_action[5]:.4f}")
                    print(f"NSGA2 Action: ε={last_action[6]:.4f}, τ={last_action[7]:.4f}, "
                          f"δ={last_action[8]:.4f}, β={last_action[9]:.2f}, "
                          f"pm={last_action[10]:.4f}, pc={last_action[11]:.4f}")
                else:
                    print(f"Action: ε={last_action[0]:.4f}, τ={last_action[1]:.4f}, "
                          f"δ={last_action[2]:.4f}, β={last_action[3]:.2f}, "
                          f"pm={last_action[4]:.4f}, pc={last_action[5]:.4f}")
            
            print(f"Metrics: HV={self.prev_metrics['hv']:.4f}, "
                  f"IGD={self.prev_metrics['igd']:.4f}, "
                  f"Diversity={self.prev_metrics['diversity']:.4f}, "
                  f"Violation={self.prev_metrics['violation']:.4f}")
    
