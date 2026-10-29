# algorithms/spea2_parallel.py
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from multiprocessing import Pool, cpu_count
from .base import BaseOptimizer
from .core_data import Individual
from .core_data import dominates
from mo_optimizer.parallel_evaluator import ParallelEvaluator
from utils.parallel_utils import ParallelExecutor, parallel_compute_fitness, parallel_compute_strength, parallel_create_offspring, parallel_select_parents

def dominates_epsilon(obj_p, obj_q, eps):
    """
    ε-dominance: 
        对所有维度：obj_p[i] <= obj_q[i] + eps
        且至少一维严格更好（小于 q[i] - eps）
    """
    # 检查输入是否为数组
    if obj_p is None or obj_q is None:
        return False
    
    # 转换为数组
    if not isinstance(obj_p, np.ndarray):
        obj_p = np.array(obj_p)
    if not isinstance(obj_q, np.ndarray):
        obj_q = np.array(obj_q)
    
    # 检查形状
    if obj_p.shape != obj_q.shape:
        return False
    
    # 检查是否包含 NaN 或 Inf
    if np.any(np.isnan(obj_p)) or np.any(np.isinf(obj_p)) or np.any(np.isnan(obj_q)) or np.any(np.isinf(obj_q)):
        return False
    
    # 如果包含NaN，直接返回False
    if np.isnan(obj_p).any() or np.isnan(obj_q).any():
        return False

    if eps < 1e-12:
        return np.all(obj_p <= obj_q) and np.any(obj_p < obj_q)

    cond1 = np.all(obj_p <= obj_q + eps)
    cond2 = np.any(obj_p < obj_q - eps)
    return cond1 and cond2


def compute_domination(args):
    """
    并行支配关系计算（用于 fast_non_dominated_sort）
    """
    i, p, population, eps_dom = args
    p.domination_count = 0
    p.dominated_solutions = set()

    for j, q in enumerate(population):
        if i == j:
            continue
        if dominates_epsilon(p.objectives, q.objectives, eps_dom):
            p.dominated_solutions.add(q)
        elif dominates_epsilon(q.objectives, p.objectives, eps_dom):
            p.domination_count += 1
    return p

def calculate_strength(args):
    """
    计算 SPEA2 Strength 值（考虑ε-dominance）
    """
    i, combined_pop, eps = args
    N = len(combined_pop)
    strength = 0
    for j in range(N):
        if i == j:
            continue
        if dominates_epsilon(combined_pop[i].objectives,
                             combined_pop[j].objectives,
                             eps):
            strength += 1
    return strength


def calculate_raw_fitness(args):
    """
    计算 SPEA2 Raw Fitness 值（考虑ε-dominance）
    """
    i, combined_pop, strengths, eps = args
    N = len(combined_pop)
    raw_fitness = 0
    for j in range(N):
        if i == j:
            continue
        if dominates_epsilon(combined_pop[j].objectives,
                             combined_pop[i].objectives,
                             eps):
            raw_fitness += strengths[j]
    return raw_fitness

def calculate_density(args):
    """
    SPEA-II 密度计算（使用动态多样性阈值 δ）
    """
    i, distances, k, delta_div = args   # delta_div 动态参数
    sorted_dist = np.sort(distances[i])
    k_index = min(k, len(sorted_dist) - 1)
    dist_k = sorted_dist[k_index]
    # RL 动态调节的 SPEA2 密度公式
    return delta_div / (dist_k + 1e-10)

class SPEA2Parallel(BaseOptimizer):
    """并行SPEA2算法实现，支持SAC参数动态调整和差分进化(DE)算子切换"""
    
    def __init__(self, simulator, pop_size: int, max_generations: int, 
                 archive_size: int, num_processes: int = None,
                 # SAC可调参数初始值
                 epsilon_dom: float = 0.0,
                 accept_threshold: float = 0.0,
                 delta_div: float = 1.0,
                 beta_penalty: float = 0.0,
                 pm: float = 0.01,
                 pc: float = 0.9,
                 # ========== 大规模优化参数 ==========
                 use_de_operator: bool = False,  # 是否使用DE算子替代GA
                 F_init: float = 0.5,           # DE初始缩放因子
                 F_end: float = 0.3,            # DE最终缩放因子
                 CR: float = 0.1,               # DE交叉概率
                 use_variable_grouping: bool = False,  # 是否使用变量分组
                 group_size: int = 10,          # 分组大小
                 local_search_rate: float = 0.3,  # 局部搜索比例
                 # 偏好采样比例（用于膝点/关键区段聚焦）
                 pref_sampling_ratio: float = 0.3,
                 config: dict = None):
        """
        初始化SPEA2算法

        参数:
        ---------
        use_de_operator : bool
            是否使用差分进化(DE)算子替代传统遗传算法(GA)
            - False: 使用SBX交叉+多项式变异（传统GA）
            - True: 使用DE变异+二项式交叉（适合大规模问题）
        F_init, F_end : float
            DE缩放因子的初始值和最终值（自适应调整）
        CR : float
            DE交叉概率
        use_variable_grouping : bool
            是否使用变量分组策略
        """
        super().__init__(simulator, pop_size, max_generations)
        self.archive_size = archive_size
        self.archive: List[Individual] = []
        
        self.evaluator = ParallelEvaluator(simulator, num_processes, config=config)
        
        # 设置并行进程数
        if num_processes:
            self.num_processes = num_processes
        else:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            # 保留2个核心给系统和其他任务
            self.num_processes = max(1, cpu_count - 1)
        
        # 初始化并行执行器
        self.parallel_executor = ParallelExecutor(n_jobs=self.num_processes)
        
        if not hasattr(self, 'evaluator'):
            self.evaluator = self._create_simple_evaluator()
        
        # -------------------------
        # SAC可调参数（由外部SAC算法动态调整）
        # -------------------------
        self.epsilon_dom = epsilon_dom        # ε: epsilon-dominance tolerance
        self.accept_threshold = accept_threshold  # τ: accept threshold
        self.delta_div = delta_div           # δ: diversity scaling factor
        self.beta_penalty = beta_penalty     # β: penalty for constraint violations
        self.pm = pm                         # pm: mutation probability
        self.pc = pc                         # pc: crossover probability
        self.pref_sampling_ratio = pref_sampling_ratio  # p: preference sampling ratio
        
        # ========== 大规模优化参数 ==========
        self.use_de_operator = use_de_operator  # 是否使用DE算子
        self.F_init = F_init
        self.F_end = F_end
        self.CR = CR
        self.use_variable_grouping = use_variable_grouping and getattr(simulator, 'K', 100) > 50
        self.group_size = group_size
        self.local_search_rate = local_search_rate
        
        # 初始化DE算子
        if self.use_de_operator:
            from .large_scale_optimizer import DifferentialEvolutionOperator, VariableGrouping
            self.de_operator = DifferentialEvolutionOperator(F=F_init, CR=CR)
            if self.use_variable_grouping:
                self.grouping = VariableGrouping(getattr(simulator, 'K', 100), group_size)
                print(f"[SPEA2]: 启用DE算子 + 变量分组，共{self.grouping.num_groups}组")
            else:
                print(f"[SPEA2]: 启用DE算子，F=[{F_init}, {F_end}], CR={CR}")
        
        # 用于记录参数变化历史（供SAC状态空间使用）
        self.param_history = {
            'epsilon_dom': [],
            'accept_threshold': [],
            'delta_div': [],
            'beta_penalty': [],
            'pm': [],
            'pc': [],
            'pref_sampling_ratio': []
        }
        
        # -------------------------
        # 统计指标（供SAC奖励函数使用）
        # -------------------------
        self.generation_stats = {
            'hv': [],                # 超体积
            'igd': [],               # 反向世代距离
            'crowding_mean': [],     # 拥挤距离均值
            'crowding_var': [],      # 拥挤距离方差
            'feasible_rate': [],     # 可行率
            'max_violation': [],     # 最大约束违约
            'avg_violation': [],     # 平均约束违约
            'coverage_ratio': [],    # 参考网格覆盖率
            'ppa_metrics': []        # PPA统计（面积、功耗、时序）
        }
        
        # -------------------------
        # 存储模拟器的属性
        # -------------------------
        try:
            sim_K = simulator.K
        except Exception:
            sim_K = getattr(simulator, 'K', None)
            if sim_K is None:
                sim_K = 1

        self.error_rates = getattr(simulator, 'error_rates', np.ones(sim_K))
        self.is_critical = getattr(simulator, 'is_critical', np.zeros(sim_K))
        self.slack = getattr(simulator, 'slack', np.ones(sim_K))
        
        # 计算归一化 slack
        min_slack = np.min(self.slack) if len(self.slack) > 0 else 0.0
        max_slack = np.max(self.slack) if len(self.slack) > 0 else 1.0
        slack_range = max_slack - min_slack if max_slack > min_slack else 1.0
        self.normalized_slack = (self.slack - min_slack) / slack_range
        
        # 计算权重（考虑关键性和松弛度）
        self.reinforcement_priority = self.error_rates * (1 + 0.2 * self.is_critical) * (1 - self.normalized_slack)
        
        if np.sum(self.error_rates) > 0:
            self.error_weights = self.error_rates / np.sum(self.error_rates)
        else:
            self.error_weights = np.ones_like(self.error_rates) / max(1, len(self.error_rates))
            
        if np.sum(self.reinforcement_priority) > 0:
            self.combined_weights = self.reinforcement_priority / np.sum(self.reinforcement_priority)
        else:
            self.combined_weights = np.ones_like(self.reinforcement_priority) / max(1, len(self.reinforcement_priority))
        
        # -------------------------
        # 初始化统一变异策略（传入动态参数）
        # -------------------------
        from utils.mutation_strategy import OptimizedMutationStrategy
        try:
            self.unified_mutator = OptimizedMutationStrategy(
                error_rates=self.error_rates,
                is_critical=self.is_critical,
                slack=self.slack,
                pm=self.pm,
                pc=self.pc,
                block_size=max(5, simulator.K // 50),  # 根据基因长度自动调整区块大小
                elite_preservation_rate=0.7,
                local_search_rate=0.3,
                eta_m=20.0
            )
        except TypeError:
            # 后备方案：如果不接受新参数
            from utils.mutation_strategy import UnifiedMutationStrategy
            self.unified_mutator = UnifiedMutationStrategy(
                error_rates=self.error_rates,
                is_critical=self.is_critical,
                slack=self.slack
            )
        
        # 初始化参数边界（收紧 epsilon 和 beta upper bound，避免策略探索到过极端值）
        self.param_bounds = {
            'epsilon_dom': (0.001, 0.05),
            'accept_threshold': (0.01, 0.5),
            'delta_div': (0.01, 0.3),
            'beta_penalty': (1.0, 50.0),
            'pm': (0.001, 0.3),
            'pc': (0.6, 1.0),
            'pref_sampling_ratio': (0.0, 1.0)
        }
        
        # 确保参数在合理范围内
        self._clip_params()
    
    def _create_simple_evaluator(self):
        """创建简单的评估器（如果没有提供）"""
        class SimpleEvaluator:
            def __init__(self, simulator):
                self.simulator = simulator
            
            def evaluate_population(self, population):
                for ind in population:
                    if ind.objectives is None:
                        # 简单的评估函数
                        area = np.sum(ind.gene) * 10
                        power = np.sum(ind.gene) * 0.5
                        slack = 0.1 - np.sum(ind.gene * getattr(self.simulator, 'is_critical', np.zeros_like(ind.gene))) * 0.01
                        sens = np.sum(getattr(self.simulator, 'error_rates', np.ones_like(ind.gene)) * (1 - ind.gene))
                        ind.objectives = np.array([area, power, -slack, sens])
                        ind.violation = 0.0
                return population
        
        return SimpleEvaluator(self.simulator)
    
    def update_parameters(self, params: Dict[str, float]):
        """
        更新算法参数（由SAC调用）
        
        参数:
        ----------
        params : Dict[str, float]
            参数字典，包含要更新的参数值
        """
        if 'epsilon_dom' in params:
            self.epsilon_dom = params['epsilon_dom']
        if 'accept_threshold' in params:
            self.accept_threshold = params['accept_threshold']
        if 'delta_div' in params:
            self.delta_div = params['delta_div']
        if 'beta_penalty' in params:
            self.beta_penalty = params['beta_penalty']
        if 'pm' in params:
            self.pm = params['pm']
            # 同时更新变异器的pm
            if hasattr(self.unified_mutator, 'pm'):
                self.unified_mutator.pm = self.pm
        if 'pc' in params:
            self.pc = params['pc']
            # 同时更新变异器的pc（如果有）
            if hasattr(self.unified_mutator, 'pc'):
                self.unified_mutator.pc = self.pc
        if 'pref_sampling_ratio' in params:
            self.pref_sampling_ratio = params['pref_sampling_ratio']
        
        # 确保参数在合理范围内
        self._clip_params()
        
        # 记录参数历史
        self._record_param_history()
    
    def get_current_parameters(self) -> Dict[str, float]:
        """
        获取当前参数值
        
        返回:
        ----------
        Dict[str, float]: 当前参数值字典
        """
        return {
            'epsilon_dom': self.epsilon_dom,
            'accept_threshold': self.accept_threshold,
            'delta_div': self.delta_div,
            'beta_penalty': self.beta_penalty,
            'pm': self.pm,
            'pc': self.pc,
            'pref_sampling_ratio': self.pref_sampling_ratio
        }
    
    def get_statistics(self, generation: int = None) -> Dict[str, Any]:
        """
        获取算法统计信息（供SAC状态空间使用）
        
        参数:
        ----------
        generation : int, optional
            指定代数的统计信息，默认返回最新一代
        
        返回:
        ----------
        Dict[str, Any]: 统计信息字典
        """
        if generation is None:
            # 返回最新一代的统计信息
            idx = -1
        else:
            idx = min(generation, len(self.generation_stats['hv'])-1)
        
        stats = {}
        for key in self.generation_stats:
            if len(self.generation_stats[key]) > 0:
                stats[key] = self.generation_stats[key][idx] if idx >= 0 else self.generation_stats[key][-1]
            else:
                stats[key] = 0.0
        
        # 添加参数历史信息
        stats['param_history'] = {k: v[-10:] for k, v in self.param_history.items() if len(v) > 0}
        
        return stats
    
    def reset(self):
        """重置SPEA2状态"""
        self.archive = []
        self.population = []
        self.generation_stats = {
            'hv': [],
            'igd': [],
            'crowding_mean': [],
            'crowding_var': [],
            'feasible_rate': [],
            'max_violation': [],
            'avg_violation': [],
            'coverage_ratio': [],
            'ppa_metrics': []
        }
        print("[SPEA2]: 重置完成")
    
    def _clip_params(self):
        """
        确保参数在合理范围内
        """
        # 保证参数在合理范围内
        self.epsilon_dom = float(np.clip(self.epsilon_dom, *self.param_bounds['epsilon_dom']))
        
        # accept_threshold 应为非负浮点数
        self.accept_threshold = float(np.clip(self.accept_threshold, 
                                             *self.param_bounds['accept_threshold']))
        
        self.delta_div = float(np.clip(self.delta_div, *self.param_bounds['delta_div']))
        self.beta_penalty = float(np.clip(self.beta_penalty, *self.param_bounds['beta_penalty']))
        self.pm = float(np.clip(self.pm, *self.param_bounds['pm']))
        self.pc = float(np.clip(self.pc, *self.param_bounds['pc']))
        self.pref_sampling_ratio = float(np.clip(self.pref_sampling_ratio, 
                                                *self.param_bounds['pref_sampling_ratio']))
    
    def _record_param_history(self):
        """
        记录参数变化历史
        """
        self.param_history['epsilon_dom'].append(self.epsilon_dom)
        self.param_history['accept_threshold'].append(self.accept_threshold)
        self.param_history['delta_div'].append(self.delta_div)
        self.param_history['beta_penalty'].append(self.beta_penalty)
        self.param_history['pm'].append(self.pm)
        self.param_history['pc'].append(self.pc)
        self.param_history['pref_sampling_ratio'].append(self.pref_sampling_ratio)
    
    def _record_generation_stats(self, population: List[Individual]):
        """
        记录当前代的统计信息
        """
        if not population:
            return
            
        # 计算可行率
        feasible_count = sum(1 for ind in population if hasattr(ind, 'violation') and ind.violation <= 0)
        feasible_rate = feasible_count / len(population) if len(population) > 0 else 0.0
        
        # 计算约束违约
        violations = []
        for ind in population:
            if hasattr(ind, 'violation'):
                violations.append(ind.violation)
        
        max_violation = max(violations) if violations else 0.0
        avg_violation = np.mean(violations) if violations else 0.0
        
        # 计算拥挤距离统计
        if len(population) > 0 and hasattr(population[0], 'crowding_distance'):
            crowding_distances = [ind.crowding_distance for ind in population 
                                 if hasattr(ind, 'crowding_distance')]
            if crowding_distances:
                crowding_mean = np.mean(crowding_distances)
                crowding_var = np.var(crowding_distances)
            else:
                crowding_mean, crowding_var = 0.0, 0.0
        else:
            crowding_mean, crowding_var = 0.0, 0.0
        
        # 记录统计信息
        self.generation_stats['feasible_rate'].append(feasible_rate)
        self.generation_stats['max_violation'].append(max_violation)
        self.generation_stats['avg_violation'].append(avg_violation)
        self.generation_stats['crowding_mean'].append(crowding_mean)
        self.generation_stats['crowding_var'].append(crowding_var)
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        快速非支配排序（矩阵化向量版 + Deb约束支配法则）
        与 NSGA-II 保持逻辑统一，确保 Pareto 前沿中只有可流片的物理可行解。
        """
        valid_pop = [ind for ind in population if ind.objectives is not None]
        if not valid_pop:
            return []

        N = len(valid_pop)
        objs = np.array([ind.objectives for ind in valid_pop])
        violations = np.array([getattr(ind, 'violation', max(0, obj[2])) for ind, obj in zip(valid_pop, objs)])
        
        diff = objs[:, np.newaxis, :] - objs[np.newaxis, :, :]
        with np.errstate(invalid='ignore'):
            cond1 = np.all(diff <= self.epsilon_dom, axis=2)
            cond2 = np.any(diff < -self.epsilon_dom, axis=2)
        pareto_dominates = cond1 & cond2
        
        V_i = violations[:, np.newaxis]
        V_j = violations[np.newaxis, :]
        
        rule1 = (V_i == 0) & (V_j > 0)
        rule2 = (V_i > 0) & (V_j > 0) & (V_i < V_j)
        rule3 = (V_i == V_j) & pareto_dominates
        
        dominates_matrix = rule1 | rule2 | rule3
        np.fill_diagonal(dominates_matrix, False)
        
        domination_counts = np.sum(dominates_matrix, axis=0)
        
        for i in range(N):
            p = valid_pop[i]
            p.domination_count = int(domination_counts[i])
            dominated_indices = np.where(dominates_matrix[i, :])[0]
            p.dominated_solutions = [valid_pop[idx] for idx in dominated_indices]

        fronts = [[]]
        for p in valid_pop:
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        if not fronts[0]:
            return []

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
            
        return fronts
    
    def calculate_fitness(self, population: List[Individual]):
        """
        改进的适应度计算（矩阵化向量版 + Deb约束支配法则）
        """
        if not population:
            return
        
        # 确保每个个体都有目标值
        valid_population = [ind for ind in population if ind.objectives is not None]
        if not valid_population:
            return
        
        N = len(valid_population)
        if N == 1:
            valid_population[0].fitness = 0.0
            return
            
        # 1. 提取目标值和时序违例度 (Constraint Violation)
        objs_list = [ind.objectives for ind in valid_population]
        objs = np.array(objs_list)
        
        # 防御性检查：确保objs是2D数组
        if objs.ndim != 2:
            print(f"[SPEA2-WARN] objectives维度错误: ndim={objs.ndim}, shape={objs.shape}")
            fixed_objs = []
            for obj in objs_list:
                if isinstance(obj, np.ndarray) and obj.ndim == 1 and len(obj) >= 4:
                    fixed_objs.append(obj[:4])
                elif isinstance(obj, (list, tuple)) and len(obj) >= 4:
                    fixed_objs.append(list(obj)[:4])
                else:
                    fixed_objs.append([1e6, 1e6, 0.0, 1.0])
            objs = np.array(fixed_objs, dtype=np.float64)
        
        # 确保是浮点数类型
        if objs.dtype != np.float64 and objs.dtype != np.float32:
            objs = objs.astype(np.float64)
        
        # 优先读取个体的 violation 属性。如果没有，默认第三个目标(index 2, slack penalty)为违例度
        violations = np.array([getattr(ind, 'violation', max(0, obj[2])) for ind, obj in zip(valid_population, objs)])
        
        # 2. 矩阵化计算支配关系 (Deb约束支配法则)
        diff = objs[:, np.newaxis, :] - objs[np.newaxis, :, :]
        with np.errstate(invalid='ignore'):
            cond1 = np.all(diff <= self.epsilon_dom, axis=2)
            cond2 = np.any(diff < -self.epsilon_dom, axis=2)
        pareto_dominates = cond1 & cond2
        
        V_i = violations[:, np.newaxis]
        V_j = violations[np.newaxis, :]
        
        # 规则1: i 不违例且 j 违例，则 i 绝对支配 j
        rule1 = (V_i == 0) & (V_j > 0)
        # 规则2: i 和 j 都违例，但 i 的违例程度更轻，则 i 支配 j
        rule2 = (V_i > 0) & (V_j > 0) & (V_i < V_j)
        # 规则3: i 和 j 违例程度相同，采用正常的 Pareto 支配
        rule3 = (V_i == V_j) & pareto_dominates
        
        dominates_matrix = rule1 | rule2 | rule3
        np.fill_diagonal(dominates_matrix, False)
        
        # 3. 计算 SPEA2 算法的核心：Strength (S) 和 Raw Fitness (R)
        # S[i] = 第 i 个个体支配的其他个体数量
        S = np.sum(dominates_matrix, axis=1)
        
        # R[i] = 支配第 i 个个体的所有个体的 S 值之和
        R = np.zeros(N)
        for i in range(N):
            # dominates_matrix[:, i] 为 True 表示行索引对应的个体支配了个体 i
            R[i] = np.sum(S[dominates_matrix[:, i]])
            
        # 4. 计算 Density (D)
        min_objs = np.min(objs, axis=0)
        max_objs = np.max(objs, axis=0)
        range_objs = max_objs - min_objs
        range_objs[range_objs < 1e-10] = 1.0
        normalized_objs = (objs - min_objs) / range_objs
        
        # 使用矩阵计算两两之间的欧氏距离
        dist_matrix = np.linalg.norm(normalized_objs[:, np.newaxis, :] - normalized_objs[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dist_matrix, float('inf'))  # 忽略自己到自己的距离
        
        # 对距离矩阵按行排序，找到第 k 近的距离
        sorted_dist_matrix = np.sort(dist_matrix, axis=1)
        k = int(np.sqrt(N)) + 1
        k_idx = max(0, min(k - 1, N - 2))  # 确保索引不越界
        kth_dists = sorted_dist_matrix[:, k_idx]
        
        # 处理距离极小的情况防止除零
        kth_dists[np.isnan(kth_dists) | (kth_dists < 1e-10)] = 1.0
        densities = self.delta_div / (kth_dists + 1e-10)
        
        # 5. 组合最终适应度
        for i in range(N):
            valid_population[i].fitness = R[i] + densities[i]
            
            # 【终极惩罚】: 如果个体有约束违反，增加巨大的 Beta 惩罚项
            # 在 SPEA2 中，fitness 越小越好 (<1 为非支配解)
            # 违例解在 R[i] 阶段就已经被可行解支配 (导致 R>=1)，这里再加上 beta 惩罚，使其彻底无法进入下一代
            if violations[i] > 0 and not np.isnan(violations[i]):
                valid_population[i].fitness += self.beta_penalty * violations[i]

    def _parallel_calculate_strength(self, i: int, population: List[Individual], eps: float) -> float:
        """并行计算强度"""
        strength = 0
        for j in range(len(population)):
            if i == j:
                continue
            if dominates_epsilon(population[i].objectives, population[j].objectives, eps):
                strength += 1
        return strength
    
    def _parallel_calculate_raw_fitness(self, i: int, population: List[Individual], 
                                       strengths: np.ndarray, eps: float) -> float:
        """并行计算原始适应度"""
        raw_fitness = 0
        for j in range(len(population)):
            if i == j:
                continue
            if dominates_epsilon(population[j].objectives, population[i].objectives, eps):
                raw_fitness += strengths[j]
        return raw_fitness
    
    def _parallel_calculate_density(self, i: int, distances: np.ndarray, k: int, delta_div: float) -> float:
        """并行计算密度"""
        sorted_dist = np.sort(distances[i])
        kth_dist = sorted_dist[min(k, len(sorted_dist)-1)]
        if np.isnan(kth_dist) or kth_dist == 0:
            kth_dist = 1.0
        return delta_div / (kth_dist + 1e-10)
    
    def _parallel_calculate_distances(self, i: int, normalized_objs: np.ndarray, N: int) -> np.ndarray:
        """并行计算距离行"""
        row = np.zeros(N)
        for j in range(N):
            if i == j:
                row[j] = float('inf')
            else:
                dist = np.linalg.norm(normalized_objs[i] - normalized_objs[j])
                if np.isnan(dist) or np.isinf(dist):
                    row[j] = 1.0
                else:
                    row[j] = dist
        return row
    
    def environmental_selection(self, population: List[Individual]) -> List[Individual]:
        """
        改进的环境选择（考虑动态参数）
        """
        # 过滤无效个体
        valid_population = [ind for ind in population if ind.objectives is not None]
        if not valid_population:
            # 如果种群为空，生成新的随机个体
            return [Individual(np.random.randint(0, 2, self.simulator.K), None) 
                    for _ in range(self.pop_size)]

        # 计算适应度（考虑动态参数）
        self.calculate_fitness(valid_population)

        # 分离当前种群和archive
        current_population = [ind for ind in valid_population if ind not in self.archive]
        combined_pop = current_population + self.archive

        if not combined_pop:
            return valid_population[:min(len(valid_population), self.pop_size)]

        # 重新计算适应度
        self.calculate_fitness(combined_pop)

        # 【关键修复】：优先选择同时满足面积/功耗和敏感性约束的解
        # 第一优先级：面积/功耗可行
        # 第二优先级：敏感性可行（sens_ratio <= delta_sens_min）
        
        # 【调试】：统计敏感性可行性
        total_sens_feasible = sum(1 for ind in combined_pop if getattr(ind, 'sensitivity_feasible', True))
        total_ap_feasible = sum(1 for ind in combined_pop if getattr(ind, 'area_power_feasible', False))
        both_feasible = sum(1 for ind in combined_pop if getattr(ind, 'area_power_feasible', False) and getattr(ind, 'sensitivity_feasible', True))
        print(f"[SPEA2环境选择] 总解={len(combined_pop)}, 面积/功耗可行={total_ap_feasible}, 敏感性可行={total_sens_feasible}, 两者都可行={both_feasible}")
        
        feasible_pop = [ind for ind in combined_pop 
                      if getattr(ind, 'area_power_feasible', False) 
                      and getattr(ind, 'sensitivity_feasible', True)]
        
        # 如果敏感性可行解不足，退而求其次选择敏感性可行的解
        if len(feasible_pop) >= self.pop_size * 0.3:
            working_pop = feasible_pop
            print(f"[SPEA2] 环境选择: {len(feasible_pop)}个完全可行解(面积/功耗/敏感性)")
        else:
            # 优先敏感性可行解
            sens_feasible = [ind for ind in combined_pop if getattr(ind, 'sensitivity_feasible', True)]
            if len(sens_feasible) >= self.pop_size * 0.3:
                working_pop = sens_feasible
                print(f"[SPEA2] 环境选择: {len(sens_feasible)}个敏感性可行解")
            else:
                sens_infeasible = [ind for ind in combined_pop if not getattr(ind, 'sensitivity_feasible', True)]
                print(f"[SPEA2] 警告: 只有{len(sens_feasible)}个敏感性可行解，{len(sens_infeasible)}个违例解")
                # 按敏感性比例排序
                sens_infeasible.sort(key=lambda x: x.objectives[3] if x.objectives is not None and len(x.objectives) >= 4 else float('inf'))
                working_pop = sens_feasible + sens_infeasible[:self.pop_size - len(sens_feasible)]

        # 1. 获取非支配个体（适应度<1+τ，使用动态τ参数）
        non_dominated_threshold = 1.0 + self.accept_threshold
        non_dominated = [ind for ind in working_pop if ind.fitness < non_dominated_threshold]

        # 2. 如果没有非支配个体，选择适应度最好的个体
        if not non_dominated:
            working_pop.sort(key=lambda x: x.fitness)
            non_dominated = working_pop[:min(self.archive_size, len(working_pop))]

        # 3. 更新archive（考虑偏好采样）
        if len(non_dominated) <= self.archive_size:
            # 如果数量足够，可以直接使用
            if self.pref_sampling_ratio > 0:
                # 执行偏好采样（聚焦膝点/关键区段）
                self.archive = self._preference_sampling(non_dominated, self.archive_size)
            else:
                self.archive = non_dominated[:]
        else:
            # 使用截断聚类保持多样性
            if self.pref_sampling_ratio > 0:
                # 先进行偏好采样
                preferred = self._preference_sampling(non_dominated, 
                                                     int(self.archive_size * self.pref_sampling_ratio))
                remaining = [ind for ind in non_dominated if ind not in preferred]
                # 剩余的从多样性好的个体中选择
                diverse = self._truncate_with_clustering(remaining, 
                                                        self.archive_size - len(preferred))
                self.archive = preferred + diverse
            else:
                self.archive = self._truncate_with_clustering(non_dominated, self.archive_size)

        # 4. 生成新种群
        new_population = []
        if len(self.archive) >= self.pop_size:
            # 从archive中选择多样性最好的个体
            self.archive.sort(key=lambda x: x.fitness)
            new_population = self.archive[:self.pop_size]
        else:
            new_population = list(self.archive)
            # 补充随机个体保持种群大小
            remaining = self.pop_size - len(new_population)
            if remaining > 0:
                # 从原种群中选择不同的个体
                other_individuals = [ind for ind in valid_population if ind not in new_population]
                if len(other_individuals) >= remaining:
                    other_individuals.sort(key=lambda x: x.fitness)
                    new_population.extend(other_individuals[:remaining])
                else:
                    new_population.extend(other_individuals)
                    # 生成随机个体
                    for _ in range(remaining - len(other_individuals)):
                        random_gene = np.random.randint(0, 2, self.simulator.K)
                        random_ind = Individual(random_gene, None)
                        # 需要评估
                        try:
                            random_ind = self.evaluator.evaluate_fitness(random_ind)
                            new_population.append(random_ind)
                        except:
                            # 如果评估失败，设置默认目标值
                            random_ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                            random_ind.violation = 0.0
                            new_population.append(random_ind)

        # 记录统计信息
        self._record_generation_stats(new_population)

        return new_population[:self.pop_size]
    
    def _preference_sampling(self, individuals: List[Individual], target_size: int) -> List[Individual]:
        """
        偏好采样 - 聚焦膝点/关键区段（使用动态偏好采样比例）
        """
        if len(individuals) <= target_size:
            return individuals[:]
        
        # 提取目标值
        objectives = np.array([ind.objectives for ind in individuals])
        
        # 计算膝点（基于最大边际效用）
        if len(objectives) >= 3 and objectives.shape[1] >= 2:
            # 计算每个解的边际效用
            utilities = []
            for i in range(len(objectives)):
                # 简单效用函数：到理想点的距离
                ideal_point = np.min(objectives, axis=0)
                nadir_point = np.max(objectives, axis=0)
                
                # 归一化
                norm_obj = (objectives[i] - ideal_point) / (nadir_point - ideal_point + 1e-10)
                
                # 加权和（可以调整权重以聚焦不同区域）
                utility = np.sum(norm_obj * self.combined_weights[:len(norm_obj)])
                utilities.append(utility)
            
            # 选择效用最好的个体
            sorted_indices = np.argsort(utilities)
            selected_indices = sorted_indices[:target_size]
            selected = [individuals[i] for i in selected_indices]
            return selected
        else:
            # 如果无法计算膝点，返回适应度最好的个体
            individuals.sort(key=lambda x: x.fitness)
            return individuals[:target_size]
    
    def _truncate_with_clustering(self, individuals: List[Individual], target_size: int) -> List[Individual]:
        """全面升级的向量化截断聚类方法 - 告别 Python 嵌套循环带来的秒级卡顿"""
        if len(individuals) <= target_size:
            return individuals[:]
        
        valid_individuals = [ind for ind in individuals 
                            if ind.objectives is not None 
                            and not np.any(np.isnan(ind.objectives))
                            and not np.any(np.isinf(ind.objectives))]
        
        if len(valid_individuals) <= target_size:
            return valid_individuals
        
        objectives = np.array([ind.objectives for ind in valid_individuals])
        
        min_objs = np.min(objectives, axis=0)
        max_objs = np.max(objectives, axis=0)
        range_objs = max_objs - min_objs
        range_objs[range_objs == 0] = 1.0  
        normalized_objs = (objectives - min_objs) / range_objs
        
        # 统一使用向量化加速的 K-means，彻底废弃引发卡顿的 O(N^3) 层次聚类
        return self._fast_kmeans_clustering(valid_individuals, normalized_objs, target_size)
    
    def _hierarchical_clustering(self, individuals: List[Individual], 
                               normalized_objs: np.ndarray, 
                               target_size: int) -> List[Individual]:
        """层次聚类方法"""
        n = len(individuals)
        if n <= target_size:
            return individuals
        
        # 计算距离矩阵
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(normalized_objs[i] - normalized_objs[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # 初始每个个体为一个簇
        clusters = [[i] for i in range(n)]
        
        # 层次聚类直到达到目标簇数
        while len(clusters) > target_size:
            # 找到距离最近的两个簇
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    # 计算簇间平均距离
                    cluster_dist = 0
                    count = 0
                    for idx_i in clusters[i]:
                        for idx_j in clusters[j]:
                            cluster_dist += distances[idx_i, idx_j]
                            count += 1
                    if count > 0:
                        avg_dist = cluster_dist / count
                        if avg_dist < min_dist:
                            min_dist = avg_dist
                            merge_i, merge_j = i, j
            
            if merge_i != -1 and merge_j != -1:
                # 合并簇
                clusters[merge_i].extend(clusters[merge_j])
                clusters.pop(merge_j)
            else:
                break
            
        # 从每个簇中选择代表性个体
        selected = []
        for cluster in clusters:
            if len(cluster) == 1:
                selected.append(individuals[cluster[0]])
            else:
                # 选择簇中适应度最好的个体
                best_idx = min(cluster, key=lambda idx: individuals[idx].fitness)
                selected.append(individuals[best_idx])
        
        return selected
    
    def _fast_kmeans_clustering(self, individuals: List[Individual], 
                              normalized_objs: np.ndarray, 
                              target_size: int) -> List[Individual]:
        """使用 Numpy 矩阵广播的极速 K-means 聚类"""
        n = len(individuals)
        k = target_size
        
        # 初始化聚类中心 (选择适应度最好的K个)
        sorted_indices = np.argsort([ind.fitness for ind in individuals])
        centers = normalized_objs[sorted_indices[:k]]
        
        # 记录每个个体所属的簇
        cluster_assignments = np.zeros(n, dtype=int)
        
        for _ in range(50): # 最大迭代次数设为50即可快速收敛
            # 【矩阵化加速】一次性计算所有个体到所有中心的距离
            # normalized_objs: (n, 1, M), centers: (1, k, M) -> diff: (n, k, M)
            diff = normalized_objs[:, np.newaxis, :] - centers[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2) # (n, k)
            
            # 极速分配簇
            new_assignments = np.argmin(distances, axis=1) # (n,)
            
            if np.array_equal(cluster_assignments, new_assignments):
                break
            cluster_assignments = new_assignments
            
            # 极速更新中心点
            new_centers = np.zeros_like(centers)
            for i in range(k):
                mask = (cluster_assignments == i)
                if np.any(mask):
                    new_centers[i] = np.mean(normalized_objs[mask], axis=0)
                else:
                    new_centers[i] = normalized_objs[np.random.randint(0, n)]
            centers = new_centers
            
        # 从每个簇中挑选适应度最好的个体
        selected = []
        selected_indices = set()
        
        fitness_array = np.array([ind.fitness for ind in individuals])
        
        for i in range(k):
            mask = (cluster_assignments == i)
            if np.any(mask):
                cluster_idx = np.where(mask)[0]
                # 在簇内找到适应度最小的索引
                best_local_idx = np.argmin(fitness_array[cluster_idx])
                best_global_idx = cluster_idx[best_local_idx]
                
                selected.append(individuals[best_global_idx])
                selected_indices.add(best_global_idx)
                
        # 兜底：如果簇有空缺，按适应度补齐
        if len(selected) < target_size:
            remaining_needed = target_size - len(selected)
            all_indices = set(range(n))
            available = list(all_indices - selected_indices)
            
            # 按适应度排序补齐
            available.sort(key=lambda idx: fitness_array[idx])
            for idx in available[:remaining_needed]:
                selected.append(individuals[idx])
                
        return selected
    
    def _diversity_preserving_selection(self, individuals: List[Individual], 
                                      normalized_objs: np.ndarray, 
                                      target_size: int) -> List[Individual]:
        """多样性保持选择 - 后备方法"""
        n = len(individuals)
        if n <= target_size:
            return individuals
        
        # 按适应度排序
        sorted_individuals = sorted(individuals, key=lambda x: x.fitness)
        
        selected = []
        selected_indices = set()
        
        # 首先选择适应度最好的几个个体
        elite_count = min(target_size // 3, 10)
        for i in range(elite_count):
            selected.append(sorted_individuals[i])
            selected_indices.add(i)
        
        # 多样性选择
        remaining_slots = target_size - len(selected)
        if remaining_slots > 0:
            # 计算已选个体的目标值中心
            if selected:
                selected_objs = np.array([ind.objectives for ind in selected])
                center = np.mean(selected_objs, axis=0)
            else:
                center = np.mean(normalized_objs, axis=0)
            
            # 选择距离中心最远的个体以保持多样性
            remaining_indices = [i for i in range(n) if i not in selected_indices]
            distances = [np.linalg.norm(normalized_objs[i] - center) for i in remaining_indices]
            
            # 按距离降序排序并选择
            sorted_remaining = sorted(zip(remaining_indices, distances), 
                                    key=lambda x: x[1], reverse=True)
            
            for i in range(min(remaining_slots, len(sorted_remaining))):
                idx = sorted_remaining[i][0]
                selected.append(individuals[idx])
        
        return selected

    # ========== 差分进化(DE)算子方法 ==========

    def _get_current_F(self, generation: int) -> float:
        """
        获取当前代的F参数（线性衰减）
        """
        progress = generation / max(1, self.max_generations - 1)
        return self.F_init + (self.F_end - self.F_init) * progress

    def _de_evolution(self, target_idx: int, population: List[Individual],
                      generation: int) -> Individual:
        """
        使用DE算子进行变异和交叉
        """
        F = self._get_current_F(generation)
        target = population[target_idx]
        indices = list(range(len(population)))

        if self.use_variable_grouping:
            trial_gene = target.gene.copy()
            num_groups = max(1, self.grouping.num_groups // 2)
            groups = np.random.choice(self.grouping.num_groups, num_groups, replace=False)

            for g_idx in groups:
                group_gene = self.grouping.get_group_genes(target.gene, g_idx)
                mutant = self.de_operator.mutate(group_gene, population, indices)
                trial_group = self.de_operator.crossover(group_gene, mutant)
                trial_gene = self.grouping.set_group_genes(trial_gene, g_idx, trial_group)

            trial = Individual(trial_gene, None)
        else:
            mutant = self.de_operator.mutate(target.gene, population, indices)
            trial_gene = self.de_operator.crossover(target.gene, mutant)
            trial = Individual(trial_gene, None)

        return trial

    def _local_search(self, individual: Individual) -> Individual:
        """
        局部搜索
        """
        if np.random.rand() > self.local_search_rate:
            return individual

        K = getattr(self, 'K', len(individual.gene))
        num_bits = max(1, K // 20)
        bit_indices = np.random.choice(K, num_bits, replace=False)

        trial_gene = individual.gene.copy()
        for bit_idx in bit_indices:
            trial_gene[bit_idx] = 1 - trial_gene[bit_idx]

        trial = Individual(trial_gene, None)
        return trial

    def run_one_generation(self, population: List[Individual], generation: int = 0) -> List[Individual]:
        """
        运行完整的一代

        根据use_de_operator选择使用:
        - True: 差分进化(DE)算子（适合大规模问题）
        - False: 传统GA算子 SBX+多项式变异
        """
        import time
        start_time = time.time()

        # 过滤无效个体
        valid_population = [ind for ind in population if ind.objectives is not None]

        if len(valid_population) < 2:
            print(f"有效个体不足，创建新种群: {len(valid_population)}")
            K = getattr(self.simulator, 'K', 30)
            valid_population = [Individual(np.random.randint(0, 2, K), None)
                               for _ in range(self.pop_size)]
            for ind in valid_population:
                area = np.sum(ind.gene) * 10
                power = np.sum(ind.gene) * 0.5
                slack = 0.1 - np.sum(ind.gene * getattr(self.simulator, 'is_critical', np.zeros_like(ind.gene))) * 0.01
                sens = np.sum(getattr(self.simulator, 'error_rates', np.ones_like(ind.gene)) * (1 - ind.gene))
                ind.objectives = np.array([area, power, -slack, sens])

        # ========== 根据算子类型选择进化策略 ==========
        if self.use_de_operator:
            # ========== 使用差分进化(DE)算子 ==========
            N = len(valid_population)
            trial_population = []

            for i in range(N):
                # DE变异+交叉
                trial = self._de_evolution(i, valid_population, generation)

                # 【关键修复】：DE产生的新个体必须使用真正的评估器评估！
                try:
                    trial = self.evaluator.evaluate_fitness(trial)
                except Exception as e:
                    # 评估失败时使用简化评估（仅作为最后手段）
                    area = np.sum(trial.gene) * 10
                    power = np.sum(trial.gene) * 0.5
                    slack = 0.1 - np.sum(trial.gene * getattr(self.simulator, 'is_critical', np.zeros_like(trial.gene))) * 0.01
                    sens = np.sum(getattr(self.simulator, 'error_rates', np.ones_like(trial.gene)) * (1 - trial.gene))
                    trial.objectives = np.array([area, power, -slack, sens])

                # 局部搜索
                trial = self._local_search(trial)

                # 【关键修复】：局部搜索修改了gene，必须重新评估！
                if trial.objectives is not None:
                    # 清除之前的评估结果，强制重新评估
                    trial.objectives = None

                try:
                    trial = self.evaluator.evaluate_fitness(trial)
                except Exception:
                    # 评估失败时使用简化评估（仅作为最后手段）
                    area = np.sum(trial.gene) * 10
                    power = np.sum(trial.gene) * 0.5
                    slack = 0.1 - np.sum(trial.gene * getattr(self.simulator, 'is_critical', np.zeros_like(trial.gene))) * 0.01
                    sens = np.sum(getattr(self.simulator, 'error_rates', np.ones_like(trial.gene)) * (1 - trial.gene))
                    trial.objectives = np.array([area, power, -slack, sens])

                trial_population.append(trial)

            # 合并种群和档案
            combined = valid_population + self.archive + trial_population

            # 计算适应度
            self.calculate_fitness(combined)

            # 环境选择
            new_population = self.environmental_selection(combined)

            operator_type = "DE"

        else:
            # ========== 使用传统GA算子 ==========
            population = valid_population

            if len(population) < 2:
                print(f"种群大小不足: {len(population)}，跳过本轮")
                return population

            # 生成后代
            offspring_genes = []
            pop_size = len(population)
            num_pairs = pop_size // 2

            # 准备父代选择任务
            parent_tasks = []
            for _ in range(num_pairs):
                parent_tasks.append((population, self.pref_sampling_ratio))

            # 使用修复的并行执行器
            parent_pairs = []
            try:
                parent_pairs = self.parallel_executor.parallel_map_fixed(
                    func_name="parallel_select_parents",
                    data=parent_tasks,
                    func_kwargs={'pref_sampling_ratio': self.pref_sampling_ratio}
                )
            except Exception as e:
                error_str = str(e)
                if "'__main__'" not in error_str and "__main__" not in error_str:
                    print(f"并行选择父代失败，使用串行: {e}")
                parent_pairs = []
                for task in parent_tasks:
                    result = parallel_select_parents(task, self.pref_sampling_ratio)
                    parent_pairs.append(result)

            # 过滤掉None值和无效对
            valid_parent_pairs = []
            for pair in parent_pairs:
                if pair is not None:
                    try:
                        if len(pair) == 2:
                            p1, p2 = pair
                            if p1 is not None and p2 is not None:
                                if hasattr(p1, 'gene') and hasattr(p2, 'gene'):
                                    valid_parent_pairs.append((p1, p2))
                        elif isinstance(pair, tuple) and len(pair) > 0:
                            p = pair[0] if len(pair) > 0 else None
                            if p is not None and hasattr(p, 'gene'):
                                valid_parent_pairs.append((p, p))
                    except Exception as e:
                        print(f"处理父代对时出错: {e}")
                        continue

            print(f"有效父代对数量: {len(valid_parent_pairs)} / {len(parent_pairs)}")

            # 如果没有有效父代对，使用随机选择
            if not valid_parent_pairs:
                print("警告：没有有效父代对，使用随机选择")
                for _ in range(num_pairs):
                    if len(population) >= 2:
                        indices = np.random.choice(len(population), 2, replace=False)
                        valid_parent_pairs.append((population[indices[0]], population[indices[1]]))
                    else:
                        if population:
                            valid_parent_pairs.append((population[0], population[0]))
                        else:
                            K = getattr(self.simulator, 'K', 30)
                            random_gene = np.random.randint(0, 2, K)
                            random_ind = Individual(random_gene, None)
                            valid_parent_pairs.append((random_ind, random_ind))

            # 准备创建后代任务
            offspring_tasks = []
            for p1, p2 in valid_parent_pairs:
                if p1 is not None and p2 is not None:
                    try:
                        if not hasattr(p1, 'gene') or not hasattr(p2, 'gene'):
                            K = getattr(self.simulator, 'K', 30)
                            p1_gene = np.random.randint(0, 2, K)
                            p2_gene = np.random.randint(0, 2, K)
                            p1 = Individual(p1_gene, None)
                            p2 = Individual(p2_gene, None)

                        offspring_tasks.append((p1, p2, self.pc, self.pm, generation))
                    except Exception as e:
                        print(f"准备后代任务时出错: {e}")
                        continue
                else:
                    print(f"发现None父代: p1={p1}, p2={p2}")

            print(f"后代任务数量: {len(offspring_tasks)}")

            # 使用并行执行器创建后代
            child_pairs = []
            if offspring_tasks:
                try:
                    child_pairs = self.parallel_executor.parallel_starmap_fixed(
                        func_name="parallel_create_offspring",
                        args_list=offspring_tasks
                    )
                except Exception as e:
                    error_str = str(e)
                    if "'__main__'" not in error_str and "__main__" not in error_str:
                        print(f"并行创建后代失败，使用串行: {e}")
                    child_pairs = []
                    for args in offspring_tasks:
                        try:
                            result = parallel_create_offspring(*args)
                            child_pairs.append(result)
                        except Exception as inner_e:
                            print(f"串行创建后代失败: {inner_e}")
                            K = getattr(self.simulator, 'K', 30)
                            random_gene = np.random.randint(0, 2, K)
                            child_pairs.append((random_gene.copy(), random_gene.copy()))
            else:
                print("警告：没有有效的后代任务，创建随机后代")
                K = getattr(self.simulator, 'K', 30)
                for _ in range(num_pairs):
                    random_gene = np.random.randint(0, 2, K)
                    child_pairs.append((random_gene.copy(), random_gene.copy()))

            # 收集后代
            offspring_count = 0
            for child_pair in child_pairs:
                if child_pair is not None:
                    try:
                        if len(child_pair) == 2:
                            child1_gene, child2_gene = child_pair
                            offspring_genes.extend([child1_gene, child2_gene])
                            offspring_count += 2
                        else:
                            K = getattr(self.simulator, 'K', 30)
                            random_gene = np.random.randint(0, 2, K)
                            offspring_genes.extend([random_gene.copy(), random_gene.copy()])
                            offspring_count += 2
                    except Exception as e:
                        print(f"处理后代对时出错: {e}")
                        K = getattr(self.simulator, 'K', 30)
                        random_gene = np.random.randint(0, 2, K)
                        offspring_genes.extend([random_gene.copy(), random_gene.copy()])
                        offspring_count += 2

            print(f"生成后代数量: {offspring_count}")

            # 限制后代数量不超过种群大小
            offspring_genes = offspring_genes[:pop_size]

            # 创建后代个体
            offspring = []
            for gene in offspring_genes:
                try:
                    if gene is not None and len(gene) > 0:
                        offspring.append(Individual(gene, None))
                    else:
                        print("发现无效基因")
                        K = getattr(self.simulator, 'K', 30)
                        random_gene = np.random.randint(0, 2, K)
                        offspring.append(Individual(random_gene, None))
                except Exception as e:
                    print(f"创建个体时出错: {e}")
                    K = getattr(self.simulator, 'K', 30)
                    random_gene = np.random.randint(0, 2, K)
                    offspring.append(Individual(random_gene, None))

            # 评估后代
            try:
                offspring = self.evaluator.evaluate_population(offspring)
            except Exception as e:
                print(f"评估后代失败: {e}")
                for ind in offspring:
                    if ind.objectives is None:
                        area = np.sum(ind.gene) * 10
                        power = np.sum(ind.gene) * 0.5
                        slack = 0.1
                        sens = 0.0
                        ind.objectives = np.array([area, power, -slack, sens])

            # 填充评估后仍为 None 的个体
            filled_off = 0
            for ind in offspring:
                if ind.objectives is None:
                    try:
                        area = np.sum(ind.gene) * 10
                        power = np.sum(ind.gene) * 0.5
                        slack = 0.1 - np.sum(ind.gene * getattr(self.simulator, 'is_critical', np.zeros_like(ind.gene))) * 0.01
                        sens = np.sum(getattr(self.simulator, 'error_rates', np.ones_like(ind.gene)) * (1 - ind.gene))
                        ind.objectives = np.array([area, power, -slack, sens])
                        ind.violation = 0.0
                        filled_off += 1
                    except Exception:
                        ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                        ind.violation = 1e6
                        filled_off += 1
            if filled_off > 0:
                print(f"已对评估后代填充目标值: {filled_off} 个体")

            # 更新精英历史信息
            if hasattr(self.unified_mutator, 'update_history'):
                for ind in offspring:
                    if hasattr(ind, 'fitness') and ind.fitness is not None:
                        is_elite = ind.fitness < np.mean([o.fitness for o in offspring if hasattr(o, 'fitness')])
                        self.unified_mutator.update_history(ind.gene, ind.fitness, is_elite)

            # 合并原种群和后代进行环境选择
            try:
                combined = population + offspring
                result = self.environmental_selection(combined)
                # 确保返回足够的个体
                if len(result) < self.pop_size:
                    for ind in population:
                        if ind not in result and len(result) < self.pop_size:
                            result.append(ind)
                    while len(result) < self.pop_size:
                        random_gene = np.random.randint(0, 2, self.simulator.K)
                        random_ind = Individual(random_gene, None)
                        try:
                            random_ind = self.evaluator.evaluate_fitness(random_ind)
                        except:
                            random_ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                            random_ind.violation = 0.0
                        result.append(random_ind)
                new_population = result[:self.pop_size]
            except Exception as e:
                print(f"环境选择失败: {e}")
                combined = population + offspring
                if len(combined) > self.pop_size:
                    combined.sort(key=lambda x: x.fitness if hasattr(x, 'fitness') and x.fitness is not None else 1e6)
                    new_population = combined[:self.pop_size]
                else:
                    result = list(combined)
                    while len(result) < self.pop_size:
                        random_gene = np.random.randint(0, 2, self.simulator.K)
                        random_ind = Individual(random_gene, None)
                        random_ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                        random_ind.violation = 0.0
                        result.append(random_ind)
                    new_population = result[:self.pop_size]

            operator_type = "DE" if self.use_de_operator else "GA"

        # 记录时间
        generation_time = time.time() - start_time
        print(f"SPEA2({operator_type}) 第{generation}代完成，耗时: {generation_time:.2f}秒")

        return new_population
    
    def _parallel_select_parents(self, population_info: Tuple) -> Tuple[Individual, Individual]:
        """并行选择父代"""
        population, pref_sampling_ratio = population_info
        
        if pref_sampling_ratio > 0 and len(population) > 10:
            pref_count = int(len(population) * pref_sampling_ratio)
            
            # 计算每个个体的适应度排名
            fitness_vals = [ind.fitness for ind in population if hasattr(ind, 'fitness')]
            if fitness_vals:
                sorted_indices = np.argsort(fitness_vals)
                pref_indices = sorted_indices[:pref_count]
                
                if np.random.rand() < 0.7 and len(pref_indices) >= 2:
                    idx1, idx2 = np.random.choice(pref_indices, 2, replace=False)
                    return population[idx1], population[idx2]
        
        # 默认随机选择
        return np.random.choice(population, 2, replace=False)
    
    def _parallel_create_offspring(self, p1: Individual, p2: Individual, pc: float, 
                                  pm: float, generation: int) -> Tuple[np.ndarray, np.ndarray]:
        """并行创建后代（使用带精英保留的优化方法）"""
        # 检查是否支持优化的交叉变异
        if hasattr(self.unified_mutator, 'crossover_with_elite_preservation'):
            # 使用带精英保留的交叉
            if np.random.rand() < pc:
                child1_gene, child2_gene = self.unified_mutator.crossover_with_elite_preservation(
                    p1.gene.astype(int), p2.gene.astype(int), generation
                )
            else:
                child1_gene, child2_gene = p1.gene.copy(), p2.gene.copy()
            
            # 使用优化的变异
            child1_gene = self._mutate_optimized(child1_gene, generation, pm)
            child2_gene = self._mutate_optimized(child2_gene, generation, pm)
        else:
            # 使用原有的交叉变异
            if np.random.rand() < pc:
                if np.random.rand() < 0.9:
                    child1_gene, child2_gene = self._uniform_crossover(p1.gene, p2.gene)
                else:
                    child1_gene, child2_gene = self._one_point_crossover(p1.gene, p2.gene)
            else:
                child1_gene, child2_gene = p1.gene.copy(), p2.gene.copy()
            
            child1_gene = self._mutate(child1_gene, generation, pm)
            child2_gene = self._mutate(child2_gene, generation, pm)
        
        return child1_gene, child2_gene
    
    def _mutate_optimized(self, gene: np.ndarray, generation: int = 0, pm: float = None) -> np.ndarray:
        """
        使用优化策略的变异
        """
        if pm is None:
            pm = self.pm
        
        if hasattr(self.unified_mutator, 'mutate'):
            return self.unified_mutator.mutate(gene, generation)
        elif hasattr(self.unified_mutator, 'block_mutation'):
            return self.unified_mutator.block_mutation(gene, generation, pm)
        else:
            # 回退到位级变异
            return self._mutate(gene, generation, pm)
    
    def run(self, eval_fn, generations_per_step: int = 1):
        """
        运行SPEA2算法（支持SAC集成）
        
        参数:
        ----------
        eval_fn : callable
            评估函数，接受基因数组返回目标值
        generations_per_step : int
            每个RL步运行多少代MOEA
        
        返回:
        ----------
        tuple: (最终种群基因, 每代前沿, 统计信息)
        """
        # 初始化种群
        self.population = [Individual(np.random.randint(0, 2, self.simulator.K), None)
                           for _ in range(self.pop_size)]
    
        # 评估函数封装
        def eval_population(pop):
            genes = np.array([ind.gene for ind in pop])
            results = eval_fn(genes)
            
            # 如果eval_fn返回元组（目标值和约束违反）
            if isinstance(results, tuple):
                if len(results) == 2:
                    # 格式：(目标值, 约束违反)
                    objs, violations = results
                    for i, ind in enumerate(pop):
                        ind.objectives = objs[i]
                        ind.violation = violations[i] if violations is not None else 0.0
                elif len(results) == 4:
                    # 格式：四个独立数组（面积、功耗、时序、敏感性）
                    # 转换为二维数组
                    area_array, power_array, slack_array, sens_array = results
                    objs = np.column_stack([area_array, power_array, slack_array, sens_array])
                    for i, ind in enumerate(pop):
                        ind.objectives = objs[i]
                        ind.violation = 0.0
                else:
                    # 其他格式，尝试作为目标值数组
                    objs = results
                    for i, ind in enumerate(pop):
                        ind.objectives = objs[i] if objs[i] is not None else np.array([1e6, 1e6, -1e6, 1e6])
                        ind.violation = 0.0
            else:
                # eval_fn直接返回目标值数组
                objs = results
                for i, ind in enumerate(pop):
                    ind.objectives = objs[i] if objs[i] is not None else np.array([1e6, 1e6, -1e6, 1e6])
                    ind.violation = 0.0
            
            return pop
    
        # 初始评估
        self.population = eval_population(self.population)
        self.archive = []
    
        # SPEA-II 主循环
        all_fronts = []
        
        for gen in range(self.max_generations):
            # 运行一代
            self.population = self.run_one_generation(self.population, gen)
            
            # 记录当前前沿
            front0 = self._get_pareto_front(self.population)
            if front0:
                all_fronts.append([ind.gene.copy() for ind in front0])
            else:
                all_fronts.append([])
            
            # 记录HV和IGD（如果提供了参考集）
            if hasattr(self, 'reference_set'):
                hv = self._calculate_hypervolume(front0)
                igd = self._calculate_igd(front0, self.reference_set)
                self.generation_stats['hv'].append(hv)
                self.generation_stats['igd'].append(igd)
        
        # 返回结果
        final_pop = np.array([ind.gene for ind in self.population])
        stats = self.get_statistics()
        
        return final_pop, all_fronts, stats
    
    def _get_pareto_front(self, population: List[Individual]) -> List[Individual]:
        """
        获取Pareto前沿
        """
        if not population:
            return []
        
        # 使用非支配排序获取第一前沿
        fronts = self.fast_non_dominated_sort(population)
        return fronts[0] if fronts else []
    
    def _calculate_hypervolume(self, front: List[Individual], ref_point: np.ndarray = None) -> float:
        """
        计算超体积
        """
        if not front:
            return 0.0
        
        # 简化实现，实际应用可能需要更精确的计算
        objectives = np.array([ind.objectives for ind in front])
        
        if ref_point is None:
            ref_point = np.max(objectives, axis=0) * 1.2
            
        # 计算每个解的超体积贡献（简化）
        hv = 0.0
        for obj in objectives:
            # 计算到参考点的矩形体积（假设目标独立）
            volume = 1.0
            for i in range(len(obj)):
                if obj[i] < ref_point[i]:
                    volume *= (ref_point[i] - obj[i])
            hv += volume
            
        return hv
    
    def _calculate_igd(self, front: List[Individual], reference_set: List[np.ndarray]) -> float:
        """
        计算反向世代距离
        """
        if not front or not reference_set:
            return float('inf')
        
        front_objs = np.array([ind.objectives for ind in front])
        ref_objs = np.array(reference_set)
        
        # 归一化
        min_vals = np.min(np.vstack([front_objs, ref_objs]), axis=0)
        max_vals = np.max(np.vstack([front_objs, ref_objs]), axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        norm_front = (front_objs - min_vals) / range_vals
        norm_ref = (ref_objs - min_vals) / range_vals
        
        # 计算每个参考点到最近前沿点的距离
        distances = []
        for ref_point in norm_ref:
            min_dist = np.min([np.linalg.norm(ref_point - front_point) 
                              for front_point in norm_front])
            distances.append(min_dist)
        
        return np.mean(distances) if distances else float('inf')
    
    def _uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """均匀交叉"""
        mask = np.random.rand(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    
    def _one_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """单点交叉"""
        c = np.random.randint(0, len(parent1))
        child1 = np.concatenate((parent1[:c], parent2[c:]))
        child2 = np.concatenate((parent2[:c], parent1[c:]))
        return child1, child2
    
    def _mutate(self, gene: np.ndarray, generation: int = 0, pm: float = None) -> np.ndarray:
        """
        使用统一的变异策略进行变异
        """
        if pm is None:
            pm = self.pm
            
        # 更新变异器的pm参数
        if hasattr(self.unified_mutator, 'pm'):
            self.unified_mutator.pm = pm
            
        return self.unified_mutator.mutate(gene, generation)