# algorithms/nsga2_parallel.py
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from multiprocessing import Pool, cpu_count
from .base import BaseOptimizer
from .core_data import Individual
from mo_optimizer.parallel_evaluator import ParallelEvaluator
from utils.parallel_utils import ParallelExecutor, dominates_epsilon

def dominates_epsilon_nsga2(obj_p, obj_q, eps):
    """
    NSGA2 ε-dominance: 
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

def compute_domination_nsga2(args):
    """
    并行支配关系计算（用于NSGA2快速非支配排序）
    """
    i, p, population, eps_dom = args
    p.domination_count = 0
    p.dominated_solutions = []

    for j, q in enumerate(population):
        if i == j:
            continue
        if dominates_epsilon_nsga2(p.objectives, q.objectives, eps_dom):
            p.dominated_solutions.append(q)
        elif dominates_epsilon_nsga2(q.objectives, p.objectives, eps_dom):
            p.domination_count += 1
    return p

class NSGA2Parallel(BaseOptimizer):
    """并行NSGA2算法实现，支持SAC参数动态调整和差分进化(DE)算子切换"""
    
    def __init__(self, simulator, pop_size: int, max_generations: int, 
                 num_processes: int = None,
                 # SAC可调参数初始值
                 epsilon_dom: float = 0.0,
                 accept_threshold: float = 0.0,  # τ: accept threshold
                 crowding_distance_threshold: float = 1e-6,
                 delta_div: float = 1.0,
                 beta_penalty: float = 0.0,
                 pm: float = 0.01,
                 pc: float = 0.9,
                 # NSGA2特有参数
                 tournament_size: int = 2,
                 eta_c: float = 20.0,  # 交叉分布指数
                 eta_m: float = 20.0,  # 变异分布指数
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
        初始化NSGA2算法

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
        
        # 设置并行进程数
        if num_processes:
            self.num_processes = num_processes
        else:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            self.num_processes = max(1, cpu_count - 2)
        
        # 初始化并行执行器
        self.parallel_executor = ParallelExecutor(n_jobs=self.num_processes)
        self.evaluator = ParallelEvaluator(simulator, num_processes, config=config)
        # -------------------------
        # SAC可调参数（由外部SAC算法动态调整）
        # -------------------------
        self.epsilon_dom = epsilon_dom  # ε: epsilon-dominance tolerance
        self.accept_threshold = accept_threshold  # τ: accept threshold for selection
        self.crowding_distance_threshold = crowding_distance_threshold  # 拥挤距离阈值
        self.delta_div = delta_div  # δ: diversity scaling factor
        self.beta_penalty = beta_penalty  # β: penalty for constraint violations
        self.pm = pm  # pm: mutation probability
        self.pc = pc  # pc: crossover probability
        self.pref_sampling_ratio = pref_sampling_ratio  # p: preference sampling ratio
        
        # NSGA2特有参数
        self.tournament_size = tournament_size
        self.eta_c = eta_c
        self.eta_m = eta_m
        
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
                print(f"[NSGA2]: 启用DE算子 + 变量分组，共{self.grouping.num_groups}组")
            else:
                print(f"[NSGA2]: 启用DE算子，F=[{F_init}, {F_end}], CR={CR}")
        
        # 用于记录参数变化历史（供SAC状态空间使用）
        self.param_history = {
            'epsilon_dom': [],
            'accept_threshold': [],
            'crowding_distance_threshold': [],
            'delta_div': [],
            'beta_penalty': [],
            'pm': [],
            'pc': [],
            'pref_sampling_ratio': [],
            'eta_c': [],
            'eta_m': []
        }
        
        # -------------------------
        # 统计指标（供SAC奖励函数使用）
        # -------------------------
        self.generation_stats = {
            'hv': [],  # 超体积
            'igd': [],  # 反向世代距离
            'crowding_mean': [],  # 拥挤距离均值
            'crowding_var': [],  # 拥挤距离方差
            'feasible_rate': [],  # 可行率
            'max_violation': [],  # 最大约束违约
            'avg_violation': [],  # 平均约束违约
            'coverage_ratio': [],  # 参考网格覆盖率
            'front_sizes': [],  # 各前沿大小
            'generation_time': []  # 每代运行时间
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
                eta_m=self.eta_m
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
            'accept_threshold': (0.01, 0.5),  # τ: accept threshold
            'crowding_distance_threshold': (1e-8, 1.0),
            'delta_div': (0.01, 0.3),
            'beta_penalty': (1.0, 50.0),
            'pm': (0.001, 0.3),
            'pc': (0.6, 1.0),
            'pref_sampling_ratio': (0.0, 1.0),
            'eta_c': (5.0, 50.0),
            'eta_m': (5.0, 50.0)
        }
        
        # 确保参数在合理范围内
        self._clip_params()

    def reset(self):
        """重置NSGA2状态"""
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
            'front_sizes': [],  
            'generation_time': [] 
        }
        print("[NSGA2]: 重置完成")
    
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
        if 'crowding_distance_threshold' in params:
            self.crowding_distance_threshold = params['crowding_distance_threshold']
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
        if 'eta_c' in params:
            self.eta_c = params['eta_c']
        if 'eta_m' in params:
            self.eta_m = params['eta_m']
        
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
            'crowding_distance_threshold': self.crowding_distance_threshold,
            'delta_div': self.delta_div,
            'beta_penalty': self.beta_penalty,
            'pm': self.pm,
            'pc': self.pc,
            'pref_sampling_ratio': self.pref_sampling_ratio,
            'eta_c': self.eta_c,
            'eta_m': self.eta_m
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
    
    def _clip_params(self):
        """
        确保参数在合理范围内
        """
        # 保证参数在合理范围内
        self.epsilon_dom = float(np.clip(self.epsilon_dom, *self.param_bounds['epsilon_dom']))
        self.accept_threshold = float(np.clip(self.accept_threshold, *self.param_bounds['accept_threshold']))
        self.crowding_distance_threshold = float(np.clip(
            self.crowding_distance_threshold, *self.param_bounds['crowding_distance_threshold']))
        self.delta_div = float(np.clip(self.delta_div, *self.param_bounds['delta_div']))
        self.beta_penalty = float(np.clip(self.beta_penalty, *self.param_bounds['beta_penalty']))
        self.pm = float(np.clip(self.pm, *self.param_bounds['pm']))
        self.pc = float(np.clip(self.pc, *self.param_bounds['pc']))
        self.pref_sampling_ratio = float(np.clip(
            self.pref_sampling_ratio, *self.param_bounds['pref_sampling_ratio']))
        self.eta_c = float(np.clip(self.eta_c, *self.param_bounds['eta_c']))
        self.eta_m = float(np.clip(self.eta_m, *self.param_bounds['eta_m']))
    
    def _record_param_history(self):
        """
        记录参数变化历史
        """
        self.param_history['epsilon_dom'].append(self.epsilon_dom)
        self.param_history['accept_threshold'].append(self.accept_threshold)
        self.param_history['crowding_distance_threshold'].append(self.crowding_distance_threshold)
        self.param_history['delta_div'].append(self.delta_div)
        self.param_history['beta_penalty'].append(self.beta_penalty)
        self.param_history['pm'].append(self.pm)
        self.param_history['pc'].append(self.pc)
        self.param_history['pref_sampling_ratio'].append(self.pref_sampling_ratio)
        self.param_history['eta_c'].append(self.eta_c)
        self.param_history['eta_m'].append(self.eta_m)
    
    def _record_generation_stats(self, population: List[Individual], fronts: List[List[Individual]], 
                           generation_time: float = 0.0):
        """
        记录当前代的统计信息
        """
        if not population:
            return
        
        # 确保所有必要的键都存在
        for key in ['front_sizes', 'generation_time']:
            if key not in self.generation_stats:
                self.generation_stats[key] = []
        
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
        
        # 计算拥挤距离统计（只考虑第一前沿）
        crowding_mean = 0.0
        crowding_var = 0.0
        if fronts and fronts[0]:
            first_front = fronts[0]
            crowding_distances = []
            for ind in first_front:
                if hasattr(ind, 'crowding_distance') and not np.isnan(ind.crowding_distance):
                    crowding_distances.append(ind.crowding_distance)
            
            if crowding_distances:
                # 过滤掉inf值
                finite_dists = [d for d in crowding_distances if not np.isinf(d)]
                if finite_dists:
                    crowding_mean = np.mean(finite_dists)
                    crowding_var = np.var(finite_dists)
                else:
                    # 如果全是inf（例如前沿只有2个点）
                    crowding_mean = 1.0
                    crowding_var = 0.0
        
        # 记录各前沿大小
        front_sizes = [len(front) for front in fronts]
        
        # 记录统计信息
        self.generation_stats['feasible_rate'].append(feasible_rate)
        self.generation_stats['max_violation'].append(max_violation)
        self.generation_stats['avg_violation'].append(avg_violation)
        self.generation_stats['crowding_mean'].append(crowding_mean)
        self.generation_stats['crowding_var'].append(crowding_var)
        self.generation_stats['front_sizes'].append(front_sizes)
        self.generation_stats['generation_time'].append(generation_time)
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        快速非支配排序（矩阵化向量版 + Deb约束支配法则）
        引入了时序违例（Slack < 0）的硬约束判断，确保 Pareto 前沿中只有可流片的物理可行解。
        """
        # 过滤无效个体
        valid_pop = [ind for ind in population if ind.objectives is not None]
        if not valid_pop:
            return []

        N = len(valid_pop)
        
        # 1. 提取所有目标值，构成形状为 (N, M) 的二维矩阵
        objs_list = [ind.objectives for ind in valid_pop]
        objs = np.array(objs_list)
        
        # 防御性检查：确保objs是2D数组
        if objs.ndim != 2:
            print(f"[NSGA2-WARN] objectives维度错误: ndim={objs.ndim}, shape={objs.shape}")
            # 尝试修复
            fixed_objs = []
            for obj in objs_list:
                if isinstance(obj, np.ndarray) and obj.ndim == 1 and len(obj) >= 4:
                    fixed_objs.append(obj[:4])
                elif isinstance(obj, (list, tuple)) and len(obj) >= 4:
                    fixed_objs.append(list(obj)[:4])
                else:
                    # 默认值
                    fixed_objs.append([1e6, 1e6, 0.0, 1.0])
            objs = np.array(fixed_objs, dtype=np.float64)
            print(f"[NSGA2-WARN] 已修复objs shape: {objs.shape}")
        
        # 确保是浮点数类型
        if objs.dtype != np.float64 and objs.dtype != np.float32:
            objs = objs.astype(np.float64)
        
        # 提取个体的时序违例程度 (Constraint Violation)
        # 优先读取个体的 violation 属性。如果没有，则默认第3个目标(index 2, slack penalty)为违例度
        # 违例度为 0 表示解是可行的 (Slack >= 0)，大于 0 表示不可行 (Slack < 0)
        violations = np.array([getattr(ind, 'violation', max(0, obj[2])) for ind, obj in zip(valid_pop, objs)])
        
        # 2. 使用 Numpy 广播机制计算两两之间的目标差值矩阵
        diff = objs[:, np.newaxis, :] - objs[np.newaxis, :, :]
        
        # 3. 矩阵化评估传统的 ε-dominance 规则
        with np.errstate(invalid='ignore'):
            cond1 = np.all(diff <= self.epsilon_dom, axis=2)  # 所有维度都不差
            cond2 = np.any(diff < -self.epsilon_dom, axis=2)  # 至少一维严格更好
        
        # 传统的 Pareto 支配关系矩阵
        pareto_dominates = cond1 & cond2
        
        # 矩阵化实现 Deb 的约束支配三大法则
        V_i = violations[:, np.newaxis] # 形状 (N, 1)
        V_j = violations[np.newaxis, :] # 形状 (1, N)
        
        # 规则1: i 可行 (不违例)，且 j 不可行 (违例) -> i 绝对支配 j
        rule1 = (V_i == 0) & (V_j > 0)
        
        # 规则2: i 和 j 都不可行，但 i 的违例程度更轻 -> i 支配 j
        rule2 = (V_i > 0) & (V_j > 0) & (V_i < V_j)
        
        # 规则3: i 和 j 处于相同的可行性状态 (都可行，或违例程度完全一样) -> 使用传统的 Pareto 支配比较 PPA
        rule3 = (V_i == V_j) & pareto_dominates
        
        # 4. 生成全局支配矩阵 (只要满足上述任一规则，i 就支配 j)
        dominates_matrix = rule1 | rule2 | rule3
        
        # 5. 修正对角线：个体自己不能支配自己
        np.fill_diagonal(dominates_matrix, False)
        
        # 6. 一次性统计每个个体的被支配数（按列求和）
        domination_counts = np.sum(dominates_matrix, axis=0)
        
        # 7. 为个体对象赋值
        for i in range(N):
            p = valid_pop[i]
            p.domination_count = int(domination_counts[i])
            dominated_indices = np.where(dominates_matrix[i, :])[0]
            p.dominated_solutions = [valid_pop[idx] for idx in dominated_indices]

        # 构建 Pareto 前沿 ----------------
        fronts = [[]]
        
        # 构建第一前沿
        for p in valid_pop:
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        # 如果第一前沿为空，直接返回
        if not fronts[0]:
            return []

        # 构建后续前沿
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
    
    def calculate_crowding_distance(self, front: List[Individual]):
        """计算拥挤距离"""
        if not front:
            return
        
        if len(front) == 1:
            front[0].crowding_distance = 1.0
            return
        
        if len(front) == 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        # 提取目标值
        objectives = np.array([ind.objectives for ind in front])
        n_individuals = len(objectives)
        n_objectives = objectives.shape[1]
        
        # 初始化拥挤距离
        for ind in front:
            ind.crowding_distance = 0.0
        
        # 对每个目标分别计算拥挤距离
        for obj_idx in range(n_objectives):
            # 按当前目标值排序
            sorted_indices = np.argsort(objectives[:, obj_idx])
            
            # 边界个体的拥挤距离设为无穷大
            front[sorted_indices[0]].crowding_distance = float('inf')
            front[sorted_indices[-1]].crowding_distance = float('inf')
            
            # 计算中间个体的拥挤距离
            min_obj = objectives[sorted_indices[0], obj_idx]
            max_obj = objectives[sorted_indices[-1], obj_idx]
            obj_range = max_obj - min_obj
            
            if obj_range < self.crowding_distance_threshold:
                obj_range = 1.0
            
            for i in range(1, n_individuals - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                
                dist = (objectives[next_idx, obj_idx] - objectives[prev_idx, obj_idx]) / obj_range
                if not np.isnan(dist) and not np.isinf(dist):
                    front[idx].crowding_distance += self.delta_div * dist

    def _clean_individuals(self, front: List[Individual]) -> List[Individual]:
        """
        清理个体，移除或修复无效的目标值

        参数:
        ----------
        front : List[Individual]
            待清理的个体列表

        返回:
        ----------
        List[Individual]: 清理后的个体列表
        """
        cleaned_front = []
        default_objectives = np.array([1e6, 1e6, -1e6, 1e6])  # 默认值

        for i, ind in enumerate(front):
            if ind.objectives is None:
                # 目标值为None，设置默认值
                ind.objectives = default_objectives.copy()
                ind.violation = 0.0
                print(f"[NSGA2-CLEAN]: 个体{i}目标值为None，设置为默认值")
                cleaned_front.append(ind)

            elif not isinstance(ind.objectives, np.ndarray):
                # 目标值不是数组，转换为数组
                try:
                    ind.objectives = np.array(ind.objectives, dtype=np.float32)
                    # 检查维度
                    if len(ind.objectives) < 4:
                        # 补充到4维
                        padding = default_objectives[len(ind.objectives):]
                        ind.objectives = np.concatenate([ind.objectives, padding])
                    print(f"[NSGA2-CLEAN]: 个体{i}目标值转换为数组，形状={ind.objectives.shape}")
                except Exception as e:
                    print(f"[NSGA2-CLEAN]: 个体{i}目标值转换失败: {e}")
                    ind.objectives = default_objectives.copy()
                ind.violation = 0.0
                cleaned_front.append(ind)

            elif np.any(np.isnan(ind.objectives)) or np.any(np.isinf(ind.objectives)):
                # 包含NaN或Inf，替换为默认值
                print(f"[NSGA2-CLEAN]: 个体{i}目标值包含NaN/Inf，原始值={ind.objectives}")

                # 尝试修复部分NaN/Inf
                repaired = ind.objectives.copy()

                # 替换NaN
                nan_mask = np.isnan(repaired)
                if np.any(nan_mask):
                    # 用列均值替换NaN
                    col_means = np.nanmean(repaired, axis=0)
                    for j in range(len(repaired)):
                        if np.any(np.isnan(repaired[j])):
                            for k in range(len(repaired[j])):
                                if np.isnan(repaired[j, k]):
                                    repaired[j, k] = col_means[k] if not np.isnan(col_means[k]) else default_objectives[k]

                # 替换Inf
                inf_mask = np.isinf(repaired)
                if np.any(inf_mask):
                    # 用列中位数替换Inf
                    col_medians = np.median(repaired[~inf_mask], axis=0) if np.any(~inf_mask) else default_objectives
                    for j in range(len(repaired)):
                        if np.any(np.isinf(repaired[j])):
                            for k in range(len(repaired[j])):
                                if np.isinf(repaired[j, k]):
                                    sign = np.sign(repaired[j, k])
                                    repaired[j, k] = col_medians[k] * (1.5 if sign > 0 else 0.5)

                # 如果修复后仍有问题，使用默认值
                if np.any(np.isnan(repaired)) or np.any(np.isinf(repaired)):
                    print(f"[NSGA2-CLEAN]: 个体{i}修复失败，使用默认值")
                    repaired = default_objectives.copy()

                ind.objectives = repaired
                ind.violation = 0.0
                cleaned_front.append(ind)

            else:
                # 目标值有效，直接添加
                cleaned_front.append(ind)

        return cleaned_front
    
    def selection(self, population: List[Individual]) -> List[Individual]:
        """
        锦标赛选择（考虑约束惩罚和偏好采样）
        """
        selected = []
        pop_size = len(population)
        
        # 计算每个个体的适应度（考虑约束惩罚）
        fitness_values = []
        for ind in population:
            # 基本适应度：基于前沿等级（越小越好）
            fitness = ind.rank if hasattr(ind, 'rank') else 0
            
            # 添加约束惩罚（使用动态β参数）
            if hasattr(ind, 'violation') and self.beta_penalty > 0:
                violation = ind.violation
                if violation > 0:
                    fitness += self.beta_penalty * violation
            
            # 考虑拥挤距离（越大越好，取倒数）
            if hasattr(ind, 'crowding_distance'):
                if ind.crowding_distance > 0:
                    fitness += 1.0 / ind.crowding_distance
            
            fitness_values.append(fitness)
        
        # 根据偏好采样比例确定精英选择数量
        elite_count = int(pop_size * self.pref_sampling_ratio) if self.pref_sampling_ratio > 0 else 0
        
        # 精英选择（选择前沿等级最好且拥挤距离最大的个体）
        if elite_count > 0:
            # 按前沿等级和拥挤距离排序
            sorted_population = sorted(population, 
                                      key=lambda x: (x.rank if hasattr(x, 'rank') else float('inf'),
                                                     -x.crowding_distance if hasattr(x, 'crowding_distance') else 0))
            elite_indices = list(range(min(elite_count, len(sorted_population))))
            selected_indices = set(elite_indices)
            selected.extend([sorted_population[i] for i in elite_indices])
        else:
            selected_indices = set()
        
        # 锦标赛选择填充剩余位置
        remaining = pop_size - len(selected)
        for _ in range(remaining):
            # 随机选择锦标赛参与者
            candidates = np.random.choice(pop_size, self.tournament_size, replace=False)
            
            # 选择适应度最好的个体
            best_idx = min(candidates, key=lambda idx: fitness_values[idx])
            while best_idx in selected_indices and len(selected_indices) < pop_size:
                # 如果已经选择过，选择次优的
                candidates = [idx for idx in candidates if idx != best_idx]
                if candidates:
                    best_idx = min(candidates, key=lambda idx: fitness_values[idx])
                else:
                    # 重新选择
                    candidates = np.random.choice(pop_size, self.tournament_size, replace=False)
                    best_idx = min(candidates, key=lambda idx: fitness_values[idx])
            
            if best_idx not in selected_indices:
                selected.append(population[best_idx])
                selected_indices.add(best_idx)
        
        return selected
    
    def crossover(self, parent1: Individual, parent2: Individual, generation: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        带精英保留的交叉算子
        优先使用区块级交叉，只在非精英区域进行交换
        """
        # 检查是否支持精英保留交叉
        if hasattr(self.unified_mutator, 'crossover_with_elite_preservation'):
            # 使用新的带精英保留的交叉
            child1_gene, child2_gene = self.unified_mutator.crossover_with_elite_preservation(
                parent1.gene.astype(int),
                parent2.gene.astype(int),
                generation
            )
            return child1_gene, child2_gene
        else:
            # 使用原有的SBX交叉
            gene1 = parent1.gene.astype(float)
            gene2 = parent2.gene.astype(float)
            
            rand = np.random.rand(len(gene1))
            
            beta = np.zeros(len(gene1))
            for i in range(len(gene1)):
                if rand[i] <= 0.5:
                    beta[i] = (2.0 * rand[i]) ** (1.0 / (self.eta_c + 1.0))
                else:
                    beta[i] = (1.0 / (2.0 * (1.0 - rand[i]))) ** (1.0 / (self.eta_c + 1.0))
            
            child1 = 0.5 * ((1 + beta) * gene1 + (1 - beta) * gene2)
            child2 = 0.5 * ((1 - beta) * gene1 + (1 + beta) * gene2)
            
            child1 = np.clip(child1, 0, 1)
            child2 = np.clip(child2, 0, 1)
            
            return (child1 > 0.5).astype(int), (child2 > 0.5).astype(int)
    
    def _greedy_guided_mutation(self, gene: np.ndarray, generation: int = 0) -> np.ndarray:
        """贪婪引导的变异算子
        
        不是随机翻转位，而是选择能带来最大benefit/cost评分的0→1翻转。
        这与贪婪算法的评分机制一致，使MOEA向贪婪方向进化。
        
        参数:
            gene: 当前个体的基因
            generation: 当前代数（用于自适应概率调整）
        返回:
            变异后的基因
        """
        # 70%概率使用贪婪引导变异，30%使用原始变异（保留探索）
        if np.random.rand() > 0.7:
            return self.mutation(gene, generation)
        
        try:
            # 获取模拟器属性（通过self.simulator访问PPAModel）
            sim = self.simulator
            K = len(gene)
            current_hardened = int(np.sum(gene))
            
            # 获取设计基线值（从simulator或parallel_evaluator）
            # 优先从simulator获取，如果不存在则使用默认值
            area_orig = getattr(sim, 'area_orig', getattr(sim, 'Area_orig', getattr(sim, 'design_area', 1.0)))
            power_orig = getattr(sim, 'power_orig', getattr(sim, 'Power_orig', getattr(sim, 'design_power', 1.0)))
            slack_orig = getattr(sim, 'slack_orig', getattr(sim, 'WNS_orig', getattr(sim, 'design_slack', 0.1)))
            sens_orig = getattr(sim, 'sens_orig', getattr(sim, 'SER_orig', getattr(sim, 'design_sensitivity', 1e-5)))
            
            # 获取TMR成本和寄存器属性
            area_per_reg = getattr(sim, 'area', np.ones(K))
            power_per_reg = getattr(sim, 'power', np.ones(K) * 0.5)
            slack_per_reg = getattr(sim, 'slack', np.ones(K) * 0.1)
            error_rates = getattr(sim, 'error_rates', np.ones(K))
            sensitivity = getattr(sim, 'sensitivity', error_rates.copy())
            
            voter_area = float(getattr(sim, 'voter_area', 0.5))
            voter_power = float(getattr(sim, 'voter_power', 0.2))
            voter_delay = float(getattr(sim, 'voter_delay', 0.13))
            
            # 计算当前状态的资源消耗
            current_area_inc = np.sum((area_per_reg * 2 + voter_area) * gene)
            current_power_inc = np.sum((power_per_reg * 2 + voter_power) * gene)
            current_worst_slack = np.min(slack_per_reg - voter_delay * gene) if current_hardened > 0 else np.min(slack_per_reg)
            current_sens_sum = np.sum(sensitivity * (1 - gene))
            total_regs = K + 2 * current_hardened
            
            # 找到所有未加固的寄存器
            candidates = np.where(gene == 0)[0]
            if len(candidates) == 0:
                return gene.copy()
            
            # 向量化计算所有候选0→1翻转的效果
            tmr_area_cost = area_per_reg[candidates] * 2 + voter_area
            tmr_power_cost = power_per_reg[candidates] * 2 + voter_power
            slack_after = np.minimum(current_worst_slack, slack_per_reg[candidates] - voter_delay)
            
            new_hardened_count = current_hardened + 1
            sens_after = (current_sens_sum - sensitivity[candidates]) / (K + 2 * new_hardened_count)
            sens_gain = (current_sens_sum / total_regs) - sens_after  # 正数表示改善
            
            # 归一化成本
            area_cost = current_area_inc + tmr_area_cost
            power_cost = current_power_inc + tmr_power_cost
            slack_cost = max(0.0, current_worst_slack - slack_after)
            
            area_norm = area_cost / max(area_orig, 1e-10)
            power_norm = power_cost / max(power_orig, 1e-10)
            slack_norm = slack_cost / max(slack_orig, 1e-10) * 10.0
            
            total_cost = area_norm + power_norm + slack_norm
            benefit_cost = sens_gain / np.maximum(total_cost, 1e-20)
            
            # 可行性过滤：面积≤20%, 功耗≤50%, slack≥0, 增益>0
            feasible = (
                (area_cost / max(area_orig, 1e-10) <= 0.2 + 1e-9) &
                (power_cost / max(power_orig, 1e-10) <= 0.5 + 1e-9) &
                (slack_after >= -1e-9) &
                (sens_gain > 0.0)
            )
            benefit_cost = np.where(feasible, benefit_cost, -np.inf)
            
            if not np.isfinite(benefit_cost).any():
                return gene.copy()
            
            # 70%选择最优，30%从top-5中随机选择（保留探索）
            if np.random.rand() < 0.7:
                best_idx = np.argmax(benefit_cost)
                new_gene = gene.copy()
                new_gene[candidates[best_idx]] = 1
                return new_gene
            else:
                top_k = min(5, len(candidates))
                top_indices = np.argsort(benefit_cost)[-top_k:]
                valid_top = top_indices[np.isfinite(benefit_cost[top_indices])]
                if len(valid_top) > 0:
                    choice = np.random.choice(valid_top)
                    new_gene = gene.copy()
                    new_gene[candidates[choice]] = 1
                    return new_gene
                return gene.copy()
                
        except Exception:
            # 如果贪婪引导失败，回退到原始变异
            return self.mutation(gene, generation)
    
    def mutation(self, gene: np.ndarray, generation: int = 0, current_fitness: float = None) -> np.ndarray:
        """
        优化的变异算子
        根据基因长度自动选择策略：
        - 短基因：精细位级变异
        - 中等基因：自适应变异
        - 长基因：区块级变异
        
        70%概率使用贪婪引导变异（向贪婪算法靠拢），30%使用原始变异（保留探索）
        """
        if np.random.rand() < 0.7:
            return self._greedy_guided_mutation(gene, generation)
        return self._original_mutation(gene, generation, current_fitness)
    
    def _original_mutation(self, gene: np.ndarray, generation: int = 0, current_fitness: float = None) -> np.ndarray:
        """原始变异算子（多项式变异或统一变异策略）"""
        if hasattr(self.unified_mutator, 'mutate'):
            # 使用新的综合变异方法
            return self.unified_mutator.mutate(gene, generation, None, current_fitness)
        else:
            # 使用原有的多项式变异
            mutated = gene.astype(float).copy()
            
            for i in range(len(gene)):
                if np.random.rand() < self.pm:
                    rand = np.random.rand()
                    if rand < 0.5:
                        delta = (2.0 * rand) ** (1.0 / (self.eta_m + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (self.eta_m + 1.0))
                    
                    mutated[i] += delta
                    mutated[i] = np.clip(mutated[i], 0, 1)
            
            return (mutated > 0.5).astype(int)
    
    def environmental_selection(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        """
        环境选择（结合父代和子代）
        """
        # 合并种群
        combined_pop = population + offspring
        
        # 过滤无效个体
        valid_pop = [ind for ind in combined_pop if ind.objectives is not None]
        if not valid_pop:
            # 如果没有有效个体，返回前pop_size个个体
            return combined_pop[:self.pop_size]
        
        # 【修复】：优先选择同时满足面积/功耗和敏感性约束的解
        # 统计敏感性可行性
        total_sens_feasible = sum(1 for ind in valid_pop if getattr(ind, 'sensitivity_feasible', True))
        total_ap_feasible = sum(1 for ind in valid_pop if getattr(ind, 'area_power_feasible', False))
        both_feasible = sum(1 for ind in valid_pop if getattr(ind, 'area_power_feasible', False) and getattr(ind, 'sensitivity_feasible', True))
        
        # 【调试】：打印敏感性判断详情
        if valid_pop and len(valid_pop) > 0:
            # 检查第一个个体的敏感性相关属性
            sample = valid_pop[0]
            has_raw_sens = hasattr(sample, 'raw_sensitivity')
            has_sens_feas = hasattr(sample, 'sensitivity_feasible')
            obj3 = sample.objectives[3] if sample.objectives is not None and len(sample.objectives) >= 4 else None
            print(f"[NSGA2环境选择] 总解={len(valid_pop)}, 面积/功耗可行={total_ap_feasible}, 敏感性可行={total_sens_feasible}, 两者都可行={both_feasible}")
            print(f"  [调试] 样本: raw_sensitivity={has_raw_sens}, sensitivity_feasible={has_sens_feas}, obj[3]={obj3}")
        
        # 优先选择两者都可行的解
        feasible_pop = [ind for ind in valid_pop if getattr(ind, 'area_power_feasible', False) and getattr(ind, 'sensitivity_feasible', True)]
        
        if len(feasible_pop) >= self.pop_size * 0.3:
            working_pop = feasible_pop
            print(f"[NSGA2] 环境选择: {len(feasible_pop)}个完全可行解")
        else:
            # 如果完全可行解不足，优先敏感性可行的解
            sens_feasible = [ind for ind in valid_pop if getattr(ind, 'sensitivity_feasible', True)]
            if len(sens_feasible) >= self.pop_size * 0.3:
                working_pop = sens_feasible
                print(f"[NSGA2] 环境选择: {len(sens_feasible)}个敏感性可行解")
            else:
                # 如果都不足，选择所有有效个体
                working_pop = valid_pop
                print(f"[NSGA2] 环境选择: {len(working_pop)}个解（无可行解）")
        
        # 非支配排序
        fronts = self.fast_non_dominated_sort(working_pop)
        
        # 如果 fronts 为空，直接返回前 pop_size 个有效个体
        if not fronts:
            return valid_pop[:self.pop_size]
        
        # 计算每个前沿的拥挤距离
        for front in fronts:
            if front and len(front) > 0:  # 只计算非空前言的拥挤距离
                self.calculate_crowding_distance(front)
        
        # 选择新种群
        new_population = []
        front_index = 0
        
        # 使用accept_threshold调整选择策略
        # τ参数用于调整选择阈值：τ越大，选择越宽松（保留更多个体）
        # 计算调整后的种群大小限制（但最终仍限制在pop_size）
        adjusted_pop_size = int(self.pop_size * (1.0 + self.accept_threshold))
        target_pop_size = min(adjusted_pop_size, self.pop_size * 2)  # 最多不超过2倍
        
        # 遍历前沿，直到达到目标种群大小
        while front_index < len(fronts) and len(new_population) < target_pop_size:
            current_front = fronts[front_index]
            
            # 如果当前前沿为空，跳过
            if not current_front:
                front_index += 1
                continue
            
            # 如果添加整个前沿不会超过目标种群大小
            if len(new_population) + len(current_front) <= target_pop_size:
                # 添加整个前沿
                new_population.extend(current_front)
            else:
                # 部分选择当前前沿
                # 先按拥挤距离排序
                current_front.sort(key=lambda x: x.crowding_distance if hasattr(x, 'crowding_distance') else 0, 
                                 reverse=True)
                remaining = target_pop_size - len(new_population)
                if remaining > 0:
                    new_population.extend(current_front[:remaining])
            
            front_index += 1
        
        # 如果选择的个体超过pop_size，需要截断到pop_size
        if len(new_population) > self.pop_size:
            # 按拥挤距离排序，保留最好的pop_size个个体
            new_population.sort(key=lambda x: x.crowding_distance if hasattr(x, 'crowding_distance') else 0, 
                              reverse=True)
            new_population = new_population[:self.pop_size]
        
        # 确保选择了足够数量的个体
        if len(new_population) < self.pop_size:
            remaining = self.pop_size - len(new_population)
            
            # 利用 Set 直接比较内存地址，干掉 O(N^2) 嵌套循环
            selected_ids = {id(ind) for ind in new_population}
            remaining_indices = [i for i, ind in enumerate(combined_pop) if id(ind) not in selected_ids]
            
            if remaining_indices:
                # 选择剩余个体
                for i in range(min(remaining, len(remaining_indices))):
                    idx = remaining_indices[i]
                    new_population.append(combined_pop[idx])
        
        return new_population[:self.pop_size]

    # ========== 差分进化(DE)算子方法 ==========

    def _get_current_F(self, generation: int) -> float:
        """
        获取当前代的F参数（线性衰减）
        早期大F（探索），晚期小F（收敛）
        """
        progress = generation / max(1, self.max_generations - 1)
        return self.F_init + (self.F_end - self.F_init) * progress

    def _de_mutation_and_crossover(self, target_idx: int, population: List[Individual],
                                   generation: int) -> Individual:
        """
        使用DE算子进行变异和交叉
        替换传统的SBX交叉+多项式变异

        公式: v = x_r1 + F * (x_r2 - x_r3)
        """
        F = self._get_current_F(generation)
        target = population[target_idx]
        indices = list(range(len(population)))

        if self.use_variable_grouping:
            # 变量分组：对部分组进行变异
            trial_gene = target.gene.copy()

            num_groups_to_mutate = max(1, self.grouping.num_groups // 3)
            groups_to_mutate = np.random.choice(
                self.grouping.num_groups,
                num_groups_to_mutate,
                replace=False
            )

            for g_idx in groups_to_mutate:
                # 对该组的基因进行DE变异
                group_gene = self.grouping.get_group_genes(target.gene, g_idx)
                mutant_gene = self.de_operator.mutate(
                    group_gene, population, indices
                )
                # 交叉
                trial_group = self.de_operator.crossover(group_gene, mutant_gene)
                # 更新
                trial_gene = self.grouping.set_group_genes(trial_gene, g_idx, trial_group)

            trial = Individual(trial_gene, None)
        else:
            # 直接对整个基因进行DE
            mutant = self.de_operator.mutate(target.gene, population, indices)
            trial_gene = self.de_operator.crossover(target.gene, mutant)
            trial = Individual(trial_gene, None)

        return trial

    def _local_search(self, individual: Individual, population: List[Individual]) -> Individual:
        """
        局部搜索：对当前解进行精细调整
        """
        if np.random.rand() > self.local_search_rate:
            return individual

        K = getattr(self, 'K', len(individual.gene))
        num_bits = max(1, K // 20)  # 5%的基因位
        bit_indices = np.random.choice(K, num_bits, replace=False)

        trial_gene = individual.gene.copy()

        for bit_idx in bit_indices:
            trial_gene[bit_idx] = 1 - trial_gene[bit_idx]

        trial = Individual(trial_gene, None)
        return trial

    def update_parameters(self, params: Dict[str, float]):
        """
        更新参数（SAC调用）

        新增DE参数更新:
        - 'F': DE缩放因子
        - 'CR': DE交叉概率
        """
        # 更新SAC参数
        if 'epsilon_dom' in params:
            self.epsilon_dom = params['epsilon_dom']
        if 'accept_threshold' in params:
            self.accept_threshold = params['accept_threshold']
        if 'crowding_distance_threshold' in params:
            self.crowding_distance_threshold = params['crowding_distance_threshold']
        if 'delta_div' in params:
            self.delta_div = params['delta_div']
        if 'beta_penalty' in params:
            self.beta_penalty = params['beta_penalty']
        if 'pm' in params:
            self.pm = params['pm']
            if hasattr(self.unified_mutator, 'pm'):
                self.unified_mutator.pm = self.pm
        if 'pc' in params:
            self.pc = params['pc']
            if hasattr(self.unified_mutator, 'pc'):
                self.unified_mutator.pc = self.pc
        if 'pref_sampling_ratio' in params:
            self.pref_sampling_ratio = params['pref_sampling_ratio']

        # 更新DE参数
        if 'F' in params:
            self.F_init = params['F']
            self.F_end = params['F']
            if hasattr(self, 'de_operator'):
                self.de_operator.F = params['F']

        if 'CR' in params:
            self.CR = params['CR']
            if hasattr(self, 'de_operator'):
                self.de_operator.CR = params['CR']

        if 'use_de_operator' in params:
            self.use_de_operator = params['use_de_operator']

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

        # 非支配排序和拥挤距离计算
        fronts = self.fast_non_dominated_sort(valid_population)
        for front in fronts:
            self.calculate_crowding_distance(front)

        # ========== 根据算子类型选择进化策略 ==========
        if self.use_de_operator:
            # ========== 使用差分进化(DE)算子 ==========
            N = len(valid_population)
            trial_population = []

            for i in range(N):
                # DE变异+交叉
                trial = self._de_mutation_and_crossover(i, valid_population, generation)

                # 【关键修复】：使用真正的评估器评估trial个体！
                try:
                    trial = self.evaluator.evaluate_fitness(trial)
                except Exception as e:
                    # 评估失败时使用简化评估（仅作为最后手段）
                    area = np.sum(trial.gene) * 10
                    power = np.sum(trial.gene) * 0.5
                    slack = 0.1 - np.sum(trial.gene * getattr(self.simulator, 'is_critical', np.zeros_like(trial.gene))) * 0.01
                    sens = np.sum(getattr(self.simulator, 'error_rates', np.ones_like(trial.gene)) * (1 - trial.gene))
                    trial.objectives = np.array([area, power, -slack, sens])
                    trial.area_power_feasible = False
                    trial.sensitivity_feasible = False

                # 局部搜索
                trial = self._local_search(trial, valid_population)
                
                # 【关键修复】：局部搜索修改了gene，必须重新评估！
                if trial.objectives is not None:
                    # 清除之前的评估结果，强制重新评估
                    trial.objectives = None
                
                try:
                    trial = self.evaluator.evaluate_fitness(trial)
                except Exception:
                    # 评估失败时使用简化评估
                    area = np.sum(trial.gene) * 10
                    power = np.sum(trial.gene) * 0.5
                    slack = 0.1 - np.sum(trial.gene * getattr(self.simulator, 'is_critical', np.zeros_like(trial.gene))) * 0.01
                    sens = np.sum(getattr(self.simulator, 'error_rates', np.ones_like(trial.gene)) * (1 - trial.gene))
                    trial.objectives = np.array([area, power, -slack, sens])
                    trial.area_power_feasible = False
                    trial.sensitivity_feasible = False

                trial_population.append(trial)

            # 合并
            combined = valid_population + trial_population

            # 选择
            new_population = self.environmental_selection(combined, [])

        else:
            # ========== 使用传统GA算子 ==========
            # 选择父代
            selected_parents = self.selection(valid_population)

            # 确保有足够的父代
            if len(selected_parents) < 2:
                selected_parents = valid_population[:2]
                if len(selected_parents) < 2:
                    K = getattr(self.simulator, 'K', 30)
                    while len(selected_parents) < 2:
                        random_gene = np.random.randint(0, 2, K)
                        random_ind = Individual(random_gene, None)
                        area = np.sum(random_gene) * 10
                        power = np.sum(random_gene) * 0.5
                        slack = 0.1
                        sens = 0.0
                        random_ind.objectives = np.array([area, power, -slack, sens])
                        selected_parents.append(random_ind)

            # 生成后代
            offspring = []

            for i in range(0, len(selected_parents), 2):
                if i + 1 < len(selected_parents):
                    p1, p2 = selected_parents[i], selected_parents[i + 1]

                    try:
                        # SBX交叉
                        if np.random.rand() < self.pc:
                            child1_gene, child2_gene = self.crossover(p1, p2, generation)
                        else:
                            child1_gene, child2_gene = p1.gene.copy(), p2.gene.copy()

                        # 多项式变异
                        child1_gene = self.mutation(child1_gene, generation)
                        child2_gene = self.mutation(child2_gene, generation)

                        offspring.append(Individual(child1_gene, None))
                        offspring.append(Individual(child2_gene, None))

                    except Exception as e:
                        print(f"生成后代时出错: {e}")
                        K = len(p1.gene) if hasattr(p1, 'gene') else 30
                        random_gene = np.random.randint(0, 2, K)
                        offspring.append(Individual(random_gene.copy(), None))
                        offspring.append(Individual(random_gene.copy(), None))

            # 限制后代数量
            offspring = offspring[:self.pop_size]

            # 【关键修复】：使用真正的评估器评估后代！
            try:
                offspring = self.evaluator.evaluate_population(offspring)
            except Exception as e:
                print(f"[NSGA2] 评估器评估失败: {e}")
                # 评估失败时使用简化评估
                for ind in offspring:
                    if ind.objectives is None:
                        area = np.sum(ind.gene) * 10
                        power = np.sum(ind.gene) * 0.5
                        slack = 0.1 - np.sum(ind.gene * getattr(self.simulator, 'is_critical', np.zeros_like(ind.gene))) * 0.01
                        sens = np.sum(getattr(self.simulator, 'error_rates', np.ones_like(ind.gene)) * (1 - ind.gene))
                        ind.objectives = np.array([area, power, -slack, sens])
                        ind.area_power_feasible = False
                        ind.sensitivity_feasible = False

            # 环境选择
            new_population = self.environmental_selection(valid_population, offspring)

        # 记录时间
        generation_time = time.time() - start_time
        operator_type = "DE" if self.use_de_operator else "GA"
        print(f"NSGA2({operator_type}) 第{generation}代完成，耗时: {generation_time:.2f}秒")

        return new_population
    
    def _parallel_crossover(self, parent1: Individual, parent2: Individual, 
                           pc: float, eta_c: float) -> Tuple[np.ndarray, np.ndarray]:
        """并行交叉工作函数"""
        if np.random.rand() < pc:
            gene1 = parent1.gene.astype(float)
            gene2 = parent2.gene.astype(float)
            
            # 生成随机数
            rand = np.random.rand(len(gene1))
            
            # 计算β值
            beta = np.zeros(len(gene1))
            for i in range(len(gene1)):
                if rand[i] <= 0.5:
                    beta[i] = (2.0 * rand[i]) ** (1.0 / (eta_c + 1.0))
                else:
                    beta[i] = (1.0 / (2.0 * (1.0 - rand[i]))) ** (1.0 / (eta_c + 1.0))
            
            # 生成子代
            child1 = 0.5 * ((1 + beta) * gene1 + (1 - beta) * gene2)
            child2 = 0.5 * ((1 - beta) * gene1 + (1 + beta) * gene2)
            
            # 确保在[0, 1]范围内
            child1 = np.clip(child1, 0, 1)
            child2 = np.clip(child2, 0, 1)
            
            # 转换为二进制
            child1_binary = (child1 > 0.5).astype(int)
            child2_binary = (child2 > 0.5).astype(int)
            
            return child1_binary, child2_binary
        else:
            return parent1.gene.copy(), parent2.gene.copy()
    
    def _parallel_mutation(self, gene: np.ndarray, pm: float, eta_m: float) -> np.ndarray:
        """并行变异工作函数"""
        mutated = gene.astype(float).copy()
        
        for i in range(len(gene)):
            if np.random.rand() < pm:
                # 计算δ值
                rand = np.random.rand()
                if rand < 0.5:
                    delta = (2.0 * rand) ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (eta_m + 1.0))
                
                # 应用变异
                mutated[i] += delta
                mutated[i] = np.clip(mutated[i], 0, 1)
        
        # 转换为二进制
        mutated_binary = (mutated > 0.5).astype(int)
        return mutated_binary
    
    def _estimate_fitness(self, individual: Individual) -> float:
        """
        估计个体的适应度（用于局部搜索）
        """
        if hasattr(individual, 'fitness') and individual.fitness is not None:
            return individual.fitness
        elif hasattr(individual, 'objectives') and individual.objectives is not None:
            # 多目标转单目标：使用加权和
            obj = individual.objectives
            weights = np.array([0.25, 0.25, 0.25, 0.25])  # 默认权重
            return float(np.dot(obj, weights))
        else:
            return float('inf')
    
    def run(self, eval_fn, generations_per_step: int = 1):
        """
        运行NSGA2算法（支持SAC集成）
        
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
        import time
        total_start_time = time.time()
        
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
        
        # NSGA2主循环
        all_fronts = []
        
        for gen in range(self.max_generations):
            # 运行一代
            self.population = self.run_one_generation(self.population, gen)
            
            # 获取当前前沿
            fronts = self.fast_non_dominated_sort(self.population)
            front0 = fronts[0] if fronts else []
            
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
            
            # 记录参考网格覆盖率
            if front0:
                coverage = self._calculate_coverage(front0)
                self.generation_stats['coverage_ratio'].append(coverage)
        
        # 返回结果
        total_time = time.time() - total_start_time
        print(f"NSGA2总运行时间: {total_time:.2f}秒")
        
        final_pop = np.array([ind.gene for ind in self.population])
        stats = self.get_statistics()
        
        return final_pop, all_fronts, stats
    
    def _calculate_hypervolume(self, front: List[Individual], ref_point: np.ndarray = None) -> float:
        """
        计算超体积
        """
        if not front:
            return 0.0
        
        # 提取目标值
        objectives = np.array([ind.objectives for ind in front])
        
        if ref_point is None:
            ref_point = np.max(objectives, axis=0) * 1.1
        
        # 计算每个解的超体积贡献（简化）
        hv = 0.0
        for obj in objectives:
            # 计算到参考点的矩形体积
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
    
    def _calculate_coverage(self, front: List[Individual], grid_size: int = 10) -> float:
        """
        计算参考网格覆盖率
        """
        if not front:
            return 0.0
        
        # 提取目标值
        objectives = np.array([ind.objectives for ind in front])
        n_objectives = objectives.shape[1]
        
        # 归一化目标值
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        norm_objs = (objectives - min_vals) / range_vals
        
        # 创建网格
        grid_cells = set()
        for obj in norm_objs:
            # 将目标值映射到网格单元格
            cell = tuple((obj * grid_size).astype(int).tolist())
            grid_cells.add(cell)
        
        # 计算覆盖率
        total_cells = grid_size ** n_objectives
        coverage = len(grid_cells) / total_cells if total_cells > 0 else 0.0
        
        return coverage
    
    def get_state_vector(self) -> np.ndarray:
        """
        获取状态向量（供SAC使用）
        
        返回:
        ----------
        np.ndarray: 状态向量
        """
        stats = self.get_statistics()
        
        # 提取关键指标
        state = []
        
        # Pareto指标
        state.append(stats.get('hv', 0.0))
        state.append(stats.get('igd', 0.0))
        
        # 拥挤距离统计
        state.append(stats.get('crowding_mean', 0.0))
        state.append(stats.get('crowding_var', 0.0))
        
        # 可行率和约束违约
        state.append(stats.get('feasible_rate', 0.0))
        state.append(stats.get('max_violation', 0.0))
        state.append(stats.get('avg_violation', 0.0))
        
        # 覆盖率
        state.append(stats.get('coverage_ratio', 0.0))
        
        # 前沿大小信息（如果有多前沿）
        front_sizes = stats.get('front_sizes', [])
        if front_sizes:
            # 添加第一前沿大小和前沿数量
            state.append(len(front_sizes[0]) if front_sizes[0] else 0)
            state.append(len(front_sizes))
        else:
            state.extend([0.0, 0.0])
        
        # 参数历史（最近的变化）
        param_history = stats.get('param_history', {})
        for param_name in ['epsilon_dom', 'delta_div', 'beta_penalty', 'pm', 'pc']:
            if param_name in param_history:
                # 取最近5个值的平均值
                recent_vals = param_history[param_name][-5:] if len(param_history[param_name]) > 0 else [0.0]
                state.append(np.mean(recent_vals))
            else:
                state.append(0.0)
        
        return np.array(state, dtype=np.float32)