# algorithms/core_data.py
import numpy as np
from typing import List, Set, Optional

class Individual:
    """表示种群中的一个个体（解）"""
    def __init__(self, gene: np.ndarray, objectives: Optional[np.ndarray] = None):
        # 确保基因是一维数组且长度正确
        self.gene = np.array(gene, dtype=int).flatten()
        self.objectives = np.array(objectives) if objectives is not None else None
        # Ensure violation attribute always exists (default 0.0)
        self.violation = 0.0
        # 面积/功耗可行性标记（True=不违例，False=违例）
        self.area_power_feasible = True  # 默认设为True，让MOEA自己优化
        # 敏感性可行性标记（True=满足sens_ratio<=0.5约束，False=违例）
        self.sensitivity_feasible = True  # 默认设为True，让MOEA自己优化
        # NSGA-II attributes
        self.rank = 0
        self.crowding_distance = 0.0
        # SPEA2 attributes
        self.fitness = 0.0
        # General attributes
        self.domination_count = 0
        self.dominated_solutions: Set[Individual] = set()
        # 原始敏感性值（用于约束计算，不受适应度权重影响）
        self.raw_sensitivity = None
        # 【新增】评估状态标记：避免重复评估
        self.evaluated = objectives is not None  # 如果初始化时提供了objectives，则认为已评估
        # 【新增】评估缓存哈希：用于快速查重
        self._cache_key = None
        # 【新增】原始目标值：保存评估前的原始目标值，用于约束违例判断（不受惩罚影响）
        self.original_objectives = None

    def get_cache_key(self) -> str:
        """生成基因的缓存键，用于快速查重"""
        if self._cache_key is None:
            self._cache_key = ''.join(map(str, self.gene.astype(int)))
        return self._cache_key

    def __repr__(self):
        return f"Individual(gene_len={len(self.gene)}, objectives={self.objectives})"

def dominates(p: Individual, q: Individual) -> bool:
    """检查解p是否支配解q"""
    if p.objectives is None or q.objectives is None:
        return False
    return np.all(p.objectives <= q.objectives) and np.any(p.objectives < q.objectives)

def dominates_epsilon(obj_p, obj_q, eps: float) -> bool:
    """
    ε-dominance 检查
    对所有维度：obj_p[i] <= obj_q[i] + eps
    且至少一维严格更好（小于 q[i] - eps）

    参数:
    ---------
    obj_p : np.ndarray
        个体p的目标值
    obj_q : np.ndarray
        个体q的目标值
    eps : float
        ε-支配容忍度

    返回:
    --------
    bool: 如果p ε-支配q则返回True
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
    if np.any(np.isnan(obj_p)) or np.any(np.isinf(obj_p)) or \
       np.any(np.isnan(obj_q)) or np.any(np.isinf(obj_q)):
        return False

    # 如果 eps 很小，使用标准支配
    if eps < 1e-12:
        return np.all(obj_p <= obj_q) and np.any(obj_p < obj_q)

    # ε-支配检查
    cond1 = np.all(obj_p <= obj_q + eps)
    cond2 = np.any(obj_p < obj_q - eps)
    return cond1 and cond2


def dominates_epsilon_nsga2(obj_p, obj_q, eps: float) -> bool:
    """NSGA2 ε-dominance 检查"""
    if obj_p is None or obj_q is None:
        return False

    if not isinstance(obj_p, np.ndarray):
        obj_p = np.array(obj_p)
    if not isinstance(obj_q, np.ndarray):
        obj_q = np.array(obj_q)

    if obj_p.shape != obj_q.shape:
        return False

    if np.any(np.isnan(obj_p)) or np.any(np.isinf(obj_p)) or \
       np.any(np.isnan(obj_q)) or np.any(np.isinf(obj_q)):
        return False

    if eps < 1e-12:
        return np.all(obj_p <= obj_q) and np.any(obj_p < obj_q)

    cond1 = np.all(obj_p <= obj_q + eps)
    cond2 = np.any(obj_p < obj_q - eps)
    return cond1 and cond2


def non_dominated_sort(individuals: List[Individual]) -> List[List[Individual]]:
    """非支配排序，将个体分为不同的 Pareto 前沿层"""
    if not individuals:
        return []

    n = len(individuals)
    domination_count = [0] * n
    dominated_set = [set() for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ind_i, ind_j = individuals[i], individuals[j]
            if ind_i.objectives is None or ind_j.objectives is None:
                continue

            if dominates(ind_i, ind_j):
                dominated_set[i].add(j)
            elif dominates(ind_j, ind_i):
                domination_count[i] += 1

        if domination_count[i] == 0:
            ind_i.rank = 0
            fronts[0].append(ind_i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for ind_i in fronts[current_front]:
            for j in dominated_set[individuals.index(ind_i)]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    individuals[j].rank = current_front + 1
                    next_front.append(individuals[j])
        current_front += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def select_pareto_elites(individuals: List[Individual], max_elites: int = 10,
                          diversity_epsilon: float = 0.01) -> List[Individual]:
    """从精英池中选择真正的 Pareto 最优解，并保持多样性"""
    if not individuals:
        return []

    evaluated = [ind for ind in individuals if ind.evaluated and ind.objectives is not None]
    if not evaluated:
        return []

    pareto_fronts = non_dominated_sort(evaluated)
    if not pareto_fronts:
        return []

    pareto_front = pareto_fronts[0]

    if len(pareto_front) <= max_elites:
        return pareto_front

    for ind in pareto_front:
        ind.crowding_distance = 0.0

    n_objectives = len(pareto_front[0].objectives)

    for m in range(n_objectives):
        sorted_front = sorted(pareto_front, key=lambda x: x.objectives[m])
        sorted_front[0].crowding_distance = float('inf')
        sorted_front[-1].crowding_distance = float('inf')
        obj_range = sorted_front[-1].objectives[m] - sorted_front[0].objectives[m]
        if obj_range > 0:
            for i in range(1, len(sorted_front) - 1):
                sorted_front[i].crowding_distance += (
                    sorted_front[i + 1].objectives[m] - sorted_front[i - 1].objectives[m]
                ) / obj_range

    selected = sorted(pareto_front, key=lambda x: -x.crowding_distance)[:max_elites]

    final_selected = []
    for ind in selected:
        too_close = False
        for selected_ind in final_selected:
            if np.allclose(ind.objectives, selected_ind.objectives, atol=diversity_epsilon):
                too_close = True
                break
        if not too_close:
            final_selected.append(ind)

    return final_selected


def calculate_diversity_score(individuals: List[Individual]) -> float:
    """计算种群的多样性得分（基于目标空间的分布）"""
    if len(individuals) < 2:
        return 0.0

    evaluated = [ind for ind in individuals if ind.objectives is not None]
    if len(evaluated) < 2:
        return 0.0

    total_distance = 0.0
    count = 0

    for i in range(len(evaluated)):
        for j in range(i + 1, len(evaluated)):
            dist = np.linalg.norm(evaluated[i].objectives - evaluated[j].objectives)
            total_distance += dist
            count += 1

    return total_distance / count if count > 0 else 0.0
    """
    NSGA2 ε-dominance 检查

    参数:
    ---------
    obj_p : np.ndarray
        个体p的目标值
    obj_q : np.ndarray
        个体q的目标值
    eps : float
        ε-支配容忍度

    返回:
    --------
    bool: 如果p ε-支配q则返回True
    """
    if obj_p is None or obj_q is None:
        return False

    if not isinstance(obj_p, np.ndarray):
        obj_p = np.array(obj_p)
    if not isinstance(obj_q, np.ndarray):
        obj_q = np.array(obj_q)

    if obj_p.shape != obj_q.shape:
        return False

    if np.any(np.isnan(obj_p)) or np.any(np.isinf(obj_p)) or \
       np.any(np.isnan(obj_q)) or np.any(np.isinf(obj_q)):
        return False

    if eps < 1e-12:
        return np.all(obj_p <= obj_q) and np.any(obj_p < obj_q)

    cond1 = np.all(obj_p <= obj_q + eps)
    cond2 = np.any(obj_p < obj_q - eps)
    return cond1 and cond2