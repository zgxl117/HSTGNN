# utils/heuristic_init.py
"""
启发式初始化策略 - 优化版 V4
用于生成高质量的初始种群，加快收敛速度

V4核心改进：
1. 使用正确的加权敏感性计算方法
2. 按加权敏感性排序选择要加固的寄存器
3. 确保生成的解满足敏感性约束 (sens_ratio <= 0.4)
"""
import numpy as np
from typing import List, Tuple, Optional
from algorithms.core_data import Individual


class HeuristicInitializer:
    """启发式初始化器 - 生成多个高质量初始解（V4优化版）"""
    
    def __init__(self, K: int, error_rates: np.ndarray, slack: np.ndarray, 
                 is_critical: np.ndarray, model=None, config: dict = None):
        """
        初始化启发式初始化器
        
        参数:
        ----------
        K : int
            基因长度
        error_rates : np.ndarray
            错误率/敏感性数组
        slack : np.ndarray
            时序裕度数组
        is_critical : np.ndarray
            关键路径标记数组
        model : object, optional
            PPA模型（用于约束感知初始化）
        config : dict, optional
            配置参数
        """
        self.K = K
        self.error_rates = np.asarray(error_rates).flatten()
        self.slack = np.asarray(slack).flatten()
        self.is_critical = np.asarray(is_critical).flatten()
        self.model = model
        self.config = config or {}
        
        # 预计算排序索引
        self.sorted_error_indices = np.argsort(self.error_rates)[::-1]
        self.sorted_slack_indices = np.argsort(self.slack)[::-1]
        self.critical_indices = np.where(self.is_critical == 1)[0]
        self.non_critical_indices = np.where(self.is_critical == 0)[0]
        
        # 时序分析
        self.positive_slack_indices = np.where(self.slack > 0)[0]
        self.negative_slack_indices = np.where(self.slack <= 0)[0]
        
        # 敏感性约束参数
        self.delta_sens_min = self.config.get('delta_sens_min', 0.4)
        
        # 计算加权敏感性权重
        self._compute_weighted_sensitivity()
        
        # 计算满足约束所需的加固数量
        self._compute_constraint_params()
        
        # 约束参数
        self.delta_area_max = self.config.get('delta_area_max', 0.2)
        self.delta_power_max = self.config.get('delta_power_max', 0.15)
        self.base_area = self.config.get('base_area', 100000.0)
        self.base_power = self.config.get('base_power', 50.0)
        self.reg_area = self.config.get('reg_area', None)
        self.reg_power = self.config.get('reg_power', None)
        
        # 计算可加固数量
        self._compute_max_hardened_count()
    
    def _compute_weighted_sensitivity(self):
        """
        计算简单均值敏感性（基于错误率）

        【敏感性定义】
        - 整体设计的敏感性 = 所有寄存器敏感性的均值 = (x1 + x2 + ... + xK) / K
        - 加固后敏感性 = (未加固寄存器敏感性之和) / (K + 2*n)

        这里的计算用于：
        1. 按敏感性贡献排序选择要加固的寄存器（使用原始敏感性排序）
        2. 计算满足敏感性约束所需的最小加固数量
        """
        p = self.error_rates.astype(np.float64)

        # 加固前的敏感性：所有原始寄存器敏感性的均值
        self.original_weighted_sens = float(np.mean(p))

        # 按原始敏感性排序（降序）- 敏感性越高的寄存器越应该被加固
        self.sorted_by_weighted_sens = np.argsort(p)[::-1]

        # 敏感性贡献（每个寄存器的敏感性值）
        self.weighted_sens_contribution = p

        print(f"[敏感性分析]")
        print(f"  总敏感性: {np.sum(p):.4f}")
        print(f"  平均敏感性: {self.original_weighted_sens:.4f}")
        print(f"  top 10 敏感性贡献: {np.sum(p[self.sorted_by_weighted_sens[:10]]):.4f}")
    
    def _compute_constraint_params(self):
        """计算满足敏感性约束所需的参数"""
        p = self.error_rates.astype(np.float64)
        total_sens = float(np.sum(p))

        self.target_weighted_sens = self.original_weighted_sens * self.delta_sens_min
        self.need_to_remove = total_sens - self.target_weighted_sens * (self.K + 2)  # 调整

        # 按加权敏感性贡献排序（从大到小）
        sorted_contrib = np.sort(self.weighted_sens_contribution)[::-1]
        cumulative = np.cumsum(sorted_contrib)

        # 找到满足约束所需的最小加固数量
        # 正确公式: (total_sens - cumulative[i]) / (K + 2*(i+1)) <= target_weighted_sens
        self.min_hardened_for_sens = None
        for i in range(self.K):
            remaining_sens = total_sens - cumulative[i]
            denom = self.K + 2 * (i + 1)
            sens_after = remaining_sens / denom
            if sens_after <= self.target_weighted_sens:
                self.min_hardened_for_sens = i + 1
                self.actual_sens_after = sens_after
                break

        if self.min_hardened_for_sens is None:
            self.min_hardened_for_sens = self.K
            self.actual_sens_after = 0.0

        self.min_hardened_ratio = self.min_hardened_for_sens / self.K
        self.actual_sens_ratio = self.actual_sens_after / self.original_weighted_sens if self.original_weighted_sens > 0 else 0.0

        print(f"[敏感性约束分析]")
        print(f"  约束要求: 加权敏感性 <= {self.delta_sens_min} (改善 >= {1-self.delta_sens_min})")
        print(f"  加权敏感性目标: {self.target_weighted_sens:.4f}")
        print(f"  需加固: top {self.min_hardened_for_sens} 个 ({100*self.min_hardened_ratio:.1f}%)")
        print(f"  加固后加权敏感性: {self.actual_sens_after:.4f}")
        print(f"  加固后比例: {self.actual_sens_ratio:.4f}")
    
    def _compute_max_hardened_count(self):
        """计算最大可加固数量"""
        if self.reg_area is not None and self.base_area > 0:
            max_area_increase = self.base_area * self.delta_area_max
            self.max_hardened_by_area = int(max_area_increase / max(np.mean(self.reg_area), 1e-8))
        else:
            self.max_hardened_by_area = self.K
        
        if self.reg_power is not None and self.base_power > 0:
            max_power_increase = self.base_power * self.delta_power_max
            self.max_hardened_by_power = int(max_power_increase / max(np.mean(self.reg_power), 1e-8))
        else:
            self.max_hardened_by_power = self.K
        
        self.max_hardened_count = min(self.max_hardened_by_area, self.max_hardened_by_power)
        self.max_hardened_count = max(1, min(self.max_hardened_count, self.K // 2))
    
    def _calculate_weighted_sens_ratio(self, gene: np.ndarray) -> float:
        """
        计算给定加固方案的敏感性比例（简单均值方法）

        【敏感性定义】
        - 加固前敏感性 = mean(sensitivity)
        - 加固后敏感性 = (未加固敏感性之和) / (K + 2*n)
        """
        p = self.error_rates.astype(np.float64)
        gene = gene.astype(np.int32)

        # 原始寄存器数量
        K = len(p)

        # 加固的寄存器数量
        num_hardened = int(np.sum(gene))

        # 加固后的敏感性
        # 分子：未加固寄存器的敏感性之和（被加固的寄存器敏感性变为0）
        p_after = p.copy()
        p_after[np.where(gene == 1)] = 0.0
        numerator = float(np.sum(p_after))

        # 分母：原始寄存器数 + 2 * 加固数量（TMR引入的新寄存器敏感性为0）
        denominator = K + 2 * num_hardened

        if denominator > 0:
            sens_after = numerator / denominator
        else:
            sens_after = 0.0

        return sens_after / self.original_weighted_sens if self.original_weighted_sens > 0 else 0.0
    
    def generate_heuristic_population(self, pop_size: int = 10, seed: int = None) -> List[Individual]:
        """生成初始种群（V4优化版）"""
        if seed is not None:
            np.random.seed(seed)
        
        population = []
        generated_genes = set()
        
        print(f"\n[启发式初始化 V4] 生成初始种群 (K={self.K})")
        
        # ============ 策略1: 精确满足敏感性约束 ============
        # 按加权敏感性排序选择
        gene = self._protect_by_weighted_sens(ratio=self.min_hardened_ratio)
        if self._is_valid_gene(gene) and tuple(gene) not in generated_genes:
            sens_ratio = self._calculate_weighted_sens_ratio(gene)
            status = "满足" if sens_ratio <= self.delta_sens_min else "违例"
            print(f"  [S1 精确] 加固{self.min_hardened_for_sens}个: sens_ratio={sens_ratio:.4f} [{status}]")
            population.append(Individual(gene.copy(), None))
            generated_genes.add(tuple(gene))
        
        # ============ 策略2-6: 不同比例的敏感性满足 ============
        greedy_seed_count = int(self.config.get('greedy_prefix_seed_count', max(8, pop_size // 5)))
        greedy_seed_count = max(0, min(greedy_seed_count, max(1, int(pop_size * 0.35))))
        for gene in self._generate_budget_greedy_prefixes(greedy_seed_count):
            if len(population) >= pop_size:
                break
            key = tuple(gene)
            if self._is_valid_gene(gene) and key not in generated_genes:
                population.append(Individual(gene.copy(), None))
                generated_genes.add(key)

        test_ratios = [0.30, 0.35, 0.40, 0.45, 0.50]
        for ratio in test_ratios:
            if len(population) >= pop_size * 0.7:  # 至少70%满足约束
                break
            gene = self._protect_by_weighted_sens(ratio=ratio)
            if self._is_valid_gene(gene) and tuple(gene) not in generated_genes:
                sens_ratio = self._calculate_weighted_sens_ratio(gene)
                status = "满足" if sens_ratio <= self.delta_sens_min else "违例"
                hardened_count = int(self.K * ratio)
                print(f"  [S{ratio:.0%}] 加固{hardened_count}个: sens_ratio={sens_ratio:.4f} [{status}]")
                population.append(Individual(gene.copy(), None))
                generated_genes.add(tuple(gene))
        
        # ============ 策略7-9: 探索性策略 ============
        explore_ratios = [0.20, 0.25, 0.15]
        for ratio in explore_ratios:
            if len(population) >= pop_size * 0.8:
                break
            gene = self._protect_by_weighted_sens(ratio=ratio)
            if self._is_valid_gene(gene) and tuple(gene) not in generated_genes:
                sens_ratio = self._calculate_weighted_sens_ratio(gene)
                status = "满足" if sens_ratio <= self.delta_sens_min else "违例"
                hardened_count = int(self.K * ratio)
                print(f"  [探索{ratio:.0%}] 加固{hardened_count}个: sens_ratio={sens_ratio:.4f} [{status}]")
                population.append(Individual(gene.copy(), None))
                generated_genes.add(tuple(gene))
        
        # ============ 补充随机解 ============
        while len(population) < pop_size:
            if np.random.random() < 0.7:
                gene = self._generate_weighted_sens_feasible()
            else:
                gene = self._generate_constrained_random()
            
            if gene is not None and tuple(gene) not in generated_genes:
                population.append(Individual(gene.copy(), None))
                generated_genes.add(tuple(gene))
        
        # 统计
        sens_feasible = sum(1 for ind in population 
                          if self._calculate_weighted_sens_ratio(ind.gene) <= self.delta_sens_min)
        print(f"[启发式初始化] 生成{pop_size}个解, {sens_feasible}个满足敏感性约束 ({100*sens_feasible/len(population):.1f}%)")
        print(f"[提示] 时序约束需在后续PPA评估中验证")
        
        return population[:pop_size]
    
    def _generate_budget_greedy_prefixes(self, count: int) -> List[np.ndarray]:
        """Generate a resource-stratified greedy front used as MOEA seed solutions."""
        if count <= 0 or self.K <= 0:
            return []

        p = np.asarray(self.error_rates, dtype=np.float64).reshape(-1)
        if p.size != self.K:
            p = np.resize(p, self.K)

        def _positive_vector(value):
            if value is None:
                return np.ones(self.K, dtype=np.float64)
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size != self.K:
                return np.ones(self.K, dtype=np.float64)
            arr = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
            fill = np.nanmedian(arr) if np.any(np.isfinite(arr)) else 1.0
            return np.where(np.isfinite(arr), arr, max(fill, 1e-12))

        area = _positive_vector(self.reg_area)
        power = _positive_vector(self.reg_power)
        area_norm = area / max(np.mean(area), 1e-12)
        power_norm = power / max(np.mean(power), 1e-12)

        slack = np.asarray(self.slack, dtype=np.float64).reshape(-1)
        if slack.size != self.K:
            slack = np.ones(self.K, dtype=np.float64)
        critical = np.asarray(self.is_critical, dtype=np.float64).reshape(-1)
        if critical.size != self.K:
            critical = np.zeros(self.K, dtype=np.float64)

        timing_risk = np.where(slack <= 0.0, 1.0, 0.0)
        cost = 0.55 * area_norm + 0.45 * power_norm
        cost *= 1.0 + 0.35 * critical + 0.25 * timing_risk
        score = p / np.maximum(cost, 1e-12)
        order = np.argsort(score)[::-1]

        max_count = int(max(1, min(self.max_hardened_count, self.K)))
        min_count = int(max(1, min(self.min_hardened_for_sens, max_count)))
        low = max(1, min(max_count, int(round(0.05 * max_count))))
        counts = np.linspace(low, max_count, num=max(2, count))
        counts = np.unique(np.clip(np.rint(counts).astype(int), 1, max_count))
        counts = np.unique(np.concatenate([counts, np.array([min_count], dtype=int)]))

        genes = []
        for n_harden in counts:
            gene = np.zeros(self.K, dtype=np.int32)
            gene[order[:int(n_harden)]] = 1
            genes.append(gene)
        return genes[:count]

    def _is_valid_gene(self, gene: np.ndarray) -> bool:
        """检查基因是否有效"""
        if gene is None or len(gene) != self.K:
            return False
        if np.sum(gene) == 0:
            return False
        return True
    
    def _protect_by_weighted_sens(self, ratio: float) -> np.ndarray:
        """
        按加权敏感性排序精确加固
        
        这是V4的核心改进：按加权敏感性贡献排序，而非单纯的敏感性
        """
        gene = np.zeros(self.K, dtype=np.int32)
        protect_count = max(1, int(self.K * ratio))
        gene[self.sorted_by_weighted_sens[:protect_count]] = 1
        return gene
    
    def _generate_weighted_sens_feasible(self) -> Optional[np.ndarray]:
        """生成满足加权敏感性约束的随机解"""
        gene = np.zeros(self.K, dtype=np.int32)
        
        target_count = self.min_hardened_for_sens
        variation = max(1, int(target_count * 0.1))
        actual_count = target_count + np.random.randint(-variation, variation + 1)
        actual_count = max(1, min(actual_count, self.max_hardened_count))
        
        # 按加权敏感性选择
        indices = self.sorted_by_weighted_sens[:actual_count].copy()
        np.random.shuffle(indices)
        gene[indices] = 1
        
        return gene
    
    def _generate_constrained_random(self) -> Optional[np.ndarray]:
        """生成约束感知的随机解"""
        gene = np.zeros(self.K, dtype=np.int32)
        
        try:
            max_count = min(self.max_hardened_count, self.K // 2)
            n_select = np.random.randint(max(1, max_count // 2), max_count + 1)
            
            # 优先选择高加权敏感性的
            candidates = list(self.sorted_by_weighted_sens[:self.K // 2])
            np.random.shuffle(candidates)
            
            gene[candidates[:min(n_select, len(candidates))]] = 1
        except Exception:
            n_select = max(1, self.K // 10)
            indices = np.random.choice(self.K, size=n_select, replace=False)
            gene[indices] = 1
        
        return gene


def generate_diverse_initial_population(K: int, model=None, config: dict = None) -> List[np.ndarray]:
    """生成多样化的初始种群"""
    np.random.seed(42)
    
    genes = []
    
    for ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        gene = np.zeros(K, dtype=np.int32)
        n_harden = max(1, int(K * ratio))
        
        if config and 'error_rates' in config:
            weights = config['error_rates']
            sorted_idx = np.argsort(weights)[::-1]
            gene[sorted_idx[:n_harden]] = 1
        else:
            indices = np.random.choice(K, size=n_harden, replace=False)
            gene[indices] = 1
        genes.append(gene)
    
    genes.append(np.zeros(K, dtype=np.int32))
    genes.append(np.ones(K, dtype=np.int32))
    
    return genes
