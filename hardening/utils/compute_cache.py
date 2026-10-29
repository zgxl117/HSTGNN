# utils/compute_cache.py
"""
计算缓存和增量更新机制
用于避免重复计算HV、IGD、多样性等昂贵操作
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from algorithms.core_data import Individual


class ComputationCache:
    """计算缓存 - 缓存HV、IGD、多样性等指标以避免重复计算"""
    
    def __init__(self):
        self.cache = {}
        self.last_population_id = None
        self.last_generation = -1
    
    def is_valid(self, population: List[Individual], generation: int) -> bool:
        """检查缓存是否有效"""
        if population is None or not population:
            return False
        
        pop_id = id(population)
        return (pop_id == self.last_population_id and 
                generation == self.last_generation)
    
    def get_hv(self, population: List[Individual], generation: int) -> Optional[float]:
        """获取缓存的HV"""
        if self.is_valid(population, generation):
            return self.cache.get('hv')
        return None
    
    def get_igd(self, population: List[Individual], generation: int) -> Optional[float]:
        """获取缓存的IGD"""
        if self.is_valid(population, generation):
            return self.cache.get('igd')
        return None
    
    def get_diversity(self, population: List[Individual], generation: int) -> Optional[float]:
        """获取缓存的多样性"""
        if self.is_valid(population, generation):
            return self.cache.get('diversity')
        return None
    
    def set(self, population: List[Individual], generation: int, 
            hv: float = None, igd: float = None, diversity: float = None):
        """设置缓存"""
        if population:
            self.last_population_id = id(population)
            self.last_generation = generation
            if hv is not None:
                self.cache['hv'] = hv
            if igd is not None:
                self.cache['igd'] = igd
            if diversity is not None:
                self.cache['diversity'] = diversity
    
    def clear(self):
        """清除缓存"""
        self.cache.clear()
        self.last_population_id = None
        self.last_generation = -1


class IncrementalParetoMaintainer:
    """增量式Pareto前沿维护
    
    而不是每个SAC步后都重新提取Pareto前沿，
    使用增量更新机制：
    1. 初始化时提取一次Pareto前沿
    2. 每代后只添加新的非支配个体
    3. 删除被支配的个体
    """
    
    def __init__(self):
        self.current_pareto: List[Individual] = []
        self.pareto_objectives: np.ndarray = None
        self.last_update_generation = 0
    
    def extract_initial_pareto(self, population: List[Individual]) -> List[Individual]:
        """初始提取Pareto前沿"""
        pareto = self._compute_pareto_front(population)
        self.current_pareto = pareto
        self.pareto_objectives = np.array([ind.objectives for ind in pareto])
        self.last_update_generation = 0
        return pareto
    
    def update_incremental(self, population: List[Individual], generation: int) -> List[Individual]:
        """增量更新Pareto前沿
        
        参数:
        ----------
        population : List[Individual]
            当前种群
        generation : int
            当前代数
            
        返回:
        ----------
        List[Individual]
            更新后的Pareto前沿
        """
        if generation <= self.last_update_generation:
            return self.current_pareto
        
        if not self.current_pareto:
            # 第一次更新：完整提取
            return self.extract_initial_pareto(population)
        
        # 增量更新：候选者 = 当前Pareto + 新种群（不进行去重，允许episode间出现相同解）
        candidates = self.current_pareto + population

        # 提取新的Pareto前沿（不进行去重）
        new_pareto = self._compute_pareto_front(candidates)
        
        # 更新缓存
        self.current_pareto = new_pareto
        self.pareto_objectives = np.array([ind.objectives for ind in new_pareto])
        self.last_update_generation = generation
        
        return new_pareto
    
    def get_current_pareto(self) -> List[Individual]:
        """获取当前缓存的Pareto前沿"""
        return self.current_pareto
    
    @staticmethod
    def _compute_pareto_front(population: List[Individual]) -> List[Individual]:
        """计算Pareto前沿（标准算法）"""
        if not population:
            return []
        
        pareto = []
        for candidate in population:
            is_dominated = False
            for incumbent in pareto:
                if IncrementalParetoMaintainer._dominates(incumbent, candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # 从pareto中移除被candidate支配的个体
                pareto = [p for p in pareto 
                         if not IncrementalParetoMaintainer._dominates(candidate, p)]
                pareto.append(candidate)
        
        return pareto
    
    @staticmethod
    def _dominates(ind1: Individual, ind2: Individual) -> bool:
        """检查ind1是否支配ind2
        
        假设所有目标都是最小化
        """
        obj1 = np.asarray(ind1.objectives)
        obj2 = np.asarray(ind2.objectives)
        
        # 检查是否至少一个目标更好
        better = np.any(obj1 < obj2)
        # 检查是否没有更差的目标
        not_worse = np.all(obj1 <= obj2)
        
        return better and not_worse
    
    @staticmethod
    def _deduplicate(population: List[Individual]) -> List[Individual]:
        """基于基因去重"""
        seen_genes = set()
        unique = []
        
        for ind in population:
            gene_tuple = tuple(ind.gene)
            if gene_tuple not in seen_genes:
                seen_genes.add(gene_tuple)
                unique.append(ind)
        
        return unique


class PerformanceProfiler:
    """性能分析 - 追踪各部分耗时"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.call_counts: Dict[str, int] = {}
        self.total_times: Dict[str, float] = {}
    
    def record(self, name: str, elapsed_time: float):
        """记录耗时"""
        if name not in self.timings:
            self.timings[name] = []
            self.call_counts[name] = 0
            self.total_times[name] = 0.0
        
        self.timings[name].append(elapsed_time)
        self.call_counts[name] += 1
        self.total_times[name] += elapsed_time
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取统计摘要"""
        summary = {}
        for name in self.timings:
            times = self.timings[name]
            summary[name] = {
                'count': self.call_counts[name],
                'total': self.total_times[name],
                'mean': np.mean(times),
                'min': np.min(times),
                'max': np.max(times),
                'std': np.std(times)
            }
        return summary
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        print("\n=== 性能统计 ===")
        for name, stats in sorted(summary.items(), key=lambda x: -x[1]['total']):
            print(f"{name}:")
            print(f"  调用次数: {stats['count']}")
            print(f"  总耗时: {stats['total']:.3f}s")
            print(f"  平均: {stats['mean']:.6f}s, 最小: {stats['min']:.6f}s, 最大: {stats['max']:.6f}s")
