# algorithms/base.py
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from .core_data import Individual

class BaseOptimizer(ABC):
    """多目标优化器基类"""
    def __init__(self, simulator, pop_size: int, max_generations: int):
        self.simulator = simulator
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.population: List[Individual] = []

    @abstractmethod
    def run(self):
        """运行优化算法"""
        pass

    def evaluate_fitness(self, individual: Individual):
        """评估个体适应度，包含约束处理

        【优化目标权重 - 敏感性平方优先策略】
        - 面积: 归一化（1倍）
        - 功耗: 归一化（1倍）
        - 时序: 负WNS（1倍）
        - 敏感性: 平方加权 - 强烈优先选择高敏感性寄存器以满足delta_sens_min约束

        策略：使用 ser^2 作为权重
        - 这确保算法优先选择敏感性更低的解（即加固了更多高敏感寄存器）
        """
        area, power, wns, ser = self.simulator.evaluate(individual.gene)

        # 从配置读取约束阈值
        config = getattr(self.simulator, 'config', {})
        constraints = config.get('constraint_thresholds', {})
        hard_threshold = constraints.get('delta_sens_min', 0.10)  # 默认10%
        
        # 计算纯敏感性比例（用于约束判断）
        pure_sens_ratio = ser / self.simulator.SER_orig if self.simulator.SER_orig > 0 else 0.0
        
        # 敏感性平方权重因子 - 目标：满足残余敏感性约束
        sensitivity_weight = 100.0

        penalty = 0.0
        if wns < 0:  # WNS约束违反
            penalty = 100.0 + abs(wns) * 10
        
        # 如果敏感性超标，施加惩罚
        if pure_sens_ratio > hard_threshold:
            penalty = max(penalty, 100.0 * (pure_sens_ratio - hard_threshold) ** 2)
        
        # 用于MOEA目标的敏感性（包含惩罚）
        raw_sens_for_objective = pure_sens_ratio + penalty
        
        objectives = [
            area / self.simulator.Area_orig + penalty,
            power / self.simulator.Power_orig + penalty,
            -wns + penalty,  # 注意：这里保持负号因为我们要最大化WNS
            sensitivity_weight * raw_sens_for_objective * raw_sens_for_objective  # 敏感性平方加权
        ]
        individual.objectives = np.array(objectives)
        # 存储纯敏感性比例（0~1范围），用于约束计算
        individual.raw_sensitivity = pure_sens_ratio