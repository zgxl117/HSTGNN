# utils/mutation_strategy.py
import numpy as np
from typing import Optional

class UnifiedMutationStrategy:
    """统一的变异策略，考虑寄存器关键性和敏感度"""
    
    def __init__(self, 
                 error_rates: np.ndarray,
                 is_critical: np.ndarray,
                 slack: np.ndarray,
                 pm: float = 0.01,
                 pc: float = 0.9):
        self.error_rates = error_rates
        self.is_critical = is_critical
        self.slack = slack
        self.pm = pm
        self.pc = pc
        
        # 计算变异权重
        self._calculate_mutation_weights()
    
    def _calculate_mutation_weights(self):
        """基于敏感度和时序计算变异权重"""
        # 高敏感度寄存器变异概率更高
        sensitivity_weights = self.error_rates / np.max(self.error_rates)
        
        # 时序紧张（slack小）的寄存器变异更谨慎
        slack_weights = self.slack / np.max(self.slack) if np.max(self.slack) > 0 else 1.0
        
        # 关键路径寄存器变异权重调整
        critical_factor = 1.0 - 0.3 * self.is_critical  # 关键路径降低30%变异概率
        
        # 综合权重
        self.mutation_weights = sensitivity_weights * slack_weights * critical_factor
        self.mutation_weights = self.mutation_weights / np.max(self.mutation_weights)
    
    def mutate(self, gene: np.ndarray, generation: int = 0, *args, **kwargs) -> np.ndarray:
        """基于权重的智能变异 - 向量化版本

        向后兼容：接受额外的 *args 和 **kwargs（例如 current_fitness、父代信息等），不会抛出错误。
        如果 generation 被以关键字形式传入（例如 mutate(gene, generation=...)），仍然能正确处理。
        """
        # 支持通过 kwargs 传入 generation（兼容旧调用）
        if 'generation' in kwargs and kwargs['generation'] is not None:
            try:
                generation = int(kwargs['generation'])
            except Exception:
                pass

        mutated = gene.copy()

        # 动态调整变异概率：前期高变异，后期低变异
        adaptive_pm = self.pm * (1.0 - generation / 1000) if generation < 1000 else self.pm * 0.1

        # 基于权重的变异概率（向量化计算）
        weighted_pm = adaptive_pm * self.mutation_weights

        # 向量化：使用随机掩码决定哪些位需要翻转
        mutation_mask = np.random.random(len(gene)) < weighted_pm

        # 翻转选中的位
        mutated[mutation_mask] = 1 - mutated[mutation_mask]

        return mutated

    def state_dict(self) -> dict:
        """返回可序列化的状态字典。

        包含关键超参数和内部权重，以便在不同Episode或不同进程间传递经验。
        """
        state = {
            'pm': float(self.pm),
            'pc': float(self.pc),
            # mutation_weights 可能是 numpy 数组，转换为 Python 列表以便序列化
            'mutation_weights': None if self.mutation_weights is None else self.mutation_weights.tolist(),
            'version': 1
        }
        return state

    def load_state(self, state: dict):
        """从之前保存的状态字典中恢复变异器状态。

        接受由 `state_dict()` 生成的字典。对于缺失字段将保持当前值。
        如果包含 'mutation_weights'，将直接使用该权重并跳过重新计算。
        否则将基于当前的 error_rates/is_critical/slack 重新计算权重。
        """
        if not isinstance(state, dict):
            raise TypeError('state must be a dict produced by state_dict()')

        if 'pm' in state and state['pm'] is not None:
            self.pm = float(state['pm'])
        if 'pc' in state and state['pc'] is not None:
            self.pc = float(state['pc'])

        mw = state.get('mutation_weights', None)
        if mw is not None:
            try:
                mw_arr = np.asarray(mw, dtype=float)
                # 简单验证长度一致性
                if mw_arr.shape[0] != self.mutation_weights.shape[0]:
                    # 若长度不匹配，尝试裁剪或补零
                    L = min(mw_arr.shape[0], self.mutation_weights.shape[0])
                    new_mw = np.zeros_like(self.mutation_weights)
                    new_mw[:L] = mw_arr[:L]
                    # 将剩余权重设置为现有权重的平均值以维持合理性
                    if L < len(new_mw):
                        new_mw[L:] = np.mean(self.mutation_weights)
                    self.mutation_weights = new_mw
                else:
                    self.mutation_weights = mw_arr.astype(float)
                # 归一化以避免数值问题
                maxv = np.max(self.mutation_weights)
                if maxv > 0:
                    self.mutation_weights = self.mutation_weights / maxv
            except Exception:
                # 回退到重新计算
                self._calculate_mutation_weights()
        else:
            # 未提供权重，重新计算（基于当前 error_rates/slack/is_critical）
            self._calculate_mutation_weights()


class OptimizedMutationStrategy(UnifiedMutationStrategy):
    """向后兼容的增强变异策略。

    这是一个轻量实现，保留 `UnifiedMutationStrategy` 的核心行为，
    并接受额外参数以便上层算法传入（例如 block_size、elite_preservation_rate、local_search_rate、eta_m）。
    """
    def __init__(self, error_rates: np.ndarray, is_critical: np.ndarray, slack: np.ndarray,
                 pm: float = 0.01, pc: float = 0.9,
                 block_size: int = 1,
                 elite_preservation_rate: float = 0.0,
                 local_search_rate: float = 0.0,
                 eta_m: float = 20.0,
                 **kwargs):
        # 初始化基础属性
        super().__init__(error_rates=error_rates, is_critical=is_critical, slack=slack, pm=pm, pc=pc)
        self.block_size = max(1, int(block_size))
        self.elite_preservation_rate = float(elite_preservation_rate)
        self.local_search_rate = float(local_search_rate)
        self.eta_m = float(eta_m)

    def mutate(self, gene: np.ndarray, generation: int = 0, *args, **kwargs) -> np.ndarray:
        """先执行基于权重的按位变异（利用父类实现），随后根据 local_search_rate 进行区块（block）局部扰动。"""
        mutated = super().mutate(gene, generation, *args, **kwargs)

        # 局部搜索：以 local_search_rate 的概率对一个区块进行扰动（翻转或随机化）
        if self.local_search_rate > 0 and np.random.rand() < self.local_search_rate:
            n = len(mutated)
            b = min(self.block_size, n)
            if b > 0 and n > 0:
                start = np.random.randint(0, n - b + 1)
                # 简单策略：对区块内位进行 50% 概率翻转
                for i in range(start, start + b):
                    if np.random.rand() < 0.5:
                        mutated[i] = 1 - mutated[i]

        return mutated

    def state_dict(self) -> dict:
        s = super().state_dict()
        s.update({
            'block_size': int(self.block_size),
            'elite_preservation_rate': float(self.elite_preservation_rate),
            'local_search_rate': float(self.local_search_rate),
            'eta_m': float(self.eta_m),
            'version': 2
        })
        return s

    def load_state(self, state: dict):
        super().load_state(state)
        if not isinstance(state, dict):
            return
        if 'block_size' in state and state['block_size'] is not None:
            self.block_size = max(1, int(state['block_size']))
        if 'elite_preservation_rate' in state and state['elite_preservation_rate'] is not None:
            self.elite_preservation_rate = float(state['elite_preservation_rate'])
        if 'local_search_rate' in state and state['local_search_rate'] is not None:
            self.local_search_rate = float(state['local_search_rate'])
        if 'eta_m' in state and state['eta_m'] is not None:
            self.eta_m = float(state['eta_m'])