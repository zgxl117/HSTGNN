# env/real_moea_wrapper2.py
import numpy as np
from typing import List, Tuple, Dict, Any
from algorithms.spea2_parallel import SPEA2Parallel
from algorithms.nsga2_parallel import NSGA2Parallel
from algorithms.core_data import Individual, select_pareto_elites, calculate_diversity_score

class PPAModel:
    """简单的寄存器属性模型，用于模拟评估计算"""
    def __init__(self, reg_table: Dict[str, Any] = None):
        reg_table = reg_table or {}
 
        # K: 预筛选寄存器数量
        self.K = int(reg_table.get('K', len(reg_table.get('area', [])) or 200))

        # 基本设计基准值
        self.design_area = float(reg_table.get('design_area', reg_table.get('area_orig', 100000.0)))
        self.design_power = float(reg_table.get('design_power', reg_table.get('power_orig', 50.0)))
        self.design_slack = float(reg_table.get('design_slack', reg_table.get('slack_orig', 0.05)))
        self.design_sensitivity = float(reg_table.get('design_sensitivity', reg_table.get('sens_orig', 1e-5)))
        # 【关键】：添加sens_orig属性，确保parallel_evaluator能正确获取敏感性基准值
        self.sens_orig = float(reg_table.get('sens_orig', self.design_sensitivity))
        # 【关键】：添加area_orig, power_orig, slack_orig属性，确保parallel_evaluator能正确获取
        self.area_orig = float(reg_table.get('area_orig', self.design_area))
        self.power_orig = float(reg_table.get('power_orig', self.design_power))
        self.slack_orig = float(reg_table.get('slack_orig', self.design_slack))

        # 基本后备值（用于出现NaN/Inf时的替代）
        self.base_area = float(reg_table.get('base_area', self.design_area))
        self.base_power = float(reg_table.get('base_power', self.design_power))
        self.base_slack = float(reg_table.get('base_slack', self.design_slack))

        # 寄存器向量属性（按长度K填充默认值）
        def _arr(key, default):
            v = reg_table.get(key, None)
            if v is None:
                return np.asarray(default, dtype=np.float32)
            return np.asarray(v, dtype=np.float32)

        self.area = _arr('area', np.ones(self.K, dtype=np.float32))
        self.power = _arr('power', np.ones(self.K, dtype=np.float32) * 0.5)
        self.slack = _arr('slack', np.ones(self.K, dtype=np.float32) * 0.1)
        self.sensitivity = _arr('sensitivity', np.random.rand(self.K).astype(np.float32))
        self.is_critical = _arr('is_critical', np.zeros(self.K, dtype=np.float32))

        # 预计算的权重/比例 - 缓存 np.sum() 结果避免重复计算
        _sens_sum = float(np.sum(self.sensitivity))
        total_sens = _sens_sum if _sens_sum > 0 else 1.0
        self.sensitivity_weights = self.sensitivity / total_sens

        # 新算法：sensitivity 直接作为 p_i（失效率），无需 base_total_failure_rate 映射
        # 原始总敏感性 = 平均失效率 = (Σ p_i) / n，归一化到 [0, 1]
        self.base_total_failure_rate = float(reg_table.get('base_total_failure_rate', 1e-2))  # 保留以兼容，但不再使用
        
        # 存储总寄存器数（包括所有寄存器，不仅仅是预筛选的K个）
        # 这用于计算敏感性时的正确分母
        self.total_num_registers = int(reg_table.get('total_num_registers', self.K))
        
        # 存储所有寄存器的敏感性总和（包括非预筛选的）
        # 如果不提供，则假设只有预筛选的寄存器
        self.total_sensitivity_sum = float(reg_table.get('total_sensitivity_sum', np.sum(self.sensitivity)))
        self.non_prefiltered_sensitivity_sum = float(reg_table.get('non_prefiltered_sensitivity_sum', 0.0))
        
        # 读取可选的 frequency 字段（单位与 slack 一致，frequency = 1/period）
        # 用户可以在寄存器表或配置中提供 frequency
        try:
            self.frequency = float(reg_table.get('frequency', reg_table.get('frequency_orig', 200e6)))
        except Exception:
            self.frequency = 200e6

        # 预筛选寄存器的索引/名称/类型
        self.original_indices = np.asarray(reg_table.get('original_indices', np.arange(self.K)))
        self.instance_names = np.asarray(reg_table.get('instance_names', [f"reg_{i}" for i in range(self.K)]))
        self.reg_types = np.asarray(reg_table.get('reg_types', ["FF"] * self.K))

        # TMR相关的默认参数
        config = reg_table.get('config', {})
        constraint_thresholds = config.get('constraint_thresholds', {}) if isinstance(config, dict) else {}
        
        # 如果reg_table中没有config，尝试从直接配置获取
        if not constraint_thresholds:
            constraint_thresholds = reg_table.get('constraint_thresholds', {})
        voter_cfg = config.get('voter', {}) if isinstance(config, dict) else {}
        
        # 如果顶层没有voter配置，检查optimization.voter
        if not voter_cfg and isinstance(config, dict):
            optimization_cfg = config.get('optimization', {})
            voter_cfg = optimization_cfg.get('voter', {}) if isinstance(optimization_cfg, dict) else {}
        
        self.voter_area = float(voter_cfg.get('voter_area', reg_table.get('voter_area', 0.5)))
        self.voter_power = float(voter_cfg.get('voter_power', reg_table.get('voter_power', 0.2)))
        self.voter_delay = float(voter_cfg.get('voter_delay', reg_table.get('voter_delay', 0.13)))
        
        # 投票器失效率
        self.voter_failure_rate = float(constraint_thresholds.get('voter_failure_rate', 1e-6))
        
        # 如果reg_table中有config，从config中读取；否则使用默认值
        config = reg_table.get('config', {})
        constraint_thresholds = config.get('constraint_thresholds', {}) if isinstance(config, dict) else {}
        
        # 如果reg_table中没有config，尝试从全局获取（通过reg_table的特殊键）
        if not constraint_thresholds:
            # 尝试从reg_table直接读取（向后兼容）
            constraint_thresholds = reg_table.get('constraint_thresholds', {})
        
        # 混合加权方法的参数（固定值，不使用配置）
        self.timing_weight_epsilon = 0.01  # 防止除零的小值（ns）
        self.timing_weight_beta = 1.0      # 时序权重系数
        self.critical_weight_alpha = 0.5   # 关键路径权重系数

        # 方便的冗余字段
        self.area_orig = float(reg_table.get('area_orig', self.design_area))
        self.power_orig = float(reg_table.get('power_orig', self.design_power))
        self.slack_orig = float(reg_table.get('slack_orig', self.design_slack))
        
        # 使用混合加权方法计算原始总敏感性（基于预筛选的K个寄存器）
        # 先计算 original_total_sensitivity，因为它会被后续代码使用
        zero_bitvec = np.zeros(self.K, dtype=np.int32)
        _, self.original_total_sensitivity = self._calculate_sensitivity_hybrid(zero_bitvec)
        
        # 默认将 sens_orig 与原始总敏感性保持一致（除非显式提供）
        self.sens_orig = float(reg_table.get('sens_orig', self.original_total_sensitivity))
        # design_sensitivity 保持与原始总敏感性一致，便于归一化计算
        self.design_sensitivity = float(reg_table.get('design_sensitivity', self.original_total_sensitivity))

        # 启动时一致性检查：如果投票器延时明显大于最小路径 slack，会导致大规模时序违例
        try:
            min_slack = float(np.min(self.slack)) if self.slack.size > 0 else None
        except Exception:
            min_slack = None

        if min_slack is not None:
            if self.voter_delay > min_slack:
                # 不自动修改模型参数，但发出警告：真实设计中这意味着可能产生负 slack
                clamped = max(1e-6, min_slack * 0.5)
                print(f"[WARN] 初始检测到 voter_delay ({self.voter_delay}) > min_slack ({min_slack})，这可能导致大量时序违例。建议在配置或寄存器表中调整 voter_delay 或 slack (示例安全值: {clamped:.6f})")
            else:
                print(f"[INFO] voter_delay={self.voter_delay}, min_slack={min_slack}")
        else:
            print(f"[INFO] 无法确定 min_slack，保持 voter_delay={self.voter_delay}")
    
    def _calculate_sensitivity_hybrid(self, bitvec):
        """
        计算加固后的系统级敏感性（基于错误率的简单均值方法）

        【敏感性定义】
        - 假设对每个寄存器注错M次，总注错次数 = K * M
        - 假设寄存器Ri的敏感性为xi（即注错M次有M*xi次会导致错误）
        - 整体设计的敏感性 = 总错误次数 / 总注错次数 = (x1 + x2 + ... + xK) / K
        - 即所有寄存器敏感性的简单均值

        【加固后敏感性】
        - 被加固的寄存器敏感性变为0
        - TMR加固会引入2个新寄存器，其敏感性也为0
        - 分母变为 K + 2*y（K为原始寄存器数，y为加固数量）

        参数:
        ----------
        bitvec : np.ndarray
            加固向量，1表示被加固

        返回:
        ----------
        tuple: (design_sensitivity_after, design_sensitivity_before)
        """
        bitvec = bitvec.astype(np.int32)

        # 总寄存器数量（所有寄存器，不仅仅是预筛选的K个）
        K_total = self.total_num_registers if hasattr(self, 'total_num_registers') else self.K

        # 所有寄存器的敏感性总和（来自reg_table）
        total_sens_sum = self.total_sensitivity_sum if hasattr(self, 'total_sensitivity_sum') else float(np.sum(self.sensitivity))

        # 加固前的敏感性：基于所有寄存器的简单均值
        design_sensitivity_before = total_sens_sum / K_total if K_total > 0 else 0.0

        # 加固的寄存器数量（基于预筛选的寄存器）
        num_hardened = int(np.sum(bitvec))

        if num_hardened == 0:
            design_sensitivity_after = design_sensitivity_before
        else:
            # 计算被加固寄存器的敏感性贡献
            # 需要从total_sensitivity中减去被加固的寄存器敏感性
            hardened_sens = float(np.sum(self.sensitivity[np.where(bitvec == 1)]))
            remaining_sens_sum = total_sens_sum - hardened_sens

            # 分母：原始寄存器数 + 2 * 加固数量（TMR引入的新寄存器敏感性为0）
            denominator = K_total + 2 * num_hardened

            if denominator > 0:
                design_sensitivity_after = remaining_sens_sum / denominator
            else:
                design_sensitivity_after = 0.0

        return design_sensitivity_after, design_sensitivity_before
    
    def evaluate_with_details(self, bitvec):
        """
        评估加固方案，返回详细结果

        返回:
        ----------
        tuple: (area_total, power_total, min_slack, total_sensitivity,
                reinforced_indices, area_increase, power_increase, slack_decrease,
                sens_improvement_ratio)
        """
        # print(f"\n=== DEBUG[PPAModel.evaluate_with_details] ===")
        # print(f"bitvec 长度: {len(bitvec)}, 加固寄存器数: {np.sum(bitvec)}")

        bitvec = np.array(bitvec).astype(np.int32)

        # 确保长度正确
        if bitvec.size != self.K:
            if bitvec.size > self.K:
                bitvec = bitvec[:self.K]
            else:
                bitvec = np.pad(bitvec, (0, self.K - bitvec.size), 'constant')

        # 计算面积
        tmr_area_cost = self.area * 2 + self.voter_area
        area_increase = np.sum(tmr_area_cost * bitvec)
        area_total = self.design_area + area_increase

        # 计算功耗
        tmr_power_cost = self.power * 2 + self.voter_power
        power_increase = np.sum(tmr_power_cost * bitvec)
        power_total = self.design_power + power_increase

        # 计算时序
        # 对所有被加固的寄存器所在路径都扣除投票器延时
        # 这样才能正确反映"非关键路径加固后可能成为新关键路径"的情况
        slack_after = self.slack.copy()
        tmr_indices = np.where(bitvec == 1)[0]  # 所有被加固的寄存器
        num_hardened = int(tmr_indices.size)
        if num_hardened > 0:
            slack_after[tmr_indices] -= self.voter_delay

        # 计算真实最差slack（可能为负，用于违例/约束）
        # 这个min_slack就是加固后的全局最差slack，已经考虑了关键路径可能变化的情况
        min_slack_raw = float(np.min(slack_after))
        original_min_slack = float(np.min(self.slack))
        slack_decrease = original_min_slack - min_slack_raw

        min_slack = min_slack_raw

        # ========== 简单均值敏感性计算（基于错误率）==========
        # 【敏感性定义】
        # - 假设对每个寄存器注错M次，总注错次数 = K * M
        # - 假设寄存器Ri的敏感性为xi（即注错M次有M*xi次会导致错误）
        # - 整体设计的敏感性 = 总错误次数 / 总注错次数 = (x1 + x2 + ... + xK) / K
        # - 即所有寄存器敏感性的简单均值

        # 总寄存器数量（所有寄存器，不仅仅是预筛选的K个）
        K_total = self.total_num_registers if hasattr(self, 'total_num_registers') else len(self.sensitivity)

        # 所有寄存器的敏感性总和（来自reg_table）
        total_sens_sum = self.total_sensitivity_sum if hasattr(self, 'total_sensitivity_sum') else float(np.sum(self.sensitivity))

        # 加固前的敏感性：基于所有寄存器的简单均值
        original_sensitivity = total_sens_sum / K_total if K_total > 0 else 0.0

        if num_hardened == 0:
            total_sensitivity = original_sensitivity
        else:
            # 计算被加固寄存器的敏感性贡献
            hardened_sens = float(np.sum(self.sensitivity[tmr_indices]))
            remaining_sens_sum = total_sens_sum - hardened_sens

            # 分母：原始寄存器数 + 2 * 加固数量（TMR引入的新寄存器敏感性为0）
            denominator = K_total + 2 * num_hardened

            if denominator > 0:
                total_sensitivity = remaining_sens_sum / denominator
            else:
                total_sensitivity = 0.0

        # 敏感性改善比例（用于快速量化）
        sens_improvement_ratio = 1.0 - (total_sensitivity / max(original_sensitivity, 1e-20))

        # 被加固的寄存器索引
        reinforced_indices = tmr_indices

        return (area_total, power_total, min_slack, total_sensitivity,
            reinforced_indices, area_increase, power_increase, slack_decrease,
            sens_improvement_ratio)
    
    def evaluate(self, bitvec):
        """
        输入:
            bitvec: 长度 = num_regs 的 0/1 向量
                1 → 对该寄存器 TMR 加固
        输出:
            (area_total, power_total, min_slack, normalized_sensitivity)
        """
        # 使用新方法计算，但只返回4个值以保持兼容性
        (area_total, power_total, min_slack, total_sensitivity,
         _, _, _, _, _) = self.evaluate_with_details(bitvec)

        # 计算归一化的敏感度：以原始总敏感性为基准，返回比例值（0~1）
        normalized_sens = total_sensitivity / max(self.original_total_sensitivity, 1e-10)

        # 检查并替换可能的NaN值
        if np.isnan(area_total) or np.isinf(area_total):
            area_total = self.base_area
        if np.isnan(power_total) or np.isinf(power_total):
            power_total = self.base_power
        if np.isnan(min_slack) or np.isinf(min_slack):
            min_slack = self.base_slack
        if np.isnan(normalized_sens) or np.isinf(normalized_sens):
            normalized_sens = 1.0

        # 【修复】：不再在这里对slack应用惩罚
        # 惩罚应该由MOEA的约束处理机制来处理，不应该影响评估值
        # 违例检测和惩罚应该在 _calculate_constraint_violations 中统一处理

        return area_total, power_total, min_slack, normalized_sens

    def evaluate_population(self, genes):
        """
        批量评估种群（供外部调用）

        输入:
            genes: 个体列表，每个个体可以是 numpy数组 或 有.gene属性的对象
        输出:
            np.ndarray: shape = (N, 4)，每行是 [area, power, -slack, normalized_sensitivity]
        """
        # 提取所有基因数组
        gene_arrays = []
        for gene in genes:
            if hasattr(gene, 'gene'):
                gene_arrays.append(gene.gene)
            else:
                gene_arrays.append(gene)

        # 确保长度一致
        max_len = max(len(g) for g in gene_arrays) if gene_arrays else self.K
        max_len = max(max_len, self.K)

        # 填充到相同长度
        padded_genes = []
        for g in gene_arrays:
            g = np.asarray(g, dtype=np.int32)
            if len(g) < max_len:
                g = np.pad(g, (0, max_len - len(g)), 'constant', constant_values=0)
            elif len(g) > max_len:
                g = g[:max_len]
            padded_genes.append(g)

        # 转换为2D数组 (N, K)
        genes_matrix = np.array(padded_genes, dtype=np.int32)

        # 调用批量向量化评估
        return self.evaluate_batch_vectorized(genes_matrix)

    def evaluate_batch_vectorized(self, genes_batch):
        """
        批量向量化评估 - 使用einsum加速

        输入:
            genes_batch: shape = (N, K) 的2D numpy数组，N是个体数，K是基因位数
                        或者是一个长度为N的列表，每个元素是长度为K的bit向量
        输出:
            np.ndarray: shape = (N, 4)，每行是 [area, power, -slack, normalized_sensitivity]

        性能优化：
        - 使用einsum代替np.dot加速矩阵乘法
        - 完全向量化，无Python循环
        - 预计算常量，避免重复计算
        """
        # 转换为numpy数组
        if isinstance(genes_batch, list):
            genes_batch = np.array(genes_batch, dtype=np.int32)
        genes_batch = np.asarray(genes_batch, dtype=np.int32)

        # 确保是2D数组 (N, K)
        if genes_batch.ndim == 1:
            genes_batch = genes_batch.reshape(1, -1)

        N = genes_batch.shape[0]

        # ========== 预计算常量 ==========
        tmr_area_cost = self.area * 2 + self.voter_area  # (K,)
        tmr_power_cost = self.power * 2 + self.voter_power  # (K,)

        # 预计算的原始总敏感性（用于归一化）
        orig_sens_norm = max(self.original_total_sensitivity, 1e-10)

        # ========== 向量化计算面积增量（使用einsum加速） ==========
        # genes_batch: (N, K), tmr_area_cost: (K,)
        # 使用einsum: 'nk,k->n' 表示对第K维求和
        area_increases = np.einsum('nk,k->n', genes_batch, tmr_area_cost)  # (N,)
        area_totals = self.design_area + area_increases  # (N,)

        # ========== 向量化计算功耗增量 ==========
        power_increases = np.einsum('nk,k->n', genes_batch, tmr_power_cost)  # (N,)
        power_totals = self.design_power + power_increases  # (N,)

        # ========== 向量化计算slack ==========
        # slack_after = slack - voter_delay * gene
        slack_after_all = self.slack[np.newaxis, :] - self.voter_delay * genes_batch  # (N, K)
        min_slacks = np.min(slack_after_all, axis=1)  # (N,)

        # ========== 向量化计算敏感性（简单均值方法）==========
        # 【敏感性定义】
        # - 假设对每个寄存器注错M次，总注错次数 = K * M
        # - 假设寄存器Ri的敏感性为xi（即注错M次有M*xi次会导致错误）
        # - 整体设计的敏感性 = 总错误次数 / 总注错次数 = (x1 + x2 + ... + xK) / K
        # - 即所有寄存器敏感性的简单均值

        # 总寄存器数量（所有寄存器，不仅仅是预筛选的K个）
        K_total = self.total_num_registers if hasattr(self, 'total_num_registers') else len(self.sensitivity)

        # 所有寄存器的敏感性总和（来自reg_table）
        total_sens_sum = self.total_sensitivity_sum if hasattr(self, 'total_sensitivity_sum') else float(np.sum(self.sensitivity))

        # 加固前的敏感性：基于所有寄存器的简单均值
        design_sens_before = total_sens_sum / K_total if K_total > 0 else 0.0

        # 计算每个个体的加固数量
        num_hardened = np.sum(genes_batch, axis=1)  # (N,)

        # 计算被加固寄存器的敏感性之和（预筛选的寄存器）
        hardened_sens_sum = np.einsum('nk,k->n', genes_batch, self.sensitivity)  # (N,)

        # 计算加固后的敏感性
        # 分子：所有寄存器敏感性总和 - 被加固寄存器的敏感性
        numerator = total_sens_sum - hardened_sens_sum  # (N,)

        # 分母：原始寄存器数 + 2 * 加固数量（TMR引入的新寄存器敏感性为0）
        denominator = K_total + 2 * num_hardened  # (N,)

        # 避免除零
        denominator = np.where(denominator > 0, denominator, 1.0)

        # 计算加固后的敏感性
        total_sensitivity = numerator / denominator  # (N,)
        normalized_sens = total_sensitivity / design_sens_before  # (N,)

        # ========== 处理NaN和Inf ==========
        area_totals = np.where(np.isfinite(area_totals), area_totals, self.base_area)
        power_totals = np.where(np.isfinite(power_totals), power_totals, self.base_power)
        min_slacks = np.where(np.isfinite(min_slacks), min_slacks, self.base_slack)
        normalized_sens = np.where(np.isfinite(normalized_sens), normalized_sens, 1.0)

        # ========== 约束惩罚：同时惩罚所有违例目标 ==========
        # 问题：MOEA优化所有4个目标，惩罚只加在slack上不够
        # 解决：同时惩罚面积和功耗目标，确保违例解在任何情况下都是"差"的
        delta_area_max = 0.5  # 允许面积增加50%
        delta_power_max = 0.5  # 允许功耗增加50%
        penalty_multiplier = 10.0  # 惩罚倍数

        # 计算面积和功耗的相对增量
        delta_area = (area_totals - self.design_area) / self.design_area  # (N,)
        delta_power = (power_totals - self.design_power) / self.design_power  # (N,)

        # 检查约束违例
        area_violated = delta_area > delta_area_max  # (N,)
        power_violated = delta_power > delta_power_max  # (N,)

        # 应用惩罚：如果约束违例，同时惩罚所有目标
        # MOEA最小化所有目标，所以需要将这些值设为很大的正数
        area_totals = np.where(area_violated, area_totals * penalty_multiplier, area_totals)
        power_totals = np.where(power_violated, power_totals * penalty_multiplier, power_totals)
        min_slacks = np.where(area_violated | power_violated, -10000.0, min_slacks)

        # ========== 组装结果 ==========
        results = np.column_stack([
            area_totals,
            power_totals,
            -min_slacks,  # 返回 -slack 以便MOEA最小化（违例时返回10000，巨大惩罚）
            normalized_sens
        ])

        return results

class RealMoeaWrapper:
    """MOEA包装器，支持SAC参数调整"""

    def __init__(self, reg_table, pop_size=100, max_gen=50, archive_size=50, num_processes=None, config = None, use_gpu=False,
                 # DE算子参数
                 use_de_operator: bool = None,  # None表示自动根据K判断
                 F_init: float = 0.5,
                 F_end: float = 0.3,
                 CR: float = 0.1,
                 use_variable_grouping: bool = None,
                 group_size: int = 10):

        self.use_gpu = use_gpu
        # 自动设置进程数
        if num_processes is None:
            import multiprocessing
            #num_processes = max(1, multiprocessing.cpu_count() - 1)  # 留一个核心给系统
            num_processes = max(1, multiprocessing.cpu_count()-1)

        # 检查是否太小而不值得并行
        if pop_size < num_processes * 4:  # 每个进程至少处理4个个体
            num_processes = 1
            print(f"INFO: 种群大小{pop_size}太小，使用单进程")

        # 保存 config（如果没有传入，则使用一个安全的默认结构）
        self.config = config or {}
        # 确保有 constraint_thresholds 等默认项
        constraints = self.config.get("constraint_thresholds", {})

        self.config["constraints"] = {
            "delta_area_max": constraints.get("delta_area_max", 0.2),
            "delta_power_max": constraints.get("delta_power_max", 0.15),
            "slack_min": constraints.get("slack_min", 0.0),  # slack最小值（更直观，默认0.0）
            "delta_sens_max": constraints.get("delta_sens_min", 0.5),  # 注意逻辑反转
        }
        # 【关键】：保存敏感性阈值到实例属性，供 _extract_pareto_front 使用
        self.delta_sens_min = constraints.get("delta_sens_min", 0.4)
        
        # violation_weights 基于 area_weight、power_weight…
        self.config["violation_weights"] = {
            "a": constraints.get("area_weight", 0.1),
            "p": constraints.get("power_weight", 0.1),
            "t": constraints.get("slack_weight", 1000.0),
            "s": constraints.get("sens_weight", 0.5),
        }
        
        # reward_params
        self.config.setdefault("reward_params", self.config.get("reward_params", {
            "alpha": 1.0, "beta": 1.0, "gamma": 0.5, "eta": 0.1, "epsilon": 1e-6,
            "zeta": 0.5,  # IGD weight
            "efficiency_weight": 0.05,
            "timing_penalty_weight": 0.5,
            "balance_weight": 0.1,
            "igd_normalization": 1.0
        }))

        # 将config添加到reg_table中，以便PPAModel可以读取敏感性计算方法配置
        if not isinstance(reg_table, dict):
            reg_table = dict(reg_table) if hasattr(reg_table, '__dict__') else {}
        reg_table['config'] = self.config

        self.model = PPAModel(reg_table)
        self.K = self.model.K

        # 存储模拟器属性
        self.error_rates = self.model.sensitivity
        self.is_critical = self.model.is_critical
        self.slack = self.model.slack

        self.num_regs = self.K
        self.sens_orig = self.model.sens_orig
        self.area_orig = self.model.area_orig
        self.power_orig = self.model.power_orig
        self.slack_orig = self.model.slack_orig

        # MOEA参数
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.archive_size = archive_size
        self.num_processes = num_processes

        moea_cfg = self.config.get("moea", {}) if isinstance(self.config, dict) else {}
        self.enable_greedy_archive = bool(moea_cfg.get("enable_greedy_archive", True))
        self.greedy_archive_size = max(0, int(moea_cfg.get("greedy_archive_size", 192)))
        self.greedy_archive_injection_count = max(0, int(moea_cfg.get("greedy_archive_injection_count", 64)))
        self.greedy_archive_max_pool = max(100, int(moea_cfg.get("greedy_archive_max_pool", 1200)))
        self.enable_greedy_trajectory = bool(moea_cfg.get("enable_greedy_trajectory", True))
        self.greedy_trajectory_refine_count = max(0, int(moea_cfg.get("greedy_trajectory_refine_count", 64)))
        self.enable_milp_archive = bool(moea_cfg.get("enable_milp_archive", True))
        self.milp_budget_targets = list(
            moea_cfg.get("milp_budget_targets", [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00])
        )
        self.milp_time_limit_sec = max(1.0, float(moea_cfg.get("milp_time_limit_sec", 10.0)))
        self.milp_mip_rel_gap = max(0.0, float(moea_cfg.get("milp_mip_rel_gap", 1e-4)))
        self.milp_max_iterations = max(1, int(moea_cfg.get("milp_max_iterations", 6)))
        self.milp_max_variables = max(1, int(moea_cfg.get("milp_max_variables", 20000)))
        self._milp_seed_genes = None
        self.ppa_pareto_only = bool(moea_cfg.get("ppa_pareto_only", True))
        self.ser_as_objective = bool(moea_cfg.get("ser_as_objective", True))
        self.scoring_aligned_hv = bool(moea_cfg.get("scoring_aligned_hv", True))
        self.score_aligned_environmental_selection = bool(
            moea_cfg.get("score_aligned_environmental_selection", True)
        )
        self.score_aligned_budget_bins = max(
            8, int(moea_cfg.get("score_aligned_budget_bins", 32))
        )
        self.repair_fill_enabled = bool(moea_cfg.get("repair_fill_enabled", True))
        self.swap_local_search_enabled = bool(moea_cfg.get("swap_local_search_enabled", True))
        self.swap_local_search_max_swaps = max(0, int(moea_cfg.get("swap_local_search_max_swaps", 24)))
        self.swap_local_search_candidate_cap = max(0, int(moea_cfg.get("swap_local_search_candidate_cap", 96)))
        self.pair_swap_local_search_enabled = bool(moea_cfg.get("pair_swap_local_search_enabled", True))
        self.pair_swap_local_search_max_swaps = max(0, int(moea_cfg.get("pair_swap_local_search_max_swaps", 3)))
        self.pair_swap_selected_cap = max(2, int(moea_cfg.get("pair_swap_selected_cap", 20)))
        self.pair_swap_unselected_cap = max(2, int(moea_cfg.get("pair_swap_unselected_cap", 40)))
        self.post_swap_fill_enabled = bool(moea_cfg.get("post_swap_fill_enabled", True))
        self.post_swap_fill_max_additions = max(
            0, int(moea_cfg.get("post_swap_fill_max_additions", 12))
        )
        self.restart_use_elite_bases = bool(moea_cfg.get("restart_use_elite_bases", True))
        self.restart_refine_candidates = bool(moea_cfg.get("restart_refine_candidates", True))
        self.restart_elite_base_ratio = float(np.clip(
            moea_cfg.get("restart_elite_base_ratio", 0.70), 0.0, 1.0
        ))
        self.memetic_refine_current_population = bool(moea_cfg.get("memetic_refine_current_population", True))
        self.memetic_refine_interval = max(1, int(moea_cfg.get("memetic_refine_interval", 3)))
        self.memetic_refine_count = max(0, int(moea_cfg.get("memetic_refine_count", 16)))
        self._base_greedy_archive_injection_count = self.greedy_archive_injection_count
        self._base_memetic_refine_count = self.memetic_refine_count
        self._base_swap_local_search_max_swaps = self.swap_local_search_max_swaps
        self._base_pair_swap_local_search_max_swaps = self.pair_swap_local_search_max_swaps
        self.warm_start_base_ratio = float(np.clip(moea_cfg.get("warm_start_base_ratio", 0.05), 0.0, 0.25))
        self.search_controls = {
            "spea2_budget_share": 0.5,
            "elite_injection_ratio": min(0.10, self.greedy_archive_injection_count / max(1, self.pop_size)),
            "local_search_strength": 0.5,
            "restart_ratio": 0.0,
        }
        self.defer_archive_until_policy_ready = bool(
            moea_cfg.get("defer_archive_until_policy_ready", True)
        )
        self.archive_unlock_updates = max(
            0, int(moea_cfg.get("archive_unlock_updates", 8))
        )
        self.sac_update_step = 0
        self.sac_policy_ready = not self.defer_archive_until_policy_ready
        self.sac_action_mode = "uninitialized"
        # SAC动作归因快照覆盖本控制步内的重启、精英注入、局部搜索和进化。
        # 回合初始化发生在快照之前，不会把历史收益错误归因给当前动作。
        self.last_pre_evolution_population = []
        self.last_evolution_population = []
        self._last_restart_count = 0
        self._step_spea2_updates = 0
        self._step_nsga2_updates = 0
        self.greedy_archive = []
        self._greedy_archive_built = False
        self._refine_cache = {}

        # 鏍规嵁鍩哄洜闀垮害K鍒ゆ柇鏄惁浣跨敤DE绠楀瓙

        # 根据基因长度K判断是否使用DE算子
        # K > 50 时使用DE算子替代传统GA
        if use_de_operator is None:
            self.use_de_operator = self.K > 50
        else:
            self.use_de_operator = use_de_operator

        # 变量分组（K > 100 时启用）
        if use_variable_grouping is None:
            self.use_variable_grouping = self.K > 100
        else:
            self.use_variable_grouping = use_variable_grouping

        self.F_init = F_init
        self.F_end = F_end
        self.CR = CR
        self.group_size = group_size

        print(f"[RealMoeaWrapper]: 初始化完成")
        print(f"  基因长度K: {self.K}")
        print(f"  使用DE算子: {self.use_de_operator}")
        print(f"  变量分组: {self.use_variable_grouping}")

        # 根据是否使用GPU选择评估器（在初始化算法之前）
        from mo_optimizer.parallel_evaluator import ParallelEvaluator
        if use_gpu:
            print("RealMoeaWrapper: 使用GPU加速模式（共享GPU评估器）")
            # 从配置中获取GPU批处理大小
            gpu_batch_size = config.get('evaluator', {}).get('gpu_batch_size', 512) if config else 512
            print(f"  GPU批处理大小: {gpu_batch_size}")

            # 使用共享的GPU评估器以节省显存并提高GPU利用率
            shared_gpu_evaluator = ParallelEvaluator(
                self,
                num_processes=1,  # GPU模式使用单一GPU worker
                batch_size=gpu_batch_size,
                use_gpu=True,
                evaluator_id='shared_gpu',
                config=config
            )

            spea2_evaluator = shared_gpu_evaluator
            nsga2_evaluator = shared_gpu_evaluator
            ## print("  SPEA2和NSGA2共享GPU评估器，优化显存使用")
        else:
            # 原有CPU并行评估器
            shared_evaluator = ParallelEvaluator(
                self,
                num_processes=num_processes,
                batch_size=None,
                use_gpu=False,
                config=config
            )
            spea2_evaluator = shared_evaluator
            nsga2_evaluator = shared_evaluator


        # 初始化算法（使用独立或共享评估器）
        self.spea2 = SPEA2Parallel(
            simulator=self,
            pop_size=pop_size,
            max_generations=max_gen,
            archive_size=archive_size,
            num_processes=num_processes,
            config=config,
            # DE参数（大规模问题时使用）
            use_de_operator=self.use_de_operator,
            F_init=self.F_init,
            F_end=self.F_end,
            CR=self.CR,
            use_variable_grouping=self.use_variable_grouping,
            group_size=self.group_size
        )
        # 使用SPEA2专用评估器
        self.spea2.evaluator = spea2_evaluator

        self.nsga2 = NSGA2Parallel(
            simulator=self,
            pop_size=pop_size,
            max_generations=max_gen,
            num_processes=num_processes,
            config=config,
            # DE参数（大规模问题时使用）
            use_de_operator=self.use_de_operator,
            F_init=self.F_init,
            F_end=self.F_end,
            CR=self.CR,
            use_variable_grouping=self.use_variable_grouping,
            group_size=self.group_size
        )
        # 使用NSGA2专用评估器
        self.nsga2.evaluator = nsga2_evaluator

        # 保存评估器引用
        self.evaluator = spea2_evaluator if use_gpu else shared_evaluator
        
        if use_gpu:
            print(f"RealMoeaWrapper: GPU评估器已创建并分配给SPEA2和NSGA2")
            # 检查GPU是否可用
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
                    print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
                else:
                    print("  警告: CUDA不可用，将回退到CPU模式")
            except ImportError:
                print("  警告: PyTorch未安装，无法使用GPU")
        
        # 历史记录
        self.history = {
            'actions': [],
            'stats': [],
            'populations': []
        }

        # 设置最大历史长度
        self.max_history_len = 100

        # 收敛检测属性
        self.convergence_history = {
            'hv': [],  # 超体积历史
            'igd': [],  # IGD历史
            'diversity': [],  # 多样性历史
            'pareto_size': []  # Pareto前沿大小历史
        }
        
        # 当前种群使用弱引用或定期清理
        self.current_population = None

        # 代计数器
        self.current_generation = 0
        self.max_generations = max_gen
        
        # 【关键修复】保留累积精英池，不在 reset() 时清空
        # 只有在明确需要时才清空（通过 force_reset 参数）
        if not hasattr(self, 'cumulative_elite_pool') or self.cumulative_elite_pool is None:
            self.cumulative_elite_pool = []
            print(f"[RealMoeaWrapper]: 累积精英池已初始化（当前为空）")
        else:
            print(f"[RealMoeaWrapper]: 保留累积精英池（当前有 {len(self.cumulative_elite_pool)} 个解）")

    def close(self):
        """释放并行评估相关资源（进程池/GPU worker/Manager），避免程序结束时阻塞退出。"""
        evaluators = []
        # 优先从算法对象上拿（更接近真实使用方）
        for obj in (getattr(self, 'spea2', None), getattr(self, 'nsga2', None)):
            ev = getattr(obj, 'evaluator', None) if obj is not None else None
            if ev is not None:
                evaluators.append(ev)
        # 兼容性字段
        if getattr(self, 'evaluator', None) is not None:
            evaluators.append(self.evaluator)

        # 去重（同一个 evaluator 可能被 SPEA2/NSGA2 共享）
        seen = set()
        for ev in evaluators:
            ev_id = id(ev)
            if ev_id in seen:
                continue
            seen.add(ev_id)
            try:
                if hasattr(ev, 'close'):
                    ev.close()
            except Exception as e:
                try:
                    print(f"[WARNING] 关闭评估器失败: {e}")
                except Exception:
                    pass

    def __del__(self):
        # 作为兜底：不保证一定执行，但能覆盖一部分直接被GC回收的场景
        try:
            self.close()
        except Exception:
            pass
        
    def _cleanup_memory(self):
        """清理内存"""
        import gc
        
        # 1. 清理历史记录
        for key in self.history:
            if len(self.history[key]) > self.max_history_len:
                self.history[key] = self.history[key][-self.max_history_len:]
        
        # 2. 清理收敛历史
        max_conv_history = 200
        for key in self.convergence_history:
            if len(self.convergence_history[key]) > max_conv_history:
                self.convergence_history[key] = self.convergence_history[key][-max_conv_history:]
        
        # 3. 清理当前种群中的冗余数据
        if self.current_population:
            for ind in self.current_population:
                # 只保留必要的属性
                if hasattr(ind, '_cache'):
                    delattr(ind, '_cache')
        
        # 4. 垃圾回收
        gc.collect()
        
        ## print(f"[内存优化] 历史记录已清理")

    def _ppa_limits(self):
        constraints = self.config.get("constraints", {})
        return (
            float(constraints.get("delta_area_max", 0.2)),
            float(constraints.get("delta_power_max", 0.5)),
            float(constraints.get("slack_min", 0.0)),
        )

    def _budget_safety_scale(self):
        return 1.0 - 1e-8

    def _normalize_gene(self, gene):
        gene = np.asarray(gene, dtype=np.int32).flatten()
        if gene.size < self.K:
            gene = np.pad(gene, (0, self.K - gene.size), "constant")
        elif gene.size > self.K:
            gene = gene[:self.K]
        return (gene > 0).astype(np.int32)

    def _ppa_arrays(self):
        model = self.model
        return (
            np.asarray(model.area, dtype=np.float64) * 2.0 + float(model.voter_area),
            np.asarray(model.power, dtype=np.float64) * 2.0 + float(model.voter_power),
            np.asarray(model.slack, dtype=np.float64),
            np.asarray(model.sensitivity, dtype=np.float64),
        )

    def _gene_metrics_fast(self, gene):
        gene = self._normalize_gene(gene)
        area_cost, power_cost, slack, sens = self._ppa_arrays()
        n_hardened = int(np.sum(gene))
        hardened_sens = float(np.dot(gene, sens))
        total_sens_sum = float(getattr(self.model, "total_sensitivity_sum", np.sum(sens)))
        total_regs = int(getattr(self.model, "total_num_registers", self.K))
        residual_ser = (total_sens_sum - hardened_sens) / max(total_regs + 2 * n_hardened, 1)
        residual_ratio = residual_ser / max(float(getattr(self, "sens_orig", self.model.sens_orig)), 1e-20)
        area_increase = float(np.dot(gene, area_cost))
        power_increase = float(np.dot(gene, power_cost))
        worst_slack = float(np.min(slack - float(self.model.voter_delay) * gene))
        area_limit, power_limit, slack_min = self._ppa_limits()
        safety = self._budget_safety_scale()
        feasible = (
            area_increase <= self.area_orig * area_limit * safety + 1e-12
            and power_increase <= self.power_orig * power_limit * safety + 1e-12
            and worst_slack >= slack_min - 1e-12
        )
        return {
            "gene": gene,
            "n": n_hardened,
            "residual_ser": residual_ser,
            "residual_ratio": residual_ratio,
            "area_increase": area_increase,
            "power_increase": power_increase,
            "worst_slack": worst_slack,
            "feasible": feasible,
        }

    def _gene_key(self, gene):
        return self._normalize_gene(gene).tobytes()

    def _refine_gene_cached(self, gene, max_swaps=None):
        gene = self._normalize_gene(gene)
        if max_swaps is None:
            max_swaps = getattr(self, "swap_local_search_max_swaps", 24)
        key = (
            gene.tobytes(),
            int(max_swaps),
            bool(getattr(self, "repair_fill_enabled", True)),
            bool(getattr(self, "pair_swap_local_search_enabled", False)),
            int(getattr(self, "pair_swap_local_search_max_swaps", 0)),
        )
        cached = getattr(self, "_refine_cache", {}).get(key)
        if cached is not None:
            return cached.copy()
        refined = self._refine_gene_by_ppa(gene, max_swaps=max_swaps)
        if not hasattr(self, "_refine_cache") or self._refine_cache is None:
            self._refine_cache = {}
        if len(self._refine_cache) > 4096:
            self._refine_cache.clear()
        self._refine_cache[key] = refined.copy()
        return refined

    def _refine_gene_preserving_budget(self, gene, max_swaps=None):
        """Improve SER without exceeding the seed's area, power, or WNS budget."""
        gene = self._normalize_gene(gene).astype(np.int32)
        if not getattr(self, "swap_local_search_enabled", True):
            return gene
        if max_swaps is None:
            max_swaps = getattr(self, "swap_local_search_max_swaps", 24)
        max_swaps = max(0, int(max_swaps))
        if max_swaps == 0 or int(np.sum(gene)) == 0:
            return gene

        area_cost, power_cost, slack, sens = self._ppa_arrays()
        voter_delay = float(self.model.voter_delay)
        seed_metrics = self._gene_metrics_fast(gene)
        area_cap = float(seed_metrics["area_increase"])
        power_cap = float(seed_metrics["power_increase"])
        wns_floor = max(float(seed_metrics["worst_slack"]), self._ppa_limits()[2])
        candidate_cap = int(getattr(self, "swap_local_search_candidate_cap", 0))

        for _ in range(max_swaps):
            selected = np.where(gene == 1)[0]
            unselected = np.where(gene == 0)[0]
            if selected.size == 0 or unselected.size == 0:
                break
            if candidate_cap > 0 and selected.size > candidate_cap:
                selected = selected[np.argsort(sens[selected])[:candidate_cap]]
            if candidate_cap > 0 and unselected.size > candidate_cap:
                unselected = unselected[np.argsort(sens[unselected])[-candidate_cap:]]

            current_area = float(np.dot(gene, area_cost))
            current_power = float(np.dot(gene, power_cost))
            best_gain = 1e-15
            best_pair = None
            for rem in selected:
                area_after = current_area - area_cost[rem] + area_cost[unselected]
                power_after = current_power - power_cost[rem] + power_cost[unselected]
                feasible = (area_after <= area_cap + 1e-12) & (power_after <= power_cap + 1e-12)
                if not np.any(feasible):
                    continue

                base_gene = gene.copy()
                base_gene[rem] = 0
                base_wns = float(np.min(slack - voter_delay * base_gene))
                next_wns = np.minimum(base_wns, slack[unselected] - voter_delay)
                gains = sens[unselected] - sens[rem]
                feasible &= (next_wns >= wns_floor - 1e-12) & (gains > best_gain)
                if not np.any(feasible):
                    continue
                masked = np.where(feasible, gains, -np.inf)
                pos = int(np.argmax(masked))
                gain = float(masked[pos])
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (int(rem), int(unselected[pos]))

            if best_pair is None:
                break
            gene[best_pair[0]] = 0
            gene[best_pair[1]] = 1

        return self._refine_gene_pair_exchange(
            gene,
            area_cap=area_cap,
            power_cap=power_cap,
            wns_floor=wns_floor,
        )

    def _refine_gene_pair_exchange(self, gene, area_cap, power_cap, wns_floor):
        """Escape a one-swap local optimum with budget-preserving 2-out/2-in moves."""
        gene = self._normalize_gene(gene).astype(np.int32)
        if not getattr(self, "pair_swap_local_search_enabled", True):
            return gene
        max_pair_swaps = int(getattr(self, "pair_swap_local_search_max_swaps", 0))
        if max_pair_swaps <= 0:
            return gene

        area_cost, power_cost, slack, sens = self._ppa_arrays()
        voter_delay = float(self.model.voter_delay)
        area_cap = float(area_cap)
        power_cap = float(power_cap)
        wns_floor = float(wns_floor)

        for _ in range(max_pair_swaps):
            selected = np.where(gene == 1)[0]
            unselected = np.where(gene == 0)[0]
            if selected.size < 2 or unselected.size < 2:
                break

            selected_cap = int(getattr(self, "pair_swap_selected_cap", 20))
            unselected_cap = int(getattr(self, "pair_swap_unselected_cap", 40))
            if selected.size > selected_cap:
                selected = selected[np.argsort(sens[selected])[:selected_cap]]
            if unselected.size > unselected_cap:
                denom = area_cost[unselected] / max(area_cap, 1e-20)
                denom += power_cost[unselected] / max(power_cap, 1e-20)
                efficiency = sens[unselected] / np.maximum(denom, 1e-20)
                unselected = unselected[np.argsort(efficiency)[-unselected_cap:]]

            rem_i, rem_j = np.triu_indices(selected.size, k=1)
            add_i, add_j = np.triu_indices(unselected.size, k=1)
            if rem_i.size == 0 or add_i.size == 0:
                break
            rem_a = selected[rem_i]
            rem_b = selected[rem_j]
            add_a = unselected[add_i]
            add_b = unselected[add_j]

            add_area = area_cost[add_a] + area_cost[add_b]
            add_power = power_cost[add_a] + power_cost[add_b]
            add_sens = sens[add_a] + sens[add_b]
            add_slack = np.minimum(slack[add_a], slack[add_b]) - voter_delay
            current_area = float(np.dot(gene, area_cost))
            current_power = float(np.dot(gene, power_cost))

            best_gain = 1e-15
            best_move = None
            for rem1, rem2 in zip(rem_a, rem_b):
                base_area = current_area - area_cost[rem1] - area_cost[rem2]
                base_power = current_power - power_cost[rem1] - power_cost[rem2]
                feasible = (
                    (base_area + add_area <= area_cap + 1e-12)
                    & (base_power + add_power <= power_cap + 1e-12)
                    & (add_slack >= wns_floor - 1e-12)
                )
                if not np.any(feasible):
                    continue

                base_gene = gene.copy()
                base_gene[rem1] = 0
                base_gene[rem2] = 0
                base_wns = float(np.min(slack - voter_delay * base_gene))
                feasible &= np.minimum(base_wns, add_slack) >= wns_floor - 1e-12
                gains = add_sens - sens[rem1] - sens[rem2]
                gains = np.where(feasible, gains, -np.inf)
                pos = int(np.argmax(gains))
                if np.isfinite(gains[pos]) and gains[pos] > best_gain:
                    best_gain = float(gains[pos])
                    best_move = (int(rem1), int(rem2), int(add_a[pos]), int(add_b[pos]))

            if best_move is None:
                break
            rem1, rem2, add1, add2 = best_move
            gene[rem1] = 0
            gene[rem2] = 0
            gene[add1] = 1
            gene[add2] = 1

        return gene

    def _build_greedy_trajectory(self):
        """Reproduce the deterministic marginal-gain baseline as an anytime archive."""
        if not getattr(self, "enable_greedy_trajectory", True):
            return []
        area_cost, power_cost, slack, sens = self._ppa_arrays()
        area_limit, power_limit, slack_min = self._ppa_limits()
        safety = self._budget_safety_scale()
        area_budget = max(float(self.area_orig) * area_limit * safety, 1e-20)
        power_budget = max(float(self.power_orig) * power_limit * safety, 1e-20)
        total_sens = float(getattr(self.model, "total_sensitivity_sum", np.sum(sens)))
        total_regs = int(getattr(self.model, "total_num_registers", self.K))
        selected = np.zeros(self.K, dtype=bool)
        area_inc = 0.0
        power_inc = 0.0
        hardened_sens = 0.0
        count = 0
        worst_slack = float(np.min(slack))
        residual_ser = total_sens / max(total_regs, 1)
        slack_after_if_hardened = slack - float(self.model.voter_delay)
        trajectory = []

        while True:
            remaining = np.where(~selected)[0]
            if remaining.size == 0:
                break
            area_after = area_inc + area_cost[remaining]
            power_after = power_inc + power_cost[remaining]
            worst_after = np.minimum(worst_slack, slack_after_if_hardened[remaining])
            residual_after = (total_sens - hardened_sens - sens[remaining]) / max(total_regs + 2 * (count + 1), 1)
            gain = residual_ser - residual_after
            normalized_cost = area_cost[remaining] / area_budget + power_cost[remaining] / power_budget
            score = gain / np.maximum(normalized_cost, 1e-20)
            feasible = (
                (area_after <= area_budget + 1e-12)
                & (power_after <= power_budget + 1e-12)
                & (worst_after >= slack_min - 1e-12)
                & (gain > 0.0)
            )
            score = np.where(feasible, score, -np.inf)
            if not np.isfinite(score).any():
                break
            pos = int(np.argmax(score))
            chosen = int(remaining[pos])
            selected[chosen] = True
            area_inc = float(area_after[pos])
            power_inc = float(power_after[pos])
            worst_slack = float(worst_after[pos])
            residual_ser = float(residual_after[pos])
            hardened_sens += float(sens[chosen])
            count += 1
            ind = self._make_individual_from_gene(selected.astype(np.int32))
            if ind.objectives is not None:
                ind.archive_source = f"greedy_trajectory:{count}"
                trajectory.append(ind)
        return trajectory

    def _elite_entry_from_individual(self, ind, source="archive"):
        if ind is None or getattr(ind, "gene", None) is None:
            return None
        return self._evaluated_elite_entry_from_gene(
            ind.gene,
            source=getattr(ind, "archive_source", source),
        )

    def _evaluated_elite_entry_from_gene(self, gene, source="archive"):
        """Create one archive record from the current unified PPA/SER model."""
        if gene is None:
            return None
        gene = self._normalize_gene(gene)
        metrics = self._gene_metrics_fast(gene)
        if not metrics.get("feasible", False):
            return None
        objectives = np.array(
            [
                float(self.area_orig) + float(metrics["area_increase"]),
                float(self.power_orig) + float(metrics["power_increase"]),
                -float(metrics["worst_slack"]),
                float(metrics["residual_ratio"]),
            ],
            dtype=np.float64,
        )
        return {
            "gene": gene.copy(),
            "objectives": objectives,
            "raw_sensitivity": float(metrics["residual_ratio"]),
            "sensitivity_feasible": True,
            "area_power_feasible": True,
            "violation": 0.0,
            "source": str(source or "archive"),
            "tmr_count": int(np.sum(gene)),
        }

    def _resource_use_fast(self, gene):
        metrics = self._gene_metrics_fast(gene)
        area_limit, power_limit, _ = self._ppa_limits()
        area_budget = max(float(self.area_orig) * area_limit * self._budget_safety_scale(), 1e-20)
        power_budget = max(float(self.power_orig) * power_limit * self._budget_safety_scale(), 1e-20)
        area_ratio = metrics["area_increase"] / area_budget
        power_ratio = metrics["power_increase"] / power_budget
        return max(area_ratio, power_ratio), metrics

    def _prune_elite_entries(self, entries, max_pool):
        entries = list(entries or [])
        max_pool = int(max_pool)
        if len(entries) <= max_pool:
            return entries

        annotated = []
        for pos, entry in enumerate(entries):
            gene = self._get_elite_entry_gene(entry)
            if gene is None:
                continue
            try:
                resource_use, metrics = self._resource_use_fast(gene)
                score = (
                    0 if metrics.get("feasible", False) else 1,
                    float(metrics.get("residual_ratio", 1e9)),
                    -int(metrics.get("n", 0)),
                    pos,
                )
                annotated.append((float(resource_use), score, entry))
            except Exception:
                annotated.append((1.0, (1, 1e9, 0, pos), entry))
        if not annotated:
            return entries[:max_pool]

        selected = []
        seen = set()

        def add(entry):
            gene = self._get_elite_entry_gene(entry)
            if gene is None:
                return False
            key = self._gene_key(gene)
            if key in seen:
                return False
            seen.add(key)
            selected.append(entry)
            return True

        # The deterministic trajectory is the hybrid algorithm's guaranteed
        # anytime baseline. Never discard it while the pool can hold it.
        trajectory_entries = [
            item for item in annotated
            if isinstance(item[2], dict)
            and str(item[2].get("source", "")).startswith("greedy_trajectory")
        ]
        for _, _, entry in sorted(trajectory_entries, key=lambda x: x[0]):
            add(entry)
            if len(selected) >= max_pool:
                return selected

        # Keep the globally best feasible SER entries first.
        for _, _, entry in sorted(annotated, key=lambda x: x[1])[: max(1, max_pool // 4)]:
            add(entry)
            if len(selected) >= max_pool:
                return selected

        # Then keep budget-diverse entries, so low/mid/high resource Pareto points survive.
        bin_count = max(8, min(64, max_pool // 8))
        bins = [[] for _ in range(bin_count)]
        for resource_use, score, entry in annotated:
            idx = int(np.clip(resource_use, 0.0, 0.999999) * bin_count)
            bins[idx].append((score, entry))
        per_bin = max(1, max_pool // bin_count)
        for bucket in bins:
            for _, entry in sorted(bucket, key=lambda x: x[0])[:per_bin]:
                add(entry)
                if len(selected) >= max_pool:
                    return selected

        # Fill any remaining slots by global score.
        for _, _, entry in sorted(annotated, key=lambda x: x[1]):
            add(entry)
            if len(selected) >= max_pool:
                break
        return selected

    def _append_to_cumulative_elite_pool(self, individuals, source="archive", max_pool=None):
        if not hasattr(self, "cumulative_elite_pool") or self.cumulative_elite_pool is None:
            self.cumulative_elite_pool = []
        max_pool = int(max_pool or getattr(self, "greedy_archive_max_pool", 1200))

        merged = []
        seen = set()
        for entry in self.cumulative_elite_pool:
            gene = self._get_elite_entry_gene(entry)
            if gene is None:
                continue
            refreshed = self._evaluated_elite_entry_from_gene(
                gene,
                source=(entry.get("source", "archive") if isinstance(entry, dict) else "archive"),
            )
            if refreshed is None:
                continue
            key = self._gene_key(gene)
            if key in seen:
                continue
            seen.add(key)
            merged.append(refreshed)

        added = 0
        for ind in individuals or []:
            entry = self._elite_entry_from_individual(ind, source=source)
            if entry is None:
                continue
            key = self._gene_key(entry["gene"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
            added += 1

        if len(merged) > max_pool:
            merged = self._prune_elite_entries(merged, max_pool)
        self.cumulative_elite_pool = merged
        return added

    def _greedy_archive_targets(self):
        if self.greedy_archive_size <= 0:
            return []
        n = max(8, min(48, self.greedy_archive_size // 4))
        dense = np.linspace(0.03, 1.0, n)
        low = np.array([0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00])
        targets = np.unique(np.round(np.concatenate([dense, low]), 4))
        return [float(t) for t in targets if t > 0]

    def _build_milp_seed_genes(self):
        """Build exact budget-spaced seeds for the linear PPA subproblem."""
        if not getattr(self, "enable_milp_archive", False):
            return []
        if self._milp_seed_genes is not None:
            return [(gene.copy(), source) for gene, source in self._milp_seed_genes]
        if self.K > getattr(self, "milp_max_variables", 20000):
            print(f"[MILP-ARCHIVE] skipped: K={self.K} exceeds milp_max_variables")
            self._milp_seed_genes = []
            return []

        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
        except Exception as exc:
            print(f"[MILP-ARCHIVE] unavailable: {exc}")
            self._milp_seed_genes = []
            return []

        area_cost, power_cost, slack, sensitivity = self._ppa_arrays()
        area_limit, power_limit, slack_min = self._ppa_limits()
        safety = self._budget_safety_scale()
        total_sensitivity = float(
            getattr(self.model, "total_sensitivity_sum", np.sum(sensitivity))
        )
        total_registers = int(getattr(self.model, "total_num_registers", self.K))
        upper = np.where(
            slack - float(self.model.voter_delay) >= slack_min - 1e-12,
            1.0,
            0.0,
        )
        bounds = Bounds(np.zeros(self.K), upper)
        integrality = np.ones(self.K, dtype=np.int8)
        targets = sorted(
            {
                float(np.clip(value, 0.01, 1.0))
                for value in getattr(self, "milp_budget_targets", [])
            }
        )

        seeds = []
        seen = set()
        for target in targets:
            constraints = LinearConstraint(
                np.vstack([area_cost, power_cost]),
                np.array([-np.inf, -np.inf]),
                np.array(
                    [
                        float(self.area_orig) * area_limit * safety * target,
                        float(self.power_orig) * power_limit * safety * target,
                    ]
                ),
            )
            residual_ser = total_sensitivity / max(total_registers, 1)
            gene = None
            for _ in range(self.milp_max_iterations):
                result = milp(
                    c=-(sensitivity + 2.0 * residual_ser),
                    integrality=integrality,
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        "time_limit": self.milp_time_limit_sec,
                        "mip_rel_gap": self.milp_mip_rel_gap,
                    },
                )
                if result.x is None:
                    break
                gene = (result.x >= 0.5).astype(np.int32)
                selected_sensitivity = float(np.dot(gene, sensitivity))
                selected_count = int(np.sum(gene))
                updated = (total_sensitivity - selected_sensitivity) / max(
                    total_registers + 2 * selected_count,
                    1,
                )
                if abs(updated - residual_ser) < 1e-10:
                    residual_ser = updated
                    break
                residual_ser = updated

            if gene is None or int(np.sum(gene)) <= 0:
                continue
            metrics = self._gene_metrics_fast(gene)
            key = self._gene_key(gene)
            if not metrics.get("feasible", False) or key in seen:
                continue
            seen.add(key)
            source = f"milp_archive:{target:.3f}"
            seeds.append((gene, source))
            print(
                f"[MILP-ARCHIVE] target={target:.3f}, n={metrics['n']}, "
                f"residual_ratio={metrics['residual_ratio']:.6f}"
            )

        self._milp_seed_genes = [(gene.copy(), source) for gene, source in seeds]
        return [(gene.copy(), source) for gene, source in seeds]

    def _build_greedy_archive(self, force=False, record_archive=True):
        if not getattr(self, "enable_greedy_archive", True) or self.greedy_archive_size <= 0:
            return []
        if self._greedy_archive_built and not force:
            if record_archive and self.greedy_archive:
                self._append_to_cumulative_elite_pool(
                    self.greedy_archive,
                    source="greedy_archive",
                )
            return list(self.greedy_archive)

        self._greedy_archive_built = True
        archive = []
        seen = set()

        def add_gene(gene, source, refine_mode="preserve"):
            if gene is None:
                return False
            if refine_mode == "full":
                gene = self._refine_gene_cached(gene)
            elif refine_mode == "preserve":
                gene = self._refine_gene_preserving_budget(gene)
            else:
                gene = self._normalize_gene(gene)
            if int(np.sum(gene)) <= 0:
                return False
            key = self._gene_key(gene)
            if key in seen:
                return False
            metrics = self._gene_metrics_fast(gene)
            if not metrics.get("feasible", False):
                return False
            ind = self._make_individual_from_gene(gene)
            if ind.objectives is None:
                return False
            ind.archive_source = source
            seen.add(key)
            archive.append(ind)
            return True

        # Keep the complete accepted trajectory. This guarantees an anytime
        # low/mid/high-budget curve instead of collapsing every seed to full budget.
        trajectory = self._build_greedy_trajectory()
        for ind in trajectory:
            if len(archive) >= self.greedy_archive_size:
                break
            add_gene(ind.gene, getattr(ind, "archive_source", "greedy_trajectory"), refine_mode="none")

        # Improve budget-spaced trajectory anchors without using more area,
        # power, or WNS than the corresponding greedy point.
        refine_count = min(len(trajectory), getattr(self, "greedy_trajectory_refine_count", 64))
        if refine_count > 0:
            positions = np.unique(np.linspace(0, len(trajectory) - 1, refine_count, dtype=int))
            for pos in positions:
                if len(archive) >= self.greedy_archive_size:
                    break
                seed = trajectory[int(pos)]
                add_gene(seed.gene, f"greedy_trajectory_swap:{int(pos) + 1}", refine_mode="preserve")

        # Exact linear-fractional seeds complement the greedy trajectory at a
        # handful of resource levels and are then evolved by both MOEAs.
        for gene, source in self._build_milp_seed_genes():
            if len(archive) >= self.greedy_archive_size:
                break
            add_gene(gene, source, refine_mode="none")

        targets = self._greedy_archive_targets()
        plans = [
            (1.0, 1.0, "balanced"),
            (1.6, 0.75, "power_guarded"),
            (0.75, 1.6, "area_guarded"),
            (2.2, 0.55, "power_strict"),
            (0.55, 2.2, "area_strict"),
        ]
        for target in targets:
            for power_bias, area_bias, label in plans:
                if len(archive) >= self.greedy_archive_size:
                    break
                target_area = min(1.0, max(0.02, target * (1.15 if area_bias <= power_bias else 1.0)))
                target_power = min(1.0, max(0.02, target * (1.15 if power_bias < area_bias else 1.0)))
                gene = self._greedy_budget_gene(
                    target_power_ratio=target_power,
                    target_area_ratio=target_area,
                    power_bias=power_bias,
                    area_bias=area_bias,
                )
                add_gene(gene, f"greedy_archive:{label}:{target:.3f}")
            if len(archive) >= self.greedy_archive_size:
                break

        # Add a few one-pass score-ordered full-budget variants as anchors.
        area_cost, power_cost, slack, sens = self._ppa_arrays()
        area_limit, power_limit, slack_min = self._ppa_limits()
        safety = self._budget_safety_scale()
        area_budget = max(float(self.area_orig) * area_limit * safety, 1e-20)
        power_budget = max(float(self.power_orig) * power_limit * safety, 1e-20)
        safe = np.where(slack - float(self.model.voter_delay) >= slack_min - 1e-12)[0]
        if safe.size:
            score_defs = [
                (sens[safe], "sens"),
                (sens[safe] / np.maximum(area_cost[safe] / area_budget + power_cost[safe] / power_budget, 1e-20), "gain_cost"),
                (sens[safe] / np.maximum(power_cost[safe] / power_budget, 1e-20), "power_eff"),
                (sens[safe] / np.maximum(area_cost[safe] / area_budget, 1e-20), "area_eff"),
            ]
            for score, label in score_defs:
                if len(archive) >= self.greedy_archive_size:
                    break
                order = safe[np.argsort(score)[::-1]]
                gene = np.zeros(self.K, dtype=np.int32)
                for idx in order:
                    trial = gene.copy()
                    trial[idx] = 1
                    if self._gene_metrics_fast(trial)["feasible"]:
                        gene = trial
                add_gene(gene, f"greedy_archive:{label}")

        self.greedy_archive = self._select_greedy_archive_individuals(self.greedy_archive_size, build=False, source_list=archive)
        added = 0
        if record_archive:
            added = self._append_to_cumulative_elite_pool(
                self.greedy_archive,
                source="greedy_archive",
            )
        print(f"[GREEDY-ARCHIVE] built {len(self.greedy_archive)} feasible greedy/local-search elites, added_to_history={added}")
        return list(self.greedy_archive)

    def _select_greedy_archive_individuals(
        self,
        max_count,
        build=True,
        source_list=None,
        record_archive=True,
    ):
        if build:
            source = self._build_greedy_archive(record_archive=record_archive)
        else:
            source = list(source_list or [])
        source = [ind for ind in source if ind is not None and getattr(ind, "gene", None) is not None]
        max_count = max(0, int(max_count))
        if max_count <= 0 or not source:
            return []
        if len(source) <= max_count:
            return list(source)

        annotated = []
        for ind in source:
            try:
                resource_use, metrics = self._resource_use_fast(ind.gene)
                annotated.append((float(resource_use), float(metrics.get("residual_ratio", 1e9)), ind))
            except Exception:
                annotated.append((1.0, 1e9, ind))
        annotated.sort(key=lambda x: (x[0], x[1]))

        selected = []
        seen = set()

        def add(ind):
            key = self._gene_key(ind.gene)
            if key in seen:
                return
            seen.add(key)
            selected.append(ind)

        # Evenly sample the budget axis, then fill by best SER.
        positions = np.linspace(0, len(annotated) - 1, max_count, dtype=int)
        for pos in positions:
            add(annotated[int(pos)][2])
            if len(selected) >= max_count:
                return selected
        for _, _, ind in sorted(annotated, key=lambda x: x[1]):
            add(ind)
            if len(selected) >= max_count:
                break
        return selected

    def _should_memetic_refine(self, context):
        if not getattr(self, "memetic_refine_current_population", True) or self.memetic_refine_count <= 0:
            return False
        if context == "final":
            return True
        if not isinstance(context, str) or not context.startswith("generation-"):
            return False
        try:
            gen = int(context.split("-", 1)[1])
        except Exception:
            return False
        return gen % max(1, self.memetic_refine_interval) == 0

    def _memetic_refine_population(self, population, max_count=None, context=""):
        if not population:
            return []
        max_count = int(max_count or self.memetic_refine_count)
        if max_count <= 0:
            return []

        candidates = []
        for ind in population:
            if ind is None or getattr(ind, "gene", None) is None:
                continue
            try:
                resource_use, metrics = self._resource_use_fast(ind.gene)
                candidates.append((float(resource_use), float(metrics.get("residual_ratio", 1e9)), ind))
            except Exception:
                continue
        if not candidates:
            return []

        # Mix best-SER candidates with budget-spaced candidates.
        selected = []
        seen = set()

        def add_candidate(item):
            ind = item[2]
            key = self._gene_key(ind.gene)
            if key in seen:
                return
            seen.add(key)
            selected.append(ind)

        for item in sorted(candidates, key=lambda x: x[1])[: max(1, max_count // 2)]:
            add_candidate(item)
        budget_sorted = sorted(candidates, key=lambda x: x[0])
        if budget_sorted:
            positions = np.linspace(0, len(budget_sorted) - 1, max_count, dtype=int)
            for pos in positions:
                add_candidate(budget_sorted[int(pos)])
                if len(selected) >= max_count:
                    break

        refined = []
        refined_seen = set()
        for ind in selected[:max_count]:
            gene = self._refine_gene_cached(ind.gene)
            key = self._gene_key(gene)
            if key in refined_seen or key == self._gene_key(ind.gene):
                continue
            refined_seen.add(key)
            metrics = self._gene_metrics_fast(gene)
            if not metrics.get("feasible", False):
                continue
            new_ind = self._make_individual_from_gene(gene)
            if new_ind.objectives is not None:
                new_ind.archive_source = f"memetic_refine:{context}"
                refined.append(new_ind)
        if refined:
            print(f"[MEMETIC] {context} repair-fill/swap refined {len(refined)} candidates")
        return refined

    def _make_individual_from_gene(self, gene):
        gene = self._normalize_gene(gene)
        try:
            details = self.model.evaluate_with_details(gene)
            area_total, power_total, min_slack, total_sensitivity = details[:4]
            sens_ratio = float(total_sensitivity) / max(float(getattr(self, "sens_orig", self.model.sens_orig)), 1e-20)
            objectives = np.array([area_total, power_total, -min_slack, sens_ratio], dtype=np.float64)
            ind = Individual(gene, objectives)
            ind.original_objectives = objectives.copy()
            ind.raw_sensitivity = sens_ratio
            ind.evaluated = True

            area_limit, power_limit, slack_min = self._ppa_limits()
            area_over = float(details[5]) / max(float(self.area_orig), 1e-20)
            power_over = float(details[6]) / max(float(self.power_orig), 1e-20)
            ind.area_power_feasible = area_over <= area_limit + 1e-12 and power_over <= power_limit + 1e-12
            ind.sensitivity_feasible = (
                True if getattr(self, "ser_as_objective", True)
                else sens_ratio <= float(getattr(self, "delta_sens_min", 1.0)) + 1e-12
            )
            ind.violation = (
                max(0.0, area_over - area_limit)
                + max(0.0, power_over - power_limit)
                + max(0.0, slack_min - float(min_slack))
                + (0.0 if getattr(self, "ser_as_objective", True)
                   else max(0.0, sens_ratio - float(getattr(self, "delta_sens_min", 1.0))))
            )
            return ind
        except Exception as exc:
            print(f"[PPA-REFINE] Failed to evaluate refined gene: {exc}")
            return Individual(gene, None)

    def _fill_gene_with_remaining_budget(self, gene, max_additions=None):
        """在不违反PPA约束的前提下，用剩余预算继续降低残余SER。"""
        gene = self._normalize_gene(gene).astype(np.int32)
        if max_additions is None:
            max_additions = self.K
        max_additions = max(0, min(self.K, int(max_additions)))
        if max_additions == 0:
            return gene

        area_cost, power_cost, slack, sens = self._ppa_arrays()
        area_limit, power_limit, slack_min = self._ppa_limits()
        safety = self._budget_safety_scale()
        area_budget = max(float(self.area_orig) * area_limit * safety, 1e-20)
        power_budget = max(float(self.power_orig) * power_limit * safety, 1e-20)
        voter_delay = float(self.model.voter_delay)
        total_sens_sum = float(getattr(self.model, "total_sensitivity_sum", np.sum(sens)))
        total_regs = int(getattr(self.model, "total_num_registers", self.K))

        for _ in range(max_additions):
            unselected = np.flatnonzero(gene == 0)
            if unselected.size == 0:
                break
            n = int(np.sum(gene))
            hard = float(np.dot(gene, sens))
            residual = (total_sens_sum - hard) / max(total_regs + 2 * n, 1)
            area_inc = float(np.dot(gene, area_cost))
            power_inc = float(np.dot(gene, power_cost))
            worst = float(np.min(slack - voter_delay * gene))
            next_area = area_inc + area_cost[unselected]
            next_power = power_inc + power_cost[unselected]
            next_worst = np.minimum(worst, slack[unselected] - voter_delay)
            next_residual = (
                total_sens_sum - hard - sens[unselected]
            ) / np.maximum(total_regs + 2 * (n + 1), 1)
            gain = residual - next_residual
            feasible = (
                (next_area <= area_budget + 1e-12)
                & (next_power <= power_budget + 1e-12)
                & (next_worst >= slack_min - 1e-12)
                & (gain > 0)
            )
            if not np.any(feasible):
                break
            cost = (
                area_cost[unselected] / area_budget
                + power_cost[unselected] / power_budget
            )
            score = np.where(feasible, gain / np.maximum(cost, 1e-20), -np.inf)
            gene[unselected[int(np.argmax(score))]] = 1

        return gene.astype(np.int32)

    def _refine_gene_by_ppa(self, gene, max_swaps=None):
        gene = self._normalize_gene(gene).astype(np.int32)
        if max_swaps is None:
            max_swaps = getattr(self, "swap_local_search_max_swaps", 24)
        max_swaps = max(0, int(max_swaps))
        area_cost, power_cost, slack, sens = self._ppa_arrays()
        area_limit, power_limit, slack_min = self._ppa_limits()
        safety = self._budget_safety_scale()
        area_budget = max(float(self.area_orig) * area_limit * safety, 1e-20)
        power_budget = max(float(self.power_orig) * power_limit * safety, 1e-20)
        voter_delay = float(self.model.voter_delay)
        total_sens_sum = float(getattr(self.model, "total_sensitivity_sum", np.sum(sens)))
        total_regs = int(getattr(self.model, "total_num_registers", self.K))

        def metrics(g):
            n = int(np.sum(g))
            hard = float(np.dot(g, sens))
            residual = (total_sens_sum - hard) / max(total_regs + 2 * n, 1)
            area_inc = float(np.dot(g, area_cost))
            power_inc = float(np.dot(g, power_cost))
            worst = float(np.min(slack - voter_delay * g))
            return residual, area_inc, power_inc, worst

        if getattr(self, "repair_fill_enabled", True):
            # repair: drop the least useful selected registers until all PPA constraints are met.
            for _ in range(self.K):
                _, area_inc, power_inc, worst = metrics(gene)
                if area_inc <= area_budget + 1e-12 and power_inc <= power_budget + 1e-12 and worst >= slack_min - 1e-12:
                    break
                selected = np.where(gene == 1)[0]
                if selected.size == 0:
                    break
                area_excess = max(0.0, area_inc - area_budget) / area_budget
                power_excess = max(0.0, power_inc - power_budget) / power_budget
                slack_excess = max(0.0, slack_min - worst)
                release = (
                    (area_cost[selected] / area_budget) * (1.0 + area_excess)
                    + (power_cost[selected] / power_budget) * (1.0 + power_excess)
                    + 5.0 * np.maximum(0.0, slack_min - (slack[selected] - voter_delay))
                    + slack_excess
                )
                damage = sens[selected] + 1e-12
                gene[selected[int(np.argmin(damage / np.maximum(release, 1e-12)))]] = 0

            # fill: greedily use the remaining budget for the best feasible SER improvement.
            for _ in range(self.K):
                residual, area_inc, power_inc, worst = metrics(gene)
                unselected = np.where(gene == 0)[0]
                if unselected.size == 0:
                    break
                n = int(np.sum(gene))
                hard = float(np.dot(gene, sens))
                next_area = area_inc + area_cost[unselected]
                next_power = power_inc + power_cost[unselected]
                next_worst = np.minimum(worst, slack[unselected] - voter_delay)
                next_residual = (total_sens_sum - hard - sens[unselected]) / np.maximum(total_regs + 2 * (n + 1), 1)
                gain = residual - next_residual
                feasible = (
                    (next_area <= area_budget + 1e-12)
                    & (next_power <= power_budget + 1e-12)
                    & (next_worst >= slack_min - 1e-12)
                    & (gain > 0)
                )
                if not np.any(feasible):
                    break
                cost = area_cost[unselected] / area_budget + power_cost[unselected] / power_budget
                score = np.where(feasible, gain / np.maximum(cost, 1e-20), -np.inf)
                gene[unselected[int(np.argmax(score))]] = 1

        if not getattr(self, "swap_local_search_enabled", True) or max_swaps <= 0:
            return gene.astype(np.int32)

        candidate_cap = int(getattr(self, "swap_local_search_candidate_cap", 0))
        for _ in range(max_swaps):
            selected = np.where(gene == 1)[0]
            unselected = np.where(gene == 0)[0]
            if selected.size == 0 or unselected.size == 0:
                break

            if candidate_cap > 0 and selected.size > candidate_cap:
                selected = selected[np.argsort(sens[selected])[:candidate_cap]]
            if candidate_cap > 0 and unselected.size > candidate_cap:
                denom = area_cost[unselected] / area_budget + power_cost[unselected] / power_budget
                score = sens[unselected] / np.maximum(denom, 1e-20)
                unselected = unselected[np.argsort(score)[-candidate_cap:]]

            best_gain = 0.0
            best_pair = None
            for rem in selected:
                base_gene = gene.copy()
                base_gene[rem] = 0
                _, area0, power0, worst0 = metrics(base_gene)
                next_area = area0 + area_cost[unselected]
                next_power = power0 + power_cost[unselected]
                next_worst = np.minimum(worst0, slack[unselected] - voter_delay)
                feasible = (
                    (next_area <= area_budget + 1e-12)
                    & (next_power <= power_budget + 1e-12)
                    & (next_worst >= slack_min - 1e-12)
                )
                if not np.any(feasible):
                    continue
                sens_gain = np.where(feasible, sens[unselected] - sens[rem], -np.inf)
                pos = int(np.argmax(sens_gain))
                if np.isfinite(sens_gain[pos]) and sens_gain[pos] > best_gain + 1e-12:
                    best_gain = float(sens_gain[pos])
                    best_pair = (int(rem), int(unselected[pos]))
            if best_pair is None:
                break
            rem, add = best_pair
            gene[rem] = 0
            gene[add] = 1

        gene = self._refine_gene_pair_exchange(
            gene,
            area_cap=area_budget,
            power_cap=power_budget,
            wns_floor=slack_min,
        )
        if getattr(self, "post_swap_fill_enabled", True):
            gene = self._fill_gene_with_remaining_budget(
                gene,
                max_additions=getattr(self, "post_swap_fill_max_additions", 12),
            )
        return gene.astype(np.int32)

    def _greedy_budget_gene(self, target_power_ratio=1.0, target_area_ratio=1.0, power_bias=1.0, area_bias=1.0):
        area_cost, power_cost, slack, sens = self._ppa_arrays()
        area_limit, power_limit, slack_min = self._ppa_limits()
        safety = self._budget_safety_scale()
        area_budget = max(float(self.area_orig) * area_limit * target_area_ratio * safety, 1e-20)
        power_budget = max(float(self.power_orig) * power_limit * target_power_ratio * safety, 1e-20)
        voter_delay = float(self.model.voter_delay)
        total_sens_sum = float(getattr(self.model, "total_sensitivity_sum", np.sum(sens)))
        total_regs = int(getattr(self.model, "total_num_registers", self.K))
        gene = np.zeros(self.K, dtype=np.int32)

        for _ in range(self.K):
            unselected = np.where(gene == 0)[0]
            if unselected.size == 0:
                break
            n = int(np.sum(gene))
            hard = float(np.dot(gene, sens))
            residual = (total_sens_sum - hard) / max(total_regs + 2 * n, 1)
            area_inc = float(np.dot(gene, area_cost))
            power_inc = float(np.dot(gene, power_cost))
            worst = float(np.min(slack - voter_delay * gene))
            next_area = area_inc + area_cost[unselected]
            next_power = power_inc + power_cost[unselected]
            next_worst = np.minimum(worst, slack[unselected] - voter_delay)
            next_residual = (total_sens_sum - hard - sens[unselected]) / np.maximum(total_regs + 2 * (n + 1), 1)
            gain = residual - next_residual
            feasible = (
                (next_area <= area_budget + 1e-12)
                & (next_power <= power_budget + 1e-12)
                & (next_worst >= slack_min - 1e-12)
                & (gain > 0)
            )
            if not np.any(feasible):
                break
            cost = area_bias * area_cost[unselected] / area_budget + power_bias * power_cost[unselected] / power_budget
            score = np.where(feasible, gain / np.maximum(cost, 1e-20), -np.inf)
            gene[unselected[int(np.argmax(score))]] = 1

        return self._refine_gene_preserving_budget(gene)

    def _budget_aware_seed_genes(self, max_genes=24, record_archive=True):
        max_genes = max(0, int(max_genes or 0))
        genes = []
        seen = set()

        def add(g, refine=True):
            if g is None or len(genes) >= max_genes:
                return False
            g = self._refine_gene_cached(g) if refine else self._normalize_gene(g)
            key = self._gene_key(g)
            if key not in seen and np.sum(g) > 0:
                seen.add(key)
                genes.append(g)
                return True
            return False

        if max_genes <= 0:
            return genes

        # 强贪婪/MILP档案不再在SAC尚未训练时提前给出答案。未解锁阶段
        # 仍可通过当前种群的memetic精修获得学习信号。
        if (
            getattr(self, "defer_archive_until_policy_ready", False)
            and not getattr(self, "sac_policy_ready", False)
        ):
            return genes

        if getattr(self, "enable_greedy_archive", True):
            archive_count = min(max_genes, max(1, getattr(self, "greedy_archive_injection_count", max_genes)))
            for ind in self._select_greedy_archive_individuals(
                archive_count,
                record_archive=record_archive,
            ):
                add(ind.gene, refine=False)
                if len(genes) >= max_genes:
                    return genes

        for target in (0.08, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00):
            add(self._greedy_budget_gene(target_power_ratio=target, target_area_ratio=target))
            if len(genes) >= max_genes:
                return genes
        add(self._greedy_budget_gene(target_power_ratio=1.0, target_area_ratio=1.0, power_bias=1.5, area_bias=0.8))
        add(self._greedy_budget_gene(target_power_ratio=1.0, target_area_ratio=1.0, power_bias=0.8, area_bias=1.5))

        area_cost, power_cost, slack, sens = self._ppa_arrays()
        area_limit, power_limit, slack_min = self._ppa_limits()
        safety = self._budget_safety_scale()
        area_budget = max(float(self.area_orig) * area_limit * safety, 1e-20)
        power_budget = max(float(self.power_orig) * power_limit * safety, 1e-20)
        safe = np.where(slack - float(self.model.voter_delay) >= slack_min - 1e-12)[0]
        if safe.size:
            for score in (
                sens[safe],
                sens[safe] / np.maximum(area_cost[safe] / area_budget + power_cost[safe] / power_budget, 1e-20),
                sens[safe] / np.maximum(power_cost[safe], 1e-20),
                sens[safe] / np.maximum(area_cost[safe], 1e-20),
            ):
                order = safe[np.argsort(score)[::-1]]
                g = np.zeros(self.K, dtype=np.int32)
                for idx in order:
                    trial = g.copy()
                    trial[idx] = 1
                    if self._gene_metrics_fast(trial)["feasible"]:
                        g = trial
                add(g)
                if len(genes) >= max_genes:
                    return genes

        if self.current_population:
            candidates = [ind for ind in self.current_population if getattr(ind, "gene", None) is not None]
            for ind in candidates[: max(4, max_genes // 3)]:
                add(ind.gene)
                if len(genes) >= max_genes:
                    break

        return genes[:max_genes]

    def _snapshot_population_for_credit(self, population):
        """复制种群的基因和目标值，避免后续原地修改污染动作归因。"""
        snapshot = []
        for ind in population or []:
            gene = getattr(ind, "gene", None)
            if gene is None:
                continue
            objectives = getattr(ind, "objectives", None)
            copied = Individual(
                np.array(gene, dtype=int, copy=True),
                None if objectives is None else np.array(objectives, copy=True),
            )
            copied.violation = float(getattr(ind, "violation", 0.0) or 0.0)
            copied.area_power_feasible = bool(getattr(ind, "area_power_feasible", True))
            copied.sensitivity_feasible = bool(getattr(ind, "sensitivity_feasible", True))
            copied.raw_sensitivity = getattr(ind, "raw_sensitivity", None)
            snapshot.append(copied)
        return snapshot

    def _install_budget_aware_elites(self, max_new=None, context="", record_archive=True):
        if max_new is None:
            max_new = max(8, min(40, self.pop_size // 10))
        max_new = max(0, int(max_new))
        refine_enabled = self._should_memetic_refine(context)
        refine_count = self.memetic_refine_count if refine_enabled else 0
        if max_new == 0 and refine_count <= 0:
            return
        try:
            seed_genes = self._budget_aware_seed_genes(
                max_new,
                record_archive=record_archive,
            )
            elite_inds = [self._make_individual_from_gene(g) for g in seed_genes]

            if refine_enabled:
                elite_inds.extend(self._memetic_refine_population(
                    self.current_population,
                    max_count=refine_count,
                    context=context,
                ))

            unique_elites = []
            seen = set()
            for ind in elite_inds:
                if ind is None or ind.objectives is None or getattr(ind, "gene", None) is None:
                    continue
                key = self._gene_key(ind.gene)
                if key in seen:
                    continue
                seen.add(key)
                unique_elites.append(ind)
            elite_inds = unique_elites
            if not elite_inds:
                return

            added_archive = 0
            if record_archive:
                added_archive = self._append_to_cumulative_elite_pool(
                    elite_inds,
                    source=f"budget_aware:{context}",
                    max_pool=getattr(self, "greedy_archive_max_pool", 1200),
                )

            def merge(pop, limit=None):
                merged = []
                seen_local = set()
                for ind in elite_inds + list(pop or []):
                    if ind is None or getattr(ind, "gene", None) is None:
                        continue
                    key = self._gene_key(ind.gene)
                    if key in seen_local:
                        continue
                    seen_local.add(key)
                    merged.append(ind)
                    if limit is not None and len(merged) >= limit:
                        break
                return merged

            self.current_population = merge(self.current_population)
            if getattr(self.spea2, "population", None) is not None:
                self.spea2.population = merge(self.spea2.population, self.pop_size)
            if getattr(self.nsga2, "population", None) is not None:
                self.nsga2.population = merge(self.nsga2.population, self.pop_size)
            print(f"[PPA-REFINE] {context} injected {len(elite_inds)} archive/local-search elites, history_added={added_archive}")
        except Exception as exc:
            print(f"[PPA-REFINE] elite injection skipped: {exc}")

    def reset(self, transfer_experience: dict = None):
        """
        重置MOEA状态（开始新的Episode）。支持从上一次Episode传递经验以进行热启动（warm-start）。

        参数:
        ----------
        transfer_experience : dict, optional
            可选的经验字典，包含：
            - 'elite_genes': list of gene arrays 或 None
            - 'spea2_params': dict
            - 'nsga2_params': dict
        """
        # 基本重置
        self.current_population = None
        self.current_generation = 0
        self.no_improvement_count = 0
        self.converged = False  # 重置收敛标志
        # ── 关键修复：只有在没有经验时才重置热启动标志 ───────────────────────
        # 有经验时热启动已在前一个 episode 成功设置，保持其状态
        if transfer_experience is None:
            self.warm_start_completed = False

        # The greedy trajectory is a fixed baseline, not an SAC achievement.
        # Build and archive it before the first policy action so rewards only
        # credit solutions that improve the common feasible baseline.
        if getattr(self, "enable_greedy_archive", False):
            try:
                self._build_greedy_archive(record_archive=True)
            except Exception as exc:
                print(f"[GREEDY-ARCHIVE] baseline initialization failed: {exc}")

        # 如果传入的是文件路径字符串，则尝试从磁盘加载
        if isinstance(transfer_experience, str):
            try:
                transfer_experience = self.load_experience(transfer_experience)
                print(f"[MOEA]: 从文件加载经验: {self.last_experience_file}")
            except Exception as e:
                print(f"[MOEA-WARN]: 加载经验文件失败: {e}")
                transfer_experience = None

        # 如果提供了 transfer_experience（字典），则进行经验加载/热启动
        if transfer_experience is not None:
            try:
                # 重置算法内部状态（如果未由经验覆盖）
                if hasattr(self.spea2, 'reset'):
                    self.spea2.reset()
                if hasattr(self.nsga2, 'reset'):
                    self.nsga2.reset()

                # 参数传递
                spea2_params = transfer_experience.get('spea2_params')
                nsga2_params = transfer_experience.get('nsga2_params')
                if spea2_params:
                    try:
                        self.spea2.update_parameters(spea2_params)
                    except Exception:
                        pass
                if nsga2_params:
                    try:
                        self.nsga2.update_parameters(nsga2_params)
                    except Exception:
                        pass

                # 【优化后的热启动逻辑】
                # 使用非支配排序选择真正的 Pareto 最优解，并动态调整精英数量
                elite_genes = transfer_experience.get('elite_genes')
                if elite_genes:
                    try:
                        elite_pool = list(elite_genes)

                        # ========== 改进1: 减少热启动比例，增加随机探索 ==========
                        # 只保留小比例固定热启动，其余精英注入比例由SAC动作控制。
                        # 外部累积档案不会被清空，因此降低工作种群注入不会损失历史最优解。
                        WARM_START_RATIO = self.warm_start_base_ratio
                        max_elite_for_warm = max(3, int(self.pop_size * WARM_START_RATIO))
                        
                        # ========== 改进2: 动态调整精英比例（基于池质量） ==========
                        base_num_elite = 3
                        evaluated_pool = [g for g in elite_pool
                                          if isinstance(g, dict) or
                                          (isinstance(g, (list, np.ndarray, Individual)) and
                                           not (isinstance(g, Individual) and g.objectives is None))]
                        
                        quality_bonus = 0
                        if len(evaluated_pool) >= 10:
                            diversity_score = self._calculate_elite_diversity(evaluated_pool)
                            if diversity_score > 0.8:
                                quality_bonus = 3
                            elif diversity_score > 0.5:
                                quality_bonus = 2
                            else:
                                quality_bonus = 1
                        
                        NUM_ELITE = min(base_num_elite + quality_bonus, max_elite_for_warm, len(elite_pool))
                        print(f"[MOEA-WARM v2] 精英池: {len(elite_pool)}, 选择: {NUM_ELITE} (上限: {max_elite_for_warm})")

                        # ========== 改进4: 基于Pareto支配的多目标精英选择 ==========
                        # 策略：优先选择Pareto最优解，再考虑约束满足度
                        
                        def _calc_constraint_violation(ind):
                            """计算约束违例程度（越小越好，0=完全满足）"""
                            obj = self._get_elite_entry_objectives(ind)
                            if obj is None:
                                obj = np.asarray(getattr(ind, 'objectives', [])).flatten()
                            # 优先使用原始未惩罚的目标值
                            orig_obj = getattr(ind, 'original_objectives', None)
                            if orig_obj is not None:
                                obj = np.asarray(orig_obj).flatten()
                            if len(obj) < 4:
                                return 100.0
                            
                            # 敏感性违例
                            raw_sens = getattr(ind, 'raw_sensitivity', None)
                            if raw_sens is None:
                                raw_sens = float(obj[3]) if len(obj) > 3 else 0.0
                            else:
                                raw_sens = float(np.asarray(raw_sens).flatten()[0]) if hasattr(raw_sens, '__len__') else float(raw_sens)
                            sens_viol = 0.0 if getattr(self, "ser_as_objective", True) else max(
                                0.0, raw_sens - getattr(self, 'delta_sens_min', 0.4)
                            )
                            
                            # 时序违例（obj[2] >= 1e5 表示违例）
                            timing_viol = 1.0 if float(obj[2]) >= 1e5 else 0.0
                            
                            # 【修复】：面积/功耗违例需要处理绝对值和比率两种情况
                            base_area = getattr(self, 'area_orig', 1.0)
                            base_power = getattr(self, 'power_orig', 1.0)
                            
                            # 检测 objectives 格式（绝对值 area > 10 说明是 um²）
                            if float(obj[0]) > 10.0:
                                # 绝对值 → 转换为比率
                                obj0_ratio = float(obj[0]) / (base_area + 1e-10)
                                obj1_ratio = float(obj[1]) / (base_power + 1e-10)
                            else:
                                # 已经是比率
                                obj0_ratio = float(obj[0])
                                obj1_ratio = float(obj[1])
                            
                            # 面积违例（基于比率）
                            area_viol = max(0.0, obj0_ratio - 1.0 - self.config["constraints"]["delta_area_max"])
                            
                            # 功耗违例
                            power_viol = max(0.0, obj1_ratio - 1.0 - self.config["constraints"]["delta_power_max"])
                            
                            # 加权总违例
                            total = 1000 * sens_viol + 50 * timing_viol + 20 * area_viol + 20 * power_viol
                            return float(total)
                        
                        def _is_better(a, b, evaluated_pool):
                            """
                            检查解a是否比解b更好（Pareto支配 + 约束满足度）
                            """
                            obj_a = self._get_elite_entry_objectives(a)
                            if obj_a is None:
                                obj_a = np.asarray(getattr(a, 'objectives', [])).flatten()
                            obj_b = self._get_elite_entry_objectives(b)
                            if obj_b is None:
                                obj_b = np.asarray(getattr(b, 'objectives', [])).flatten()
                            
                            if len(obj_a) < 4 or len(obj_b) < 4:
                                return a is not b
                            
                            # 计算违例
                            viol_a = _calc_constraint_violation(a)
                            viol_b = _calc_constraint_violation(b)
                            
                            # 约束满足度差异很大时，约束更优的解更好
                            if abs(viol_a - viol_b) > 100:
                                return viol_a < viol_b
                            
                            # 否则用Pareto支配判断
                            a_dominates_b = np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)
                            b_dominates_a = np.all(obj_b <= obj_a) and np.any(obj_b < obj_a)
                            
                            if a_dominates_b and not b_dominates_a:
                                return True
                            if b_dominates_a and not a_dominates_b:
                                return False
                            
                            # 无支配关系时，约束违例少的更好
                            return viol_a < viol_b
                        
                        # 收集所有有评估结果的精英
                        evaluated_elites = []
                        for g in elite_pool:
                            obj = self._get_elite_entry_objectives(g)
                            if obj is not None:
                                obj = np.asarray(obj).flatten()
                                if len(obj) >= 4:
                                    evaluated_elites.append(g)
                        
                        # 基于Pareto支配和约束满足度进行排序
                        if len(evaluated_elites) > 1:
                            # 计算每个解的约束违例
                            elite_with_viol = [(ind, _calc_constraint_violation(ind)) for ind in evaluated_elites]
                            # 按违例排序（低违例优先）
                            elite_with_viol.sort(key=lambda x: x[1])
                            # 取前 NUM_ELITE 个
                            selected_individuals = [x[0] for x in elite_with_viol[:NUM_ELITE]]
                        else:
                            selected_individuals = evaluated_elites[:NUM_ELITE] if evaluated_elites else []
                        
                        selected_elites = []
                        for ind in selected_individuals:
                            gene = self._get_elite_entry_gene(ind)
                            if gene is not None:
                                selected_elites.append(np.array(gene).flatten())
                            elif isinstance(ind, Individual):
                                selected_elites.append(ind.gene.flatten())
                        
                        # 补足数量
                        while len(selected_elites) < NUM_ELITE and len(elite_pool) > len(selected_elites):
                            # 将已选中的基因转换为 tuple 用于比较
                            selected_tuples = set(tuple(e.flatten().astype(int)) for e in selected_elites)
                            remaining = []
                            for g in elite_pool:
                                g_gene = self._get_elite_entry_gene(g)
                                if g_gene is None:
                                    continue
                                g_gene = np.array(g_gene).flatten()
                                g_tuple = tuple(g_gene.flatten().astype(int))
                                if g_tuple not in selected_tuples:
                                    remaining.append(g)
                            
                            if remaining:
                                r = remaining[0]
                                r_gene = self._get_elite_entry_gene(r)
                                if r_gene is not None:
                                    selected_elites.append(np.array(r_gene).flatten())
                                else:
                                    break
                            else:
                                break
                        
                        # 打印选择的精英信息
                        if selected_individuals:
                            viols = [_calc_constraint_violation(ind) for ind in selected_individuals]
                            print(f"[MOEA-WARM v2] 选择了 {len(selected_elites)} 个精英（约束违例: min={min(viols):.2f}, mean={np.mean(viols):.2f}）")

                        # ========== 优化3: 避免重复评估 ==========
                        # 构建已评估解的缓存键集合
                        evaluated_cache = set()
                        if hasattr(self, 'evaluation_cache'):
                            evaluated_cache = self.evaluation_cache

                        spea2_pop = []
                        nsga2_pop = []

                        # 添加精英解
                        for gene in selected_elites:
                            gene_array = np.array(gene).flatten()
                            cache_key = ''.join(map(str, gene_array.astype(int)))

                            # 检查是否已评估
                            if cache_key in evaluated_cache:
                                # 已评估，创建 Individual 并标记为已评估
                                ind = Individual(gene_array, None)
                                ind.evaluated = True
                                ind._cache_key = cache_key
                            else:
                                ind = Individual(gene_array, None)

                            spea2_pop.append(ind)
                            nsga2_pop.append(Individual(gene_array.copy(), None))

                        # # 【关键改进】大幅增加随机解比例
                        random_count = self.pop_size - len(selected_elites)
                        for i in range(random_count):
                            gene = np.random.randint(0, 2, self.K)
                            spea2_pop.append(Individual(gene.copy(), None))
                            nsga2_pop.append(Individual(gene.copy(), None))

                        self.spea2.population = spea2_pop
                        self.nsga2.population = nsga2_pop
                        self.current_population = spea2_pop + nsga2_pop
                        self.warm_start_completed = True

                        print(f"[MOEA-WARM] 种群初始化: {len(selected_elites)} 精英 + {random_count} 随机 = {self.pop_size}")

                        # 评估种群
                        try:
                            self.warm_start_evaluate(self.spea2.population)
                            self.warm_start_evaluate(self.nsga2.population)

                            # 标记新评估的解
                            for ind in self.spea2.population + self.nsga2.population:
                                if ind.objectives is not None:
                                    ind.evaluated = True

                            initial_pareto = self._extract_pareto_front(self.spea2.population)
                            initial_hv = self._calculate_hypervolume(initial_pareto)
                            if initial_pareto:
                                sens_values = []
                                for ind in initial_pareto:
                                    if ind.objectives is not None:
                                        obj = np.asarray(ind.objectives).flatten()
                                        if len(obj) >= 4:
                                            sens_values.append(float(obj[3]))
                                if sens_values:
                                    print(f"[MOEA-WARM v2] 初始Pareto敏感性: min={min(sens_values):.4f}, mean={np.mean(sens_values):.4f}")
                            
                            print(f"[MOEA-WARM v2] 初始HV={initial_hv:.4f}")
                        except Exception as e:
                            import traceback
                            print(f"[MOEA-WARM] 种群评估失败: {e}")
                            traceback.print_exc()
                    except Exception as e:
                        print(f"[MOEA-WARM]: 热启动精英基因失败: {e}")
                        import traceback
                        traceback.print_exc()
                    except Exception as e:
                        print(f"[MOEA-WARM]: 热启动精英基因失败: {e}")
                else:
                    # 如果没有精英基因（第一个Episode），生成随机种群
                    try:
                        spea2_pop = []
                        nsga2_pop = []
                        
                        # 【多样性增强】第一个Episode使用混合策略初始化
                        # 策略：部分随机 + 部分启发式初始化
                        HEURISTIC_RATIO = 0.3  # 30%启发式，70%随机
                        heuristic_count = int(self.pop_size * HEURISTIC_RATIO)
                        random_count = self.pop_size - heuristic_count
                        
                        # 1. 生成启发式初始解（基于问题特征）
                        # 根据敏感性(is_critical)和松弛量(slack)来决定是否选择寄存器
                        if hasattr(self, 'is_critical') and hasattr(self, 'slack'):
                            # 策略1：选择低敏感性的寄存器（安全选择）
                            safe_genes = self._create_heuristic_gene_low_sensitivity(heuristic_count // 3)
                            # 策略2：选择高松弛的寄存器（低风险）
                            low_risk_genes = self._create_heuristic_gene_high_slack(heuristic_count // 3)
                            # 策略3：混合策略（平衡）
                            mixed_genes = self._create_heuristic_gene_balanced(heuristic_count - len(safe_genes) - len(low_risk_genes))
                            
                            heuristic_genes = safe_genes + low_risk_genes + mixed_genes
                        else:
                            heuristic_genes = []
                        
                        # 添加启发式解
                        for gene in heuristic_genes:
                            spea2_pop.append(Individual(gene.copy(), None))
                            nsga2_pop.append(Individual(gene.copy(), None))
                        
                        # 添加随机解
                        for i in range(random_count):
                            gene = np.random.randint(0, 2, self.K)
                            spea2_pop.append(Individual(gene.copy(), None))
                            nsga2_pop.append(Individual(gene.copy(), None))
                        
                        self.spea2.population = spea2_pop
                        self.nsga2.population = nsga2_pop
                        self.current_population = spea2_pop + nsga2_pop
                        self.warm_start_completed = True
                        print(f"[MOEA-WARM] 第一个Episode，启发式+随机混合初始化: {len(heuristic_genes)} 个启发式 + {random_count} 个随机")
                    except Exception as e:
                        print(f"[MOEA-WARM]: 混合种群初始化失败，回退到纯随机: {e}")
                        # 回退到纯随机
                        spea2_pop = []
                        nsga2_pop = []
                        for i in range(self.pop_size):
                            gene = np.random.randint(0, 2, self.K)
                            spea2_pop.append(Individual(gene.copy(), None))
                            nsga2_pop.append(Individual(gene.copy(), None))
                        self.spea2.population = spea2_pop
                        self.nsga2.population = nsga2_pop
                        self.current_population = spea2_pop + nsga2_pop
                        self.warm_start_completed = True
                    except Exception as e:
                        print(f"[MOEA-WARM]: 随机种群初始化失败: {e}")
            except Exception as e:
                print(f"[MOEA-WARN] 应用 transfer_experience 时出错: {e}")

        # 重置历史记录（保留部分信息用于统计分析）
        self.history = {
            'actions': [],
            'stats': [],
            'populations': []
        }
        
        # 重置收敛历史
        self.convergence_history = {
            'hv': [],
            'igd': [],
            'diversity': [],
            'pareto_size': [],
            'improvement': []
        }

        print("MOEA已重置，准备开始新Episode")

    def warm_start_evaluate(self, population):
        """
        热启动后评估种群（设置objectives属性）

        这个方法专门用于热启动后，确保每个Individual的objectives被正确设置。
        使用与MOEA运行时相同的评估方式。
        """
        if not population:
            return
        
        evaluated_count = 0
        failed_count = 0
        
        # 获取敏感性阈值
        delta_sens_min = getattr(self, 'delta_sens_min', 0.4)
        
        # 收集所有基因用于批量评估
        genes_list = []
        for ind in population:
            if hasattr(ind, 'gene') and ind.gene is not None:
                genes_list.append(ind.gene.flatten())
            else:
                genes_list.append(np.zeros(self.K))
        
        if not genes_list:
            return
        
        # 转换为数组
        genes_batch = np.array(genes_list)
        
        try:
            # 使用与MOEA运行时相同的批量评估方法
            # 【修复】：检查 shared_gpu_evaluator 而不是 gpu_evaluator
            if hasattr(self, 'shared_gpu_evaluator') and self.shared_gpu_evaluator is not None:
                # GPU评估模式
                area_batch, power_batch, slack_batch, ser_batch = self.shared_gpu_evaluator.gpu_evaluator.evaluate_batch_gpu(genes_batch)
                
                # 更新每个个体的objectives
                for i, ind in enumerate(population):
                    if hasattr(ind, 'gene') and ind.gene is not None:
                        if i < len(area_batch):
                            area = float(area_batch[i])
                            power = float(power_batch[i])
                            slack = float(slack_batch[i])
                            ser = float(ser_batch[i])
                        else:
                            area, power, slack, ser = 1e6, 1e6, -1e6, 1.0
                        
                        # 统一归一化敏感性为 0~1 比例，避免与后续约束判断口径不一致
                        ser_ratio = ser / (self.sens_orig + 1e-10) if ser > 1.0 else float(ser)
                        
                        # 转换为MOEA目标格式: [area, power, -slack, ser_ratio]
                        ind.objectives = np.array([area, power, -slack, ser_ratio], dtype=np.float64)
                        ind.original_objectives = ind.objectives.copy()  # 保存原始值用于违例判断
                        ind.raw_sensitivity = float(ser)
                        
                        # 敏感性约束检查（基于归一化比例）
                        ind.sensitivity_feasible = bool(ser_ratio <= delta_sens_min)
                        
                        # 【Bug2修复】：显式计算并赋值 area_power_feasible
                        # 使用配置中的阈值计算 delta
                        delta_area_th = self.config["constraints"]["delta_area_max"]  # 0.2
                        delta_power_th = self.config["constraints"]["delta_power_max"]  # 0.15
                        base_area = getattr(self, 'area_orig', getattr(self.model, 'base_area', 1.0))
                        base_power = getattr(self, 'power_orig', getattr(self.model, 'base_power', 1.0))
                        delta_area = (area - base_area) / (base_area + 1e-10)
                        delta_power = (power - base_power) / (base_power + 1e-10)
                        ind.area_power_feasible = bool(delta_area <= delta_area_th and delta_power <= delta_power_th)
                        
                        evaluated_count += 1
            else:
                # CPU评估模式：使用model的批量评估
                if hasattr(self.model, 'evaluate_population'):
                    results = self.model.evaluate_population(genes_batch)
                    # model.evaluate_population返回 [area, power, -slack, normalized_sens]
                    for i, ind in enumerate(population):
                        if hasattr(ind, 'gene') and ind.gene is not None and i < len(results):
                            ind.objectives = np.array(results[i], dtype=np.float64)
                            ind.raw_sensitivity = float(results[i][3])
                            # 【修复】：确保比较是标量
                            sens_val = float(results[i][3])
                            ind.sensitivity_feasible = bool(sens_val <= delta_sens_min)
                            # 【Bug2修复】：显式计算 area_power_feasible
                            delta_area_th = self.config["constraints"]["delta_area_max"]
                            delta_power_th = self.config["constraints"]["delta_power_max"]
                            base_area = getattr(self, 'area_orig', getattr(self.model, 'base_area', 1.0))
                            base_power = getattr(self, 'power_orig', getattr(self.model, 'base_power', 1.0))
                            obj0, obj1 = float(results[i][0]), float(results[i][1])
                            delta_area = (obj0 - base_area) / (base_area + 1e-10)
                            delta_power = (obj1 - base_power) / (base_power + 1e-10)
                            ind.area_power_feasible = bool(delta_area <= delta_area_th and delta_power <= delta_power_th)
                            evaluated_count += 1
                else:
                    # 后备：逐个评估
                    for ind in population:
                        if hasattr(ind, 'gene') and ind.gene is not None:
                            try:
                                objs = self.evaluate(ind.gene)
                                if isinstance(objs, (tuple, list, np.ndarray)) and len(objs) >= 4:
                                    # 已经是正确的格式
                                    objs = np.asarray(objs).flatten()
                                    ind.objectives = np.array(objs[:4], dtype=np.float64)
                                    ind.raw_sensitivity = float(objs[3])
                                    # 【修复】：确保比较是标量
                                    ind.sensitivity_feasible = bool(float(objs[3]) <= delta_sens_min)
                                    # 【Bug2修复】：显式计算 area_power_feasible
                                    delta_area_th = self.config["constraints"]["delta_area_max"]
                                    delta_power_th = self.config["constraints"]["delta_power_max"]
                                    base_area = getattr(self, 'area_orig', getattr(self.model, 'base_area', 1.0))
                                    base_power = getattr(self, 'power_orig', getattr(self.model, 'base_power', 1.0))
                                    delta_area = (float(objs[0]) - base_area) / (base_area + 1e-10)
                                    delta_power = (float(objs[1]) - base_power) / (base_power + 1e-10)
                                    ind.area_power_feasible = bool(delta_area <= delta_area_th and delta_power <= delta_power_th)
                                    evaluated_count += 1
                                else:
                                    failed_count += 1
                                    ind.sensitivity_feasible = False
                                    ind.area_power_feasible = False
                            except Exception:
                                failed_count += 1
                                ind.sensitivity_feasible = False
                                ind.area_power_feasible = False
        except Exception as e:
            print(f"[WARM_EVAL] 批量评估失败: {e}")
            import traceback
            traceback.print_exc()
            # 后备：逐个评估
            for ind in population:
                if hasattr(ind, 'gene') and ind.gene is not None:
                    try:
                        objs = self.evaluate(ind.gene)
                        if isinstance(objs, (tuple, list, np.ndarray)) and len(objs) >= 4:
                            objs = np.asarray(objs).flatten()
                            ind.objectives = np.array(objs[:4], dtype=np.float64)
                            ind.raw_sensitivity = float(objs[3])
                            ind.sensitivity_feasible = bool(float(objs[3]) <= delta_sens_min)
                            delta_area_th = self.config["constraints"]["delta_area_max"]
                            delta_power_th = self.config["constraints"]["delta_power_max"]
                            base_area = getattr(self, 'area_orig', getattr(self.model, 'base_area', 1.0))
                            base_power = getattr(self, 'power_orig', getattr(self.model, 'base_power', 1.0))
                            delta_area = (float(objs[0]) - base_area) / (base_area + 1e-10)
                            delta_power = (float(objs[1]) - base_power) / (base_power + 1e-10)
                            ind.area_power_feasible = bool(delta_area <= delta_area_th and delta_power <= delta_power_th)
                            evaluated_count += 1
                        else:
                            failed_count += 1
                            ind.sensitivity_feasible = False
                            ind.area_power_feasible = False
                    except Exception:
                        failed_count += 1
                        ind.sensitivity_feasible = False
                        ind.area_power_feasible = False
        
        print(f"[WARM_EVAL] 评估完成: 成功={evaluated_count}, 失败={failed_count}, 总数={len(population)}")
    
    def save_experience(self, filepath: str = None, overwrite: bool = False) -> dict:
        """
        保存Episode经验以用于下一个Episode的热启动（并可持久化到磁盘）。

        参数:
        - filepath: 可选，持久化路径（如果未提供则使用 results/experiences/experience_<ts>.pkl）
        - overwrite: 当 filepath 存在时是否覆盖

        返回：经验字典（并在 filepath 不为空时同时保存到磁盘）
        """
        import pickle
        from pathlib import Path
        from datetime import datetime

        experience = {}
        
        # 【关键修复】累积式精英池：合并历史上所有精英解（保存完整个体信息）
        all_elite_entries = []
        
        # 1. 首先加载之前累积的精英解（如果存在）
        if hasattr(self, 'cumulative_elite_pool') and self.cumulative_elite_pool:
            all_elite_entries.extend(self.cumulative_elite_pool)
            print(f"[MOEA-ELITE] 累积精英池: 之前有 {len(self.cumulative_elite_pool)} 个解")
        
        # 2. 添加当前Episode的新精英解
        try:
            final_pareto = self._extract_pareto_front(self.current_population) if self.current_population else []
            
            # 合并去重
            new_elites = 0
            for ind in final_pareto:
                gene = ind.gene.copy() if ind.gene is not None else None
                if gene is None:
                    continue
                is_duplicate = False
                for existing_entry in all_elite_entries:
                    existing_gene = existing_entry.get('gene') if isinstance(existing_entry, dict) else getattr(existing_entry, 'gene', None)
                    if existing_gene is not None and np.array_equal(gene, existing_gene):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    # 保存完整的个体信息（修复：不再只保存基因）
                    entry = {
                        'gene': gene,
                        'objectives': ind.objectives.copy() if ind.objectives is not None else None,
                        'raw_sensitivity': float(ind.raw_sensitivity) if hasattr(ind, 'raw_sensitivity') and ind.raw_sensitivity is not None else None,
                        'sensitivity_feasible': bool(getattr(ind, 'sensitivity_feasible', False)),
                        'area_power_feasible': bool(getattr(ind, 'area_power_feasible', True)),
                        'violation': float(ind.violation) if hasattr(ind, 'violation') and ind.violation is not None else None,
                    }
                    all_elite_entries.append(entry)
                    new_elites += 1
            
            print(f"[MOEA-ELITE] 当前Episode新增 {new_elites} 个精英解，总计 {len(all_elite_entries)} 个")
        except Exception as e:
            print(f"[MOEA-ELITE] 提取当前精英解失败: {e}")
        
        # 3. 更新累积精英池
        self.cumulative_elite_pool = all_elite_entries.copy()
        
        # 4. 保存到经验中（限制最大数量避免内存爆炸）
        # 4. 保存到经验中（限制最大数量避免内存爆炸，并保留预算多样性）
        MAX_ELITE_POOL = int(self.config.get('moea', {}).get('elite_pool_save_limit', 800)) if isinstance(self.config, dict) else 800
        if len(all_elite_entries) > MAX_ELITE_POOL:
            all_elite_entries = self._prune_elite_entries(all_elite_entries, MAX_ELITE_POOL)
            self.cumulative_elite_pool = all_elite_entries.copy()
            print(f"[MOEA-ELITE] 精英池过大，按预算多样性压缩到 {len(all_elite_entries)} 个")

        experience['elite_genes'] = all_elite_entries

        # 算法参数
        try:
            experience['spea2_params'] = self.spea2.get_current_parameters() if hasattr(self.spea2, 'get_current_parameters') else {}
        except Exception:
            experience['spea2_params'] = {}
        try:
            experience['nsga2_params'] = self.nsga2.get_current_parameters() if hasattr(self.nsga2, 'get_current_parameters') else {}
        except Exception:
            experience['nsga2_params'] = {}
        experience['action_schema'] = (
            self.config.get('action_schema', 'legacy_hyperparameters')
            if isinstance(self.config, dict) else 'legacy_hyperparameters'
        )
        experience['search_controls'] = dict(getattr(self, 'search_controls', {}))

        # 变异器可序列化状态（如果存在）
        try:
            experience['spea2_mutator'] = self.spea2.unified_mutator.state_dict() if hasattr(self.spea2, 'unified_mutator') and hasattr(self.spea2.unified_mutator, 'state_dict') else None
        except Exception:
            experience['spea2_mutator'] = None
        try:
            experience['nsga2_mutator'] = self.nsga2.unified_mutator.state_dict() if hasattr(self.nsga2, 'unified_mutator') and hasattr(self.nsga2.unified_mutator, 'state_dict') else None
        except Exception:
            experience['nsga2_mutator'] = None

        # 记录时间戳
        experience['timestamp'] = datetime.utcnow().isoformat()

        # 持久化（可选）
        if filepath is None:
            # 默认路径
            folder = Path('results') / 'experiences'
            folder.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filepath = folder / f'experience_{ts}.pkl'
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            if filepath.exists() and not overwrite:
                raise FileExistsError(f"经验文件已存在: {filepath}")

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(experience, f)
            self.last_experience_file = str(filepath)
            print(f"[MOEA]: 经验已持久化到 {filepath}")
        except Exception as e:
            print(f"[MOEA-WARN]: 保存经验到磁盘失败: {e}")

        return experience

    def load_experience(self, filepath: str) -> dict:
        """
        从磁盘加载经验并尝试恢复到 MOEA（返回加载的经验字典）
        """
        import pickle
        from pathlib import Path

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"经验文件不存在: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                experience = pickle.load(f)
        except Exception as e:
            print(f"[MOEA-WARN]: 加载经验失败: {e}")
            raise

        # 将经验应用到当前MOEA
        try:
            spea2_params = experience.get('spea2_params')
            nsga2_params = experience.get('nsga2_params')
            if spea2_params:
                try:
                    self.spea2.update_parameters(spea2_params)
                except Exception:
                    pass
            if nsga2_params:
                try:
                    self.nsga2.update_parameters(nsga2_params)
                except Exception:
                    pass

            # 恢复变异器状态
            spea2_mut = experience.get('spea2_mutator')
            if spea2_mut and hasattr(self.spea2, 'unified_mutator') and hasattr(self.spea2.unified_mutator, 'load_state'):
                try:
                    self.spea2.unified_mutator.load_state(spea2_mut)
                except Exception as e:
                    print(f"[MOEA-WARN]: 恢复 SPEA2 变异器状态失败: {e}")
            nsga2_mut = experience.get('nsga2_mutator')
            if nsga2_mut and hasattr(self.nsga2, 'unified_mutator') and hasattr(self.nsga2.unified_mutator, 'load_state'):
                try:
                    self.nsga2.unified_mutator.load_state(nsga2_mut)
                except Exception as e:
                    print(f"[MOEA-WARN]: 恢复 NSGA2 变异器状态失败: {e}")

            # 恢复精英基因到种群（如果存在）
            elite = experience.get('elite_genes')
            if elite:
                from algorithms.core_data import Individual
                K = self.K
                # 用精英以及轻微扰动初始化两个算法的种群
                self.spea2.population = [Individual(g.copy(), None) for g in elite[:self.pop_size]]
                self.nsga2.population = [Individual(g.copy(), None) for g in elite[-self.pop_size:]]
                while len(self.spea2.population) < self.pop_size:
                    self.spea2.population.append(Individual(np.random.randint(0, 2, K), None))
                while len(self.nsga2.population) < self.pop_size:
                    self.nsga2.population.append(Individual(np.random.randint(0, 2, K), None))
                
                # 【关键修复】恢复累积精英池
                self.cumulative_elite_pool = [g.copy() for g in elite]
                print(f"[MOEA-ELITE] 从经验文件恢复累积精英池: {len(elite)} 个解")

            # 记录路径
            self.last_experience_file = str(filepath)
        except Exception as e:
            print(f"[MOEA-WARN]: 应用加载经验时出错: {e}")

        return experience

    def _create_initial_population_for_algorithm(self, algorithm_name: str, seed_population: list = None) -> list:
        """
        为指定算法创建一个独立的初始种群。
        - 若提供 seed_population，则基于其中的个体进行采样/扰动以保持多样性
        - 否则生成随机种群
        - 返回的个体已评估（objectives 已设置）
        """
        from algorithms.core_data import Individual
        pop = []
        K = self.K
        if seed_population:
            # 从seed_population中采样并随机扰动
            for i in range(self.pop_size):
                base = seed_population[np.random.randint(0, len(seed_population))]
                gene = base.gene.copy() if hasattr(base, 'gene') else np.array(base, dtype=int)
                # 以小概率翻转位来生成多样性
                mask = np.random.rand(len(gene)) < 0.02  # 2% 位随机扰动
                gene = gene.copy()
                gene[mask] = 1 - gene[mask]
                ind = Individual(gene, None)
                # 如果seed中有评估结果，直接复制
                if hasattr(base, 'objectives') and base.objectives is not None:
                    ind.objectives = base.objectives.copy()
                else:
                    ind.objectives = None
                pop.append(ind)
        else:
            for _ in range(self.pop_size):
                pop.append(Individual(np.random.randint(0, 2, K), None))
        
        # 评估种群中所有 objectives 为 None 的个体（使用直接评估）
        needs_eval = [ind for ind in pop if ind.objectives is None]
        if needs_eval:
            try:
                # 直接调用模拟器的evaluate方法
                genes = np.array([ind.gene for ind in needs_eval])
                results = self.evaluate(genes)
                
                if isinstance(results, tuple) and len(results) == 2:
                    objs, violations = results
                    for i, ind in enumerate(needs_eval):
                        ind.objectives = objs[i]
                        ind.violation = violations[i] if violations is not None else 0.0
                else:
                    # 直接作为目标值处理
                    for i, ind in enumerate(needs_eval):
                        ind.objectives = results[i]
                        ind.violation = 0.0
            except Exception as e:
                print(f"[MOEA-WARN]: 初始种群评估失败 ({algorithm_name}): {e}")
                # 使用简单评估
                for ind in needs_eval:
                    if ind.objectives is None:
                        area = np.sum(ind.gene) * 10
                        power = np.sum(ind.gene) * 0.5
                        slack = 0.1
                        sens = 0.0
                        ind.objectives = np.array([area, power, -slack, sens])
                        ind.violation = 0.0
        
        return pop

    def check_convergence(self, threshold=1e-4, patience=5):
        """
        检查MOEA是否收敛
        """
        if len(self.convergence_history['hv']) < patience + 1:
            return False
        
        # 获取最近patience代的HV
        recent_hv = self.convergence_history['hv'][-patience-1:]
        
        # 计算相对改进率
        improvements = []
        for i in range(1, len(recent_hv)):
            if recent_hv[i-1] > 0:
                improvement = abs(recent_hv[i] - recent_hv[i-1]) / recent_hv[i-1]
                improvements.append(improvement)
        
        # 检查是否收敛
        if improvements:
            avg_improvement = np.mean(improvements)
            max_improvement = max(improvements)
            
            # 收敛条件：平均改进率低于阈值，且最大改进率也较低
            is_converged = (avg_improvement < threshold and 
                           max_improvement < threshold * 2)
            
            return is_converged
        
        return False
    
    def configure_search_controls(self, controls: Dict[str, Any] = None):
        """应用SAC给出的混合搜索控制量，不改动外部全局精英档案。"""
        controls = controls or {}
        self.search_controls = {
            "spea2_budget_share": float(np.clip(
                controls.get("spea2_budget_share", self.search_controls.get("spea2_budget_share", 0.5)),
                0.05, 0.95,
            )),
            "elite_injection_ratio": float(np.clip(
                controls.get("elite_injection_ratio", self.search_controls.get("elite_injection_ratio", 0.1)),
                0.0, 0.5,
            )),
            "local_search_strength": float(np.clip(
                controls.get("local_search_strength", self.search_controls.get("local_search_strength", 0.5)),
                0.0, 1.0,
            )),
            "restart_ratio": float(np.clip(
                controls.get("restart_ratio", self.search_controls.get("restart_ratio", 0.0)),
                0.0, 0.5,
            )),
        }

        strength = self.search_controls["local_search_strength"]
        self.greedy_archive_injection_count = int(round(
            self.pop_size * self.search_controls["elite_injection_ratio"]
        ))
        self.memetic_refine_count = int(round(self._base_memetic_refine_count * strength))
        self.swap_local_search_max_swaps = int(round(self._base_swap_local_search_max_swaps * strength))
        self.pair_swap_local_search_max_swaps = int(round(
            self._base_pair_swap_local_search_max_swaps * strength
        ))

    def set_sac_training_context(self, update_step=0, action_mode="stochastic_sample"):
        """延迟强基线档案，直到SAC已经获得最小训练经验。"""
        self.sac_update_step = max(0, int(update_step or 0))
        self.sac_action_mode = str(action_mode or "stochastic_sample")
        self.sac_policy_ready = (
            not self.defer_archive_until_policy_ready
            or self.sac_update_step >= self.archive_unlock_updates
        )

    def _algorithm_active_for_generation(self, algorithm: str, generation: int, total_generations: int) -> bool:
        """将连续预算比例均匀摊到当前SAC步的各代中。"""
        total_generations = max(1, int(total_generations))
        generation = max(0, int(generation))
        share = float(self.search_controls.get("spea2_budget_share", 0.5))
        if share >= 0.5:
            quota = total_generations if algorithm == "spea2" else max(
                1, int(round(total_generations * (1.0 - share) / 0.5))
            )
        else:
            quota = total_generations if algorithm == "nsga2" else max(
                1, int(round(total_generations * share / 0.5))
            )
        quota = min(total_generations, quota)
        before = (generation * quota) // total_generations
        after = ((generation + 1) * quota) // total_generations
        return after > before

    def _perturb_gene_with_expected_flips(self, gene, expected_flips):
        """优先执行等量0/1交换，避免重启个体的资源占用发生剧烈漂移。"""
        result = self._normalize_gene(gene).copy()
        flips = max(1, min(self.K, int(round(float(expected_flips)))))
        selected = np.flatnonzero(result)
        unselected = np.flatnonzero(result == 0)
        pair_count = min(flips // 2, len(selected), len(unselected))
        if pair_count > 0:
            result[np.random.choice(selected, pair_count, replace=False)] = 0
            result[np.random.choice(unselected, pair_count, replace=False)] = 1
        remaining = flips - 2 * pair_count
        if remaining > 0:
            indices = np.random.choice(self.K, min(remaining, self.K), replace=False)
            result[indices] = 1 - result[indices]
        return result

    def _restart_base_genes(self):
        """收集可用于受控重启的可行精英基因，并保持跨档案去重。"""
        if not getattr(self, "restart_use_elite_bases", True):
            return []
        if (
            getattr(self, "defer_archive_until_policy_ready", False)
            and not getattr(self, "sac_policy_ready", False)
        ):
            return []

        bases = []
        seen = set()
        sources = [
            getattr(self, "cumulative_elite_pool", []) or [],
            getattr(self, "greedy_archive", []) or [],
        ]
        for source in sources:
            for entry in source:
                gene = (
                    getattr(entry, "gene", None)
                    if isinstance(entry, Individual)
                    else self._get_elite_entry_gene(entry)
                )
                if gene is None:
                    continue
                normalized = self._normalize_gene(gene)
                key = normalized.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                bases.append(normalized.copy())
        return bases

    def _apply_controlled_restart(self):
        """混合精英邻域扰动与随机扰动，重启后修复可能的PPA违例。"""
        ratio = float(self.search_controls.get("restart_ratio", 0.0))
        count = min(self.pop_size, int(round(self.pop_size * ratio)))
        self._last_restart_count = 0
        if count <= 0:
            return 0

        elite_bases = self._restart_base_genes()

        def restart_population(population, expected_flips):
            if not population:
                return 0
            replace_count = min(count, len(population))
            source = list(population)
            replacement = []
            elite_count = min(
                replace_count,
                int(round(replace_count * self.restart_elite_base_ratio)),
            ) if elite_bases else 0
            for index in range(replace_count):
                if index < elite_count:
                    base_gene = elite_bases[np.random.randint(0, len(elite_bases))]
                else:
                    base = source[np.random.randint(0, len(source))]
                    base_gene = base.gene
                gene = self._perturb_gene_with_expected_flips(base_gene, expected_flips)
                candidate = self._make_individual_from_gene(gene)
                if (
                    candidate is not None
                    and getattr(self, "restart_refine_candidates", True)
                    and float(getattr(candidate, "violation", 0.0) or 0.0) > 1e-12
                ):
                    gene = self._refine_gene_by_ppa(gene, max_swaps=0)
                    candidate = self._make_individual_from_gene(gene)
                if candidate is not None:
                    replacement.append(candidate)
            if replacement:
                population[-len(replacement):] = replacement
            return len(replacement)

        spea2_flips = max(1.0, float(getattr(self.spea2, "pm", 1.0 / max(1, self.K))) * self.K)
        nsga2_flips = max(1.0, float(getattr(self.nsga2, "pm", 1.0 / max(1, self.K))) * self.K)
        restarted = restart_population(getattr(self.spea2, "population", None), spea2_flips)
        restarted = max(restarted, restart_population(getattr(self.nsga2, "population", None), nsga2_flips))
        restarted = max(restarted, restart_population(self.current_population, 0.5 * (spea2_flips + nsga2_flips)))
        self._last_restart_count = restarted
        return restarted

    def update_parameters(self, params: Dict[str, Any]):
        """
        更新算法参数
        """
        # 检查参数格式
        if 'spea2' in params and 'nsga2' in params:
            # 新格式：分别为两个算法设置独立参数
            self.spea2.update_parameters(params['spea2'])
            self.nsga2.update_parameters(params['nsga2'])
            if 'search_controls' in params:
                self.configure_search_controls(params['search_controls'])
        else:
            # 旧格式：两个算法使用相同参数
            print("警告: NSGA2和SPEA2将使用相同参数")
            self.spea2.update_parameters(params)
            self.nsga2.update_parameters(params)
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        获取当前参数
        """
        spea2_params = self.spea2.get_current_parameters()
        nsga2_params = self.nsga2.get_current_parameters()
        
        # 检查两个算法的参数是否相同
        if spea2_params == nsga2_params:
            # 参数相同，返回共享格式
            return spea2_params
        else:
            # 参数不同，返回独立格式
            return {
                'spea2': spea2_params,
                'nsga2': nsga2_params
            }
        
    def evaluate(self, genes):
        """
        Evaluate一个加固方案：
        输入：genes (0/1 vector)
        输出：四目标 + 违例信息
        """
        if self.use_gpu:
            # GPU模式评估
            area_new, power_new, slack_new, sens_new = self.model.evaluate(genes)
            
            # 获取真实的slack值（可能为负，用于约束检查）
            # evaluate返回的slack_new是截断后的值，需要重新计算真实值
            genes_array = np.asarray(genes).astype(int)
            if len(genes_array) != self.model.K:
                if len(genes_array) > self.model.K:
                    genes_array = genes_array[:self.model.K]
                else:
                    genes_array = np.pad(genes_array, (0, self.model.K - len(genes_array)), 'constant')
            
            # 计算真实的slack（可能为负）
            slack_after = self.model.slack.copy()
            tmr_indices = np.where(genes_array == 1)[0]
            if len(tmr_indices) > 0:
                slack_after[tmr_indices] -= self.model.voter_delay
            slack_real = float(np.min(slack_after))  # 真实的worst_slack，可能为负
            
            # 计算违例
            delta_area = (area_new - self.model.base_area) / self.model.base_area
            delta_power = (power_new - self.model.base_power) / self.model.base_power
            delta_sens = (1.0 - sens_new) / 1.0
            
            # 约束检查
            area_viol = max(0, delta_area - self.config["constraints"]["delta_area_max"])
            power_viol = max(0, delta_power - self.config["constraints"]["delta_power_max"])
            sens_viol = 0.0 if getattr(self, "ser_as_objective", True) else max(
                0, self.config["constraints"]["delta_sens_max"] - delta_sens
            )
            
            # Slack约束：使用真实的slack值（可能为负）进行检查
            slack_min = self.config["constraints"].get("slack_min", 0.0)
            if slack_real < slack_min:
                # slack小于阈值，给予很大惩罚
                delay_viol = 1000.0  # 大惩罚值，确保这类解被严重惩罚
            else:
                delay_viol = 0.0  # slack满足要求，无违例
            
            # 加权违约
            violation = (
                self.config["violation_weights"]["a"] * area_viol +
                self.config["violation_weights"]["p"] * power_viol +
                self.config["violation_weights"]["t"] * delay_viol +
                self.config["violation_weights"]["s"] * sens_viol
            )
            
            return np.array([area_new, power_new, -slack_new, sens_new, violation])
        else:
            genes = np.asarray(genes).astype(int)

            # 使用PPAModel的简化评估方法
            area_new, power_new, slack_new, sens_new = self.model.evaluate(genes)

            # 获取真实的slack值（可能为负，用于约束检查）
            # evaluate返回的slack_new是截断后的值，需要重新计算真实值
            if len(genes) != self.model.K:
                if len(genes) > self.model.K:
                    genes = genes[:self.model.K]
                else:
                    genes = np.pad(genes, (0, self.model.K - len(genes)), 'constant')
            
            # 计算真实的slack（可能为负）
            slack_after = self.model.slack.copy()
            tmr_indices = np.where(genes == 1)[0]
            if len(tmr_indices) > 0:
                slack_after[tmr_indices] -= self.model.voter_delay
            slack_real = float(np.min(slack_after))  # 真实的worst_slack，可能为负

            # 获取基准值
            area_base = self.model.base_area
            power_base = self.model.base_power
            slack_base = float(np.min(self.model.slack))
            sens_base = 1.0  # 归一化的基准敏感度

            # 计算Δ值
            delta_area = (area_new - area_base) / area_base
            delta_power = (power_new - power_base) / power_base
            delta_sens = (sens_base - sens_new) / sens_base

            # 约束检查
            area_viol = max(0, delta_area - self.config["constraints"]["delta_area_max"])
            power_viol = max(0, delta_power - self.config["constraints"]["delta_power_max"])
            sens_viol = 0.0 if getattr(self, "ser_as_objective", True) else max(
                0, self.config["constraints"]["delta_sens_max"] - delta_sens
            )
            
            # Slack约束：使用真实的slack值（可能为负）进行检查
            slack_min = self.config["constraints"].get("slack_min", 0.0)
            if slack_real < slack_min:
                # slack小于阈值，给予很大惩罚
                delay_viol = 1000.0  # 大惩罚值，确保这类解被严重惩罚
            else:
                delay_viol = 0.0  # slack满足要求，无违例

            # 加权违约
            violation = (
                self.config["violation_weights"]["a"] * area_viol +
                self.config["violation_weights"]["p"] * power_viol +
                self.config["violation_weights"]["t"] * delay_viol +
                self.config["violation_weights"]["s"] * sens_viol
            )

        # 返回数组格式 - 只返回4个目标值，违例值单独处理
        return np.array([area_new, power_new, -slack_new, sens_new, violation])

    def get_statistics(self):
        """
        获取统计信息
        """
        stats = {
            'population_size': len(self.current_population) if self.current_population else 0,
            'current_generation': self.current_generation,
            'hv_history_length': len(self.convergence_history['hv']) if hasattr(self, 'convergence_history') else 0,
        }

        # 添加Pareto前沿信息
        if self.current_population:
            pareto_front = self._extract_pareto_front(self.current_population)
            stats['pareto_size'] = len(pareto_front)

            if pareto_front:
                # 计算目标值的统计信息
                objectives = []
                for ind in pareto_front:
                    if ind.objectives is not None:
                        objectives.append(ind.objectives)

                if objectives:
                    objectives = np.array(objectives)
                    stats['objectives_mean'] = np.mean(objectives, axis=0).tolist()
                    stats['objectives_std'] = np.std(objectives, axis=0).tolist()
                    stats['objectives_min'] = np.min(objectives, axis=0).tolist()
                    stats['objectives_max'] = np.max(objectives, axis=0).tolist()

        # 添加HV信息
        if hasattr(self, 'convergence_history') and self.convergence_history['hv']:
            stats['current_hv'] = self.convergence_history['hv'][-1]
            stats['max_hv'] = max(self.convergence_history['hv'])
            stats['hv_improvement'] = self.convergence_history['hv'][-1] - self.convergence_history['hv'][0] if len(self.convergence_history['hv']) > 0 else 0

        return stats
    
    def _compute_dynamic_constraint_thresholds(self):
        """
        基于违约权重计算动态约束阈值，实现优先级控制。
        
        机制：
        - 权重越高的指标 -> 约束越严格（阈值动态收紧）
        - 权重越低的指标 -> 约束越宽松（阈值动态放宽）
        
        具体实现：
        - 权重归一化：w_i_norm = w_i / sum(weights)
        - 动态阈值：实际阈值 = 基准阈值 * (1.0 + (1.0 - w_i_norm) * adjustment_factor)
        - 高权重指标阈值收紧，低权重指标阈值放宽
        
        调整因子：通过 _weight_adjustment_factor 控制最大放宽程度
        """
        # 获取权重
        w_area = self.config["violation_weights"].get("a", 0.1)
        w_power = self.config["violation_weights"].get("p", 0.1)
        w_timing = self.config["violation_weights"].get("t", 1000.0)
        w_sens = self.config["violation_weights"].get("s", 0.5)
        
        weights = [w_area, w_power, w_timing, w_sens]
        total = sum(weights)
        if total == 0:
            # 默认：所有权重相等
            w_area_norm = w_power_norm = w_timing_norm = w_sens_norm = 0.25
        else:
            w_area_norm = w_area / total
            w_power_norm = w_power / total
            w_timing_norm = w_timing / total
            w_sens_norm = w_sens / total
        
        # 调整因子：控制低权重指标的放宽程度
        # adjustment_factor = 0.5 表示低权重指标的阈值最多放宽 50%
        self._weight_adjustment_factor = 0.5
        
        # 获取基准约束阈值
        base_area = self.config["constraints"]["delta_area_max"]  # 面积：越小越严格（≤基准值）
        base_power = self.config["constraints"]["delta_power_max"]  # 功耗：越小越严格（≤基准值）
        base_sens = self.delta_sens_min  # 敏感性：越小越严格（≤基准值）
        base_slack = self.config["constraints"]["slack_min"]  # 时序：越大越严格（≥基准值）
        
        # 动态阈值计算：
        # 高权重 -> w_norm大 -> (1 - w_norm)小 -> 阈值收紧
        # 低权重 -> w_norm小 -> (1 - w_norm)大 -> 阈值放宽
        
        self._dynamic_area_max = base_area * (1.0 + (1.0 - w_area_norm) * self._weight_adjustment_factor)
        self._dynamic_power_max = base_power * (1.0 + (1.0 - w_power_norm) * self._weight_adjustment_factor)
        self._dynamic_sens_max = base_sens * (1.0 + (1.0 - w_sens_norm) * self._weight_adjustment_factor)
        self._dynamic_slack_min = base_slack * (1.0 - (1.0 - w_timing_norm) * self._weight_adjustment_factor)
        
        # 打印调试信息
        print(f"[权重分析] 归一化权重: area={w_area_norm:.3f}, power={w_power_norm:.3f}, timing={w_timing_norm:.3f}, sens={w_sens_norm:.3f}")
        print(f"[权重分析] 基准阈值: area≤{base_area}, power≤{base_power}, slack≥{base_slack}, sens≤{base_sens}")
        print(f"[权重分析] 动态阈值: area≤{self._dynamic_area_max:.4f}, power≤{self._dynamic_power_max:.4f}, slack≥{self._dynamic_slack_min:.4f}, sens≤{self._dynamic_sens_max:.4f}")
    
    def _extract_ppa_feasible_pareto(self, population):
        """Return the strict PPA-feasible area/residual-SER Pareto front."""
        if not population:
            return []
        area_limit, power_limit, slack_min = self._ppa_limits()
        unique = {}
        candidates = []
        points = []
        for ind in population:
            gene = getattr(ind, "gene", None)
            objectives = getattr(ind, "original_objectives", None)
            if objectives is None:
                objectives = getattr(ind, "objectives", None)
            if gene is None or objectives is None:
                continue
            gene = self._normalize_gene(gene)
            key = self._gene_key(gene)
            if key in unique:
                continue
            try:
                obj = np.asarray(objectives, dtype=np.float64).flatten()
                if obj.size < 4 or not np.all(np.isfinite(obj[:4])):
                    continue
                if obj[0] > 10.0:
                    area_over = float(obj[0]) / max(float(self.area_orig), 1e-20) - 1.0
                    power_over = float(obj[1]) / max(float(self.power_orig), 1e-20) - 1.0
                else:
                    area_over = float(obj[0]) - 1.0
                    power_over = float(obj[1]) - 1.0
                wns = -float(obj[2])
                residual = getattr(ind, "raw_sensitivity", None)
                residual = float(obj[3]) if residual is None else float(np.asarray(residual).flatten()[0])
                if residual > 1.5:
                    residual /= max(float(self.sens_orig), 1e-20)
                if area_over > area_limit + 1e-12 or power_over > power_limit + 1e-12 or wns < slack_min - 1e-12:
                    continue
                unique[key] = ind
                candidates.append(ind)
                # Power and WNS are hard constraints. The external scorer and
                # training HV both optimize area overhead and residual SER.
                points.append([max(0.0, area_over), residual])
                ind.area_power_feasible = True
                ind.sensitivity_feasible = True
                ind.raw_sensitivity = residual
            except Exception:
                continue
        if not candidates:
            return []

        objectives = np.asarray(points, dtype=np.float64)
        dominated = np.zeros(len(candidates), dtype=bool)
        for i, point in enumerate(objectives):
            le = np.all(objectives <= point + 1e-14, axis=1)
            lt = np.any(objectives < point - 1e-14, axis=1)
            le[i] = False
            if np.any(le & lt):
                dominated[i] = True
        return [ind for i, ind in enumerate(candidates) if not dominated[i]]

    def _extract_pareto_front(self, population):
        """
        使用 Numpy 向量化加速的 Pareto 前沿提取（带权重优先级过滤和去重）。
        
        固定阈值 + 权重优先级策略：
        - 阈值固定为 yaml 配置值，不动态调整
        - 优化优先级按权重分配：高权重指标的可行解优先保留
        - 自动去除重复的解
        
        优先级顺序（从高到低）：
        1. 敏感性 (sens_weight: 1000)
        2. 时序 (slack_weight: 50)
        3. 面积/功耗 (area/power_weight: 20)
        """
        if not population:
            return []
        if getattr(self, "ppa_pareto_only", True):
            return self._extract_ppa_feasible_pareto(population)
        
        # 0. 【去重处理】按基因分组，只保留唯一解
        unique_genes = {}  # gene_str -> ind (保留第一个遇到的)
        for ind in population:
            if ind is None or ind.gene is None:
                continue
            genes_arr = np.array(ind.gene).flatten()
            gene_str = ''.join(str(int(g)) for g in genes_arr)
            if gene_str not in unique_genes:
                unique_genes[gene_str] = ind
        
        population = list(unique_genes.values())
        if len(unique_genes) < len(population):
            print(f"[去重] _extract_pareto_front: 去重后 {len(population)} 个唯一解")
        
        # 1. 过滤无效个体
        valid_pop = [ind for ind in population if ind.objectives is not None]
        if not valid_pop:
            return []
        
        # 2. 获取固定阈值（使用yaml配置值）
        delta_area_max = self.config["constraints"]["delta_area_max"]  # 0.2
        delta_power_max = self.config["constraints"]["delta_power_max"]  # 0.5
        slack_min = self.config["constraints"]["slack_min"]  # 0.0
        delta_sens_min = getattr(self, 'delta_sens_min', 0.4)  # 0.4
        
        # 3. 获取权重（用于计算优先级得分）
        w_area = self.config["violation_weights"].get("a", 0.1)
        w_power = self.config["violation_weights"].get("p", 0.1)
        w_timing = self.config["violation_weights"].get("t", 1000.0)
        w_sens = self.config["violation_weights"].get("s", 0.5)
        
        # 4. 计算每个解的约束违例程度（用于优先级排序）
        def _get_violation_info(ind: Individual) -> dict:
            """计算解的约束违例信息"""
            # 提取目标值（优先使用原始未惩罚的目标值）
            obj = getattr(ind, 'original_objectives', None)
            if obj is None:
                obj = ind.objectives
            if obj is None or len(obj) < 4:
                return {'total_viol': float('inf'), 'sens_viol': 1.0, 'timing_viol': 1.0, 
                        'area_viol': 1.0, 'power_viol': 1.0}
            
            obj = np.asarray(obj).flatten()
            obj0 = float(obj[0])
            obj1 = float(obj[1])
            obj2 = float(obj[2])
            obj3 = float(obj[3])
            
            # 获取基准值用于比率转换
            base_area = getattr(self, 'area_orig', 1.0)
            base_power = getattr(self, 'power_orig', 1.0)
            
            # 【关键修复】：检测 objectives 是否为绝对值（area > 10 说明是绝对值 um²）
            # real_moea_wrapper.evaluate_population 返回绝对值；_calculate_objectives_and_violation 返回比率
            if obj0 > 10.0:
                # 绝对值 → 转换为比率
                obj0_ratio = obj0 / (base_area + 1e-10)
                obj1_ratio = obj1 / (base_power + 1e-10)
            else:
                # 已经是比率
                obj0_ratio = obj0
                obj1_ratio = obj1
            
            # 面积违例（基于比率）
            area_viol = max(0.0, obj0_ratio - 1.0 - delta_area_max)
            
            # 功耗违例（基于比率）
            power_viol = max(0.0, obj1_ratio - 1.0 - delta_power_max)
            
            # 敏感性违例（确保使用标量）
            raw_sens = getattr(ind, 'raw_sensitivity', None)
            if raw_sens is None:
                if obj3 > 10:
                    raw_sens = obj3 / 100.0
                else:
                    raw_sens = obj3
            else:
                # 确保是标量
                raw_sens = float(np.asarray(raw_sens).flatten()[0]) if hasattr(raw_sens, '__len__') else float(raw_sens)
            
            sens_viol = max(0.0, raw_sens - delta_sens_min)
            
            # 时序违例（obj[2]是-wns，obj[2] >= 1e5 表示违例）
            timing_viol = 1.0 if obj2 >= 1e5 else 0.0
            
            # 加权总违例
            total_viol = (w_area * area_viol + w_power * power_viol + 
                         w_timing * timing_viol + w_sens * sens_viol)
            
            return {
                'total_viol': float(total_viol),
                'sens_viol': float(sens_viol),
                'timing_viol': float(timing_viol),
                'area_viol': float(area_viol),
                'power_viol': float(power_viol)
            }
        
        # 4. 动态计算优先级顺序（按权重降序）
        priority_order = [
            ('sens', '敏感性', w_sens),
            ('timing', '时序', w_timing),
            ('area', '面积', w_area),
            ('power', '功耗', w_power),
        ]
        # 按权重降序排序
        priority_order.sort(key=lambda x: x[2], reverse=True)
        
        print(f"[Pareto优先级] 动态权重顺序: " + 
              ", ".join([f"{name}({w:.1f})" for _, name, w in priority_order]))
        
        # 6. 动态分层过滤：按权重优先级逐层筛选
        current_set = valid_pop  # 从所有有效解开始
        
        for priority_key, priority_name, priority_weight in priority_order:
            feasible = []
            infeasible = []
            viol_key = f'{priority_key}_viol'
            
            for ind in current_set:
                viol_info = _get_violation_info(ind)
                viol_val = float(viol_info[viol_key])
                if viol_val <= 1e-3:  # 使用容差，允许极小的违例值通过
                    feasible.append((ind, viol_info))
                else:
                    infeasible.append((ind, viol_info))
            
            # 如果可行解足够，继续细化；否则使用当前可行解
            if len(feasible) >= self.pop_size * 0.3:
                current_set = [x[0] for x in feasible]
            else:
                # 可行解不足，包含一些违例解
                if feasible:
                    feasible.sort(key=lambda x: float(x[1][viol_key]))
                    current_set = feasible + infeasible[:max(0, self.pop_size - len(feasible))]
                else:
                    current_set = infeasible
                break
        
        # 7. 提取最终工作集
        working_pop = current_set
        
        # 8. 统计各级别的可行率（确保使用标量比较）
        all_viol_info = {ind: _get_violation_info(ind) for ind in valid_pop}
        sens_feas_count = sum(1 for ind in valid_pop if float(all_viol_info[ind]['sens_viol']) <= 0)
        timing_feas_count = sum(1 for ind in valid_pop if float(all_viol_info[ind]['timing_viol']) <= 0)
        area_feas_count = sum(1 for ind in valid_pop if float(all_viol_info[ind]['area_viol']) <= 0)
        power_feas_count = sum(1 for ind in valid_pop if float(all_viol_info[ind]['power_viol']) <= 0)
        
        print(f"[Pareto统计] 敏感性可行: {sens_feas_count}/{len(valid_pop)}, "
              f"时序可行: {timing_feas_count}/{len(valid_pop)}, "
              f"面积可行: {area_feas_count}/{len(valid_pop)}, "
              f"功耗可行: {power_feas_count}/{len(valid_pop)}")
        
        # 9. 从工作集中提取 Pareto 前沿
        if not working_pop:
            return []
        
        # 确保 working_pop 是 Individual 对象列表
        if isinstance(working_pop[0], tuple):
            feasible_inds = [x[0] for x in working_pop]
        else:
            feasible_inds = working_pop
        
        # 提取 objectives，确保是 2D 数组
        obj_list = []
        for ind in feasible_inds:
            obj = np.asarray(ind.objectives).flatten()
            if len(obj) >= 4:
                obj_list.append(obj[:4])
            else:
                obj_list.append(np.array([1e6, 1e6, -1e6, 1.0]))
        objectives = np.array(obj_list)
        
        N = len(feasible_inds)
        is_dominated = np.zeros(N, dtype=bool)
        
        # 向量化支配判断
        for i in range(N):
            obj_i = objectives[i]
            less_equal = np.all(objectives <= obj_i, axis=1)
            strict_less = np.any(objectives < obj_i, axis=1)
            if np.any(less_equal & strict_less):
                is_dominated[i] = True
        
        # 返回未被支配的精英解
        return [feasible_inds[i] for i in range(N) if not is_dominated[i]]
    
    def _assign_crowding_distance(self, front):
        """向量化批量计算拥挤距离，一次性赋给整个前沿"""
        n_individuals = len(front)
        if n_individuals == 0:
            return
        if n_individuals <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
            
        objectives = np.array([ind.objectives for ind in front])
        n_objs = objectives.shape[1]
        distances = np.zeros(n_individuals)
        
        for m in range(n_objs):
            sorted_idx = np.argsort(objectives[:, m])
            obj_values = objectives[sorted_idx, m]
            
            # 边界赋予无穷大
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            
            obj_range = obj_values[-1] - obj_values[0]
            if obj_range == 0:
                continue
                
            # 向量化计算中间点的距离 (利用 numpy 切片平移)
            distances[sorted_idx[1:-1]] += (obj_values[2:] - obj_values[:-2]) / obj_range
            
        # 一次性写回个体属性
        for i, ind in enumerate(front):
            ind.crowding_distance = distances[i] / n_objs
    
    @staticmethod
    def _get_elite_entry_gene(entry):
        """统一从精英条目提取基因（兼容旧格式数组和新格式 dict）。"""
        if isinstance(entry, dict):
            return entry.get('gene')
        if isinstance(entry, Individual):
            return entry.gene
        return entry if isinstance(entry, np.ndarray) else None

    @staticmethod
    def _get_elite_entry_objectives(entry):
        """统一从精英条目提取 objectives（兼容旧格式数组和新格式 dict）。"""
        if isinstance(entry, dict):
            return entry.get('objectives')
        if isinstance(entry, Individual):
            return entry.objectives
        return None

    def _calculate_elite_diversity(self, elite_pool: list) -> float:
        """
        计算精英池的多样性分数
        
        基于精英解的敏感性分布计算多样性：
        - 如果敏感性分布范围大（从低到高都有），多样性高
        - 如果敏感性都接近，多样性低
        
        返回: 0.0~1.0 的多样性分数
        """
        if not elite_pool or len(elite_pool) < 2:
            return 0.5  # 默认中等多样性
        
        sensitivities = []
        for g in elite_pool:
            obj = self._get_elite_entry_objectives(g)
            if obj is not None:
                obj = np.asarray(obj).flatten()
                if len(obj) > 0:
                    sensitivities.append(obj[0])
        
        if len(sensitivities) < 2:
            return 0.5
        
        # 计算敏感性分布的归一化范围
        sens_min, sens_max = min(sensitivities), max(sensitivities)
        sens_range = sens_max - sens_min
        
        # 归一化到 [0, 1]
        # 范围越大，多样性越高
        # 假设最大可能范围是 1.0（从0到1）
        diversity = min(sens_range / 1.0, 1.0)
        
        return diversity

    def _create_heuristic_gene_low_sensitivity(self, count):
        """策略1：创建低敏感性基因（选择敏感度低的寄存器）"""
        genes = []
        sensitivity = getattr(self, 'error_rates', None)
        if sensitivity is None:
            return genes
        
        sorted_indices = np.argsort(sensitivity)  # 从低到高排序
        
        for i in range(min(count, len(sorted_indices))):
            gene = np.zeros(self.K, dtype=int)
            # 选择敏感性最低的 i+1 个位置（基因=1表示选择该寄存器）
            selected = sorted_indices[:i+1]
            gene[selected] = 1
            genes.append(gene)
        
        return genes
    
    def _create_heuristic_gene_high_slack(self, count):
        """策略2：创建高松弛量基因（选择松弛量大的寄存器）"""
        genes = []
        slack = getattr(self, 'slack', None)
        if slack is None:
            return genes
        
        sorted_indices = np.argsort(slack)[::-1]  # 从高到低排序
        
        for i in range(min(count, len(sorted_indices))):
            gene = np.zeros(self.K, dtype=int)
            # 选择松弛量最高的 i+1 个位置
            selected = sorted_indices[:i+1]
            gene[selected] = 1
            genes.append(gene)
        
        return genes
    
    def _create_heuristic_gene_balanced(self, count):
        """策略3：创建平衡基因（同时考虑敏感性和松弛量）"""
        genes = []
        sensitivity = getattr(self, 'error_rates', None)
        slack = getattr(self, 'slack', None)
        
        if sensitivity is None or slack is None:
            return genes
        
        # 计算综合分数：低敏感性 + 高松弛 = 好寄存器
        # 归一化
        sens_norm = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min() + 1e-10)
        slack_norm = (slack - slack.min()) / (slack.max() - slack.min() + 1e-10)
        
        # 综合分数（越低越好）
        score = sens_norm - slack_norm  # 低敏感 + 高松弛 = 低分 = 好
        sorted_indices = np.argsort(score)
        
        for i in range(min(count, len(sorted_indices))):
            gene = np.zeros(self.K, dtype=int)
            # 选择综合分数最低的 i+1 个位置
            selected = sorted_indices[:i+1]
            gene[selected] = 1
            genes.append(gene)
        
        return genes
    
    def _calculate_current_hv(self):
        """
        计算当前种群的超体积(HV)
        """
        if self.current_population is None or len(self.current_population) == 0:
            return 0.0
        
        # 提取有效个体
        valid_population = [ind for ind in self.current_population 
                            if ind.objectives is not None and 
                            not np.any(np.isnan(ind.objectives)) and
                            not np.any(np.isinf(ind.objectives))]
        
        if len(valid_population) == 0:
            return 0.0
        
        # 提取目标值
        objectives = []
        for ind in valid_population:
            # 确保目标值是数组
            if isinstance(ind.objectives, np.ndarray) and len(ind.objectives) >= 4:
                # 检查是否包含NaN或Inf
                if not np.any(np.isnan(ind.objectives)) and not np.any(np.isinf(ind.objectives)):
                    objectives.append(ind.objectives)
        
        if len(objectives) == 0:
            return 0.0
        
        objectives = np.array(objectives)

        # 归一化目标值
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)

        # 避免除零
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0

        normalized_objs = (objectives - min_vals) / ranges
        
        # 设置参考点（比最大值大20%）
        reference_point = np.ones(len(min_vals)) * 1.2      ## 使用归一化的最大值1
        ## max_vals = np.max(objectives, axis=0) * 1.2
        
        # 简化HV计算
        sorted_idx = np.argsort(normalized_objs[:, 0])
        sorted_objs = normalized_objs[sorted_idx]
        
        # 计算HV
        hv = 0.0
        prev_obj = None
        
        for i, obj in enumerate(sorted_objs):
            if i == 0:
                # 第一个解
                volume = 1.0
                for d in range(len(obj)):
                    volume *= (reference_point[d] - obj[d])
                hv += volume
            else:
                # 计算与前一个解的差异
                volume = 1.0
                for d in range(len(obj)):
                    volume *= (reference_point[d] - obj[d])
                
                # 减去重叠部分（近似）
                prev_volume = 1.0
                for d in range(len(prev_obj)):
                    prev_volume *= (reference_point[d] - prev_obj[d])
                
                hv += volume - prev_volume * 0.5  # 近似减去重叠
            
            prev_obj = obj
        
        # 确保HV非负
        hv = max(0.0, hv)
        
        # 打印调试信息
        # print(f"DEBUG[HV]: 计算HV={hv:.6f}, 种群大小={len(valid_population)}, 归一化目标形状={normalized_objs.shape}")
        
        return hv
    
    def _calculate_scoring_hv_2d(self, population):
        """Exact fixed-reference HV used by the external area/residual scorer."""
        front = self._extract_ppa_feasible_pareto(population)
        if not front:
            return 0.0
        area_limit, _, _ = self._ppa_limits()
        points = []
        for ind in front:
            try:
                obj = getattr(ind, "original_objectives", None)
                if obj is None:
                    obj = ind.objectives
                obj = np.asarray(obj, dtype=np.float64).flatten()
                if obj[0] > 10.0:
                    area_over = float(obj[0]) / max(float(self.area_orig), 1e-20) - 1.0
                else:
                    area_over = float(obj[0]) - 1.0
                residual = getattr(ind, "raw_sensitivity", None)
                residual = float(obj[3]) if residual is None else float(np.asarray(residual).flatten()[0])
                if residual > 1.5:
                    residual /= max(float(self.sens_orig), 1e-20)
                x = float(np.clip(area_over / max(area_limit, 1e-20), 0.0, 1.0))
                y = float(np.clip(residual, 0.0, 1.0))
                points.append((x, y))
            except Exception:
                continue
        if not points:
            return 0.0
        points.sort(key=lambda p: (p[0], p[1]))
        hv = 0.0
        best_y = 1.0
        for x, y in points:
            if y < best_y:
                hv += max(1.0 - x, 0.0) * (best_y - y)
                best_y = y
        return float(np.clip(hv, 0.0, 1.0))

    def get_feasible_archive_hv(self, extra_population=None):
        population = list(extra_population or [])
        if hasattr(self, "cumulative_elite_pool"):
            for entry in self.cumulative_elite_pool or []:
                gene = self._get_elite_entry_gene(entry)
                if gene is None:
                    continue
                try:
                    population.append(self._make_individual_from_gene(gene))
                except Exception:
                    continue
        return self._calculate_scoring_hv_2d(population)

    def _calculate_hypervolume(self, pareto_front):
        """
        计算超体积，使用更准确的算法
        确保返回标量
        """
        if getattr(self, "scoring_aligned_hv", True):
            return self._calculate_scoring_hv_2d(pareto_front)
        if isinstance(pareto_front, list):
            if not pareto_front:
                return 0.0
            # 提取目标值
            objectives = []
            for ind in pareto_front:
                if ind.objectives is not None:
                    obj = np.asarray(ind.objectives).flatten()
                    if len(obj) >= 4 and not np.any(np.isnan(obj)):
                        objectives.append(obj[:4])

            if not objectives:
                return 0.0

            objectives = np.array(objectives)
        elif isinstance(pareto_front, np.ndarray):
            if pareto_front.size == 0:
                return 0.0
            objectives = pareto_front
        else:
            return 0.0
        
        # 确保 objectives 是 2D 数组
        if objectives.ndim == 1:
            objectives = objectives.reshape(1, -1)

        # 归一化目标值,归一化到[0,1]范围
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0  # 避免除零

        normalized = (objectives - min_vals) / ranges

        # 设置参考点
        reference_point = np.ones(objectives.shape[1]) * 1.2

        # 使用改进的HV计算方法
        n_objs = objectives.shape[1]
        if n_objs == 4:
            # 4维问题使用改进的算法
            return self._calculate_hypervolume_4d_improved(normalized, reference_point)
        else:
            # 其他维度使用通用方法
            return self._calculate_hypervolume_general_improved(normalized, reference_point)
    
    def _calculate_hypervolume_4d_improved(self, objectives, ref_point):
        """4维HV的改进计算方法"""
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
            
            # 减去与前面解的重叠部分（更精确的计算）
            if i > 0:
                # 计算重叠体积
                overlap_volume = self._calculate_overlap_volume_4d_improved(
                    sorted_objs[:i], obj, ref_point
                )
                volume = max(0.0, volume - overlap_volume)
            
            hv += volume
        
        return float(hv)
    
    def _calculate_overlap_volume_4d_improved(self, prev_objs, current_obj, ref_point):
        """计算4维重叠体积（改进版）"""
        if len(prev_objs) == 0:
            return 0.0
        
        # 找到支配current_obj的所有前面解
        dominated_by = []
        for prev_obj in prev_objs:
            if np.all(prev_obj <= current_obj) and np.any(prev_obj < current_obj):
                dominated_by.append(prev_obj)
        
        if not dominated_by:
            return 0.0
        
        # 计算重叠体积：使用更精确的方法
        # 对于4维，计算被支配区域的体积
        max_overlap = 0.0
        for dom_obj in dominated_by:
            # 计算被支配区域的体积
            overlap = 1.0
            for d in range(len(current_obj)):
                # 重叠区域的上界是current_obj，下界是dom_obj
                upper_bound = min(current_obj[d], ref_point[d])
                lower_bound = max(dom_obj[d], 0.0)
                if upper_bound > lower_bound:
                    overlap *= (upper_bound - lower_bound)
                else:
                    overlap = 0.0
                    break
            max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def _calculate_hypervolume_general_improved(self, objectives, ref_point):
        """通用HV计算方法（改进版）"""
        if len(objectives) == 0:
            return 0.0
        
        # 按第一个目标排序
        sorted_idx = np.argsort(objectives[:, 0])
        sorted_objs = objectives[sorted_idx]
        
        hv = 0.0
        
        for i, obj in enumerate(sorted_objs):
            volume = 1.0
            for d in range(len(obj)):
                if obj[d] < ref_point[d]:
                    volume *= (ref_point[d] - obj[d])
            
            if i == 0:
                hv += volume
            else:
                # 改进的重叠计算
                prev_obj = sorted_objs[i-1]
                overlap = 1.0
                for d in range(len(obj)):
                    upper = min(obj[d], ref_point[d])
                    lower = max(prev_obj[d], 0.0)
                    if upper > lower:
                        overlap *= (upper - lower)
                    else:
                        overlap = 0.0
                        break
                hv += max(0.0, volume - overlap * 0.7)  # 改进的重叠系数
        
        return float(hv)
    
    def run_with_convergence(self, eval_fn, max_generations=50, convergence_threshold=0.02, patience=6, 
                         generations_per_batch: int = 5, reset: bool = True, ignore_early_convergence: bool = False,
                         min_generations_before_check: int = 5):
        """
        运行MOEA直到收敛或达到最大代数

        参数:
        ----------
        eval_fn : callable
            评估函数
        max_generations : int
            最大进化代数
        convergence_threshold : float
            收敛阈值（HV相对改进率），默认改为0.02
        patience : int
            连续多少代没有显著改进则收敛，默认改为6
        generations_per_batch : int
            每批运行的代数（用于显示进度和分批检查）
        reset : bool
            是否在运行前重置MOEA状态
        ignore_early_convergence : bool
            是否忽略MOEA内部收敛判定，强制跑完预算代数
        min_generations_before_check : int
            至少运行多少代后才开始检查收敛

        返回:
        ----------
        tuple: (最终种群, 历史前沿, 统计信息, 收敛信息)
        """
        print(f"MOEA开始运行，最大代数={max_generations}, K={self.K}")

        # 确保convergence_threshold是浮点数
        try:
            convergence_threshold = float(convergence_threshold)
        except (ValueError, TypeError):
            convergence_threshold = 1e-4
            print(f"WARNING: 无法将convergence_threshold转换为浮点数，使用默认值{convergence_threshold}")
    
        # 如果eval_fn为None，使用默认评估函数
        if eval_fn is None:
            def eval_fn(genes):
                return self.model.evaluate_population(genes)

        # 如果明确要求重置，则先重置种群与收敛状态
        if reset:
            print(f"[MOEA]: 重置MOEA")
            self.reset()

        # 重置当前步的局部收敛标记，避免上一步的判定永久卡死后续SAC步
        self.converged = False
        self.last_pre_evolution_population = []
        self.last_evolution_population = []

        # 1. 如果种群为空或需要初始化，初始化种群
        if self.current_population is None or len(self.current_population) == 0:
            print(f"[MOEA]: 初始化种群，大小={self.pop_size}")
            genes_list = []
            for i in range(self.pop_size):
                gene = np.random.randint(0, 2, self.K)
                genes_list.append(gene)
            
            # 批量评估
            if genes_list:
                genes_array = np.array(genes_list)
                objectives = eval_fn(genes_array)
                print(f"[MOEA]: 初始评估完成，目标值形状={objectives.shape}")
                
                self.current_population = []
                for i, gene in enumerate(genes_list):
                    ind = Individual(gene, None)
                    if i < len(objectives):
                        ind.objectives = objectives[i]
                    else:
                        ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                    ind.violation = 0.0
                    self.current_population.append(ind)

        self._step_spea2_updates = 0
        self._step_nsga2_updates = 0
        # 当前种群是本次SAC动作执行前的基线。后续由动作控制的重启、
        # 精英注入、局部精修和异构进化均计入稠密动作归因奖励。
        self.last_pre_evolution_population = self._snapshot_population_for_credit(
            self.current_population
        )
        self._apply_controlled_restart()
        elite_count = int(round(
            self.pop_size * float(self.search_controls.get("elite_injection_ratio", 0.0))
        ))
        self._install_budget_aware_elites(
            max_new=elite_count,
            context="pre-run",
            record_archive=False,
        )

        # 检查种群是否有效
        valid_pop = [ind for ind in self.current_population if ind.objectives is not None]
        if len(valid_pop) == 0:
            print("ERROR: 种群中没有有效个体，无法运行MOEA")
            return [], [], {}, {'is_converged': False, 'final_generation': 0, 'final_hv': 0.0}
        
        # 2. 计算初始HV（用于收敛检测）
        pareto_front = self._extract_pareto_front(self.current_population)
        initial_hv = self._calculate_hypervolume(pareto_front)
        print(f"[MOEA]: 初始HV: {initial_hv:.6f}")
        
        # 诊断：打印收敛历史运行前快照
        try:
            hv_history_before = list(self.convergence_history.get('hv', []))
        except Exception:
            hv_history_before = []
        print(f"[MOEA-SNAP BEFORE] hv_history_len={len(hv_history_before)}, hv_history_tail={hv_history_before[-5:] if hv_history_before else []}")

        # 3. 更新收敛历史
        if 'hv' not in self.convergence_history:
            self.convergence_history['hv'] = []
        self.convergence_history['hv'].append(initial_hv)
        print(f"[MOEA-SNAP AFTER-APPEND] new_hv_len={len(self.convergence_history['hv'])}, hv_tail={self.convergence_history['hv'][-5:]}")

        # 4. 主循环
        generation = 0
        # 每个SAC动作必须获得独立的进化响应窗口。若沿用上一步的无改进计数，
        # 后续动作会在一代后立即收敛，无法形成可辨识的参数-性能样本。
        self.no_improvement_count = 0
        no_improvement_count = 0
        is_converged = False
        
        # 每5代清理一次内存
        if generation % 5 == 0:
            self._cleanup_memory()

        print(f"[MOEA]: 开始进化，目标代数={max_generations}")
        
        while generation < max_generations and not is_converged:
            try:
                current_gen = self.current_generation + 1
                print(f"[MOEA]: 开始第 {current_gen} 代")
                
                # 4.1 确保当前种群已评估
                for i, ind in enumerate(self.current_population):
                    if ind.objectives is None:
                        try:
                            results = eval_fn([ind.gene])
                            if results is not None and len(results) > 0:
                                ind.objectives = results[0]
                            else:
                                ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                        except Exception as e:
                            print(f"[MOEA]: 个体{i}评估失败: {e}")
                            ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                        ind.violation = 0.0
                
                # 4.2 并行运行SPEA2和NSGA2一代（使用线程 + 异步机制优化）
                from concurrent.futures import ThreadPoolExecutor
                import threading

                # 使用线程运行（避免进程序列化开销），但通过异步机制减少阻塞
                # 关键洞察：GIL 只在 Python 代码执行时有效，GPU 评估期间 GIL 被释放
                current_pop = self.current_population
                spea2_active = self._algorithm_active_for_generation("spea2", generation, max_generations)
                nsga2_active = self._algorithm_active_for_generation("nsga2", generation, max_generations)
                print(
                    f"[MOEA-CONTROL]: gen={generation + 1}/{max_generations}, "
                    f"SPEA2={'on' if spea2_active else 'off'}, "
                    f"NSGA2={'on' if nsga2_active else 'off'}"
                )

                # 确保两个算法有独立的内部种群（独立演化）
                try:
                    if not getattr(self.spea2, 'population', None):
                        # 使用当前种群作为种子进行初始化（或随机初始化）
                        self.spea2.population = self._create_initial_population_for_algorithm('spea2', current_pop)
                    if not getattr(self.nsga2, 'population', None):
                        # 用不同随机扰动的种子初始化NSGA2，以保证多样性与独立性
                        self.nsga2.population = self._create_initial_population_for_algorithm('nsga2', current_pop)
                except Exception as e:
                    print(f"[MOEA-WARN]: 初始化子算法种群失败: {e}")

                def run_spea2_thread():
                    """运行SPEA2一代（线程模式），使用SPEA2自己的内部种群"""
                    import time
                    thread_start = time.time()
                    if not spea2_active:
                        return list(getattr(self.spea2, 'population', []) or []), None, 0.0
                    try:
                        # 在线程中运行 SPEA2（GPU 评估期间 GIL 被释放，实现真正并行）
                        result = self.spea2.run_one_generation(self.spea2.population, current_gen-1)
                        # 将返回的新种群写回对象以便下一代继续使用
                        try:
                            self.spea2.population = result
                        except Exception:
                            pass
                        elapsed = time.time() - thread_start
                        print(f"[MOEA-PARALLEL]: SPEA2线程完成，耗时={elapsed:.2f}秒")
                        return result, None, elapsed
                    except Exception as e:
                        elapsed = time.time() - thread_start
                        print(f"DEBUG[MOEA-PARALLEL]: SPEA2线程失败: {e}，耗时={elapsed:.2f}秒")
                        import traceback
                        traceback.print_exc()
                        # 回退：返回当前内部种群
                        return list(getattr(self.spea2, 'population', [])), str(e), elapsed

                def run_nsga2_thread():
                    """运行NSGA2一代（线程模式），使用NSGA2自己的内部种群"""
                    import time
                    thread_start = time.time()
                    if not nsga2_active:
                        return list(getattr(self.nsga2, 'population', []) or []), None, 0.0
                    try:
                        result = self.nsga2.run_one_generation(self.nsga2.population, current_gen-1)
                        try:
                            self.nsga2.population = result
                        except Exception:
                            pass
                        elapsed = time.time() - thread_start
                        print(f"[MOEA-PARALLEL]: NSGA2线程完成，耗时={elapsed:.2f}秒")
                        return result, None, elapsed
                    except Exception as e:
                        elapsed = time.time() - thread_start
                        print(f"DEBUG[MOEA-PARALLEL]: NSGA2线程失败: {e}，耗时={elapsed:.2f}秒")
                        import traceback
                        traceback.print_exc()
                        return list(getattr(self.nsga2, 'population', [])), str(e), elapsed
                
                # 初始化结果变量
                spea2_pop = None
                nsga2_pop = None
                spea2_exec_time = 0.0
                nsga2_exec_time = 0.0
                
                # GPU使用监控
                gpu_stats_before = None
                gpu_stats_after = None
                try:
                    import torch
                    if torch.cuda.is_available() and self.use_gpu:
                        gpu_stats_before = {
                            'allocated': torch.cuda.memory_allocated(0) / 1024**3,
                            'reserved': torch.cuda.memory_reserved(0) / 1024**3
                        }
                except:
                    pass
                
                try:
                    import time
                    start_time = time.time()
                    
                    # 使用 ThreadPoolExecutor 启动两个线程
                    # 关键：GPU 评估期间 GIL 被释放，所以线程能真正并行执行 GPU 操作
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        # 提交两个任务到线程池
                        spea2_future = executor.submit(run_spea2_thread)
                        nsga2_future = executor.submit(run_nsga2_thread)
                        
                        # 并行等待（两个线程同时运行）
                        # GPU 操作不受 GIL 限制，所以这里能真正实现并行
                        spea2_result, spea2_error, spea2_exec_time = spea2_future.result(timeout=300)
                        nsga2_result, nsga2_error, nsga2_exec_time = nsga2_future.result(timeout=300)
                        self._step_spea2_updates += int(spea2_active and spea2_error is None)
                        self._step_nsga2_updates += int(nsga2_active and nsga2_error is None)
                        
                        print(f"[MOEA-PARALLEL]: SPEA2线程耗时={spea2_exec_time:.2f}s, NSGA2线程耗时={nsga2_exec_time:.2f}s")
                        
                        if spea2_error is None:
                            spea2_pop = spea2_result
                        else:
                            print(f"DEBUG[MOEA-PARALLEL]: SPEA2错误: {spea2_error}")
                            spea2_pop = list(self.current_population) if self.current_population else []
                        
                        if nsga2_error is None:
                            nsga2_pop = nsga2_result
                        else:
                            print(f"DEBUG[MOEA-PARALLEL]: NSGA2错误: {nsga2_error}")
                            nsga2_pop = list(self.current_population) if self.current_population else []
                    
                    elapsed_time = time.time() - start_time
                    print(f"[MOEA]: 并行执行完成 - SPEA2: {len(spea2_pop) if spea2_pop else 0}, "
                          f"NSGA2: {len(nsga2_pop) if nsga2_pop else 0}, 耗时: {elapsed_time:.2f}秒")
                    
                    # GPU使用监控
                    try:
                        import torch
                        if torch.cuda.is_available() and self.use_gpu:
                            gpu_stats_after = {
                                'allocated': torch.cuda.memory_allocated(0) / 1024**3,
                                'reserved': torch.cuda.memory_reserved(0) / 1024**3
                            }
                            if gpu_stats_before:
                                allocated_diff = gpu_stats_after['allocated'] - gpu_stats_before['allocated']
                                # print(f"  GPU内存变化: {allocated_diff:+.3f}GB (当前: {gpu_stats_after['allocated']:.3f}GB)")
                    except:
                        pass
                        
                except Exception as e:
                    print(f"[MOEA]: 线程并行执行失败，回退到串行: {e}")
                    import traceback
                    traceback.print_exc()
                    # 回退到串行执行
                    try:
                        spea2_pop = (
                            self.spea2.run_one_generation(self.current_population, current_gen-1)
                            if spea2_active else list(getattr(self.spea2, 'population', []) or self.current_population)
                        )
                        self._step_spea2_updates += int(spea2_active)
                    except Exception as e2:
                        print(f"[MOEA]: SPEA2 串行执行失败: {e2}")
                        spea2_pop = list(self.current_population) if self.current_population else []
                    try:
                        nsga2_pop = (
                            self.nsga2.run_one_generation(self.current_population, current_gen-1)
                            if nsga2_active else list(getattr(self.nsga2, 'population', []) or self.current_population)
                        )
                        self._step_nsga2_updates += int(nsga2_active)
                    except Exception as e2:
                        print(f"[MOEA]: NSGA2 串行执行失败: {e2}")
                        nsga2_pop = list(self.current_population) if self.current_population else []
                
                # 4.4 优化合并种群：延迟合并，直接处理两个种群
                # 避免创建中间列表，直接在环境选择中处理
                if spea2_pop is None:
                    spea2_pop = list(current_pop) if current_pop else []
                if nsga2_pop is None:
                    nsga2_pop = list(current_pop) if current_pop else []
                
                print(f"[MOEA]: SPEA2种群大小={len(spea2_pop)}, NSGA2种群大小={len(nsga2_pop)}")
                
                # 4.5 环境选择，直接处理两个种群，避免中间合并
                old_size = len(self.current_population)
                if spea2_pop and nsga2_pop:
                    # 直接从两个种群中选择，避免创建大的合并列表
                    self.current_population = self._environmental_selection_optimized(
                        spea2_pop, nsga2_pop
                    )
                elif spea2_pop:
                    self.current_population = self._environmental_selection(spea2_pop)
                elif nsga2_pop:
                    self.current_population = self._environmental_selection(nsga2_pop)
                else:
                    self.current_population = list(current_pop) if current_pop else []

                # 保存最近一次环境选择快照作为异常回退；动作窗口完成后
                # 会用最终精修后的种群覆盖，完整反映当前动作控制的搜索收益。
                self.last_evolution_population = self._snapshot_population_for_credit(
                    self.current_population
                )

                if (
                    (elite_count > 0 or self.memetic_refine_count > 0)
                    and (generation == 0 or generation % 3 == 2)
                ):
                    self._install_budget_aware_elites(
                        max_new=max(1, min(elite_count, max(6, min(24, self.pop_size // 16)))),
                        context=f"generation-{current_gen}",
                        record_archive=False,
                    )
                
                print(f"[MOEA]: 环境选择: {old_size} -> {len(self.current_population)}")
                
                # 清理临时变量，释放内存
                if 'spea2_pop' in locals():
                    del spea2_pop
                if 'nsga2_pop' in locals():
                    del nsga2_pop
                import gc
                gc.collect()
                
                # 4.6 更新代数计数器
                self.current_generation = current_gen
                generation += 1
                
                # 4.7 计算当前HV
                current_pareto = self._extract_pareto_front(self.current_population)
                current_hv = self._calculate_hypervolume(current_pareto)
                
                # 确保current_hv是标量
                if isinstance(current_hv, np.ndarray):
                    if current_hv.size == 1:
                        current_hv = float(current_hv.item())
                    else:
                        current_hv = float(np.mean(current_hv))
                
                self.convergence_history['hv'].append(current_hv)
                
                # 4.8 检查改进
                if len(self.convergence_history['hv']) >= 2:
                    prev_hv = self.convergence_history['hv'][-2]
                    
                    # 确保prev_hv是标量
                    if isinstance(prev_hv, np.ndarray):
                        if prev_hv.size == 1:
                            prev_hv = float(prev_hv.item())
                        else:
                            prev_hv = float(np.mean(prev_hv))
                    
                    if prev_hv > 0:
                        improvement = (current_hv - prev_hv) / prev_hv
                        
                        # 确保improvement是标量
                        if isinstance(improvement, np.ndarray):
                            if improvement.size == 1:
                                improvement = float(improvement.item())
                            else:
                                improvement = float(np.mean(improvement))
                            
                        print(f"[MOEA-CONV]: 第{current_gen}代，HV={current_hv:.6f}, "
                              f"改进率={improvement:.6f}, 阈值={convergence_threshold}")
                        
                        if 'improvement' not in self.convergence_history:
                            self.convergence_history['improvement'] = []
                        self.convergence_history['improvement'].append(improvement)
    
                        print(f"[MOEA]: 第{current_gen}代，HV={current_hv:.6f}, 改进率={improvement:.6f}")
    
                        # 使用正确的比较：确保improvement是标量
                        if improvement > convergence_threshold:
                            no_improvement_count = 0
                            self.no_improvement_count = 0  # 同步到全局
                            print(f"[MOEA-CONV]: 有显著改进({improvement:.6f} > {convergence_threshold})，重置计数")
                        else:
                            no_improvement_count += 1
                            self.no_improvement_count = no_improvement_count  # 同步到全局
                            if improvement < 0:
                                print(f"[MOEA-CONV]: HV退化 {improvement:.6f}, 无改进计数 {no_improvement_count}/{patience}")
                            else:
                                print(f"[MOEA-CONV]: 改进不足 {improvement:.6f}, 无改进计数 {no_improvement_count}/{patience}")
                        
                        # 【灾难性变异】当连续无改进超过阈值时，触发大变异以增加探索
                        # 注意：在 fixed_budget 模式或 ignore_early_convergence 模式下，应跳过此机制
                        CATASTROPHIC_MUTATION_THRESHOLD = 5  # 连续5代无改进触发（从3提高到5）
                        CATASTROPHIC_MUTATION_RATE = 0.10   # 随机翻转10%的基因（从25%降低到10%）
                        CATASTROPHIC_MUTATION_COUNT = 8     # 对8个解进行大变异（从10降低到8）
                        
                        # 仅在非固定预算模式下允许灾难性变异
                        if not ignore_early_convergence and no_improvement_count >= CATASTROPHIC_MUTATION_THRESHOLD:
                            if no_improvement_count == CATASTROPHIC_MUTATION_THRESHOLD:
                                print(f"[MOEA-DIVERSITY]: 触发灾难性变异！连续{no_improvement_count}代无改进")
                                # 记录触发的代数
                                self.catastrophic_mutation_triggered_at = current_gen
                            
                            # 对当前种群中的部分解进行大变异
                            mutated_count = 0
                            # 选择拥挤距离最小的个体（大变异可能伤害好解）
                            self._assign_crowding_distance(self.current_population)
                            self.current_population.sort(
                                key=lambda x: x.crowding_distance if hasattr(x, 'crowding_distance') 
                                              and not np.isinf(x.crowding_distance) else 0
                            )
                            
                            for i in range(min(CATASTROPHIC_MUTATION_COUNT, len(self.current_population))):
                                ind = self.current_population[i]
                                gene = ind.gene.copy()
                                # 随机翻转 CATASTROPHIC_MUTATION_RATE 比例的基因
                                mutation_mask = np.random.random(self.K) < CATASTROPHIC_MUTATION_RATE
                                gene[mutation_mask] = 1 - gene[mutation_mask]
                                ind.gene = gene
                                ind.objectives = None  # 标记需要重新评估
                                mutated_count += 1
                            
                            print(f"[MOEA-DIVERSITY]: 已对 {mutated_count} 个解进行灾难性变异（翻转{CATASTROPHIC_MUTATION_RATE*100:.0f}%基因）")
                            
                            # 重新评估变异的解
                            for ind in self.current_population[:mutated_count]:
                                try:
                                    results = eval_fn([ind.gene])
                                    if results is not None and len(results) > 0:
                                        ind.objectives = results[0]
                                    else:
                                        ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                                except:
                                    ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                                ind.violation = 0.0
                            
                            # 重新分配拥挤距离
                            self._assign_crowding_distance(self.current_population)
    
                    # 检查收敛（至少运行 min_generations_before_check 代才开始检查）
                    if not ignore_early_convergence and generation >= min_generations_before_check and no_improvement_count >= patience:
                        is_converged = True
                        # 仅在非固定预算模式下设置全局收敛标记，避免卡死后续SAC步
                        if not ignore_early_convergence:
                            self.converged = True
                            self.no_improvement_count = no_improvement_count  # 同步到全局
                        print(f"[MOEA-CONV]: MOEA已触发收敛！全局第 {self.current_generation} 代 (超出耐心值 {patience})")
                        break       # 立即跳出当前 while 循环
                
                # 4.9 显示进度
                if generation % 1 == 0 or is_converged or generation >= max_generations:
                    pareto_front = self._extract_pareto_front(self.current_population)
                    pareto_size = len(pareto_front)
                    print(f"进度: {generation}/{max_generations} 代, "
                          f"HV: {current_hv:.6f}, "
                          f"Pareto前沿大小: {pareto_size}")
                if generation >= max_generations:
                    # 达到最大代数，检查是否真正收敛
                    if no_improvement_count >= patience:
                        is_converged = True
                        # fixed_budget 模式下只记录本轮收敛信息，不永久标记，避免后续 SAC 步全部跳过
                        if not ignore_early_convergence:
                            self.converged = True
                            self.no_improvement_count = no_improvement_count  # 同步到全局
                            print(f"[MOEA]: 真正收敛于第 {generation} 代")
                        else:
                            print(f"[MOEA]: 达到最大代数 {max_generations}，本轮无改进但固定预算模式下不永久标记收敛")
                    else:
                        is_converged = False
                        print(f"[MOEA]: 达到最大代数 {max_generations}，但未真正收敛")
                    
            except Exception as e:
                print(f"[MOEA]: 第{generation+1}代运行出错: {e}")
                import traceback
                traceback.print_exc()
                generation += 1
        
        # 5. 最终统计
        self._install_budget_aware_elites(
            max_new=elite_count,
            context="final",
            record_archive=False,
        )
        self.last_evolution_population = self._snapshot_population_for_credit(
            self.current_population
        )
        final_pareto = self._extract_pareto_front(self.current_population)
        final_hv = self._calculate_hypervolume(final_pareto)
        
        # 确保final_hv是标量
        if isinstance(final_hv, np.ndarray):
            if final_hv.size == 1:
                final_hv = float(final_hv.item())
            else:
                final_hv = float(np.mean(final_hv))

        # 诊断：返回前打印收敛历史对比与Pareto前沿大小
        try:
            hv_history_after = list(self.convergence_history.get('hv', []))
        except Exception:
            hv_history_after = []
        final_pareto = self._extract_pareto_front(self.current_population)
        final_pareto_size = len(final_pareto)
        print(f"[MOEA-SNAP FINAL] initial_hv={initial_hv:.6f}, final_hv={final_hv:.6f}, hv_history_len_before={len(hv_history_before) if 'hv_history_before' in locals() else 'N/A'}, hv_history_len_after={len(hv_history_after)}, hv_tail_after={hv_history_after[-5:] if hv_history_after else []}, final_pareto_size={final_pareto_size}")
        
        # 确保initial_hv是标量
        if isinstance(initial_hv, np.ndarray):
            if initial_hv.size == 1:
                initial_hv = float(initial_hv.item())
            else:
                initial_hv = float(np.mean(initial_hv))
        
        convergence_info = {
            'is_converged': is_converged,
            'final_generation': generation,
            'final_hv': final_hv,
            'initial_hv': initial_hv,
            'hv_improvement': final_hv - initial_hv if initial_hv > 0 else 0.0,
            'spea2_updates': self._step_spea2_updates,
            'nsga2_updates': self._step_nsga2_updates,
            'algorithm_update_ratio': (
                (self._step_spea2_updates + self._step_nsga2_updates) /
                max(1.0, 2.0 * generation)
            ),
            'restart_count': self._last_restart_count,
            'search_controls': dict(self.search_controls),
        }
        
        print(f"  [MOEA]: MOEA运行完成:")
        print(f"  总运行代数: {generation}")
        print(f"  是否收敛: {is_converged}")
        print(f"  初始HV: {initial_hv:.6f}")
        print(f"  最终HV: {final_hv:.6f}")
        print(f"  HV改进: {final_hv - initial_hv:.6f}")
        
        stats = self.get_statistics()
        return self.current_population, [], stats, convergence_info
    
    def _score_aligned_point(self, ind):
        gene = getattr(ind, "gene", None)
        if gene is None:
            return None
        try:
            metrics = self._gene_metrics_fast(gene)
            if not metrics.get("feasible", False):
                return None
            area_limit, _, _ = self._ppa_limits()
            area_budget = max(float(self.area_orig) * area_limit, 1e-20)
            return (
                float(np.clip(metrics["area_increase"] / area_budget, 0.0, 1.0)),
                float(np.clip(metrics["residual_ratio"], 0.0, 1.0)),
            )
        except Exception:
            return None

    def _truncate_front_by_hv_contribution(self, front, limit):
        """SMS-EMOA style truncation for the fixed-reference two-objective front."""
        limit = max(0, int(limit))
        if limit <= 0:
            return []
        unique = {}
        for ind in front or []:
            point = self._score_aligned_point(ind)
            if point is None:
                continue
            coord_key = (round(point[0], 14), round(point[1], 14))
            unique.setdefault(coord_key, (point, ind))
        active = sorted(unique.values(), key=lambda item: (item[0][0], item[0][1]))
        if len(active) <= limit:
            return [item[1] for item in active]

        while len(active) > limit:
            contributions = np.full(len(active), np.inf, dtype=np.float64)
            if len(active) > 2:
                for i in range(1, len(active) - 1):
                    prev_y = active[i - 1][0][1]
                    x, y = active[i][0]
                    next_x = active[i + 1][0][0]
                    contributions[i] = max(0.0, next_x - x) * max(0.0, prev_y - y)
            elif len(active) == 2 and limit == 1:
                # Retain the point with the larger standalone dominated area.
                contributions[0] = (1.0 - active[0][0][0]) * (1.0 - active[0][0][1])
                contributions[1] = (1.0 - active[1][0][0]) * (1.0 - active[1][0][1])
                remove_at = int(np.argmin(contributions))
                active.pop(remove_at)
                continue
            remove_at = int(np.argmin(contributions))
            active.pop(remove_at)
        return [item[1] for item in active]

    def _fill_score_aligned_survivors(self, selected, candidates, limit):
        """Fill dominated survivors by budget strata without reintroducing 4-D noise."""
        selected = list(selected or [])
        limit = max(0, int(limit))
        if len(selected) >= limit:
            return selected[:limit]
        seen = {
            self._gene_key(ind.gene)
            for ind in selected
            if getattr(ind, "gene", None) is not None
        }
        bins = [[] for _ in range(self.score_aligned_budget_bins)]
        infeasible = []
        for ind in candidates or []:
            gene = getattr(ind, "gene", None)
            if gene is None or self._gene_key(gene) in seen:
                continue
            point = self._score_aligned_point(ind)
            if point is None:
                infeasible.append(ind)
                continue
            idx = min(self.score_aligned_budget_bins - 1, int(point[0] * self.score_aligned_budget_bins))
            bins[idx].append((point[1], point[0], ind))
        for bucket in bins:
            bucket.sort(key=lambda item: (item[0], item[1]))

        depth = 0
        while len(selected) < limit:
            added = False
            for bucket in bins:
                if depth >= len(bucket):
                    continue
                ind = bucket[depth][2]
                key = self._gene_key(ind.gene)
                if key not in seen:
                    seen.add(key)
                    selected.append(ind)
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break
            depth += 1

        if len(selected) < limit:
            infeasible.sort(key=lambda ind: float(getattr(ind, "violation", float("inf")) or 0.0))
            for ind in infeasible:
                key = self._gene_key(ind.gene)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(ind)
                if len(selected) >= limit:
                    break
        return selected

    def _repair_gene_to_ppa(self, gene):
        """Repair hard PPA violations without greedily collapsing to one filled solution."""
        gene = self._normalize_gene(gene).astype(np.int32)
        area_cost, power_cost, slack, sens = self._ppa_arrays()
        area_limit, power_limit, slack_min = self._ppa_limits()
        safety = self._budget_safety_scale()
        area_budget = max(float(self.area_orig) * area_limit * safety, 1e-20)
        power_budget = max(float(self.power_orig) * power_limit * safety, 1e-20)
        voter_delay = float(self.model.voter_delay)

        for _ in range(self.K):
            selected = np.flatnonzero(gene)
            if selected.size == 0:
                break
            area_inc = float(np.dot(gene, area_cost))
            power_inc = float(np.dot(gene, power_cost))
            worst = float(np.min(slack - voter_delay * gene))
            if (
                area_inc <= area_budget + 1e-12
                and power_inc <= power_budget + 1e-12
                and worst >= slack_min - 1e-12
            ):
                break
            release = (
                area_cost[selected] / area_budget
                + power_cost[selected] / power_budget
                + 5.0 * np.maximum(0.0, slack_min - (slack[selected] - voter_delay))
            )
            damage = sens[selected] + 1e-12
            gene[selected[int(np.argmin(damage / np.maximum(release, 1e-12)))]] = 0
        return gene

    def _restore_score_aligned_population_size(self, selected, candidates, limit):
        """Restore a fixed unique population after archive/warm-start deduplication."""
        selected = list(selected or [])
        limit = max(0, int(limit))
        if len(selected) >= limit:
            return selected[:limit]

        seen = {
            self._gene_key(ind.gene)
            for ind in selected
            if getattr(ind, "gene", None) is not None
        }
        bases = [ind for ind in selected if getattr(ind, "gene", None) is not None]
        if not bases:
            bases = [ind for ind in candidates or [] if getattr(ind, "gene", None) is not None]
        if not bases:
            return selected

        attempts = 0
        max_attempts = max(200, 12 * limit)
        flip_schedule = (2, 4, 6, 8, 12, 16, 24, 32)
        while len(selected) < limit and attempts < max_attempts:
            base = bases[attempts % len(bases)]
            flips = min(self.K, flip_schedule[(attempts // max(1, len(bases))) % len(flip_schedule)])
            gene = self._perturb_gene_with_expected_flips(base.gene, flips)
            gene = self._repair_gene_to_ppa(gene)
            key = self._gene_key(gene)
            attempts += 1
            if key in seen:
                continue
            individual = self._make_individual_from_gene(gene)
            if individual.objectives is None:
                continue
            seen.add(key)
            individual.archive_source = "diversity_restore"
            selected.append(individual)

        # Fixed-size evolutionary operators are preferable to silently running
        # a smaller population. Duplicates are only a last-resort fallback.
        clone_index = 0
        while selected and len(selected) < limit:
            base = selected[clone_index % len(selected)]
            clone = self._make_individual_from_gene(base.gene)
            clone.archive_source = "fixed_size_fallback"
            selected.append(clone)
            clone_index += 1
        return selected[:limit]

    def _environmental_selection(self, population):
        """
        改进的环境选择方法
        考虑多样性和拥挤距离（已接入 Numpy 向量化加速）
        """
        if not population:
            return []

        if getattr(self, "score_aligned_environmental_selection", False):
            pareto_front = self._extract_ppa_feasible_pareto(population)
            selected = self._truncate_front_by_hv_contribution(pareto_front, self.pop_size)
            selected = self._fill_score_aligned_survivors(selected, population, self.pop_size)
            return self._restore_score_aligned_population_size(selected, population, self.pop_size)

        if len(population) <= self.pop_size:
            return population
        
        # 提取Pareto前沿 (此时已经使用的是向量化的高速提取)
        pareto_front = self._extract_pareto_front(population)
        
        if len(pareto_front) >= self.pop_size:
            # ==========================================
            # 如果Pareto前沿足够大，直接一次性批量赋予拥挤距离
            # ==========================================
            self._assign_crowding_distance(pareto_front)
            
            # 按拥挤距离排序，选择多样性最好的
            pareto_front.sort(key=lambda x: x.crowding_distance if hasattr(x, 'crowding_distance') 
                              and not np.isinf(x.crowding_distance) else float('inf'), reverse=True)
            return pareto_front[:self.pop_size]
        else:
            # 否则补充非Pareto个体
            selected = list(pareto_front)
            remaining = self.pop_size - len(selected)
            
            # 从非Pareto个体中选择
            non_pareto = [ind for ind in population if ind not in pareto_front]
            
            # 按拥挤距离或适应度排序
            if non_pareto:
                self._assign_crowding_distance(non_pareto)
                
                if hasattr(non_pareto[0], 'crowding_distance'):
                    non_pareto.sort(key=lambda x: x.crowding_distance if hasattr(x, 'crowding_distance') 
                                      and not np.isinf(x.crowding_distance) else float('inf'), reverse=True)
                elif hasattr(non_pareto[0], 'fitness'):
                    non_pareto.sort(key=lambda x: x.fitness if hasattr(x, 'fitness') else float('inf'))
            
            selected.extend(non_pareto[:remaining])
            return selected
    
    def _environmental_selection_optimized(self, spea2_pop, nsga2_pop):
        """
        优化的环境选择：直接从两个种群中选择，避免创建大的合并列表
        """
        if getattr(self, "score_aligned_environmental_selection", False):
            return self._environmental_selection(list(spea2_pop or []) + list(nsga2_pop or []))

        # 1. 分别提取两个种群的Pareto前沿（已过滤敏感性违例）
        spea2_front = self._extract_pareto_front(spea2_pop)
        nsga2_front = self._extract_pareto_front(nsga2_pop)
        
        # 2. 合并前沿（而不是合并整个种群）
        combined_front = spea2_front + nsga2_front
        
        # 3. 如果前沿足够大，直接选择
        if len(combined_front) >= self.pop_size:
            # 批量计算拥挤距离并选择
            self._assign_crowding_distance(combined_front)
            
            combined_front.sort(key=lambda x: x.crowding_distance if hasattr(x, 'crowding_distance') 
                              and not np.isinf(x.crowding_distance) else 0, reverse=True)
            return combined_front[:self.pop_size]
        
        # 4. 如果前沿不够大，补充非Pareto个体
        selected = list(combined_front)
        remaining = self.pop_size - len(selected)
        
        # 从两个种群的非Pareto个体中选择
        spea2_non_pareto = [ind for ind in spea2_pop if ind not in spea2_front]
        nsga2_non_pareto = [ind for ind in nsga2_pop if ind not in nsga2_front]
        all_non_pareto = spea2_non_pareto + nsga2_non_pareto
        
        # 按拥挤距离或适应度排序
        if all_non_pareto:
            if hasattr(all_non_pareto[0], 'crowding_distance'):
                all_non_pareto.sort(key=lambda x: x.crowding_distance if hasattr(x, 'crowding_distance') 
                                  and not np.isinf(x.crowding_distance) else 0, reverse=True)
            elif hasattr(all_non_pareto[0], 'fitness'):
                all_non_pareto.sort(key=lambda x: x.fitness if hasattr(x, 'fitness') else float('inf'))
        
        selected.extend(all_non_pareto[:remaining])
        return selected
    
    def _calculate_simple_crowding_distance(self, ind, front):
        """简单计算拥挤距离"""
        if len(front) < 2:
            return float('inf')
        
        objectives = np.array([x.objectives for x in front])
        n_objs = objectives.shape[1]
        
        crowding = 0.0
        for obj_idx in range(n_objs):
            sorted_indices = np.argsort(objectives[:, obj_idx])
            obj_values = objectives[sorted_indices, obj_idx]
            
            # 找到当前个体在排序中的位置
            ind_idx = np.where(sorted_indices == front.index(ind))[0]
            if len(ind_idx) > 0:
                idx = ind_idx[0]
                if idx == 0 or idx == len(front) - 1:
                    crowding += float('inf')
                else:
                    obj_range = obj_values[-1] - obj_values[0]
                    if obj_range > 0:
                        crowding += (obj_values[idx+1] - obj_values[idx-1]) / obj_range
        
        return crowding / n_objs if not np.isinf(crowding) else float('inf')
    
    def get_last_pareto(self, include_history=True):
        """
        获取最后的Pareto前沿

        参数:
            include_history: 是否包含累积精英池（默认True）
        
        返回:
            如果 include_history=True: 当前种群Pareto + 累积精英池（去重）
            如果 include_history=False: 仅当前种群Pareto
        """
        # Re-evaluate every gene with the current unified model. Saved objectives
        # are diagnostic metadata only and must not decide the exported front.
        candidates = []
        seen = set()
        baseline_keys = set()
        source_by_key = {}

        for entry in getattr(self, "greedy_archive", []) or []:
            gene = self._get_elite_entry_gene(entry)
            if gene is None:
                continue
            key = self._gene_key(gene)
            baseline_keys.add(key)
            source_by_key[key] = "fixed_baseline"

        for entry in getattr(self, "cumulative_elite_pool", []) or []:
            gene = self._get_elite_entry_gene(entry)
            if gene is None:
                continue
            key = self._gene_key(gene)
            entry_source = entry.get("source", "historical") if isinstance(entry, dict) else getattr(
                entry, "archive_source", "historical"
            )
            source_by_key.setdefault(key, str(entry_source))

        def add_gene(gene, historical, source):
            if gene is None:
                return
            key = self._gene_key(gene)
            if key in seen:
                return
            fresh = self._make_individual_from_gene(gene)
            if fresh is None or fresh.objectives is None:
                return
            seen.add(key)
            fresh.is_historical = bool(historical)
            fresh.is_fixed_baseline = key in baseline_keys
            fresh.archive_source = source_by_key.get(
                key, str(source or ("historical" if historical else "current"))
            )
            candidates.append(fresh)

        for ind in self.current_population or []:
            add_gene(getattr(ind, "gene", None), False, getattr(ind, "archive_source", "current"))

        if include_history:
            history_sources = [
                getattr(self, "greedy_archive", []) or [],
                getattr(self, "cumulative_elite_pool", []) or [],
            ]
            for source_list in history_sources:
                for entry in source_list:
                    gene = self._get_elite_entry_gene(entry)
                    entry_source = entry.get("source", "historical") if isinstance(entry, dict) else getattr(
                        entry, "archive_source", "historical"
                    )
                    add_gene(gene, True, entry_source)

        if getattr(self, "ppa_pareto_only", True):
            return self._extract_ppa_feasible_pareto(candidates)
        return self._extract_pareto_front(candidates)

    def get_search_attribution(self, pareto_front=None):
        """Quantify score-aligned improvement beyond the fixed seed baseline."""
        final_front = list(pareto_front) if pareto_front is not None else self.get_last_pareto(True)

        baseline_individuals = []
        baseline_keys = set()
        for entry in getattr(self, "greedy_archive", []) or []:
            gene = self._get_elite_entry_gene(entry)
            if gene is None:
                continue
            key = self._gene_key(gene)
            if key in baseline_keys:
                continue
            baseline_keys.add(key)
            individual = self._make_individual_from_gene(gene)
            if individual is not None:
                baseline_individuals.append(individual)

        baseline_front = self._extract_ppa_feasible_pareto(baseline_individuals)
        baseline_hv = float(self._calculate_scoring_hv_2d(baseline_front))
        final_hv = float(self._calculate_scoring_hv_2d(final_front))
        nonbaseline_points = sum(
            1 for individual in final_front
            if self._gene_key(individual.gene) not in baseline_keys
        )
        exploration_step_points = sum(
            1 for individual in final_front
            if str(getattr(individual, "archive_source", "")).startswith("sac_exploration_step:")
        )
        policy_step_points = sum(
            1 for individual in final_front
            if str(getattr(individual, "archive_source", "")).startswith("sac_policy_step:")
        )
        return {
            "baseline_hv": baseline_hv,
            "final_hv": final_hv,
            "search_hv_gain": max(0.0, final_hv - baseline_hv),
            "baseline_front_size": len(baseline_front),
            "final_front_size": len(final_front),
            "nonbaseline_front_points": nonbaseline_points,
            "exploration_step_front_points": exploration_step_points,
            "policy_step_front_points": policy_step_points,
        }

    def get_current_pareto(self):
        """获取当前种群Pareto前沿（不包含历史）"""
        return self.get_last_pareto(include_history=False)

    def get_historical_pareto(self):
        """获取累积精英池的Pareto前沿（仅历史解）"""
        if not hasattr(self, 'cumulative_elite_pool') or not self.cumulative_elite_pool:
            return []

        individuals = []
        seen = set()
        for entry in self.cumulative_elite_pool:
            gene = self._get_elite_entry_gene(entry)
            if gene is None:
                continue
            key = self._gene_key(gene)
            if key in seen:
                continue
            try:
                ind = self._make_individual_from_gene(gene)
                if ind is None or ind.objectives is None:
                    continue
                seen.add(key)
                ind.is_historical = True
                individuals.append(ind)
            except Exception:
                pass

        return self._extract_pareto_front(individuals) if individuals else []
        
    def save_pareto_solution_to_csv(self, individual, save_path, solution_id=1):
        """
        保存单个Pareto解为CSV文件
        """
        from pathlib import Path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 1. 获取基因
            genes = individual.gene.flatten()

            # 2. 使用详细的评估方法
            results = self.model.evaluate_with_details(genes)

            # 确保我们有足够的返回值
            if len(results) >= 9:
                (area_total, power_total, min_slack, remaining_absolute_sensitivity,
                 reinforced_indices_prefiltered, area_increase, power_increase, slack_decrease,
                 sens_improvement_ratio) = results
            else:
                print(f"警告: evaluate_with_details 返回 {len(results)} 个值，期望至少 9 个")
                return False

            # 3. 计算加固后的所有路径slack（用于保存path_slack列）
            # 确保genes长度正确
            if len(genes) != self.model.K:
                if len(genes) > self.model.K:
                    genes = genes[:self.model.K]
                else:
                    genes = np.pad(genes, (0, self.model.K - len(genes)), 'constant')
            
            # 3.1 计算加固前的敏感性（使用混合加权方法，确保与加固后使用相同方法）
            zero_bitvec = np.zeros(self.model.K, dtype=np.int32)
            _, original_sensitivity_before = self.model._calculate_sensitivity_hybrid(zero_bitvec)
            
            # 计算加固后的路径slack
            slack_after = self.model.slack.copy()
            tmr_indices = np.where(np.array(genes) == 1)[0]  # 所有被加固的寄存器
            if len(tmr_indices) > 0:
                slack_after[tmr_indices] -= self.model.voter_delay
            
            # 计算worst_slack（所有路径的最小值，已考虑加固影响）
            # 使用真实值（可能为负），不截断，以便在CSV中显示真实情况
            worst_slack_real = float(np.min(slack_after))

            # 4. 准备CSV内容
            csv_lines = []

            # 4.1 整体PPA信息
            # 导出 frequency（如果模型中存在）以及根据 slack_min 计算的 frequency_after_tmr
            slack_min = float(self.config.get('constraints', {}).get('slack_min', 0.0))

            # 计算所有路径的原始 path_delay（使用 model.frequency 和 model.slack）
            # path_delay = 1/frequency - path_slack
            try:
                freq0 = float(getattr(self.model, 'frequency', 200e6))  # Hz
                # 将频率转换为周期（纳秒）： period_ns = 1e9 / freq(Hz)
                period_ns = 1e9 / freq0
                # 使用当前解的加固后 slack（单位 ns）来计算 path_delays
                path_delays = period_ns - np.array(slack_after, dtype=np.float64)
            except Exception:
                freq0 = 200e6
                period_ns = 1e9 / freq0
                path_delays = period_ns - np.array(slack_after, dtype=np.float64)

            # 对每条路径计算允许的最大频率上界：freq <= 1 / (slack_min + path_delay + voter_delay)
            # 假定 voter_delay 单位为纳秒 (ns)
            voter = float(getattr(self.model, 'voter_delay', self.model.voter_delay if hasattr(self.model, 'voter_delay') else 0.0))
            # 避免分母为0或负值：如果 (slack_min + path_delay + voter) <= 0，则认为频率上界为 +inf（不受约束）
            # slack_min 应为纳秒单位（ns）
            slack_min_ns = float(slack_min)
            denom = slack_min_ns + path_delays + voter
            with np.errstate(divide='ignore', invalid='ignore'):
                freq_bounds = np.where(denom > 0, 1.0 / denom, np.inf)

            # 选择对所有路径都安全的最大频率：取最小的上界
            # freq_bounds 的单位为 1/ns，因此需要转换回 Hz：freq(Hz) = 1 / (denom_ns * 1e-9) = 1e9 / denom_ns
            # 我们先取路径允许的频率上界（Hz）为: 1e9 / denom
            with np.errstate(divide='ignore', invalid='ignore'):
                freq_bounds_hz = np.where(denom > 0, 1e9 / denom, np.inf)

            frequency_after_tmr_hz = float(np.min(freq_bounds_hz)) if freq_bounds_hz.size > 0 else freq0
            # 不应超过原始频率（Hz）
            if np.isfinite(frequency_after_tmr_hz):
                frequency_after_tmr_hz = min(frequency_after_tmr_hz, freq0)
            else:
                frequency_after_tmr_hz = freq0

            frequency_after_tmr = frequency_after_tmr_hz

            # 根据用户期望：如果配置中的 slack_min 大于真实最差 slack，则
            # 我们需要将导出结果中的最差 slack 调整为 slack_min，并将所有路径
            # 的 slack 平移 delta = slack_min - worst_slack_real，使得行与 header 一致。
            # 这样可以同时导出 frequency_after_tmr 表示在保证 slack >= slack_min
            # 条件下的最大可达频率。
            exported_worst_slack = float(worst_slack_real)
            shift_delta = 0.0
            shift_applied = False
            try:
                if slack_min is not None and float(slack_min) > worst_slack_real:
                    shift_delta = float(slack_min) - worst_slack_real
                    exported_worst_slack = float(slack_min)
                    shift_applied = True
            except Exception:
                shift_delta = 0.0
                shift_applied = False

            # 标注单位：worst_slack/ns, frequency 和 frequency_after_tmr 输出为 MHz
            # design_sensitivity_before: 加固前的系统级敏感性（简单均值方法）
            # design_sensitivity_after: 加固后的系统级敏感性（简单均值方法）
            # 格式：total_area, total_power, worst_slack(ns), design_sensitivity_before, design_sensitivity_after, frequency(MHz), frequency_after_tmr(MHz)
            csv_lines.append("total_area,total_power,worst_slack(ns),design_sensitivity_before,design_sensitivity_after,frequency(MHz),frequency_after_tmr(MHz)")
            csv_lines.append(f"{area_total:.2f},{power_total:.2f},{exported_worst_slack:.6f},{original_sensitivity_before:.6f},{remaining_absolute_sensitivity:.6f},{round(freq0/1e6,6):.6f},{round(frequency_after_tmr/1e6,6):.6f}")
            csv_lines.append("")  # 空行

            # 4.2 寄存器信息标题（path_slack 单位为 ns）
            csv_lines.append("index,instance_name,reg_type,area,reg_power,path_slack(ns),sensitivity")

            # 4.3 被加固寄存器信息
            # 注意：CSV中的sensitivity列显示原始敏感性（加固前的值）
            # 虽然加固后实际敏感性为0，但保留原始值便于分析对比

            processed_count = 0
            path_slacks_in_csv = []
            
            # reinforced_indices_prefiltered 是被加固的索引（预筛选后的索引，0到K-1）
            for pre_idx in reinforced_indices_prefiltered:
                # 确保pre_idx在有效范围内
                if pre_idx < len(self.model.original_indices):
                    # 获取原始索引
                    original_idx = self.model.original_indices[pre_idx]

                    # 确保pre_idx在属性数组范围内
                    if pre_idx < len(self.model.area) and pre_idx < len(slack_after):
                        # 获取寄存器信息
                        instance_name = self.model.instance_names[pre_idx] if pre_idx < len(self.model.instance_names) else f"reg_{original_idx}"
                        reg_type = self.model.reg_types[pre_idx] if pre_idx < len(self.model.reg_types) else "FF"
                        area = float(self.model.area[pre_idx])
                        power = float(self.model.power[pre_idx])
                        # 使用加固后的路径slack
                        raw_path_slack = float(slack_after[pre_idx])
                        if shift_applied:
                            path_slack_after = raw_path_slack + shift_delta
                        else:
                            path_slack_after = raw_path_slack
                        path_slacks_in_csv.append(path_slack_after)
                        
                        # 获取原始敏感性值（加固前的敏感性）
                        if pre_idx < len(self.model.sensitivity):
                            original_sensitivity = float(self.model.sensitivity[pre_idx])
                        else:
                            original_sensitivity = 0.0
                        
                        # 在CSV中显示原始敏感性（虽然加固后实际为0，但保留原始值供参考）
                        sensitivity_after = original_sensitivity

                        csv_lines.append(
                            f"{original_idx},{instance_name},{reg_type},{area:.2f},{power:.4f},{path_slack_after:.6f},{sensitivity_after:.6f}"
                        )
                        processed_count += 1
                    else:
                        print(f"  警告: 预筛选索引 {pre_idx} 超出属性数组范围")
                else:
                    print(f"  警告: 预筛选索引 {pre_idx} 超出原始索引映射范围")

            ## print(f"  成功处理 {processed_count}/{len(reinforced_indices_prefiltered)} 个寄存器")

            # 5. 写入文件
            with open(save_path, 'w') as f:
                f.write('\n'.join(csv_lines))

            print(f"INFO: Pareto解 {solution_id} 已保存到: {save_path}")
            print(f"  worst_slack_real={worst_slack_real:.6f} (真实), exported_worst_slack={exported_worst_slack:.6f}")
            print(f"  design_sensitivity_before={original_sensitivity_before:.6f}, design_sensitivity_after={remaining_absolute_sensitivity:.6f}")
            print(f"  frequency={freq0:.6f}, frequency_after_tmr={frequency_after_tmr:.6f}")

            # 6. 验证保存的内容
            ## print(f"  保存了 {processed_count} 行寄存器数据")
            ## print(f"  文件行数: {len(csv_lines)}")
            ## if csv_lines:
            ##     print(f"  最后几行内容:")
            ##     for i, line in enumerate(csv_lines[-min(5, len(csv_lines)):]):
            ##         print(f"    {i}: {line}")

            return True

        except Exception as e:
            print(f"保存Pareto解时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def save_pareto_front_to_csv(self, pareto_front, save_dir, filename_prefix="solution", 
                                  deduplicate=True, separate_current_history=True):
        """
        保存整个Pareto前沿
        
        参数:
            pareto_front: Pareto前沿个体列表
            save_dir: 保存目录
            filename_prefix: 文件名前缀
            deduplicate: 是否去重（默认True）- 只保存唯一的解
            separate_current_history: 是否区分当前解/历史解（默认True）
        """
        import pandas as pd
        from pathlib import Path

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        current_count = 0
        history_count = 0
        summary_data = []

        # ========== 【去重处理】在保存前先去除重复的解 ==========
        if deduplicate:
            # 按基因分组，记录每个唯一解
            unique_genes = {}  # gene_str -> (ind, is_historical)
            
            for ind in pareto_front:
                if ind is None or ind.gene is None or ind.objectives is None:
                    continue
                
                genes_arr = np.array(ind.gene).flatten()
                gene_str = ''.join(str(int(g)) for g in genes_arr)
                
                is_hist = getattr(ind, 'is_historical', True)  # 默认作为历史解
                
                if gene_str not in unique_genes:
                    unique_genes[gene_str] = (ind, is_hist)
                elif not is_hist:
                    # 如果新解是当前解但已存在历史解，保留当前解
                    unique_genes[gene_str] = (ind, False)
            
            # 用去重后的解替换原来的
            original_count = len(pareto_front)
            pareto_front = [v[0] for v in unique_genes.values()]
            print(f"[去重] 原始解数量: {original_count}, 去重后: {len(pareto_front)}, 去除重复: {original_count - len(pareto_front)}")
        
        # ========== 【区分当前/历史解】 ==========
        current_solutions = []
        history_solutions = []
        
        if separate_current_history:
            for ind in pareto_front:
                is_hist = getattr(ind, 'is_historical', True)
                if is_hist:
                    history_solutions.append(ind)
                else:
                    current_solutions.append(ind)
            
            print(f"[解分类] 当前解: {len(current_solutions)}, 历史解: {len(history_solutions)}")
        else:
            current_solutions = pareto_front
        
        # ========== 保存当前解 ==========
        def _save_solutions(solutions, prefix, source_name):
            nonlocal saved_count, current_count, history_count
            
            if not solutions:
                return 0
            
            count = 0
            for i, ind in enumerate(solutions):
                if ind is None or ind.gene is None or ind.objectives is None:
                    continue

                genes_arr = np.array(ind.gene).flatten()
                gene_str = ''.join(str(int(g)) for g in genes_arr)

                # 保存 individual CSV
                solution_path = save_dir / f"{prefix}_{i+1:03d}.csv"
                success = self.save_pareto_solution_to_csv(
                    individual=ind,
                    save_path=solution_path,
                    solution_id=i+1
                )

                if not success:
                    continue

                count += 1
                saved_count += 1

                reinforced_count = int(np.sum(genes_arr == 1))
                summary_data.append({
                    'solution_id': i + 1,
                    'file': solution_path.name,
                    'source': getattr(ind, 'archive_source', source_name),
                    'is_fixed_baseline': bool(getattr(ind, 'is_fixed_baseline', False)),
                    'reinforced_count': reinforced_count,
                    'genes': gene_str
                })
            
            return count
        
        # 保存当前解 (前缀: solution_current)
        current_count = _save_solutions(current_solutions, f"{filename_prefix}_current", "current")
        
        # 保存历史解 (前缀: solution_history)
        history_count = _save_solutions(history_solutions, f"{filename_prefix}_history", "historical")

        # 保存汇总信息
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = save_dir / "summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"Pareto前沿汇总已保存到: {summary_path}")
            print(f"  └─ 当前解: {current_count}, 历史解: {history_count}, 总计: {saved_count}")

        return {'total': saved_count, 'current': current_count, 'history': history_count}
