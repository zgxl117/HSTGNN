# mo_optimizer/simulator.py
import numpy as np
from typing import Tuple, Optional, Dict, Any
import csv
from pathlib import Path

class ICDesignSimulator:
    """一个物理上合理的集成电路设计模拟器。"""
    
    def __init__(self, csv_file_path: Optional[str] = None, 
                 num_registers: int = 1000, 
                 K: int = 200, 
                 error_rates: list = None,
                 frequency: float = 200e6,
                 config: Dict[str, Any] = None):
        """
        初始化模拟器
        
        参数:
        ----------
        csv_file_path : str, optional
            CSV文件路径，包含寄存器信息和原始PPA值
        num_registers : int
            如果未提供CSV，则生成随机寄存器的数量
        K : int
            预筛选的寄存器数量
        error_rates : list, optional
            如果提供，则使用给定的错误率
        """
        # --- 读取CSV文件或使用默认值 ---
        self.config = config or {}
        self.base_total_failure_rate = float(self.config.get('constraint_thresholds', {}).get('base_total_failure_rate', self.config.get('base_total_failure_rate', 1e-2)))
        self.voter_failure_rate = float(self.config.get('constraint_thresholds', {}).get('voter_failure_rate', self.config.get('voter_failure_rate', 1e-6)))
        self.use_csv_failure_prob = bool(self.config.get('constraint_thresholds', {}).get('use_csv_failure_prob', self.config.get('use_csv_failure_prob', True)))
        # --- 读取confiig文件或使用默认值 ---
        voter_cfg = config.get('voter', {})
        self.voter_delay = voter_cfg.get('voter_delay', 0.13)
        self.voter_area = voter_cfg.get('voter_area', 0.5)
        self.voter_power = voter_cfg.get('voter_power', 0.2)
        self.min_slack = float(self.config.get('constraint_thresholds', {}).get('min_slack', 0.0))  

        if csv_file_path and Path(csv_file_path).exists():
            self._load_from_csv(csv_file_path, num_registers)
        else:
            # 如果没有CSV文件，使用默认值
            self.num_registers = num_registers
            # frequency can be user supplied for generated CSVs (默认 200MHz)
            self.frequency = float(frequency)
            self._setup_default_values()
        
        # 验证K的合理性（修复问题4）
        if K > self.num_registers:
            print(f"警告: K ({K}) 大于寄存器总数 ({self.num_registers})，将K设置为 {self.num_registers}")
            K = self.num_registers
        if K <= 0:
            raise ValueError(f"K必须大于0，当前值: {K}")
        
        self.K = K  # 预筛选的寄存器数量
        
        # --- 预筛选 ---
        # 优先使用 CSV 中提供的 failure_prob（如果存在并且非全 NaN），否则使用敏感性权重
        fp = self.all_reg_properties.get('failure_prob', None)
        if fp is not None and not np.all(np.isnan(fp)):
            # 使用 failure_prob 排序（降序）
            sorted_indices = np.argsort(fp)[::-1]
        else:
            sorted_indices = np.argsort(self.all_reg_properties['sensitivity'])[::-1]
        self.SETFF_indices = sorted_indices[:K]
        self.map_to_original = {i: orig_i for i, orig_i in enumerate(self.SETFF_indices)}
        
        # 存储前K个寄存器的属性
        # error_rates 代表用于优化/加固优先级的“单寄存器失效率”或敏感性权重
        if fp is not None and not np.all(np.isnan(fp)):
            self.error_rates = fp[self.SETFF_indices]
        else:
            self.error_rates = self.all_reg_properties['sensitivity'][self.SETFF_indices]
        self.is_critical = self.all_reg_properties['is_critical'][self.SETFF_indices]
        self.slack = self.all_reg_properties['slack'][self.SETFF_indices]
        
        # 计算错误率权重
        self.error_weights = self.error_rates / np.sum(self.error_rates) if np.sum(self.error_rates) > 0 else np.ones(self.K) / self.K
        
        # 计算slack权重
        epsilon = 1e-6
        slack_weights = 1.0 / (self.slack + epsilon)
        slack_weights = slack_weights / np.sum(slack_weights) if np.sum(slack_weights) > 0 else np.ones(self.K) / self.K
        
        # 计算归一化 slack
        min_slack = np.min(self.slack) if len(self.slack) > 0 else 0.0
        max_slack = np.max(self.slack) if len(self.slack) > 0 else 1.0
        slack_range = max_slack - min_slack if max_slack > min_slack else 1.0
        normalized_slack = (self.slack - min_slack) / slack_range
        
        # 计算加固优先级
        self.reinforcement_priority = self.error_rates * (1 + 0.2 * self.is_critical) * (1 - normalized_slack)
        
        # 归一化优先级作为权重
        self.combined_weights = self.reinforcement_priority / np.sum(self.reinforcement_priority) if np.sum(self.reinforcement_priority) > 0 else np.ones(self.K) / self.K
        
        # 计算时序敏感度
        self.timing_sensitivity = 1.0 - normalized_slack

        # 将字典转换为 numpy 数组，避免重复查找
        self.map_indices_array = np.array([self.map_to_original[i] for i in range(self.K)])
        
        # 打印原始PPA值
        print(f"原始PPA值:")
        print(f"  总面积: {self.Area_orig}")
        print(f"  总功耗: {self.Power_orig}")
        print(f"  最差时序裕量: {self.WNS_orig}")
        print(f"  设计敏感性: {self.SER_orig}")
        print(f"  寄存器数量: {self.num_registers}")
    
    def _load_from_csv(self, csv_file_path: str, num_registers: int):
        """
        从CSV文件加载数据
        """
        print(f"从CSV文件加载数据: {csv_file_path}")
        
        # 读取CSV文件
        with open(csv_file_path, 'r') as f:
            lines = f.readlines()
        
        print(f"CSV文件行数: {len(lines)}")
        # 解析整体设计指标（假设在前几行）
        header_found = False
        design_metrics = None
        data_start_line = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            
            # 查找整体设计指标行
            if line.startswith('total_area'):
                # 这是整体设计指标行
                parts = line.split(',')
                # 支持可选的frequency字段：total_area,total_power,worst_slack,design_sensitivity[,frequency]
                if len(parts) >= 4:
                    # 下一行应该是数值
                    if i+1 < len(lines):
                        values_line = lines[i+1].strip()
                        values = values_line.split(',')
                        if len(values) >= 4:
                            try:
                                self.Area_orig = float(values[0])
                                self.Power_orig = float(values[1])
                                self.WNS_orig = float(values[2])
                                self.SER_orig = float(values[3])
                                # 如果CSV中提供frequency字段，则读取它
                                if len(values) >= 5 and values[4] != '':
                                    try:
                                        self.frequency = float(values[4])
                                    except Exception:
                                        print(f"警告: 无法解析 frequency 字段: {values[4]}，使用默认 frequency=200e6")
                                        self.frequency = 200e6
                                else:
                                    # 如果未提供，则使用默认200MHz
                                    self.frequency = 200e6

                                design_metrics = {
                                    'total_area': self.Area_orig,
                                    'total_power': self.Power_orig,
                                    'worst_slack': self.WNS_orig,
                                    'design_sensitivity': self.SER_orig
                                }
                                
                                data_start_line = i + 2  # 跳过标题行和数据行
                                break
                            except ValueError as e:
                                print(f"警告: 无法解析设计指标值: {values}")
                                print(f"解析设计指标值时出错: {e}")
                                print(f"值列表: {values}")
        
        # 如果没有找到设计指标，使用默认值
        if design_metrics is None:
            print("警告: 未在CSV文件中找到整体设计指标，使用默认值")
            self._setup_default_values()
            return
        
        # 解析寄存器数据
        registers_data = []
        current_index = 0
        
        for i in range(data_start_line, len(lines)):
            line = lines[i].strip()
            
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            
            # 检查是否到达寄存器数据部分
            if line.startswith('index'):
                # 这是寄存器数据标题行，跳过
                continue
            
            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    index = int(parts[0])
                    instance_name = parts[1]
                    reg_type = parts[2]
                    area = float(parts[3])
                    # merged_registers.csv 列顺序: area, reg_power, reg_dynamic_power, reg_leakage_power, path_slack, sensitivity
                    path_power = float(parts[4])  # reg_power(uW)
                    path_slack = float(parts[7])  # path_slack(ns) - 第8列
                    sensitivity = float(parts[8])  # sensitivity - 第9列
                    # 支持可选的 failure_prob 列（第10列，索引9）
                    failure_prob = None
                    if len(parts) >= 10 and parts[9] != '':
                        try:
                            failure_prob = float(parts[9])
                        except Exception:
                            failure_prob = None
                    
                    registers_data.append({
                        'index': index,
                        'instance_name': instance_name,
                        'reg_type': reg_type,
                        'area': area,
                        'path_power': path_power,
                        'path_slack': path_slack,
                        'sensitivity': sensitivity,
                        'failure_prob': failure_prob
                    })
                    current_index += 1
                except (ValueError, IndexError) as e:
                    print(f"警告: 解析寄存器数据时出错 (行 {i+1}): {e}")
        
        self.num_registers = len(registers_data)
        print(f"从CSV读取了 {self.num_registers} 个寄存器的数据")
        
        if self.num_registers == 0:
            print("错误: 未从CSV文件中读取到寄存器数据")
            self._setup_default_values()
            return
        
        # 提取寄存器属性
        areas = np.array([reg['area'] for reg in registers_data])
        powers = np.array([reg['path_power'] for reg in registers_data])
        slacks = np.array([reg['path_slack'] for reg in registers_data])
        sensitivities = np.array([reg['sensitivity'] for reg in registers_data])
        # 读取可选的 failure_prob 列（若存在）
        failure_probs = np.array([reg.get('failure_prob') if reg.get('failure_prob') is not None else np.nan for reg in registers_data], dtype=np.float64)
        
        # 确定关键路径寄存器（slack较小的寄存器）
        # 假设slack小于平均值的60%为关键路径
        slack_threshold = np.mean(slacks) * 0.6
        is_critical = (slacks < slack_threshold).astype(int)
        
        # 计算toggle_rate（从功耗估计）
        avg_power_per_bit = np.mean(powers)
        toggle_rates = powers / avg_power_per_bit if avg_power_per_bit > 0 else np.ones_like(powers)
        
        # 从area估计bitwidth
        avg_area_per_bit = np.mean(areas) / 16  # 假设平均每个bit 1/16的面积
        bitwidths = (areas / avg_area_per_bit).astype(int)
        bitwidths = np.clip(bitwidths, 8, 64)  # 限制在8-64位
        
        # 计算基于 sensitivity 权重的默认 failure probability
        sens_sum = np.sum(sensitivities) if np.sum(sensitivities) > 0 else 1.0
        sensitivity_weights = sensitivities / sens_sum
        computed_failure_probs = sensitivity_weights * float(self.base_total_failure_rate)

        # 决定最终使用的 p_i（若 use_csv_failure_prob 且 CSV 中提供了值，则优先使用 CSV，缺失值回退到计算值）
        if self.use_csv_failure_prob and not np.all(np.isnan(failure_probs)):
            final_p = np.where(np.isnan(failure_probs), computed_failure_probs, failure_probs)
            p_source = 'csv_mixed'
        else:
            final_p = computed_failure_probs
            p_source = 'computed'

        # 设计故障概率 = 平均失效率（未加固时的平均失效率）
        design_failure_prob = float(np.sum(final_p) / len(final_p))

        instance_names = np.array([reg['instance_name'] for reg in registers_data])
        reg_types = np.array([reg['reg_type'] for reg in registers_data])
        
        self.all_reg_properties = {
            'sensitivity': sensitivities,
            'failure_prob': final_p,
            'failure_prob_source': p_source,
            'computed_failure_prob': computed_failure_probs,
            'is_critical': is_critical,
            'slack': slacks,
            'bitwidth': bitwidths,
            'toggle_rate': toggle_rates,
            'area': areas,
            'power': powers,
            'instance_name': instance_names,   # 添加：寄存器名称
            'reg_type': reg_types              # 添加：寄存器类型
        }

        # 记录设计级别的 failure sum（总失效率量级）以便其他模块读取
        self.design_failure_prob = design_failure_prob
        # 为兼容旧代码，把 sens_orig 指向 design_failure_prob（表示系统级失效率基准）
        self.sens_orig = design_failure_prob
        
        # 验证数据
        total_csv_area = np.sum(areas)
        total_csv_power = np.sum(powers)
        worst_csv_slack = np.min(slacks)
        total_csv_sens = np.sum(sensitivities)
        
        # 使用简单均值方法计算整体敏感性（基于错误率）
        # 整体敏感性 = 所有寄存器敏感性的均值 = (x1 + x2 + ... + xK) / K
        calculated_sensitivity = float(np.sum(sensitivities) / self.num_registers) if self.num_registers > 0 else 0.0
        
        # 验证CSV中的值与计算值是否一致（允许大的误差范围，因为两者可能使用不同的计算方法）
        # 【修复】：CSV中的design_sensitivity应该作为最终基准值，不应被覆盖
        # 这里只打印警告，不覆盖SER_orig
        if abs(self.SER_orig - calculated_sensitivity) > 0.01:
            print(f"警告: CSV中的design_sensitivity ({self.SER_orig:.6f}) "
                  f"与计算值 ({calculated_sensitivity:.6f}) 存在差异")
            print(f"注意: 使用CSV中的值作为基准（这是正确的做法）")
        else:
            print(f"验证通过: CSV中的design_sensitivity ({self.SER_orig:.6f}) "
                  f"与计算值 ({calculated_sensitivity:.6f}) 一致")
        
        print(f"CSV数据统计:")
        print(f"  寄存器总面积: {total_csv_area:.2f} (vs 设计总面积: {self.Area_orig:.2f})")
        print(f"  寄存器总功耗: {total_csv_power:.2f} (vs 设计总功耗: {self.Power_orig:.2f})")
        print(f"  最差slack: {worst_csv_slack:.6f} (vs 设计最差slack: {self.WNS_orig:.6f})")
        print(f"  总敏感度: {total_csv_sens:.6f}")
        print(f"  平均敏感度 (Σs_i/n): {calculated_sensitivity:.6f}")
        print(f"  设计敏感度 (SER_orig): {self.SER_orig:.6f}")
        print(f"  设计级别失效率总和 (design_failure_prob - Σp_i): {self.design_failure_prob:.8e} (source: {p_source})")
        print(f"  关键路径寄存器数量: {np.sum(is_critical)}")
    
    def _setup_default_values(self):
        """
        设置默认值（当没有CSV文件时）
        注意：self.num_registers 应该在调用此方法前已经设置
        """
        # 确保使用传入的 num_registers，而不是硬编码的1000
        # self.num_registers 已经在 __init__ 中设置，这里不需要重新设置
        
        # --- 原始设计基准值 ---
        self.Area_orig = 100000.0
        self.Power_orig = 50.0
        # 最差时序裕量应基于生成的寄存器slack赋值为最小值
        # 先给一个占位值，稍后在生成slack后覆盖
        self.WNS_orig = 0.05
        # 默认频率（周期的倒数），单位与 slack 一致，默认 200MHz
        self.frequency = getattr(self, 'frequency', 200e6)
        # self.SER_orig 将在生成敏感度后根据公式计算
        
        # --- 创建虚拟的寄存器属性 ---
        np.random.seed(42)
        
        # 生成寄存器slack信息
        slack = np.random.normal(loc=0.1, scale=0.05, size=self.num_registers)
        is_critical = np.random.choice([0, 1], self.num_registers, p=[0.9, 0.1])
        slack[is_critical == 1] *= 0.5
        slack = np.abs(slack)
        
        # 生成随机敏感度（包含一些敏感性为0的寄存器）
        # 假设约30%的寄存器敏感性为0（实际电路中很多寄存器可能不敏感）
        zero_sensitivity_ratio = 0.3  # 30%的寄存器敏感性为0
        num_zero_sensitivity = int(self.num_registers * zero_sensitivity_ratio)
        num_nonzero_sensitivity = self.num_registers - num_zero_sensitivity
        
        # 生成敏感性：一部分为0，一部分在(0, 1)之间
        sensitivity = np.zeros(self.num_registers)
        # 非零敏感性在(0, 1)之间随机分布
        sensitivity[:num_nonzero_sensitivity] = np.random.uniform(0.01, 0.99, num_nonzero_sensitivity)
        # 打乱顺序，使0值随机分布
        np.random.shuffle(sensitivity)
        
        # 使用简单均值方法计算整体敏感性（基于错误率）
        # 整体敏感性 = 所有寄存器敏感性的均值 = (x1 + x2 + ... + xK) / K
        self.SER_orig = float(np.sum(sensitivity) / self.num_registers) if self.num_registers > 0 else 0.0

        # 将 WNS_orig 设置为生成的 slacks 的最小值，确保整体最差时序与各路径一致
        try:
            self.WNS_orig = float(np.min(slack))
        except Exception:
            pass
        
        print(f"生成的寄存器敏感度统计:")
        print(f"  总寄存器数: {self.num_registers}")
        print(f"  敏感性为0的寄存器数: {num_zero_sensitivity}")
        print(f"  敏感性非0的寄存器数: {num_nonzero_sensitivity}")
        print(f"  敏感性范围: [{np.min(sensitivity):.4f}, {np.max(sensitivity):.4f}]")
        ## print(f"  根据公式计算的整体敏感性 (Σs_i/n): {self.SER_orig:.6f}")
        
        self.all_reg_properties = {
            'sensitivity': sensitivity,
            'is_critical': is_critical,
            'slack': slack,
            'bitwidth': np.random.choice([8, 16, 32, 64], self.num_registers, p=[0.4, 0.3, 0.2, 0.1]),
            'toggle_rate': np.random.rand(self.num_registers),
            'area': np.random.normal(20.0, 5.0, self.num_registers),
            'power': np.random.normal(0.01, 0.002, self.num_registers)
        }
        # 计算并填充基于 sensitivity 权重的默认 failure_prob（当没有 CSV 时使用）
        try:
            sens_sum_def = np.sum(sensitivity) if np.sum(sensitivity) > 0 else 1.0
            sensitivity_weights_def = sensitivity / sens_sum_def
            computed_failure_probs_def = sensitivity_weights_def * float(getattr(self, 'base_total_failure_rate', 1e-2))
            self.all_reg_properties['failure_prob'] = computed_failure_probs_def
            self.all_reg_properties['failure_prob_source'] = 'computed'
            self.design_failure_prob = float(np.sum(computed_failure_probs_def))
            self.sens_orig = self.design_failure_prob
        except Exception:
            self.all_reg_properties['failure_prob'] = np.full(self.num_registers, np.nan)
            self.all_reg_properties['failure_prob_source'] = 'none'
            self.design_failure_prob = None
    
    def evaluate(self, chromosome_k: np.ndarray) -> Tuple[float, float, float, float]:
        """
        评估加固方案的绝对 PPA 和敏感度指标。
        """
        # 0. 预处理
        chromosome_k = np.array(chromosome_k).flatten()
        mapped_slacks = self.all_reg_properties['slack'][self.map_indices_array]
        invalid_mask = (mapped_slacks - self.voter_delay) < getattr(self, 'min_slack', 0.0)
        chromosome_k[invalid_mask] = 0
        # 1. 映射加固索引
        active_indices = self.map_indices_array[chromosome_k == 1]
        tmr_count = len(active_indices)
        
        # ---------------- A. 面积计算 (Area) ----------------
        if tmr_count > 0:
            # 这里的逻辑：每个加固点增加 2 倍原寄存器面积 + 1 个投票器面积
            original_areas = self.all_reg_properties['area'][active_indices]
            # 计算总增量
            area_increase = np.sum(original_areas * 2) + (self.voter_area * tmr_count)
            area = self.Area_orig + area_increase
        else:
            area = self.Area_orig

        # ---------------- B. 功耗计算 (Power) ----------------
        if tmr_count > 0:
            # 这里的逻辑：每个加固点增加 2 倍原寄存器功耗 + 1 个投票器功耗
            original_powers = self.all_reg_properties['power'][active_indices]
            # 计算总增量
            power_increase = np.sum(original_powers * 2) + (self.voter_power * tmr_count)
            power = self.Power_orig + power_increase
        else:
            power = self.Power_orig

        # ---------------- C. 时序计算 (WNS) ----------------
        # 使用 .copy()，否则多次调用 evaluate 会不断扣减原始数据，导致出现负值
        current_all_slacks = self.all_reg_properties['slack'].copy()
        
        if tmr_count > 0:
            # 物理真实逻辑：只要路径上加了 TMR，该路径的延迟就一定会增加（Slack 减少）
            current_all_slacks[active_indices] -= self.voter_delay
            
        # 全电路最差 Slack = 原始 3000 个路径（未加固路径）与（加固后路径）的全局最小值
        wns = np.min(current_all_slacks)
        
        # 额外的物理一致性保护：WNS 理论上绝对不能超过原始设计值
        wns = min(wns, self.WNS_orig)

        # ---------------- D. 敏感性计算 (Sensitivity/SER) ----------------
        # 构建全电路的掩码 (0/1 向量)
        full_solution = np.zeros(self.num_registers)
        full_solution[active_indices] = 1
        
        # 只保留未加固寄存器（gene=0）的敏感度
        unprotected_mask = (full_solution == 0)
        # SER = (未加固寄存器敏感性和) / (原始寄存器数 + 2 * 加固数量)
        # 注意：TMR加固引入2个新寄存器，分母需要加上 2 * tmr_count
        # 与 greedy_hardening.py 中的 FairEvaluator.evaluate_matrix 保持一致
        ser = np.sum(self.all_reg_properties['sensitivity'][unprotected_mask]) / (self.num_registers + 2 * tmr_count)

        return float(area), float(power), float(wns), float(ser)
    
    def generate_register_info_csv(self, file_path: str):
        """
        生成寄存器信息CSV文件
        
        Args:
            file_path: CSV文件保存路径
        """
        import csv
        
        with open(file_path, 'w', newline='') as csvfile:
            # 写入整体设计指标（增加 frequency 字段）
            writer = csv.writer(csvfile)
            # 标注单位：worst_slack 与 path_slack 使用纳秒 (ns)，frequency 使用 MHz
            writer.writerow(['total_area', 'total_power', 'worst_slack(ns)', 'design_sensitivity', 'frequency(MHz)'])
            # 精度与下面的数据列一致：worst_slack(ns) 保留 6 位小数，design_sensitivity 保留 4 位小数
            writer.writerow([self.Area_orig, self.Power_orig, round(self.WNS_orig, 6), round(self.SER_orig, 4), round(self.frequency / 1e6, 6)])
            writer.writerow([])  # 空行
            
            # 写入寄存器数据标题
            writer.writerow(['index', 'instance_name', 'reg_type', 'area', 'path_power', 'path_slack(ns)', 'sensitivity'])
            
            # 写入每个寄存器的数据
            for i in range(self.num_registers):
                # 生成虚拟的实例名和寄存器类型
                instance_name = f"reg_{i}"
                reg_type = "FF"
                
                # 假定内部 slack 单位已为纳秒 (ns)。若不是，请在生成 CSV 前转换。
                raw_slack = float(self.all_reg_properties['slack'][i])
                writer.writerow([
                    i,
                    instance_name,
                    reg_type,
                    round(float(self.all_reg_properties['area'][i]), 2),
                    round(float(self.all_reg_properties['power'][i]), 4),
                    round(raw_slack, 6),
                    round(float(self.all_reg_properties['sensitivity'][i]), 4)
                ])
        
        print(f"寄存器信息CSV文件已生成: {file_path}")
        print(f"共生成 {self.num_registers} 个寄存器的信息")
        print(f"整体设计指标:")
        print(f"  total_area: {self.Area_orig}")
        print(f"  total_power: {self.Power_orig}")
        print(f"  worst_slack: {self.WNS_orig}")
        # print(f"  design_sensitivity: {self.SER_orig:.6f} (根据公式 s_all = Σs_i/n 计算)")
        
        # 验证生成的CSV文件中的整体敏感性是否正确  DEBUG
        ## generated_sensitivity_sum = np.sum(self.all_reg_properties['sensitivity'])
        ## calculated_sensitivity = generated_sensitivity_sum / self.num_registers
        ## print(f"验证: 生成的寄存器敏感度总和 = {generated_sensitivity_sum:.6f}")
        ## print(f"验证: 计算的整体敏感性 (Σs_i/n) = {calculated_sensitivity:.6f}")
        ## print(f"验证: 与SER_orig一致: {abs(calculated_sensitivity - self.SER_orig) < 1e-6}")