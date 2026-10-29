# mo_optimizer/gpu_evaluator.py
import numpy as np
import torch
import time
from typing import List, Tuple, Dict, Any
from algorithms.core_data import Individual


class GPUAcceleratedEvaluator:
    """GPU加速的并行评估器"""
    
    def __init__(self, simulator, device=None, batch_size=512, config=None):
        """
        初始化GPU评估器
        
        参数:
        ----------
        simulator : ICDesignSimulator
            模拟器对象
        device : str, optional
            设备 ('cuda' 或 'cpu')
        batch_size : int
            GPU批处理大小
        config : dict, optional
            配置字典（包含约束阈值 constraint_thresholds）
        """
        self.simulator = simulator
        self.config = config or {}
        
        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print("GPU评估器: 使用CUDA加速")
            else:
                device = 'cpu'
                print("GPU评估器: CUDA不可用，使用CPU")
        
        self.device = torch.device(device)
        
        # 获取原始值
        self.area_orig = getattr(simulator, 'Area_orig', 100000.0)
        self.power_orig = getattr(simulator, 'Power_orig', 50.0)
        self.slack_orig = getattr(simulator, 'WNS_orig', 0.05)
        self.sens_orig = getattr(simulator, 'sens_orig', getattr(simulator, 'SER_orig', 1e-5))
        
        # 获取约束阈值（从配置读取）
        constraints = self.config.get('constraint_thresholds', {})
        self.delta_area_max = constraints.get('delta_area_max', 0.8)
        self.delta_power_max = constraints.get('delta_power_max', 1.3)
        self.slack_min = constraints.get('slack_min', 0.0)
        self.delta_sens_min = constraints.get('delta_sens_min', 0.3)  # 从配置读取，而不是硬编码 0.5

        # 获取 voter 参数（从配置读取，兼容 'delay' 和 'voter_delay'）
        voter_cfg = self.config.get('voter', {})
        self.voter_delay = voter_cfg.get('delay', voter_cfg.get('voter_delay', 0.2))
        self.voter_area = voter_cfg.get('area', voter_cfg.get('voter_area', 10.0))
        self.voter_power = voter_cfg.get('power', voter_cfg.get('voter_power', 0.02))
        
        # 获取寄存器属性
        self.K = getattr(simulator, 'K', 200)
        
        # 【关键修复】：获取总寄存器数（用于敏感性计算的正确分母）
        # 与 greedy_hardening.py 中的 FairEvaluator 保持一致
        self.total_num_registers = getattr(simulator, 'total_num_registers', self.K)
        
        # 获取所有寄存器的敏感性总和
        self.total_sensitivity_sum = getattr(simulator, 'total_sensitivity_sum', None)
        if self.total_sensitivity_sum is None:
            # 如果没有提供，尝试从 model 获取（RealMoeaWrapper 的情况）
            if hasattr(simulator, 'model'):
                self.total_sensitivity_sum = getattr(simulator.model, 'total_sensitivity_sum', None)
            if self.total_sensitivity_sum is None:
                # 最后后备：使用预筛选寄存器的敏感性总和
                self.total_sensitivity_sum = float(np.sum(getattr(simulator, 'sensitivity', np.array([0.0]))))
        
        # 获取预筛选寄存器属性并转移到GPU
        self._prepare_gpu_data(simulator)
        
        # 批处理设置
        self.batch_size = batch_size
        
        # 性能统计
        self.eval_count = 0
        self.total_time = 0.0
        self.gpu_memory_usage = 0.0
        
        # print(f"GPU评估器初始化完成: 设备={self.device}, 批处理大小={batch_size}")
        
        # 显示GPU信息
        ## if self.device.type == 'cuda':
        ##     print(f"GPU名称: {torch.cuda.get_device_name(self.device)}")
        ##     print(f"GPU内存: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f}GB")
    
    def _prepare_gpu_data(self, simulator):
        """准备GPU数据"""
        # 获取寄存器属性
        if hasattr(simulator, 'all_reg_properties') and simulator.all_reg_properties:
            # 检查是否有必需的键
            required_keys = ['area', 'power', 'slack', 'sensitivity', 'is_critical']
            if all(key in simulator.all_reg_properties for key in required_keys):
                # 获取预筛选寄存器的属性
                if hasattr(simulator, 'SETFF_indices') and simulator.SETFF_indices is not None:
                    indices = simulator.SETFF_indices
                    self.area = simulator.all_reg_properties['area'][indices]
                    self.power = simulator.all_reg_properties['power'][indices]
                    self.slack = simulator.all_reg_properties['slack'][indices]
                    self.sensitivity = simulator.all_reg_properties['sensitivity'][indices]
                    self.is_critical = simulator.all_reg_properties['is_critical'][indices]
                else:
                    # 如果没有预筛选，使用所有寄存器
                    self.area = simulator.all_reg_properties['area']
                    self.power = simulator.all_reg_properties['power']
                    self.slack = simulator.all_reg_properties['slack']
                    self.sensitivity = simulator.all_reg_properties['sensitivity']
                    self.is_critical = simulator.all_reg_properties['is_critical']
                
                # 确保数据类型正确
                self.area = np.asarray(self.area, dtype=np.float32)
                self.power = np.asarray(self.power, dtype=np.float32)
                self.slack = np.asarray(self.slack, dtype=np.float32)
                self.sensitivity = np.asarray(self.sensitivity, dtype=np.float32)
                self.is_critical = np.asarray(self.is_critical, dtype=np.float32)
            else:
                # 缺少必需的键，使用后备方案
                missing_keys = [key for key in required_keys if key not in simulator.all_reg_properties]
                print(f"警告: all_reg_properties缺少键: {missing_keys}，使用后备方案")
                self._use_fallback_data(simulator)
                return  # 后备方案已经创建了GPU张量
        else:
            # 后备：使用模拟器的直接属性
            self._use_fallback_data(simulator)
            return  # 后备方案已经创建了GPU张量
        
        # 转换到GPU（只有在使用all_reg_properties路径时才执行到这里）
        self._create_gpu_tensors()
    
    def _use_fallback_data(self, simulator):
        """使用后备数据方案"""
        # 尝试从model属性获取（RealMoeaWrapper的情况）
        if hasattr(simulator, 'model'):
            model = simulator.model
            self.area = np.asarray(getattr(model, 'area', np.ones(self.K)), dtype=np.float32)
            self.power = np.asarray(getattr(model, 'power', np.ones(self.K) * 0.5), dtype=np.float32)
            self.slack = np.asarray(getattr(model, 'slack', np.ones(self.K) * 0.1), dtype=np.float32)
            self.sensitivity = np.asarray(getattr(model, 'sensitivity', np.random.rand(self.K)), dtype=np.float32)
            self.is_critical = np.asarray(getattr(model, 'is_critical', np.zeros(self.K)), dtype=np.float32)
        else:
            # 最后的后备方案：使用默认值
            self.area = np.ones(self.K, dtype=np.float32)
            self.power = np.ones(self.K, dtype=np.float32) * 0.5
            self.slack = np.ones(self.K, dtype=np.float32) * 0.1
            self.sensitivity = np.random.rand(self.K).astype(np.float32)
            self.is_critical = np.zeros(self.K, dtype=np.float32)
        
        # 确保数据类型正确
        self.area = np.asarray(self.area, dtype=np.float32)
        self.power = np.asarray(self.power, dtype=np.float32)
        self.slack = np.asarray(self.slack, dtype=np.float32)
        self.sensitivity = np.asarray(self.sensitivity, dtype=np.float32)
        self.is_critical = np.asarray(self.is_critical, dtype=np.float32)
        
        # 转换到GPU
        self._create_gpu_tensors()
    
    def _create_gpu_tensors(self):
        """创建GPU张量"""
        # 转换到GPU
        self.area_gpu = torch.tensor(self.area, device=self.device)
        self.power_gpu = torch.tensor(self.power, device=self.device)
        self.slack_gpu = torch.tensor(self.slack, device=self.device)
        self.sensitivity_gpu = torch.tensor(self.sensitivity, device=self.device)
        self.is_critical_gpu = torch.tensor(self.is_critical, device=self.device)
        
        # 设计基准值
        self.design_area = torch.tensor(self.area_orig, device=self.device)
        self.design_power = torch.tensor(self.power_orig, device=self.device)
        
        # TMR参数 - 使用从配置读取的值（在 __init__ 中已设置）
        self.voter_area = torch.tensor(float(self.voter_area), device=self.device)
        self.voter_power = torch.tensor(float(self.voter_power), device=self.device)
        self.voter_delay = torch.tensor(float(self.voter_delay), device=self.device)
        
        print(f"数据准备完成: K={len(self.area_gpu)}")
    
    def evaluate_batch_gpu(self, genes_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GPU批量评估
        
        参数:
        ----------
        genes_batch : np.ndarray
            形状为 (batch_size, K) 的基因批次
            
        返回:
        ----------
        tuple: (area, power, wns, ser) 四个numpy数组
        """
        if len(genes_batch) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # GPU内存使用统计（仅在CUDA设备上）
        if self.device.type == 'cuda':
            mem_before = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        
        # 转换为GPU张量
        genes_tensor = torch.tensor(genes_batch, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # 1. 面积计算: area_total = design_area + sum(bitvec * (area*2 + voter_area))
            area_cost_per_reg = self.area_gpu * 2 + self.voter_area
            area_increase = torch.matmul(genes_tensor, area_cost_per_reg)
            area_total = self.design_area + area_increase
            
            # 2. 功耗计算: power_total = design_power + sum(bitvec * (power*2 + voter_power))
            power_cost_per_reg = self.power_gpu * 2 + self.voter_power
            power_increase = torch.matmul(genes_tensor, power_cost_per_reg)
            power_total = self.design_power + power_increase
            
            # 3. 时序计算: slack_after = slack - voter_delay * bitvec
            # 【修复】：所有被加固的寄存器都扣除 voter_delay（不限于关键路径）
            # TMR加固会在每个被加固的寄存器上添加投票器，增加路径延迟
            slack_after = self.slack_gpu.unsqueeze(0) - self.voter_delay * genes_tensor
            # 找到每个个体的最差slack
            min_slack, _ = torch.min(slack_after, dim=1)
            
            # 4. 敏感性计算: 加固后敏感性 = (Σ原始敏感性 - Σ被加固的敏感性) / (n + 2*t)
            # 其中 n = 总寄存器数，t = 被加固的寄存器数
            sensitivity_sum = torch.matmul(genes_tensor, self.sensitivity_gpu)  # Σ被加固的敏感性
            total_sensitivity = self.total_sensitivity_sum  # Σ原始敏感性（所有寄存器）
            num_reinforced = torch.sum(genes_tensor, dim=1)  # 每个个体被加固的寄存器数
            # 【关键修复】：使用 total_num_registers（所有寄存器数），而不是 K（预筛选寄存器数）
            # 与 greedy_hardening.py 中的 FairEvaluator.evaluate_matrix 保持一致
            new_denominator = self.total_num_registers + 2.0 * num_reinforced
            # 加固后的总敏感性（绝对值）
            remaining_sensitivity = total_sensitivity - sensitivity_sum
            # 加固后的平均敏感性（基于新的寄存器数）
            ser = remaining_sensitivity / new_denominator
            
        # GPU内存使用统计（仅在CUDA设备上）
        if self.device.type == 'cuda':
            mem_after = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            mem_peak = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
            # 每100次评估输出一次统计信息
            self.eval_count += len(genes_batch)
            ## if self.eval_count % 100 == 0:
            ##     print(f"GPU评估统计: 已评估{self.eval_count}个个体, "
            ##           f"GPU内存: {mem_after:.1f}MB (峰值: {mem_peak:.1f}MB)")
        
        # 返回CPU上的numpy数组
        return (
            area_total.cpu().numpy(),
            power_total.cpu().numpy(),
            min_slack.cpu().numpy(),
            ser.cpu().numpy()
        )
    
    def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """评估整个种群 - GPU加速版本"""
        if not population:
            return []
        
        start_time = time.time()
        
        # 1. 提取所有基因
        genes_list = []
        valid_indices = []
        
        for i, ind in enumerate(population):
            if ind.gene is not None:
                genes_list.append(ind.gene.flatten())
                valid_indices.append(i)
        
        if not genes_list:
            return population
        
        genes_batch = np.array(genes_list)
        
        # 2. GPU批量评估
        area_batch, power_batch, slack_batch, ser_batch = self.evaluate_batch_gpu(genes_batch)
        
        # 3. 更新个体目标值
        for idx, batch_idx in enumerate(valid_indices):
            ind = population[batch_idx]
            area = area_batch[idx]
            power = power_batch[idx]
            slack = slack_batch[idx]
            ser = ser_batch[idx]
            
            ind.objectives, ind.violation, ind.area_power_feasible, ind.original_objectives = self._calculate_objectives_and_violation(area, power, slack, ser)
        
        # 4. 性能统计
        elapsed = time.time() - start_time
        self.eval_count += len(population)
        self.total_time += elapsed
        
        # GPU内存使用统计
        if self.device.type == 'cuda':
            self.gpu_memory_usage = torch.cuda.memory_allocated(self.device) / 1024**3
        
        # 打印性能信息
        # speed = len(population) / elapsed if elapsed > 0 else 0
        # print(f"GPU评估: {len(population)}个个体, 耗时{elapsed:.3f}秒, 速度{speed:.1f}个体/秒")
        # if self.device.type == 'cuda':
        #     print(f"  GPU内存使用: {self.gpu_memory_usage:.3f}GB")
        
        return population
    
    def _calculate_objectives_and_violation(self, area: float, power: float, wns: float, ser: float):
        """计算目标值和约束违例
        
        参数:
        -----
        wns : float
            最差 slack（ns），应该 >= 0
            - wns < 0: 时序违例（绝对不允许）
            - wns >= 0: 时序可行
        """
        if np.isnan(area) or np.isnan(power) or np.isnan(wns) or np.isnan(ser):
            area, power, wns, ser = 1e6, 1e6, -1e6, 1e6
        
        area_ratio = area / (self.area_orig + 1e-10)
        power_ratio = power / (self.power_orig + 1e-10)
        
        # 【关键修复】：检查 ser 是否已经是归一化比例值（0~1）
        # PPAModel.evaluate 返回的 normalized_sensitivity 是 0~1 的比例值
        # 如果 ser 在 0~1 之间，说明已经是归一化值，不需要再除
        # 如果 ser > 1，说明是绝对敏感性值，需要归一化
        if 0 <= ser <= 1.0:
            ser_ratio = ser  # 已经是归一化比例值
        else:
            ser_ratio = ser / (self.sens_orig + 1e-10)  # 绝对值，需要归一化
        
        delta_area = (area - self.area_orig) / (self.area_orig + 1e-10)
        delta_power = (power - self.power_orig) / (self.power_orig + 1e-10)
        delta_slack = self.slack_orig - wns
        
        area_viol = max(0, delta_area - self.delta_area_max)
        power_viol = max(0, delta_power - self.delta_power_max)
        
        # 时序约束：对负 slack（时序违例）施加最强惩罚
        # wns 为最差 slack 值，应该 >= slack_min（通常为 0）
        if wns < self.slack_min:
            # 时序违例：每负化 1ps 加 100x 惩罚（相当于数千倍的其他约束）
            slack_viol = abs(wns - self.slack_min) * 100.0  # 非常严厉：低于阈值越多，惩罚越重
        else:
            # 非违例情况：传统的约束检查
            slack_viol = max(0, delta_slack - 0.05)
        
        # 敏感性约束：使用归一化敏感性比例
        # ser_ratio 是 0~1 的归一化值，delta_sens_min=0.2 表示敏感性不能超过 20%
        sens_viol = max(0.0, ser_ratio - self.delta_sens_min)
        
        total_violation = (0.1 * area_viol + 
                         0.1 * power_viol + 
                         1000.0 * slack_viol + 
                         0.5 * sens_viol)
        
        # 保留原始 wns（用于输出），但目标值应该对负 slack 进行惩罚。
        # 之前的实现在 wns < 0 时将目标设为 0，会误导优化（把违例当作好值）。
        # 修复：wns<0时，wns_objective设为一个很大的正值，使MOEA自动淘汰时序违例解
        if wns < self.slack_min:
            # 时序违例：wns_objective设为一个很大的正值，确保MOEA认为这是极差的解
            wns_objective = 1e6  # 极大正值，MOEA最小化时会自动淘汰
        else:
            wns_objective = -wns  # 正常情况：wns越大越好，对应目标值越小
        
        objectives = [
            area_ratio,
            power_ratio,
            wns_objective,
            ser_ratio
        ]
        
        # 检查面积或功耗是否有约束违例（这两个目标MOEA需要最小化）
        has_area_power_violation = (area_viol > 0) or (power_viol > 0)
        
        # 【关键修复】：如果面积或功耗超过阈值，直接将惩罚应用到所有目标值上，
        # 但同时保存原始目标值用于约束违例判断
        original_objectives = objectives.copy()
        if has_area_power_violation:
            penalty_multiplier = 100.0
            objectives = [o * penalty_multiplier for o in objectives]
        
        objectives = [0.0 if np.isnan(x) else x for x in objectives]
        
        return np.array(objectives), total_violation, not has_area_power_violation, original_objectives
    
    def evaluate_individual(self, individual: Individual) -> Individual:
        """评估单个个体"""
        if individual.gene is None:
            return individual
        
        # 转换为批次格式
        genes_batch = np.array([individual.gene.flatten()])
        
        # GPU评估
        area, power, slack, ser = self.evaluate_batch_gpu(genes_batch)
        
        if len(area) > 0:
            individual.objectives, individual.violation, individual.area_power_feasible, individual.original_objectives = self._calculate_objectives_and_violation(
                area[0], power[0], slack[0], ser[0]
            )
        
        return individual
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            #print("GPU缓存已清理")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        avg_time = self.total_time / self.eval_count if self.eval_count > 0 else 0
        
        return {
            'eval_count': self.eval_count,
            'total_time': self.total_time,
            'avg_time_per_eval': avg_time,
            'gpu_memory_usage_gb': self.gpu_memory_usage,
            'device': str(self.device)
        }
    
    def __del__(self):
        """析构函数，清理资源"""
        self.clear_cache()