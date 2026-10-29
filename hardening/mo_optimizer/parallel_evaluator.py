# mo_optimizer/parallel_evaluator.py
import numpy as np
from multiprocessing import Pool, cpu_count, get_context
from typing import List, Tuple, Dict, Any
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from algorithms.core_data import Individual

# 全局进程池管理器
class PoolManager:
    _instance = None
    _pool = None
    _pool_size = 1
    
    def __new__(cls, pool_size=None):
        if cls._instance is None:
            cls._instance = super(PoolManager, cls).__new__(cls)
            if pool_size is not None:
                cls._pool_size = max(1, pool_size)
        return cls._instance
    
    @classmethod
    def get_pool(cls, pool_size=None):
        if pool_size is not None:
            cls._pool_size = max(1, pool_size)
        
        if cls._pool is None:
            print(f"PoolManager: 创建进程池，大小={cls._pool_size}")
            try:
                # 使用 spawn 上下文创建进程池
                ctx = get_context('spawn')
                cls._pool = ctx.Pool(processes=cls._pool_size, maxtasksperchild=50)
            except Exception as e:
                print(f"PoolManager: 创建进程池失败: {e}")
                cls._pool = None
        
        return cls._pool
    
    @classmethod
    def close_pool(cls):
        if cls._pool is not None:
            try:
                cls._pool.close()
                cls._pool.join()
                cls._pool = None
                print("PoolManager: 进程池已关闭")
            except:
                pass


# 模块级函数，用于子进程执行
def evaluate_batch_worker(args):
    """工作函数：在子进程中评估一批基因"""
    batch_data, simulator_method = args
    
    try:
        # 调用模拟器方法
        results = []
        for genes in batch_data['genes']:
            try:
                # 使用传递过来的模拟器方法
                if simulator_method == 'evaluate':
                    result = batch_data['simulator'].evaluate(genes)
                else:
                    # 其他方法处理
                    result = [1e6, 1e6, -1e6, 1e6]
                
                # 处理结果
                if isinstance(result, tuple):
                    if len(result) >= 4:
                        results.append((result[0], result[1], result[2], result[3]))
                        continue
                elif isinstance(result, np.ndarray):
                    if result.shape[0] >= 4:
                        results.append((result[0], result[1], result[2], result[3]))
                        continue
                elif isinstance(result, list):
                    if len(result) >= 4:
                        results.append((result[0], result[1], result[2], result[3]))
                        continue
                
                # 默认值
                results.append((1e6, 1e6, -1e6, 1e6))
            except Exception as e:
                print(f"子进程评估失败: {e}")
                results.append((1e6, 1e6, -1e6, 1e6))
        
        return results
    except Exception as e:
        print(f"工作函数执行失败: {e}")
        return [(1e6, 1e6, -1e6, 1e6)] * len(batch_data['genes'])


class ParallelEvaluator:
    """高性能并行评估器 - 异步优化版本"""
    
    def __init__(self, simulator, num_processes: int = None, batch_size: int = None, 
                 use_gpu: bool = False, evaluator_id: str = None, config: dict = None):
        """
        初始化评估器
        
        参数:
        ----------
        use_gpu : bool
            是否使用GPU模式
        evaluator_id : str, optional
            评估器标识符（用于区分不同的评估器实例，如'spea2'或'nsga2'）
        config : dict, optional
            配置字典（包含约束阈值等信息）
        """
        self.simulator = simulator
        self.use_gpu = use_gpu
        self.evaluator_id = evaluator_id or 'default'
        self.config = config or {}
        
        if use_gpu:
            # GPU模式：使用集中式 GPU worker 进程，避免多个进程各自创建 CUDA 上下文
            from .gpu_worker import GPUWorker
            # print("ParallelEvaluator: 使用集中式GPU worker模式")

            # 准备传递到 GPU worker 的模拟器属性（轻量化，避免直接传 simulator 对象）
            # 检查是否是RealMoeaWrapper（有model属性）
            if hasattr(simulator, 'model'):
                # RealMoeaWrapper的情况：从model中提取数据
                model = simulator.model
                all_reg_properties = {
                    'area': np.asarray(model.area, dtype=np.float32),
                    'power': np.asarray(model.power, dtype=np.float32),
                    'slack': np.asarray(model.slack, dtype=np.float32),
                    'sensitivity': np.asarray(model.sensitivity, dtype=np.float32),
                    'is_critical': np.asarray(model.is_critical, dtype=np.float32)
                }
                K = getattr(model, 'K', len(model.area))
                # 【关键修复】：获取 total_num_registers 和 total_sensitivity_sum（用于敏感性计算）
                total_num_registers = getattr(model, 'total_num_registers', K)
                total_sensitivity_sum = float(getattr(model, 'total_sensitivity_sum', np.sum(model.sensitivity)))
                Area_orig = getattr(model, 'area_orig', getattr(model, 'design_area', 100000.0))
                Power_orig = getattr(model, 'power_orig', getattr(model, 'design_power', 50.0))
                WNS_orig = getattr(model, 'slack_orig', getattr(model, 'design_slack', 0.05))
                sens_orig = getattr(model, 'sens_orig', getattr(model, 'design_sensitivity', 1e-5))
                SETFF_indices = None  # RealMoeaWrapper已经预筛选过了
            else:
                # 标准模拟器的情况：直接使用all_reg_properties
                all_reg_properties = getattr(simulator, 'all_reg_properties', {})
                K = getattr(simulator, 'K', None)
                # 【关键修复】：获取 total_num_registers 和 total_sensitivity_sum
                total_num_registers = getattr(simulator, 'total_num_registers', K if K is not None else len(all_reg_properties.get('area', [])))
                total_sensitivity_sum = float(getattr(simulator, 'total_sensitivity_sum', np.sum(all_reg_properties.get('sensitivity', [0.0]))))
                Area_orig = getattr(simulator, 'Area_orig', getattr(simulator, 'design_area', 100000.0))
                Power_orig = getattr(simulator, 'Power_orig', getattr(simulator, 'design_power', 50.0))
                WNS_orig = getattr(simulator, 'WNS_orig', getattr(simulator, 'design_slack', 0.05))
                sens_orig = getattr(simulator, 'sens_orig', getattr(simulator, 'design_sensitivity', 1e-5))
                SETFF_indices = getattr(simulator, 'SETFF_indices', None)
            
            sim_props = {
                'all_reg_properties': all_reg_properties,
                'SETFF_indices': SETFF_indices,
                'K': K,
                'total_num_registers': total_num_registers,  # 【关键修复】：添加总寄存器数
                'total_sensitivity_sum': total_sensitivity_sum,  # 【关键修复】：添加总敏感性和
                'Area_orig': Area_orig,
                'Power_orig': Power_orig,
                'WNS_orig': WNS_orig,
                'sens_orig': sens_orig
            }
            
            # 验证数据完整性
            if not all_reg_properties or 'area' not in all_reg_properties:
                raise ValueError(f"无法从模拟器提取寄存器属性。simulator类型: {type(simulator)}, "
                               f"has model: {hasattr(simulator, 'model')}, "
                               f"has all_reg_properties: {hasattr(simulator, 'all_reg_properties')}")
            # 启动 GPU worker
            # 使用传入的batch_size，如果没有则使用默认值512
            gpu_batch_size = batch_size if batch_size is not None else 512
            try:
                self.gpu_worker = GPUWorker(sim_props, batch_size=gpu_batch_size, config=self.config)
                self.use_gpu = True
                worker_name = f"[{self.evaluator_id}]" if self.evaluator_id != 'default' else ""
                # print(f"ParallelEvaluator{worker_name}: GPU worker已启动，批处理大小={gpu_batch_size}")
            except Exception as e:
                print(f"无法启动 GPU worker，回退到 CPU 模式: {e}")
                import traceback
                traceback.print_exc()
                self.use_gpu = False

            # GPU 模式下不使用额外进程
            self.num_processes = 1
            self.batch_size = gpu_batch_size  # GPU批处理大小
            
            # 【修复】：从配置中读取约束阈值
            constraints = self.config.get('constraint_thresholds', {})
            self.delta_area_max = constraints.get('delta_area_max', 0.2)
            self.delta_power_max = constraints.get('delta_power_max', 0.15)
            self.slack_min_threshold = constraints.get('slack_min', 0.0)
            self.delta_sens_min = constraints.get('delta_sens_min', 0.5)
            
            # 获取原始值（尝试多种可能的属性名以提高兼容性）
            def _first_attr(obj, names, default):
                for n in names:
                    if hasattr(obj, n):
                        return getattr(obj, n)
                return default

            self.area_orig = _first_attr(simulator, ['Area_orig', 'area_orig', 'design_area', 'base_area'], 100000.0)
            self.power_orig = _first_attr(simulator, ['Power_orig', 'power_orig', 'design_power', 'base_power'], 50.0)
            self.slack_orig = _first_attr(simulator, ['WNS_orig', 'slack_orig', 'design_slack', 'base_slack'], 0.05)
            self.sens_orig = _first_attr(simulator, ['sens_orig', 'SER_orig', 'design_sensitivity', 'base_sens'], 1e-5)
            
        else:
            # CPU并行模式（原有逻辑）
            from multiprocessing import cpu_count as get_cpu_count
            available_cpus = get_cpu_count()

            try:
                import torch
                has_gpu = torch.cuda.is_available()
            except ImportError:
                has_gpu = False
            
            # 设置进程数
            if num_processes is None:
                if has_gpu:
                    self.num_processes = max(1, available_cpus // 2)
                else:
                    self.num_processes = max(1, available_cpus - 2)
            else:
                self.num_processes = min(num_processes, available_cpus)
            
            # print(f"CPU核心数: {available_cpus}, 使用进程数: {self.num_processes}")
            # if has_gpu:
            #     print(f"检测到GPU，调整CPU进程数以平衡资源")
            
            # 动态调整批处理大小
            if batch_size is None:
                if has_gpu:
                    self.batch_size = max(2, 50 // self.num_processes)
                else:
                    self.batch_size = max(5, 100 // self.num_processes)
            else:
                self.batch_size = batch_size
            
            # 获取原始值（兼容多种模拟器实现）
            def _first_attr(obj, names, default):
                for n in names:
                    if hasattr(obj, n):
                        return getattr(obj, n)
                return default

            self.area_orig = _first_attr(simulator, ['Area_orig', 'area_orig', 'design_area', 'base_area'], 100000.0)
            self.power_orig = _first_attr(simulator, ['Power_orig', 'power_orig', 'design_power', 'base_power'], 50.0)
            self.slack_orig = _first_attr(simulator, ['WNS_orig', 'slack_orig', 'design_slack', 'base_slack'], 0.05)
            self.sens_orig = _first_attr(simulator, ['sens_orig', 'SER_orig', 'design_sensitivity', 'base_sens'], 1e-5)

            # 【修复】：从配置中读取约束阈值
            constraints = self.config.get('constraint_thresholds', {})
            self.delta_area_max = constraints.get('delta_area_max', 0.2)
            self.delta_power_max = constraints.get('delta_power_max', 0.15)
            self.slack_min_threshold = constraints.get('slack_min', 0.0)
            self.delta_sens_min = constraints.get('delta_sens_min', 0.5)

            print(f"ParallelEvaluator: 配置 {self.num_processes} 个进程，批次大小 {self.batch_size}")

        # 内存监控
        from utils.memory_saver import memory_saver
        self.memory_saver = memory_saver

        # 使用单例进程池，避免重复创建
        self._pool = None

        # 性能统计
        self.eval_count = 0
        self.total_time = 0.0
        self.avg_eval_time = 0.0

    def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """评估种群"""
        if not population:
            return []
        
        # 检查内存使用
        self.memory_saver.check_and_clean()
        
        if self.use_gpu:
            # GPU 模式：把种群的基因打包成 batch 发送给 GPU worker
            genes_list = [ind.gene.flatten() for ind in population if ind.gene is not None]
            if not genes_list:
                return population

            genes_batch = np.array(genes_list)
            try:
                results = self.gpu_worker.evaluate_population(genes_batch)
                # results 为 per-individual tuples
                valid_idx = 0
                for ind in population:
                    if ind.gene is None:
                        continue
                    if valid_idx < len(results):
                        area, power, wns, ser = results[valid_idx]
                        ind.objectives, ind.violation, ind.area_power_feasible, ind.original_objectives = self._calculate_objectives_and_violation(area, power, wns, ser)
                    else:
                        ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                        ind.violation = 1e6
                        ind.area_power_feasible = False
                        ind.original_objectives = np.array([1e6, 1e6, -1e6, 1e6])
                    valid_idx += 1
                result = population
            except Exception as e:
                print(f"GPU worker 评估失败，回退到串行: {e}")
                result = self._serial_evaluate(population)
        else:
            # CPU并行模式（原有逻辑）
            # 根据种群大小决定使用哪种评估方式
            if len(population) <= 5 or self.num_processes <= 1:
                result = self._serial_evaluate(population)
            elif len(population) <= 20:
                result = self._simple_parallel_evaluate(population)
            else:
                result = self._async_parallel_evaluate(population)
        # 评估后清理内存
        self._cleanup_after_evaluation()

        return result
    
    def _cleanup_after_evaluation(self):
        """评估后清理"""
        # 清理进程池
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
                self._pool = None
            except:
                pass
        
        # 清理numpy缓存
        import numpy as np
        if hasattr(np, 'getbuffer'):
            try:
                np.getbuffer  # 触发内部清理
            except:
                pass
        
        # 强制垃圾回收
        import gc
        gc.collect()

    def _async_parallel_evaluate(self, population: List[Individual]) -> List[Individual]:
        """异步并行评估 - 最大化CPU利用率"""
        start_time = time.time()
        
        # 使用ProcessPoolExecutor而不是Pool，提供更好的异步支持
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # 准备任务
            futures = {}
            for idx, ind in enumerate(population):
                # 提交任务到进程池
                future = executor.submit(
                    self._evaluate_single_worker, 
                    ind.gene.copy()
                )
                futures[future] = idx
            
            # 异步收集结果
            results = [None] * len(population)
            completed = 0
            
            # 使用as_completed获取已完成的任务，避免等待
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    area, power, wns, ser = future.result(timeout=60)
                    results[idx] = (area, power, wns, ser)
                    completed += 1
                    
                    # 进度显示
                    ## if completed % 10 == 0:
                    ##     print(f"评估进度: {completed}/{len(population)}")
                        
                except Exception as e:
                    print(f"评估失败: {e}")
                    results[idx] = (1e6, 1e6, -1e6, 1e6)
            
            # 处理结果
            for i, (ind, result) in enumerate(zip(population, results)):
                if result is not None:
                    area, power, wns, ser = result
                    ind.objectives, ind.violation, ind.area_power_feasible, ind.original_objectives = self._calculate_objectives_and_violation(
                        area, power, wns, ser
                    )
                else:
                    # 保护性赋值，确保任何情况下个体都有合理的目标值与违例值
                    ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                    ind.violation = 1e6
                    ind.area_power_feasible = False
                    ind.original_objectives = np.array([1e6, 1e6, -1e6, 1e6])
        
        elapsed = time.time() - start_time
        self.eval_count += len(population)
        self.total_time += elapsed
        
        ## avg_speed = self.eval_count / max(self.total_time, 0.001)
        ## print(f"异步评估完成: {len(population)} 个体, 耗时 {elapsed:.2f} 秒, "
        ##       f"平均速度 {avg_speed:.1f} 个体/秒, CPU效率: {min(100, elapsed*100/(elapsed+0.1)):.0f}%")
        
        return population
    
    def _simple_parallel_evaluate(self, population: List[Individual]) -> List[Individual]:
        """简单的多进程评估 - 适合中等种群"""
        start_time = time.time()
        
        # 使用spawn上下文创建进程池
        ctx = get_context('spawn')
        
        with ctx.Pool(processes=self.num_processes) as pool:
            # 批量提交任务
            tasks = []
            for ind in population:
                tasks.append(pool.apply_async(
                    self._evaluate_single_worker,
                    (ind.gene.copy(),)
                ))
            
            # 收集结果
            for i, (ind, task) in enumerate(zip(population, tasks)):
                try:
                    area, power, wns, ser = task.get(timeout=30)
                    ind.objectives, ind.violation, ind.area_power_feasible, ind.original_objectives = self._calculate_objectives_and_violation(
                        area, power, wns, ser
                    )
                except Exception as e:
                    print(f"评估失败: {e}")
                    ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                    ind.violation = 1e6
                    ind.area_power_feasible = False
                    ind.original_objectives = np.array([1e6, 1e6, -1e6, 1e6])
        
        elapsed = time.time() - start_time
        print(f"简单并行评估完成: {len(population)} 个体, 耗时 {elapsed:.2f} 秒")
        
        return population
    
    def _evaluate_single_worker(self, genes):
        """工作函数 - 在子进程中执行"""
        try:
            result = self.simulator.evaluate(genes)

            # 处理不同的返回格式
            if isinstance(result, tuple):
                # 如果是元组，取前4个值
                if len(result) >= 4:
                    return self._process_result(result[:4])
                else:
                    print(f"警告: evaluate返回元组长度不足: {len(result)}")
                    return (1e6, 1e6, -1e6, 1e6)
            elif isinstance(result, np.ndarray):
                # 如果是数组，取前4个元素
                if result.size >= 4:
                    return self._process_result(result[:4])
                else:
                    print(f"警告: evaluate返回数组长度不足: {result.size}")
                    return (1e6, 1e6, -1e6, 1e6)
            else:
                print(f"警告: evaluate返回未知类型: {type(result)}")
                return (1e6, 1e6, -1e6, 1e6)

        except Exception as e:
            print(f"工作函数评估失败: {e}")
            import traceback
            traceback.print_exc()
            return (1e6, 1e6, -1e6, 1e6)
    
    def _serial_evaluate(self, population: List[Individual]) -> List[Individual]:
        """串行评估 - 用于小种群"""
        start_time = time.time()
        
        for ind in population:
            try:
                result = self.simulator.evaluate(ind.gene)
                area, power, wns, ser = self._process_result(result)
                ind.objectives, ind.violation, ind.area_power_feasible, ind.original_objectives = self._calculate_objectives_and_violation(
                    area, power, wns, ser
                )
            except Exception as e:
                print(f"评估失败: {e}")
                ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                ind.violation = 1e6
                ind.area_power_feasible = False
                ind.original_objectives = np.array([1e6, 1e6, -1e6, 1e6])
        
        elapsed = time.time() - start_time
        if elapsed > 0.5:
            print(f"串行评估完成: {len(population)} 个体, 耗时 {elapsed:.2f} 秒")
        
        return population
    
    @staticmethod
    def _process_result(result) -> Tuple[float, float, float, float]:
        """处理模拟器返回结果，确保返回4个值"""
        try:
            if isinstance(result, tuple):
                if len(result) >= 4:
                    # 确保是浮点数
                    return (float(result[0]), float(result[1]), 
                            float(result[2]), float(result[3]))
            elif isinstance(result, np.ndarray):
                if result.size >= 4:
                    # 取前4个元素
                    if result.ndim > 0:
                        return (float(result[0]), float(result[1]), 
                                float(result[2]), float(result[3]))
                    else:
                        return (float(result), 1e6, -1e6, 1e6)
            elif isinstance(result, list):
                if len(result) >= 4:
                    return (float(result[0]), float(result[1]), 
                            float(result[2]), float(result[3]))
            
            # 默认值
            return (1e6, 1e6, -1e6, 1e6)
        except Exception as e:
            print(f"处理结果时出错: {e}")
            return (1e6, 1e6, -1e6, 1e6)
    
    def _calculate_objectives_and_violation(self, area: float, power: float, wns: float, ser: float):
        """计算目标值和约束违例"""
        if np.isnan(area) or np.isnan(power) or np.isnan(wns) or np.isnan(ser):
            area, power, wns, ser = 1e6, 1e6, -1e6, 1e6
        
        area_ratio = area / (self.area_orig + 1e-10)
        power_ratio = power / (self.power_orig + 1e-10)
        
        # 【关键修复】：检查 ser 是否已经是归一化比例值（0~1）
        # PPAModel.evaluate 返回的 normalized_sensitivity 是 0~1 的比例值
        # 而 self.sens_orig 是绝对敏感性（约0.0003），两者不能直接相除
        # 如果 ser 在 0~1 之间，说明已经是归一化值，不需要再除
        # 如果 ser > 1，说明是绝对敏感性值，需要归一化
        if 0 <= ser <= 1.0:
            ser_ratio = ser  # 已经是归一化比例值
        else:
            ser_ratio = ser / (self.sens_orig + 1e-10)  # 绝对值，需要归一化
        
        delta_area = (area - self.area_orig) / (self.area_orig + 1e-10)
        delta_power = (power - self.power_orig) / (self.power_orig + 1e-10)
        # 稳健处理 slack 基线：支持多种模拟器属性名或缺失情形
        try:
            slack_baseline = float(self.slack_orig)
        except Exception:
            slack_baseline = float(wns)

        # 如果 baseline 或 wns 为 NaN/Inf，则回退到 0
        if np.isnan(slack_baseline) or np.isinf(slack_baseline):
            slack_baseline = float(wns)
        if np.isnan(wns) or np.isinf(wns):
            wns = float(slack_baseline)

        delta_slack = slack_baseline - wns
        
        # 【修复】：敏感性改善量计算
        # ser 可能是归一化值（0~1）或绝对值
        # 归一化情况下：原始值=1.0，加固后=ser，改善量=1.0-ser
        # 绝对值情况下：原始值=sens_orig，加固后=ser，改善量=sens_orig-ser
        if 0 <= ser <= 1.0:
            # 归一化敏感性
            original_normalized_sens = 1.0
            delta_sens = original_normalized_sens - ser  # 敏感性改善量
            # 违例：如果敏感性没有改善到阈值以下
            # delta_sens_min 意味着残余敏感性不超过该阈值（从配置文件读取）
            sens_viol = max(0.0, ser - self.delta_sens_min)  # ser > delta_sens_min 则有违例
        else:
            # 绝对敏感性值
            delta_sens = self.sens_orig - ser
            # 【修复】：使用 delta_sens_min 作为阈值比例
            sens_viol = max(0.0, self.delta_sens_min * self.sens_orig - delta_sens)

        # 【修复】：使用配置中的阈值而不是硬编码值
        area_viol = max(0, delta_area - self.delta_area_max)
        power_viol = max(0, delta_power - self.delta_power_max)
        slack_viol = max(0.0, delta_slack - 0.05)  # 保持原有的delta_slack阈值0.05用于变化量限制
        
        total_violation = (0.1 * area_viol + 
                         0.1 * power_viol + 
                         1000.0 * slack_viol + 
                         0.5 * sens_viol)
        
        # wns_objective 应该反映真实的 slack 违例程度
        # -WNS 表示 worst negative slack 的负值（如果 slack 负得越多，-slack 越大）
        # 直接使用 -wns，让 slack 越负的值越大，在目标函数中越差
        wns_objective = -wns  # 直接取负值
        
        objectives = [
            area_ratio,
            power_ratio,
            wns_objective,
            ser_ratio
        ]
        
        # 检查面积或功耗是否有约束违例
        has_area_power_violation = (area_viol > 0) or (power_viol > 0)
        
        # 【关键修复】：wns<0时，wns_objective设为一个很大的正值，确保MOEA自动淘汰时序违例解
        slack_min = getattr(self, 'slack_min', 0.0)
        if wns < slack_min:
            objectives[2] = 1e6  # 时序违例：wns_objective设为极大正值
        
        # 硬约束：残余敏感性超过阈值的解直接惩罚
        hard_threshold = self.delta_sens_min  # 从配置读取
        if ser_ratio > hard_threshold:
            penalty_multiplier = 100.0
            objectives = [o * penalty_multiplier for o in objectives]
        
        # 【关键修复】：如果面积或功耗超过阈值，直接将惩罚应用到所有目标值上
        # 但同时保存原始目标值用于约束违例判断
        original_objectives = objectives.copy()
        if has_area_power_violation:
            penalty_multiplier = 100.0
            objectives = [o * penalty_multiplier for o in objectives]
        
        objectives = [0.0 if np.isnan(x) else x for x in objectives]
        
        return np.array(objectives), total_violation, not has_area_power_violation, original_objectives
    
    def _parallel_evaluate_safe(self, population: List[Individual]) -> List[Individual]:
        """安全的并行评估 - 使用局部进程池"""
        start_time = time.time()
        
        # 检查模拟器是否可pickle
        try:
            import pickle
            pickle.dumps(self.simulator)
            simulator_picklable = True
        except:
            simulator_picklable = False
            print("警告: 模拟器不可pickle，可能影响并行评估")
        
        # 获取全局进程池
        pool = PoolManager.get_pool(self.num_processes)
        if pool is None:
            print("无法获取进程池，使用串行评估")
            return self._serial_evaluate(population)
        
        try:
            # 准备批次数据
            batches = []
            indices = []
            
            for i in range(0, len(population), self.batch_size):
                batch = population[i:i+self.batch_size]
                batch_genes = [ind.gene.copy() for ind in batch]
                
                # 准备批次数据
                batch_data = {
                    'genes': batch_genes,
                    'simulator': self.simulator if simulator_picklable else None
                }
                
                batches.append(batch_data)
                indices.append((i, min(i+self.batch_size, len(population))))
            
            # 并行处理批次
            if len(batches) == 1:
                # 单个批次直接处理
                results = [evaluate_batch_worker((batches[0], 'evaluate'))]
            else:
                # 多个批次并行处理
                tasks = [(batch, 'evaluate') for batch in batches]
                results = list(pool.map(evaluate_batch_worker, tasks))
            
            # 处理结果
            total_individuals = 0
            for batch_idx, (start, end) in enumerate(indices):
                if batch_idx < len(results):
                    batch_results = results[batch_idx]
                    for j, ind in enumerate(population[start:end]):
                        if j < len(batch_results):
                            area, power, wns, ser = batch_results[j]
                            ind.objectives, ind.violation, ind.area_power_feasible, ind.original_objectives = self._calculate_objectives_and_violation(
                                area, power, wns, ser
                            )
                            total_individuals += 1
                        else:
                            # 如果批次结果缺失，则给出默认惩罚值
                            ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                            ind.violation = 1e6
                            ind.area_power_feasible = False
                            ind.original_objectives = np.array([1e6, 1e6, -1e6, 1e6])
                            total_individuals += 1
            
            elapsed = time.time() - start_time
            print(f"并行评估完成: {total_individuals} 个体, 耗时 {elapsed:.2f} 秒")
            
            return population
            
        except Exception as e:
            print(f"并行评估过程中出错: {e}")
            raise
        finally:
            # 注意：我们不在这里关闭池，而是由PoolManager管理
            pass
    
    def _batch_evaluate(self, population: List[Individual]) -> List[Individual]:
        """批量评估"""
        genes = [ind.gene.copy() for ind in population]
        results = self.simulator.evaluate_population(genes)
        
        for i, ind in enumerate(population):
            if i < len(results):
                area, power, wns, ser = self._process_result(results[i])
                ind.objectives, ind.violation, ind.area_power_feasible, ind.original_objectives = self._calculate_objectives_and_violation(area, power, wns, ser)
            else:
                # 结果缺失时赋默认值
                ind.objectives = np.array([1e6, 1e6, -1e6, 1e6])
                ind.violation = 1e6
                ind.area_power_feasible = False
                ind.original_objectives = np.array([1e6, 1e6, -1e6, 1e6])
        
        return population
    
    def _evaluate_single(self, individual: Individual):
        """评估单个个体"""
        try:
            result = self.simulator.evaluate(individual.gene)
            area, power, wns, ser = self._process_result(result)
            individual.objectives, individual.violation, individual.area_power_feasible, individual.original_objectives = self._calculate_objectives_and_violation(
                area, power, wns, ser
            )
        except Exception as e:
            print(f"评估失败: {e}")
            individual.objectives = np.array([1e6, 1e6, -1e6, 1e6])
            individual.violation = 1e6
            individual.area_power_feasible = False
            individual.original_objectives = np.array([1e6, 1e6, -1e6, 1e6])
    
    def evaluate_individual(self, individual: Individual) -> Individual:
        """评估单个个体"""
        self._evaluate_single(individual)
        return individual
    
    def evaluate_fitness(self, individual: Individual):
        """评估个体适应度"""
        return self.evaluate_individual(individual)
    
    def close(self):
        """关闭评估器"""
        PoolManager.close_pool()
        # 关闭 GPU worker（如果存在）
        try:
            if hasattr(self, 'gpu_worker') and self.gpu_worker is not None:
                self.gpu_worker.shutdown()
        except Exception:
            pass