# utils/parallel_utils.py
import numpy as np
from typing import List, Callable, Any
from multiprocessing import Pool, cpu_count, Manager, get_context
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from functools import partial
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Windows兼容性：使用spawn上下文
if sys.platform == 'win32':
    _mp_context = get_context('spawn')
else:
    _mp_context = None  # 使用默认上下文

class ParallelExecutor:
    """修复的并行执行器，避免pickle问题"""
    
    def __init__(self, n_jobs: int = -1):
        """
        初始化并行执行器
        
        参数:
        ----------
        n_jobs : int
            并行任务数，-1表示使用所有可用CPU核心减1
        """
        if n_jobs == -1:
            self.n_jobs = max(1, cpu_count() - 1)
        else:
            self.n_jobs = max(1, min(n_jobs, cpu_count()))
        
        self.pool = None
        
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None
    
    def parallel_map_fixed(self, func_name: str, data: List, 
                          func_args: tuple = (), func_kwargs: dict = None) -> List:
        """
        安全的并行映射，避免pickle问题
        
        参数:
        ----------
        func_name : str
            函数名称（必须是模块级函数，不是实例方法）
        data : List
            输入数据列表
        func_args : tuple
            函数的额外位置参数
        func_kwargs : dict
            函数的额外关键字参数
        
        返回:
        ----------
        List: 并行计算的结果列表
        """
        if not data:
            return []
        
        # 使用全局函数，而不是实例方法
        if func_kwargs is None:
            func_kwargs = {}
        
        # 创建部分函数
        func = partial(_global_func_wrapper, func_name, func_args, func_kwargs)
        
        # 对于小任务，使用串行执行（避免multiprocessing开销）
        if len(data) <= 10 or self.n_jobs <= 1:
            return [func(item) for item in data]
        
        # 使用进程池（Windows兼容）
        try:
            if _mp_context is not None:
                # Windows: 使用spawn上下文
                with _mp_context.Pool(processes=self.n_jobs) as pool:
                    results = pool.map(func, data, chunksize=1)
            else:
                # Unix/Linux: 使用默认上下文
                with Pool(processes=self.n_jobs) as pool:
                    results = pool.map(func, data, chunksize=1)
            return results
        except (AttributeError, ImportError) as e:
            # Windows上的__main__错误是已知问题，静默处理
            error_str = str(e)
            if "'__main__'" in error_str or "__main__" in error_str:
                # Windows multiprocessing的已知限制，静默回退到串行
                return [func(item) for item in data]
            else:
                # 其他导入错误，记录警告
                logger.debug(f"并行执行失败（导入错误），回退到串行: {e}")
                return [func(item) for item in data]
        except Exception as e:
            # 其他错误，记录调试信息（不是警告）
            logger.debug(f"并行执行失败，回退到串行: {e}")
            return [func(item) for item in data]
    
    def parallel_starmap_fixed(self, func_name: str, args_list: List, 
                              func_kwargs: dict = None) -> List:
        """
        安全的并行starmap
        
        参数:
        ----------
        func_name : str
            函数名称
        args_list : List
            参数列表，每个元素是函数的参数元组
        func_kwargs : dict
            函数的额外关键字参数
        
        返回:
        ----------
        List: 并行计算的结果列表
        """
        if not args_list:
            return []
        
        if func_kwargs is None:
            func_kwargs = {}
        
        # 对于小任务，使用串行执行（避免multiprocessing开销）
        if len(args_list) <= 10 or self.n_jobs <= 1:
            return [_global_starmap_wrapper(func_name, args, func_kwargs) for args in args_list]
        
        # 包装参数
        wrapped_args = []
        for args in args_list:
            wrapped_args.append((func_name, args, func_kwargs))
        
        # 使用进程池（Windows兼容）
        try:
            if _mp_context is not None:
                # Windows: 使用spawn上下文
                with _mp_context.Pool(processes=self.n_jobs) as pool:
                    results = pool.starmap(_global_starmap_wrapper, wrapped_args)
            else:
                # Unix/Linux: 使用默认上下文
                with Pool(processes=self.n_jobs) as pool:
                    results = pool.starmap(_global_starmap_wrapper, wrapped_args)
            return results
        except (AttributeError, ImportError) as e:
            # Windows上的__main__错误是已知问题，静默处理
            error_str = str(e)
            if "'__main__'" in error_str or "__main__" in error_str:
                # Windows multiprocessing的已知限制，静默回退到串行
                return [_global_starmap_wrapper(func_name, args, func_kwargs) for args in args_list]
            else:
                # 其他导入错误，记录调试信息
                logger.debug(f"并行执行失败（导入错误），回退到串行: {e}")
                return [_global_starmap_wrapper(func_name, args, func_kwargs) for args in args_list]
        except Exception as e:
            # 其他错误，记录调试信息（不是警告）
            logger.debug(f"并行执行失败，回退到串行: {e}")
            return [_global_starmap_wrapper(func_name, args, func_kwargs) for args in args_list]

# 全局函数包装器
def _global_func_wrapper(func_name: str, func_args: tuple, 
                        func_kwargs: dict, data_item):
    """
    全局函数包装器，避免pickle问题
    """
    # 安全构建调用参数，支持两种调用模式：
    # 1) data_item 提供主参数（通常是 tuple/list），func_args 为额外参数
    # 2) 通过 starmap 提供 args（此时 data_item 为 None，args 作为 func_args 传入）
    if func_kwargs is None:
        func_kwargs = {}

    call_args = ()

    # data_item 优先（可能为 tuple/list 或单一值）
    if data_item is not None:
        if isinstance(data_item, (tuple, list)):
            call_args += tuple(data_item)
        else:
            call_args += (data_item,)

    # func_args 作为附加参数（可能为 tuple/list 或单一值）
    if func_args:
        if isinstance(func_args, (tuple, list)):
            call_args += tuple(func_args)
        else:
            call_args += (func_args,)

    try:
        safe_kwargs = dict(func_kwargs or {})
        if func_name == "parallel_select_parents":
            # 兼容多种调用约定：
            # 1) data_item 是 (population, ratio) -> 直接作为 task_data 传入
            # 2) data_item 是 population，func_args 提供 ratio -> 将其打包为 (population, ratio)
            # 3) data_item 为 None（starmap 情况）-> 展开 func_args 调用
            safe_kwargs = dict(func_kwargs or {})
            # 如果 data_item 为长度为2 的序列，并且第二个元素看起来像 ratio（数值或 None），
            # 则把它当作 (population, ratio) 的 task_data
            if data_item is not None and isinstance(data_item, (tuple, list)) and len(data_item) == 2 and (
                isinstance(data_item[1], (int, float)) or data_item[1] is None
            ):
                # 优先使用 data_item 中的 ratio，移除重复的关键字
                safe_kwargs.pop('pref_sampling_ratio', None)
                return parallel_select_parents(tuple(data_item), **safe_kwargs)
            # 如果 data_item 看起来像 population（长度>2 的序列），而 func_args 提供 ratio，则打包
            if data_item is not None and isinstance(data_item, (tuple, list)) and len(data_item) > 2 and func_args:
                ratio = None
                if isinstance(func_args, (tuple, list)) and len(func_args) > 0:
                    ratio = func_args[0]
                elif 'pref_sampling_ratio' in (func_kwargs or {}):
                    ratio = (func_kwargs or {}).pop('pref_sampling_ratio')
                safe_kwargs.pop('pref_sampling_ratio', None)
                return parallel_select_parents((data_item, ratio), **safe_kwargs)
            if data_item is not None and not isinstance(data_item, (tuple, list)):
                # data_item 是单一对象（非序列），将其打包为 task_data = (population, ratio)
                ratio = None
                if func_args and isinstance(func_args, (tuple, list)) and len(func_args) > 0:
                    ratio = func_args[0]
                elif 'pref_sampling_ratio' in (func_kwargs or {}):
                    ratio = (func_kwargs or {}).pop('pref_sampling_ratio')
                safe_kwargs.pop('pref_sampling_ratio', None)
                return parallel_select_parents((data_item, ratio), **safe_kwargs)
            if data_item is None:
                # starmap: func_args 已为参数元组，直接调用（可能为 (population, ratio)）
                return parallel_select_parents(*func_args, **safe_kwargs)
            # 兜底：把 data_item 作为 task_data 传入
            return parallel_select_parents(data_item, **safe_kwargs)
        elif func_name == "parallel_create_offspring":
            return parallel_create_offspring(*call_args, **safe_kwargs)
        elif func_name == "parallel_compute_strength":
            # parallel_compute_strength expects a single task_data argument (i, population, epsilon)
            # Ensure we pass the task_data as one positional parameter and not expand it.
            if data_item is not None:
                task = data_item
            else:
                task = func_args if func_args else None
            return parallel_compute_strength(task, **safe_kwargs)
        elif func_name == "parallel_compute_fitness":
            # parallel_compute_fitness also expects a single task_data argument (i, population, strengths, epsilon)
            if data_item is not None:
                task = data_item
            else:
                task = func_args if func_args else None
            return parallel_compute_fitness(task, **safe_kwargs)
        else:
            raise ValueError(f"未知函数名: {func_name}")
    except Exception:
        logger.exception("Error in _global_func_wrapper while calling %s", func_name)
        raise

def _global_starmap_wrapper(func_name: str, args: tuple, func_kwargs: dict):
    """
    全局starmap包装器
    """
    # 当通过 starmap 调用时，args 已经是要展开的参数元组，传入为 func_args
    return _global_func_wrapper(func_name, args, func_kwargs, None)

# 全局并行函数
def parallel_select_parents(task_data, pref_sampling_ratio=0.3):
    """
    并行选择父代（模块级函数）
    """
    if task_data is None:
        return None
    
    try:
        population, ratio = task_data
        
        # 检查population是否有效
        if population is None or len(population) < 2:
            # 返回随机个体作为后备
            if population and len(population) > 0:
                # 如果只有一个个体，则复制它
                if len(population) == 1:
                    return population[0], population[0]
                else:
                    # 随机选择两个不同的个体
                    indices = np.random.choice(len(population), 2, replace=False)
                    return population[indices[0]], population[indices[1]]
            else:
                return None
        
        # 获取实际使用的比例
        actual_ratio = pref_sampling_ratio if pref_sampling_ratio is not None else ratio
        
        if actual_ratio > 0 and len(population) > 10:
            pref_count = int(len(population) * actual_ratio)
            
            # 计算每个个体的适应度排名
            fitness_vals = []
            for ind in population:
                if hasattr(ind, 'fitness') and not np.isnan(ind.fitness):
                    fitness_vals.append(ind.fitness)
                else:
                    fitness_vals.append(float('inf'))  # 无效适应度设为无穷大
            
            if any(np.isfinite(fitness_vals)):
                sorted_indices = np.argsort(fitness_vals)
                pref_indices = sorted_indices[:pref_count]
                
                if np.random.rand() < 0.7 and len(pref_indices) >= 2:
                    idx1, idx2 = np.random.choice(pref_indices, 2, replace=False)
                    return population[idx1], population[idx2]
        
        # 默认随机选择
        indices = np.random.choice(len(population), 2, replace=False)
        return population[indices[0]], population[indices[1]]
        
    except Exception as e:
        print(f"选择父代时出错: {e}")
        return None

def parallel_create_offspring(p1, p2, pc, pm, generation):
    """
    并行创建后代（模块级函数）
    参数: p1, p2, pc, pm, generation
    确保所有参数都有值
    """
    try:
        # 检查所有参数是否有效
        if p1 is None or p2 is None:
            raise ValueError(f"父代为None: p1={p1}, p2={p2}")
        
        # 检查父代是否有gene属性
        if not hasattr(p1, 'gene') or not hasattr(p2, 'gene'):
            raise ValueError(f"父代缺少gene属性: p1={type(p1)}, p2={type(p2)}")
        
        # 检查gene是否为有效数组
        if p1.gene is None or p2.gene is None:
            raise ValueError("父代gene为None")
        
        if len(p1.gene) == 0 or len(p2.gene) == 0:
            raise ValueError(f"父代gene长度为零: p1={len(p1.gene)}, p2={len(p2.gene)}")
        
        # 检查其他参数
        if pc is None or pm is None or generation is None:
            raise ValueError(f"参数为None: pc={pc}, pm={pm}, generation={generation}")
        
        # 确保参数类型正确
        pc = float(pc) if pc is not None else 0.9
        pm = float(pm) if pm is not None else 0.01
        generation = int(generation) if generation is not None else 0
        
        # 执行交叉
        if np.random.rand() < pc:
            if np.random.rand() < 0.9:
                child1_gene, child2_gene = uniform_crossover(p1.gene, p2.gene)
            else:
                child1_gene, child2_gene = one_point_crossover(p1.gene, p2.gene)
        else:
            child1_gene, child2_gene = p1.gene.copy(), p2.gene.copy()
        
        # 执行变异
        child1_gene = mutate_gene(child1_gene, generation, pm)
        child2_gene = mutate_gene(child2_gene, generation, pm)
        
        return child1_gene, child2_gene
        
    except Exception as e:
        print(f"创建后代时出错: {e}")
        print(f"参数详情: p1={type(p1)}, p2={type(p2)}, pc={pc}, pm={pm}, generation={generation}")
        
        # 尝试获取基因长度
        try:
            if p1 is not None and hasattr(p1, 'gene') and p1.gene is not None:
                K = len(p1.gene)
            elif p2 is not None and hasattr(p2, 'gene') and p2.gene is not None:
                K = len(p2.gene)
            else:
                K = 30  # 默认基因长度
        except:
            K = 30
        
        # 创建随机基因作为后备
        random_gene = np.random.randint(0, 2, K)
        return random_gene.copy(), random_gene.copy()

def parallel_compute_strength(task_data, eps=0.0):
    """
    并行计算强度（模块级函数）
    """
    i, population, epsilon = task_data
    strength = 0
    for j in range(len(population)):
        if i == j:
            continue
        if dominates_epsilon(population[i].objectives, population[j].objectives, epsilon):
            strength += 1
    return strength

def parallel_compute_fitness(task_data, eps=0.0):
    """
    并行计算适应度（模块级函数）
    """
    i, population, strengths, epsilon = task_data
    raw_fitness = 0
    for j in range(len(population)):
        if i == j:
            continue
        if dominates_epsilon(population[j].objectives, population[i].objectives, epsilon):
            raw_fitness += strengths[j]
    return raw_fitness

# 辅助函数（也必须是模块级别的）
def uniform_crossover(parent1, parent2):
    """均匀交叉"""
    mask = np.random.rand(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

def one_point_crossover(parent1, parent2):
    """单点交叉"""
    c = np.random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:c], parent2[c:]))
    child2 = np.concatenate((parent2[:c], parent1[c:]))
    return child1, child2

def mutate_gene(gene, generation, pm):
    """变异函数"""
    mutated = gene.copy()
    for i in range(len(gene)):
        if np.random.rand() < pm:
            mutated[i] = 1 - mutated[i]  # 翻转
    return mutated

def dominates_epsilon(obj_p, obj_q, eps):
    """
    ε-dominance: 
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