# main.py
import numpy as np
import warnings
import torch
import yaml
import argparse
from pathlib import Path
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any

# 添加路径
sys.path.append(str(Path(__file__).parent))

from env.sac_moea_env import SACMOEAEnv
from env.real_moea_wrapper import RealMoeaWrapper
from mo_optimizer.simulator import ICDesignSimulator
from sac.sac_agent import SACAgent
from utils.logger import Logger
from utils.logger import get_logger
from utils.constraint_analyzer import ConstraintAnalyzer

def _deduplicate_population(population):
    """Return unique individuals by gene while preserving order."""
    unique = []
    seen = set()
    for ind in population or []:
        gene = getattr(ind, 'gene', None)
        if gene is None:
            continue
        try:
            key = tuple(np.asarray(gene).flatten().astype(int).tolist())
        except Exception:
            key = id(ind)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ind)
    return unique

def _filter_constraint_feasible_population(population, simulator, config):
    """Filter candidates with the same hard constraints as the optimizer."""
    if simulator is None:
        return []

    orig_vals = ConstraintAnalyzer._get_original_values(simulator)
    area_orig = max(float(orig_vals.get('area_orig', 0.0)), 1e-20)
    power_orig = max(float(orig_vals.get('power_orig', 0.0)), 1e-20)
    slack_orig = float(orig_vals.get('slack_orig', 0.0))
    sens_orig = max(float(orig_vals.get('sens_orig', 0.0)), 1e-20)

    thresholds = config.get('constraint_thresholds', {})
    ser_as_objective = bool(config.get('moea', {}).get('ser_as_objective', True))
    delta_area_max = float(thresholds.get('delta_area_max', 0.2))
    delta_power_max = float(thresholds.get('delta_power_max', 0.15))
    slack_min = float(thresholds.get('slack_min', slack_orig - 0.05))
    try:
        delta_sens_min = float(thresholds.get('delta_sens_min', 0.5))
        if not (0.0 < delta_sens_min < 1.0):
            delta_sens_min = 0.5
    except Exception:
        delta_sens_min = 0.5

    feasible = []
    for ind in _deduplicate_population(population):
        gene = getattr(ind, 'gene', None)
        if gene is None:
            continue
        try:
            gene = np.asarray(gene).flatten()
            if hasattr(simulator, 'evaluate_with_details'):
                details = simulator.evaluate_with_details(gene)
                if len(details) < 4:
                    continue
                area_total = float(details[0])
                power_total = float(details[1])
                min_slack = float(details[2])
                total_sensitivity = float(details[3])

                area_overhead = (area_total - area_orig) / area_orig
                power_overhead = (power_total - power_orig) / power_orig
                sens_ratio = total_sensitivity / sens_orig

                is_feasible = (
                    area_overhead <= delta_area_max + 1e-12 and
                    power_overhead <= delta_power_max + 1e-12 and
                    min_slack >= slack_min - 1e-12 and
                    (ser_as_objective or sens_ratio <= delta_sens_min + 1e-12)
                )
            else:
                is_feasible = (
                    bool(getattr(ind, 'area_power_feasible', False)) and
                    (ser_as_objective or bool(getattr(ind, 'sensitivity_feasible', True))) and
                    float(getattr(ind, 'violation', 0.0)) <= 1e-12
                )
            if is_feasible:
                feasible.append(ind)
        except Exception:
            fallback_feasible = (
                bool(getattr(ind, 'area_power_feasible', False)) and
                (ser_as_objective or bool(getattr(ind, 'sensitivity_feasible', True))) and
                float(getattr(ind, 'violation', 0.0)) <= 1e-12
            )
            if fallback_feasible:
                feasible.append(ind)
    return feasible

def export_constraint_violation_views(moea_wrapper, config, save_dir, all_candidates=None):
    """Save constraint plots for all candidates, current Pareto, and feasible set."""
    if moea_wrapper is None or not hasattr(moea_wrapper, 'model'):
        return []

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = moea_wrapper.model

    if all_candidates is None:
        all_candidates = moea_wrapper.get_last_pareto(include_history=True)
    all_candidates = _deduplicate_population(all_candidates)

    if hasattr(moea_wrapper, 'get_current_pareto'):
        current_pareto = moea_wrapper.get_current_pareto()
    else:
        current_pareto = moea_wrapper.get_last_pareto(include_history=False)
    current_pareto = _deduplicate_population(current_pareto)

    feasible_set = _filter_constraint_feasible_population(all_candidates, model, config)

    views = [
        ('all_exploration_candidates', 'all exploration candidates',
         all_candidates, 'constraint_violations_all_exploration.png'),
        ('current_pareto_front', 'current Pareto front',
         current_pareto, 'constraint_violations_current_pareto.png'),
        ('final_feasible_set', 'final feasible set',
         feasible_set, 'constraint_violations_feasible_set.png'),
    ]

    summary_rows = []
    for view_key, view_name, population, filename in views:
        row = {
            'view': view_key,
            'description': view_name,
            'population_size': len(population),
            'plot_file': filename,
        }
        if population:
            try:
                stats = ConstraintAnalyzer.analyze_population_constraints(
                    population=population,
                    simulator=model,
                    config=config,
                    save_path=str(save_dir / filename),
                )
                for key, value in stats.items():
                    try:
                        row[key] = float(value)
                    except Exception:
                        row[key] = value
            except Exception as exc:
                row['error'] = str(exc)
                print(f"  ! Constraint plot failed for {view_key}: {exc}")
        else:
            row['status'] = 'empty'
            print(f"  ! Constraint plot skipped for {view_key}: empty population")
        summary_rows.append(row)

    summary_path = save_dir / 'constraint_violation_views_summary.csv'
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  Constraint violation views saved to: {save_dir}")
    return summary_rows

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证配置完整性
        required_keys = ['sac', 'moea', 'training', 'reward_params']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件中缺少必需的键: {key}")
        
        # 设置默认值
        config.setdefault('state_dim', 24)
        config.setdefault('action_low', [0.001, 0.01, 0.01, 1.0, 0.001, 0.6])
        config.setdefault('action_high', [0.2, 0.5, 0.3, 200.0, 0.3, 1.0])
        config.setdefault('initial_action', [0.05, 0.1, 0.05, 10.0, 0.01, 0.8])

        if config.get('action_schema') == 'search_control_v2':
            action_low = np.asarray(config['action_low'], dtype=np.float64)
            action_high = np.asarray(config['action_high'], dtype=np.float64)
            initial_action = np.asarray(config['initial_action'], dtype=np.float64)
            if action_low.shape != (8,) or action_high.shape != (8,) or initial_action.shape != (8,):
                raise ValueError('search_control_v2 必须配置8维 action_low/action_high/initial_action')
            if not np.all(np.isfinite(action_low)) or not np.all(np.isfinite(action_high)):
                raise ValueError('search_control_v2 动作边界必须是有限数值')
            if np.any(action_low >= action_high):
                raise ValueError('search_control_v2 每一维都必须满足 action_low < action_high')
            if np.any(initial_action < action_low) or np.any(initial_action > action_high):
                raise ValueError('search_control_v2 的 initial_action 必须位于动作边界内')

        # 设置约束阈值默认值
        config.setdefault('constraint_thresholds', {
            'delta_area_max': 0.2,
            'delta_power_max': 0.15,
            'delta_sens_min': 0.5,
            'area_weight': 0.1,
            'power_weight': 0.1,
            'slack_weight': 1000.0,
            'sens_weight': 0.5
        })
        
        # MOEA配置默认值
        config['moea'].setdefault('pop_size', 100)
        config['moea'].setdefault('max_generations', 50)
        config['moea'].setdefault('archive_size', 50)
        config['moea'].setdefault('num_processes', 4)
        config['moea'].setdefault('generations_per_step', 2)
        
        # SAC配置默认值
        config['sac'].setdefault('gamma', 0.99)
        config['sac'].setdefault('tau', 0.005)
        config['sac'].setdefault('lr', 3e-4)
        config['sac'].setdefault('batch_size', 128)
        config['sac'].setdefault('buffer_size', 1000000)
        config['sac'].setdefault('hidden_dim', 256)
        config['sac'].setdefault('automatic_entropy_tuning', True)
        config['sac'].setdefault('alpha', 0.2)
        
        # 训练配置默认值
        config['training'].setdefault('max_steps', 500)
        config['training'].setdefault('eval_interval', 50)
        config['training'].setdefault('log_interval', 10)
        config['training'].setdefault('checkpoint_dir', 'checkpoints')
        config['training'].setdefault('save_interval', 10)
        
        # 奖励参数默认值
        config['reward_params'].setdefault('alpha', 1.0)
        config['reward_params'].setdefault('beta', 1.0)
        config['reward_params'].setdefault('gamma', 0.5)
        config['reward_params'].setdefault('eta', 0.1)
        config['reward_params'].setdefault('epsilon', 1e-6)
        
        # 获取日志记录器（如果已初始化）
        try:
            logger = get_logger()
            logger.log(f"配置文件加载成功: {config_path}")
            logger.log(f"MOEA配置: 种群大小={config['moea']['pop_size']}, "
                      f"最大代数={config['moea']['max_generations']}")
            logger.log(f"SAC配置: 学习率={config['sac']['lr']}, "
                      f"批次大小={config['sac']['batch_size']}")
            logger.log(f"训练配置: 最大步数={config['training']['max_steps']}, "
                      f"评估间隔={config['training']['eval_interval']}")
        except:
            # 如果日志记录器未初始化，使用print
            print(f"配置文件加载成功: {config_path}")
            print(f"MOEA配置: {config['moea']}")
            print(f"SAC配置: {config['sac']}")
            print(f"训练配置: {config['training']}")
        
        return config
    
    except FileNotFoundError:
        error_msg = f"配置文件未找到: {config_path}"
        print(f"错误: {error_msg}")
        raise
    except yaml.YAMLError as e:
        error_msg = f"配置文件YAML格式错误: {e}"
        print(f"错误: {error_msg}")
        raise
    except Exception as e:
        error_msg = f"加载配置文件时发生未知错误: {e}"
        print(f"错误: {error_msg}")
        raise

def create_reg_table(simulator: ICDesignSimulator, config: Dict = None, logger=None) -> Dict[str, np.ndarray]:
    """
    从模拟器创建寄存器表
    
    参数:
    ----------
    simulator : ICDesignSimulator
        集成电路设计模拟器
    config : Dict, optional
        配置字典（包含voter等参数）
    logger : Logger, optional
        日志记录器
    
    返回:
    ----------
    Dict[str, np.ndarray]: 寄存器属性表
    """
    try:
        # 确保config不为None
        if config is None:
            config = {}
        
        # 记录寄存器信息
        if logger:
            logger.log("开始创建寄存器表...")
            logger.log(f"总寄存器数量: {simulator.num_registers}")
            logger.log(f"预筛选寄存器数量: {simulator.K}")
            logger.log(f"原始PPA值:")
            logger.log(f"  总面积: {simulator.Area_orig}")
            logger.log(f"  总功耗: {simulator.Power_orig}")
            logger.log(f"  最差时序裕量: {simulator.WNS_orig}")
            logger.log(f"  设计敏感性: {simulator.SER_orig}")
        
        num_registers = simulator.num_registers
        
        # 提取预筛选寄存器的属性
        if hasattr(simulator, 'SETFF_indices'):
            # 使用预筛选的寄存器索引
            indices = simulator.SETFF_indices
            
            # 创建寄存器表 - 只包含预筛选的寄存器
            reg_table = {
                'area': simulator.all_reg_properties['area'][indices],
                'power': simulator.all_reg_properties['power'][indices],
                'slack': simulator.all_reg_properties['slack'][indices],
                'sensitivity': simulator.all_reg_properties['sensitivity'][indices],
                'is_critical': simulator.all_reg_properties['is_critical'][indices]
            }
            
            # 关键：添加设计基准值
            reg_table['design_area'] = simulator.Area_orig
            reg_table['design_power'] = simulator.Power_orig
            reg_table['design_slack'] = simulator.WNS_orig
            
            # 【修复】：直接使用simulator中的SER_orig作为design_sensitivity
            # simulator.SER_orig已经在加载CSV时被正确设置为原始设计敏感性
            reg_table['design_sensitivity'] = simulator.SER_orig
            # 【关键】：同时设置sens_orig，确保所有代码路径都能获取正确的敏感性基准值
            reg_table['sens_orig'] = simulator.SER_orig
            
            # 添加总寄存器数（用于敏感性计算的正确分母）
            reg_table['total_num_registers'] = simulator.num_registers
            
            # 【关键】：添加config，以便PPAModel可以读取voter等参数
            reg_table['config'] = config
            
            # 添加所有寄存器的敏感性总和（用于计算加固后的全局敏感性）
            all_sensitivities = simulator.all_reg_properties['sensitivity']
            reg_table['total_sensitivity_sum'] = float(np.sum(all_sensitivities))
            reg_table['non_prefiltered_sensitivity_sum'] = float(np.sum(all_sensitivities) - np.sum(simulator.all_reg_properties['sensitivity'][indices]))
            
            # 添加原始索引映射
            reg_table['original_indices'] = indices
            
            # 添加实例名和寄存器类型
            if 'instance_name' in simulator.all_reg_properties:
                reg_table['instance_names'] = simulator.all_reg_properties['instance_name'][indices]
            else:
                # 创建默认实例名
                reg_table['instance_names'] = np.array([f"reg_{idx}" for idx in indices])
            
            if 'reg_type' in simulator.all_reg_properties:
                reg_table['reg_types'] = simulator.all_reg_properties['reg_type'][indices]
            else:
                reg_table['reg_types'] = np.array(["FF"] * len(indices))
            
            # 计算一些统计信息
            area_stats = {
                'mean': float(np.mean(reg_table['area'])),
                'std': float(np.std(reg_table['area'])),
                'min': float(np.min(reg_table['area'])),
                'max': float(np.max(reg_table['area']))
            }
            
            power_stats = {
                'mean': float(np.mean(reg_table['power'])),
                'std': float(np.std(reg_table['power'])),
                'min': float(np.min(reg_table['power'])),
                'max': float(np.max(reg_table['power']))
            }
            
            slack_stats = {
                'mean': float(np.mean(reg_table['slack'])),
                'std': float(np.std(reg_table['slack'])),
                'min': float(np.min(reg_table['slack'])),
                'max': float(np.max(reg_table['slack']))
            }
            
            sensitivity_stats = {
                'mean': float(np.mean(reg_table['sensitivity'])),
                'std': float(np.std(reg_table['sensitivity'])),
                'min': float(np.min(reg_table['sensitivity'])),
                'max': float(np.max(reg_table['sensitivity']))
            }
            
            critical_count = int(np.sum(reg_table['is_critical']))
            critical_ratio = critical_count / len(indices)
            
            if logger:
                logger.log(f"寄存器表创建成功，包含 {len(indices)} 个预筛选寄存器")
                logger.log(f"设计基准值: 面积={simulator.Area_orig}, 功耗={simulator.Power_orig}, "
                          f"slack={simulator.WNS_orig}, 敏感度={simulator.SER_orig}")
                
                try:
                    import pandas as pd
                    import os
                    
                    data_dir = Path("data/register_info")
                    data_dir.mkdir(parents=True, exist_ok=True)
                    
                    df = pd.DataFrame({
                        'register_id': indices,
                        'instance_name': reg_table.get('instance_names', [f"reg_{i}" for i in range(len(indices))]),
                        'reg_type': reg_table.get('reg_types', ["FF"] * len(indices)),
                        'area': reg_table['area'],
                        'power': reg_table['power'],
                        'slack': reg_table['slack'],
                        'sensitivity': reg_table['sensitivity'],
                        'is_critical': reg_table['is_critical']
                    })
                    
                    csv_path = data_dir / "register_table.csv"
                    df.to_csv(csv_path, index=False)
                    
                    np_path = data_dir / "register_table.npz"
                    np.savez(np_path, **reg_table)
                    
                    design_info = pd.DataFrame({
                        'design_area': [simulator.Area_orig],
                        'design_power': [simulator.Power_orig],
                        'design_slack': [simulator.WNS_orig],
                        'design_sensitivity': [simulator.SER_orig]
                    })
                    design_path = data_dir / "design_info.csv"
                    design_info.to_csv(design_path, index=False)
                    
                except Exception as e:
                    logger.log(f"保存寄存器表到文件时出错: {e}", "WARNING")
            
            return reg_table
            
        else:
            # 如果没有预筛选索引，使用所有寄存器
            reg_table = {
                'area': simulator.all_reg_properties['area'],
                'power': simulator.all_reg_properties['power'],
                'slack': simulator.all_reg_properties['slack'],
                'sensitivity': simulator.all_reg_properties['sensitivity'],
                'is_critical': simulator.all_reg_properties['is_critical']
            }
            
            reg_table['design_area'] = simulator.Area_orig
            reg_table['design_power'] = simulator.Power_orig
            reg_table['design_slack'] = simulator.WNS_orig
            reg_table['design_sensitivity'] = simulator.SER_orig
            reg_table['total_num_registers'] = simulator.num_registers
            
            # 【关键】：添加config，以便PPAModel可以读取voter等参数
            reg_table['config'] = config
            
            all_sensitivities = simulator.all_reg_properties['sensitivity']
            reg_table['total_sensitivity_sum'] = float(np.sum(all_sensitivities))
            reg_table['non_prefiltered_sensitivity_sum'] = 0.0
            reg_table['original_indices'] = np.arange(simulator.num_registers)
            
            if 'instance_name' in simulator.all_reg_properties:
                reg_table['instance_names'] = simulator.all_reg_properties['instance_name']
            else:
                reg_table['instance_names'] = np.array([f"reg_{i}" for i in range(simulator.num_registers)])
            
            if 'reg_type' in simulator.all_reg_properties:
                reg_table['reg_types'] = simulator.all_reg_properties['reg_type']
            else:
                reg_table['reg_types'] = np.array(["FF"] * simulator.num_registers)
            
            return reg_table
    
    except AttributeError as e:
        error_msg = f"模拟器缺少必需的属性: {e}"
        raise

def validate_reg_table(reg_table: Dict[str, np.ndarray], logger=None) -> bool:
    """
    验证寄存器表的数据完整性
    
    参数:
    ----------
    reg_table : Dict[str, np.ndarray]
        寄存器表
    logger : Logger, optional
        日志记录器
    
    返回:
    ----------
    bool: 验证是否通过
    """
    try:
        required_keys = ['area', 'power', 'slack', 'sensitivity', 'is_critical']
        
        # 检查必需键
        for key in required_keys:
            if key not in reg_table:
                if logger:
                    logger.log(f"错误: 寄存器表缺少必需的键: {key}", "ERROR")
                return False
        
        # 检查数据维度
        n_registers = len(reg_table['area'])
        for key in required_keys:
            if len(reg_table[key]) != n_registers:
                if logger:
                    logger.log(f"错误: 键 {key} 的数据维度不一致", "ERROR")
                return False
        
        # 检查数据有效性
        # 面积应为正数
        if np.any(reg_table['area'] <= 0):
            if logger:
                logger.log("警告: 发现非正面积值", "WARNING")
        
        # 功耗应为非负数
        if np.any(reg_table['power'] < 0):
            if logger:
                logger.log("警告: 发现负功耗值", "WARNING")
        
        # 时序裕量应为实数（可能为负）
        if np.any(np.isnan(reg_table['slack'])):
            if logger:
                logger.log("警告: 发现NaN时序裕量值", "WARNING")
        
        # 敏感度应在[0,1]范围内
        if np.any(reg_table['sensitivity'] < 0) or np.any(reg_table['sensitivity'] > 1):
            if logger:
                logger.log("警告: 敏感度值超出[0,1]范围", "WARNING")
        
        # 关键路径标识应为0或1
        unique_values = np.unique(reg_table['is_critical'])
        if not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [1]) and not np.array_equal(unique_values, [0, 1]):
            if logger:
                logger.log("警告: is_critical包含非0/1值", "WARNING")
        
        if logger:
            logger.log(f"寄存器表验证通过，包含 {n_registers} 个寄存器")
            
            # 记录数据统计
            stats = {}
            for key in required_keys:
                data = reg_table[key]
                stats[key] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                }
            
            # 记录统计信息
            for key, stat in stats.items():
                logger.log(f"{key}: 均值={stat['mean']:.4f}, 标准差={stat['std']:.4f}, "
                          f"范围=[{stat['min']:.4f}, {stat['max']:.4f}]")
        
        return True
    
    except Exception as e:
        error_msg = f"验证寄存器表时发生错误: {e}"
        if logger:
            logger.log(f"错误: {error_msg}", "ERROR")
        else:
            print(f"错误: {error_msg}")
        return False

def train_sac(config, reg_table, log_dir, simulator):
    """训练SAC代理 - 多步Episode设计"""
    from utils.logger import get_logger
    import gc
    import psutil

    # 内存监控工具
    try:
        from utils.memory_saver import MemorySaver
        memory_saver = MemorySaver(memory_threshold=0.75) 
    except ImportError:
        class SimpleMemoryMonitor:
            def check_and_clean(self):
                memory_info = psutil.virtual_memory()
                used_percent = memory_info.percent / 100.0
                if used_percent > 0.75:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return True
                return False
        memory_saver = SimpleMemoryMonitor()

    # 创建日志记录器
    logger = get_logger(
        log_dir=str(log_dir),
        experiment_name="sac_moea_training",
        use_tensorboard=False
    )
    
    logger.log("开始SAC-MOEA训练（多步Episode设计）...")

    # 应用最新的 max_episodes 设置，防止被旧配置覆盖
    if 'max_episodes' not in config['training']:
        config['training']['max_episodes'] = 50
    
    if 'max_steps_per_episode' not in config.get('sac', {}):
        config['sac']['max_steps_per_episode'] = 10
        logger.log("警告: 配置中缺少max_steps_per_episode参数，使用默认值 10")
    
    # 创建MOEA包装器
    moea_wrapper = RealMoeaWrapper(
        reg_table=reg_table,
        pop_size=config['moea']['pop_size'],
        max_gen=config['moea']['max_generations'],
        archive_size=config['moea']['archive_size'],
        num_processes=config['moea']['num_processes'],
        config=config,
        use_gpu=config['evaluator'].get('use_gpu', False) 
    )
    
    # 创建SAC环境
    env = SACMOEAEnv(moea_wrapper=moea_wrapper, config=config)
    
    # 强制指定使用NVIDIA GPU
    device = 'cpu'
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i).lower()
            if 'nvidia' in gpu_name or 'geforce' in gpu_name or '1660' in gpu_name:
                device = f'cuda:{i}'
                break
                
    # 创建SAC代理
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_space=env.action_space,
        config=config['sac'],
        device=device
    )

    logger.log(f"SAC设备: {agent.device}")
    
    # 训练参数
    max_episodes = config['training'].get('max_episodes', 50)
    max_sac_steps_per_episode = config['sac'].get('max_steps_per_episode', 10)
    eval_interval = config['training'].get('eval_interval', 10)
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    moea_generations_per_sac_step = config['moea'].get('generations_per_step', 2)
    sac_updates_per_step = max(1, int(config['sac'].get('updates_per_step', 1)))
    sac_min_replay_size = max(1, int(config['sac'].get('min_replay_size', agent.batch_size)))
    sac_training_started = False
    sac_cfg = config.get('sac', {})
    pure_exploration_steps = int(sac_cfg.get('pure_exploration_steps', max(8, sac_min_replay_size)))
    deterministic_update_start = int(sac_cfg.get('deterministic_update_start', 32))
    use_deterministic_after_updates = bool(sac_cfg.get('use_deterministic_after_updates', True))
    eval_noise_scale = float(sac_cfg.get('eval_noise_scale', 0.01))
    action_smoothing = float(sac_cfg.get('action_smoothing', 0.65))
    action_smoothing_start_step = int(sac_cfg.get('action_smoothing_start_step', pure_exploration_steps))
    early_stop_on_stagnation = bool(sac_cfg.get('early_stop_on_stagnation', True))
    stagnation_min_steps = int(sac_cfg.get('stagnation_min_steps', 4))
    stagnation_patience = int(sac_cfg.get('stagnation_patience', 3))
    best_feasible_gain_eps = float(sac_cfg.get('best_feasible_gain_eps', 1e-6))
    last_action = None
    
    all_parameters_history = []
    best_reward = -float('inf')
    episode_rewards = []
    episode_lengths = []
    convergence_history = []
    
    print(f"\n{'='*80}")
    print(f"开始训练: {max_episodes}个Episode (强化学习全局回合数)")
    print(f"每个Episode: {max_sac_steps_per_episode}步SAC动作")
    print(f"每步SAC: 内部跑{moea_generations_per_sac_step}代MOEA用于物理仿真")
    print(f"{'='*80}\n")
    
    # Episode训练循环 (外层循环，这就是标准的RL结构)
    for episode in range(max_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {episode+1}/{max_episodes} 开始 (这代表代理的第 {episode+1} 次episode)")
        print(f"{'='*80}")

        if hasattr(env, "set_episode_index"):
            env.set_episode_index(episode + 1)
        
        memory_saver.check_and_clean()
        
        # 重置环境（调用热启动+灾难变异的逻辑）
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_converged = False
        total_moea_generations = 0
        episode_parameters_history = []
        episode_no_global_improvement_steps = 0
        
        # 内层循环：每个Episode内部跑多个SAC动作决策
        for step in range(max_sac_steps_per_episode):
            sac_step_number = step + 1
            print(f"\n--- Episode {episode+1}, SAC步 {sac_step_number}/{max_sac_steps_per_episode} ---")
            
            if step % 2 == 0:  
                memory_saver.check_and_clean()

            global_step = episode * max_sac_steps_per_episode + step
            use_eval_action = (
                use_deterministic_after_updates
                and global_step >= pure_exploration_steps
                and agent.update_step >= deterministic_update_start
            )
            raw_action = agent.select_action(
                state,
                evaluate=use_eval_action,
                noise_scale=eval_noise_scale if use_eval_action else None,
            )
            if (
                last_action is not None
                and global_step >= action_smoothing_start_step
                and 0.0 < action_smoothing < 1.0
            ):
                action = action_smoothing * last_action + (1.0 - action_smoothing) * raw_action
                action = np.clip(action, env.action_space.low, env.action_space.high)
                action_smoothed = True
            else:
                action = raw_action
                action_smoothed = False

            fixed_execution_action = getattr(env, '_fixed_execution_action', None)
            if (
                getattr(env, 'operator_control_mode', 'adaptive') == 'fixed'
                and fixed_execution_action is not None
            ):
                action = fixed_execution_action.copy()
                action_smoothed = False
            last_action = action.copy() if hasattr(action, 'copy') else action
            if getattr(env, 'operator_control_mode', 'adaptive') == 'fixed':
                action_mode = 'fixed_ablation'
            else:
                action_mode = 'deterministic_mean' if use_eval_action else 'stochastic_sample'
            if action_smoothed:
                action_mode += '+smooth'

            if hasattr(env, 'set_sac_training_context'):
                env.set_sac_training_context(
                    update_step=agent.update_step,
                    action_mode=action_mode,
                )
            
            # 将超参数下发给MOEA，执行几代仿真，并获取下一个状态
            next_state, reward, terminated, truncated, info = env.step(action)

            # 收集经验存入回放池
            agent.store_transition(state, action, reward, next_state, terminated)
            
            # 只要池子里的数据足够，就开始反向传播，更新Actor和Critic权重
            replay_size = len(agent.replay_buffer)
            replay_threshold = max(agent.batch_size, sac_min_replay_size)
            if replay_size >= replay_threshold:
                updates_before = agent.update_step
                for _ in range(sac_updates_per_step):
                    agent.update()
                if (not sac_training_started) and agent.update_step > updates_before:
                    sac_training_started = True
                    print(
                        f"[SAC-TRAIN] started at episode {episode+1}, "
                        f"SAC step {sac_step_number}, replay_size={replay_size}, "
                        f"batch_size={agent.batch_size}, updates_per_step={sac_updates_per_step}"
                    )
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_moea_generations = info.get('total_moea_generations', 0)
            
            action_params = info.get('action_params', {})
            
            # 记录历史
            if isinstance(action_params, dict) and 'spea2' in action_params:
                step_record = {
                    'episode': episode + 1,
                    'sac_step': sac_step_number,
                    'global_step': global_step,
                    'epsilon_spea2': action_params['spea2'].get('epsilon', 0.0),
                    'tau_spea2': action_params['spea2'].get('tau', 0.0),
                    'delta_spea2': action_params['spea2'].get('delta', 0.0),
                    'beta_spea2': action_params['spea2'].get('beta', 0.0),
                    'pm_spea2': action_params['spea2'].get('pm', 0.0),
                    'pc_spea2': action_params['spea2'].get('pc', 0.0),
                    'epsilon_nsga2': action_params['nsga2'].get('epsilon', 0.0),
                    'tau_nsga2': action_params['nsga2'].get('tau', 0.0),
                    'delta_nsga2': action_params['nsga2'].get('delta', 0.0),
                    'beta_nsga2': action_params['nsga2'].get('beta', 0.0),
                    'pm_nsga2': action_params['nsga2'].get('pm', 0.0),
                    'pc_nsga2': action_params['nsga2'].get('pc', 0.0),
                    'mutation_flips_spea2': action_params['spea2'].get('mutation_flips', None),
                    'mutation_flips_nsga2': action_params['nsga2'].get('mutation_flips', None),
                    'spea2_budget_share': action_params.get('controls', {}).get('spea2_budget_share', None),
                    'elite_injection_ratio': action_params.get('controls', {}).get('elite_injection_ratio', None),
                    'local_search_strength': action_params.get('controls', {}).get('local_search_strength', None),
                    'restart_ratio': action_params.get('controls', {}).get('restart_ratio', None),
                    'reward': reward,
                    'hv': info.get('hv', 0.0),
                    'hv_improvement': info.get('hv_improvement', 0.0),
                    'moea_generations': info.get('generations', 0),
                    'total_moea_generations': total_moea_generations,
                    'is_converged': info.get('is_converged', False),
                    'population_size': info.get('population_size', 0),
                    'replay_size': replay_size,
                    'sac_update_step': agent.update_step,
                    'action_mode': action_mode,
                    'best_feasible_ser': info.get('best_feasible_ser', None),
                    'best_feasible_global_gain': info.get('best_feasible_global_gain', 0.0),
                    'best_feasible_step_gain': info.get('best_feasible_step_gain', 0.0),
                    'archive_hv': info.get('archive_hv', 0.0),
                    'archive_hv_gain': info.get('archive_hv_gain', 0.0),
                    'population_hv_before': info.get('population_hv_before', 0.0),
                    'population_hv_after': info.get('population_hv_after', 0.0),
                    'population_hv_gain': info.get('population_hv_gain', 0.0),
                    'population_ser_before': info.get('population_ser_before', None),
                    'population_ser_after': info.get('population_ser_after', None),
                    'population_ser_gain': info.get('population_ser_gain', 0.0),
                    'population_hv_raw_before': info.get('population_hv_raw_before', 0.0),
                    'population_ser_raw_before': info.get('population_ser_raw_before', None),
                    'population_credit_source': info.get('population_credit_source', ''),
                    'preparation_quality_excluded': info.get('preparation_quality_excluded', False),
                    'archive_policy_ready': info.get('archive_policy_ready', True),
                    'sac_update_step_context': info.get('sac_update_step_context', 0),
                    'sac_action_mode_context': info.get('sac_action_mode_context', action_mode),
                    'population_hv_credit': info.get('population_hv_credit', 0.0),
                    'population_ser_credit': info.get('population_ser_credit', 0.0),
                    'population_hv_credit_scale': info.get('population_hv_credit_scale', None),
                    'population_ser_credit_scale': info.get('population_ser_credit_scale', None),
                    'action_quality': info.get('action_quality', 0.0),
                    'global_quality_reward': info.get('global_quality_reward', 0.0),
                    'spea2_updates': info.get('spea2_updates', 0),
                    'nsga2_updates': info.get('nsga2_updates', 0),
                    'algorithm_update_ratio': info.get('algorithm_update_ratio', 1.0),
                    'restart_count': info.get('restart_count', 0),
                }
            else:
                step_record = {
                    'episode': episode + 1,
                    'sac_step': sac_step_number,
                    'global_step': global_step,
                    'epsilon': action_params.get('epsilon', 0.0),
                    'tau': action_params.get('tau', 0.0),
                    'delta': action_params.get('delta', 0.0),
                    'beta': action_params.get('beta', 0.0),
                    'pm': action_params.get('pm', 0.0),
                    'pc': action_params.get('pc', 0.0),
                    'reward': reward,
                    'hv': info.get('hv', 0.0),
                    'hv_improvement': info.get('hv_improvement', 0.0),
                    'moea_generations': info.get('generations', 0),
                    'total_moea_generations': total_moea_generations,
                    'is_converged': info.get('is_converged', False),
                    'population_size': info.get('population_size', 0),
                    'replay_size': replay_size,
                    'sac_update_step': agent.update_step,
                    'action_mode': action_mode,
                    'best_feasible_ser': info.get('best_feasible_ser', None),
                    'best_feasible_global_gain': info.get('best_feasible_global_gain', 0.0),
                    'best_feasible_step_gain': info.get('best_feasible_step_gain', 0.0),
                    'archive_hv': info.get('archive_hv', 0.0),
                    'archive_hv_gain': info.get('archive_hv_gain', 0.0),
                    'population_hv_before': info.get('population_hv_before', 0.0),
                    'population_hv_after': info.get('population_hv_after', 0.0),
                    'population_hv_gain': info.get('population_hv_gain', 0.0),
                    'population_ser_before': info.get('population_ser_before', None),
                    'population_ser_after': info.get('population_ser_after', None),
                    'population_ser_gain': info.get('population_ser_gain', 0.0),
                    'population_hv_raw_before': info.get('population_hv_raw_before', 0.0),
                    'population_ser_raw_before': info.get('population_ser_raw_before', None),
                    'population_credit_source': info.get('population_credit_source', ''),
                    'preparation_quality_excluded': info.get('preparation_quality_excluded', False),
                    'archive_policy_ready': info.get('archive_policy_ready', True),
                    'sac_update_step_context': info.get('sac_update_step_context', 0),
                    'sac_action_mode_context': info.get('sac_action_mode_context', action_mode),
                    'population_hv_credit': info.get('population_hv_credit', 0.0),
                    'population_ser_credit': info.get('population_ser_credit', 0.0),
                    'population_hv_credit_scale': info.get('population_hv_credit_scale', None),
                    'population_ser_credit_scale': info.get('population_ser_credit_scale', None),
                    'action_quality': info.get('action_quality', 0.0),
                    'global_quality_reward': info.get('global_quality_reward', 0.0),
                }
            
            episode_parameters_history.append(step_record)
            all_parameters_history.append(step_record)
            
            best_gain = float(info.get('best_feasible_global_gain', 0.0) or 0.0)
            archive_hv_gain = float(info.get('archive_hv_gain', 0.0) or 0.0)
            if best_gain > best_feasible_gain_eps or archive_hv_gain > best_feasible_gain_eps:
                episode_no_global_improvement_steps = 0
            else:
                episode_no_global_improvement_steps += 1

            if info.get('is_converged', False):
                episode_converged = True
            
            print(
                f"性能指标: 奖励: {reward:.4f}, HV: {info.get('hv', 0.0):.4f}, "
                f"档案HV: {info.get('archive_hv', 0.0):.4f}, 档案增益: {archive_hv_gain:.6f}, "
                f"action_mode={action_mode}, best_SER_gain={best_gain:.6f}"
            )

            if (
                early_stop_on_stagnation
                and sac_step_number >= stagnation_min_steps
                and info.get('is_converged', False)
                and episode_no_global_improvement_steps >= stagnation_patience
            ):
                print(
                    f"  [SAC-EARLY-STOP] Episode {episode+1} 在第 {sac_step_number} 步停止："
                    f"MOEA已收敛且连续 {episode_no_global_improvement_steps} 步未刷新全局best-feasible SER或档案HV"
                )
                break
            
            # --- 关键逻辑：当前Episode成功收敛或超时，打断小循环，进入下一次大循环 ---
            if terminated or truncated:
                if info.get('terminated_by_convergence', False):
                    print(f"  √ 局部最优突破完毕！Episode {episode+1} 在第 {sac_step_number} 步物理收敛，本回合结束。")
                else:
                    print(f"  ! Episode {episode+1} 达到步数上限提前终止，本回合结束。")
                break
        
        # 记录Episode统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        #保存Pareto前沿（每个Episode结束或收敛时保存成果）
        save_interval = config['training'].get('save_interval', 10)
        is_final_episode = (episode + 1) == max_episodes
        output_cfg = config.get('output', {})
        save_episode_artifacts = bool(output_cfg.get('save_episode_artifacts', False))
        save_unconverged_episode_artifacts = bool(output_cfg.get('save_unconverged_episode_artifacts', False))
        save_final_episode_artifacts = bool(output_cfg.get('save_final_episode_artifacts', False))
        should_save = save_episode_artifacts and (
            ((episode + 1) % save_interval == 0) or
            (save_unconverged_episode_artifacts and (not episode_converged)) or
            (save_final_episode_artifacts and is_final_episode)
        )
        
        if should_save:
            # 获取当前的步数 n (注意：SAC 步数通常从 1 开始计数比较直观)
            sac_step_n = info.get('episode_step', 0) 
            status = f"已在第 {sac_step_n} 步收敛" if episode_converged else f"未收敛 (已达 {sac_step_n} 步)"
            
            print(f"\n[Episode {episode+1}] 正在导出数据至 SAC_{sac_step_n} 文件夹...")
            
            try:
                pareto_front = moea_wrapper.get_last_pareto()
                if pareto_front and len(pareto_front) > 0:
                    episode_dir = Path("results") / f"episode_{episode+1}"
                    pareto_dir = episode_dir / f"SAC_{sac_step_n}"
                    pareto_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 1. 保存方案 CSV
                    moea_wrapper.save_pareto_front_to_csv(pareto_front, save_dir=pareto_dir)
                    
                    # 2. 绘制约束违约分析图
                    # 【修复】：传入moea_wrapper.model（PPAModel）而不是simulator（ICDesignSimulator）
                    # PPAModel有evaluate_with_details方法，可以返回精确的真实PPA值
                    # ICDesignSimulator只有evaluate方法，无法提供精确值
                    export_constraint_violation_views(
                        moea_wrapper=moea_wrapper,
                        config=config,
                        save_dir=pareto_dir,
                        all_candidates=pareto_front,
                    )

                    # 3. 绘制全维度 Pareto 前沿对比图
                    # 【修复】：使用moea_wrapper.model而不是simulator来获取PPA值
                    try:
                        from utils.plot_pareto_4d import plot_pareto_4d
                        total_area_list, total_power_list, worst_slack_list, sens_list = [], [], [], []
                        
                        model = moea_wrapper.model
                        for ind in pareto_front:
                            try:
                                # 使用PPAModel.evaluate_with_details获取精确值
                                if hasattr(model, 'evaluate_with_details'):
                                    details = model.evaluate_with_details(ind.gene)
                                    area_new = details[0]
                                    power_new = details[1]
                                    slack_new = details[2]  # 这已经是计算好的min_slack
                                    sens_new = details[3]
                                else:
                                    res = simulator.evaluate(ind.gene)
                                    area_new = res[0]
                                    power_new = res[1]
                                    slack_new = res[2]
                                    sens_new = res[3]
                                
                                total_area_list.append(area_new)
                                total_power_list.append(power_new)
                                worst_slack_list.append(slack_new)
                                sens_list.append(sens_new)
                            except Exception:
                                continue
                        
                        if total_area_list:
                            pareto_data = {
                                "total_area": np.array(total_area_list),
                                "total_power": np.array(total_power_list),
                                "worst_slack": np.array(worst_slack_list),
                                "design_sensitivity": np.array(sens_list)
                            }
                            plot_pareto_4d(pareto_data, save_path=str(pareto_dir / "pareto_front_4d.png"))
                            print(f"  √ 全维度图表已保存: {pareto_dir}")
                    except Exception as e_pareto:
                        print(f"  ! Pareto 绘图失败: {e_pareto}")
                        
                else:
                    print(f"  ! 警告: Episode {episode+1} 步数 {sac_step_n} 未能提取有效前沿")
            except Exception as e:
                print(f"  ! 核心保存逻辑出错: {e}")
        
        convergence_history.append({
            'episode': episode + 1,
            'is_converged': episode_converged,
            'moea_generations': total_moea_generations,
            'final_hv': info.get('hv', 0.0),
            'hv_improvement': info.get('hv_improvement', 0.0),
            'episode_reward': episode_reward,
            'episode_length': episode_length
        })
        
        episode_params_df = pd.DataFrame(episode_parameters_history)
        episode_params_path = log_dir / f'parameters_episode_{episode+1}.csv'
        episode_params_df.to_csv(episode_params_path, index=False)
        
        # 定期保存网络权重模型
        if (episode + 1) % eval_interval == 0 or (episode + 1) == max_episodes:
            current_checkpoint = checkpoint_dir / f'checkpoint_episode_{episode+1}.pth'
            agent.save_checkpoint(str(current_checkpoint))
            
            recent_episodes = min(eval_interval, len(episode_rewards))
            if recent_episodes > 0:
                recent_avg_reward = np.mean(episode_rewards[-recent_episodes:])
                if recent_avg_reward > best_reward:
                    best_reward = recent_avg_reward
                    best_checkpoint = checkpoint_dir / 'best_model.pth'
                    agent.save_checkpoint(str(best_checkpoint))
                    print(f"  ★ 保存新的最佳网络模型 (平均奖励: {best_reward:.4f})")
        
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        print(f"\n训练进度: {episode+1}/{max_episodes} Episode完成, 历史平均奖励: {avg_reward:.4f}")
    
    # 训练完成后的最终保存
    all_params_df = pd.DataFrame(all_parameters_history)
    all_params_path = log_dir / 'all_parameters_history.csv'
    all_params_df.to_csv(all_params_path, index=False)
    
    final_checkpoint = checkpoint_dir / 'final_model.pth'
    agent.save_checkpoint(str(final_checkpoint))
    print(f"\n最终SAC模型已保存到: {final_checkpoint}")

    convergence_df = pd.DataFrame(convergence_history)
    convergence_path = log_dir / 'convergence_history.csv'
    convergence_df.to_csv(convergence_path, index=False)

    output_cfg = config.get('output', {})
    final_results_root = Path(output_cfg.get('final_results_dir', 'results'))
    final_results_dir = final_results_root / log_dir.name
    final_results_dir.mkdir(parents=True, exist_ok=True)

    # ========== 保存最终Pareto前沿 ==========
    print("\n" + "="*80)
    print("保存最终Pareto前沿...")
    print("="*80)

    try:
        # 获取moea_wrapper
        moea_wrapper = env.moea if hasattr(env, 'moea') else None

        if moea_wrapper is not None and hasattr(moea_wrapper, 'get_last_pareto'):
            # 获取最终的Pareto前沿
            final_pareto = moea_wrapper.get_last_pareto()

            if final_pareto and len(final_pareto) > 0:
                # 使用PPAModel的评估方法计算详细指标
                pareto_rows = []
                pareto_front_size = len(final_pareto)

                # 获取原始设计敏感性
                original_sensitivity = getattr(moea_wrapper, 'sens_orig', 0.175777)
                design_area = getattr(moea_wrapper, 'area_orig', 228662.0)
                design_power = getattr(moea_wrapper, 'power_orig', 29210.0)

                # 获取约束限制
                constraints = config.get('constraint_thresholds', {})
                area_limit = constraints.get('delta_area_max', 0.2)
                power_limit = constraints.get('delta_power_max', 0.5)
                slack_min = constraints.get('slack_min', 0.0)

                for idx, ind in enumerate(final_pareto):
                    try:
                        genes = ind.gene.flatten()

                        # 使用model的详细评估
                        if hasattr(moea_wrapper, 'model') and moea_wrapper.model is not None:
                            results = moea_wrapper.model.evaluate_with_details(genes)
                            if len(results) >= 9:
                                (area_total, power_total, min_slack, total_sensitivity,
                                 reinforced_indices, area_increase, power_increase, slack_decrease,
                                 sens_improvement_ratio) = results
                            else:
                                continue
                        else:
                            continue

                        # 计算关键指标（与贪婪算法一致）
                        tmr_count = int(np.sum(genes))
                        residual_ratio = total_sensitivity / max(original_sensitivity, 1e-20)
                        ser_reduction_pct = (1.0 - residual_ratio) * 100.0

                        area_overhead = area_increase / max(design_area, 1e-20)
                        power_overhead = power_increase / max(design_power, 1e-20)

                        # 约束违例检查
                        feasible = (
                            area_overhead <= area_limit + 1e-12 and
                            power_overhead <= power_limit + 1e-12 and
                            min_slack >= slack_min - 1e-12
                        )

                        # 构建基因字符串
                        gene_str = ''.join(str(int(g)) for g in genes)

                        pareto_rows.append({
                            'method': 'sac_moea',
                            'seed': 'trained',
                            'eval_index': idx,
                            'solution_id': f'sac_pareto_{idx:04d}',
                            'tmr_count': tmr_count,
                            'residual_ser': total_sensitivity,
                            'residual_ratio': residual_ratio,
                            'ser_reduction_pct': ser_reduction_pct,
                            'area_total': area_total,
                            'power_total': power_total,
                            'worst_slack': min_slack,
                            'area_overhead': area_overhead,
                            'power_overhead': power_overhead,
                            'area_overhead_pct': area_overhead * 100.0,
                            'power_overhead_pct': power_overhead * 100.0,
                            'feasible': feasible,
                            'sens_improvement_ratio': sens_improvement_ratio,
                            'archive_source': getattr(ind, 'archive_source', 'unknown'),
                            'is_fixed_baseline': bool(getattr(ind, 'is_fixed_baseline', False)),
                            'genes': gene_str
                        })
                    except Exception as e:
                        print(f"  警告: 处理Pareto解 {idx} 时出错: {e}")
                        continue

                if pareto_rows:
                    # 保存完整Pareto前沿
                    pareto_df = pd.DataFrame(pareto_rows)

                    # 保存完整Pareto前沿
                    pareto_front_path = final_results_dir / 'sac_moea_pareto_front.csv'
                    pareto_df.to_csv(pareto_front_path, index=False)
                    if hasattr(moea_wrapper, 'get_search_attribution'):
                        attribution = moea_wrapper.get_search_attribution(final_pareto)
                        attribution_path = final_results_dir / 'search_attribution.csv'
                        pd.DataFrame([attribution]).to_csv(attribution_path, index=False)
                        print(
                            "  Search attribution: "
                            f"baseline HV={attribution['baseline_hv']:.6f}, "
                            f"final HV={attribution['final_hv']:.6f}, "
                            f"gain={attribution['search_hv_gain']:.6f}, "
                            f"nonbaseline points={attribution['nonbaseline_front_points']}, "
                            f"exploration points={attribution['exploration_step_front_points']}, "
                            f"learned-policy points={attribution['policy_step_front_points']}"
                        )
                    final_register_dir = final_results_dir / 'final_solution_registers'
                    try:
                        export_stats = moea_wrapper.save_pareto_front_to_csv(
                            final_pareto,
                            save_dir=final_register_dir,
                            filename_prefix='solution',
                            deduplicate=True,
                            separate_current_history=True,
                        )
                        print(f"  Final per-solution register CSVs saved to: {final_register_dir}")
                        print(f"  Export stats: {export_stats}")
                    except Exception as e_export:
                        print(f"  ! Failed to export final per-solution register CSVs: {e_export}")
                    if output_cfg.get('generate_final_constraint_views', True):
                        export_constraint_violation_views(
                            moea_wrapper=moea_wrapper,
                            config=config,
                            save_dir=final_results_dir,
                            all_candidates=final_pareto,
                        )
                    print(f"  Pareto前沿已保存到: {pareto_front_path}")
                    print(f"  Pareto前沿大小: {len(pareto_df)}")

                    # 找出最佳可行解
                    feasible_df = pareto_df[pareto_df['feasible'] == True]
                    if not feasible_df.empty:
                        # 按residual_ratio排序，找最佳解
                        best_feasible = feasible_df.sort_values('residual_ratio').iloc[0]

                        # 保存最佳可行解
                        best_path = final_results_dir / 'sac_moea_best_feasible.csv'
                        best_feasible.to_frame().T.to_csv(best_path, index=False)
                        print(f"\n  【SAC+MOEA 最佳可行解】")
                        print(f"    TMR加固数量: {int(best_feasible['tmr_count'])}")
                        print(f"    残余敏感性比例: {best_feasible['residual_ratio']*100:.2f}%")
                        print(f"    敏感性改善: {best_feasible['ser_reduction_pct']:.1f}%")
                        print(f"    面积增加: {best_feasible['area_overhead_pct']:.1f}%")
                        print(f"    功耗增加: {best_feasible['power_overhead_pct']:.1f}%")
                        print(f"    最差WNS: {best_feasible['worst_slack']:.4f} ns")
                    else:
                        print(f"  警告: Pareto前沿中没有可行解！")

                else:
                    print(f"  警告: 无法提取有效的Pareto解")
            else:
                print(f"  警告: Pareto前沿为空")
        else:
            print(f"  警告: 无法获取MOEA wrapper或Pareto前沿")

    except Exception as e:
        print(f"  保存Pareto前沿时出错: {e}")
        import traceback
        traceback.print_exc()

    return agent, env, logger, convergence_history, all_params_df

def main():
    np.seterr(all='ignore')
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    import gc
    import torch

    parser = argparse.ArgumentParser(description="SAC for MOEA Parameter Tuning")
    parser.add_argument('--config', type=str, default='configs/sac_config.yaml', help='配置文件路径')
    parser.add_argument('--csv_file', type=str, default=None, help='寄存器数据CSV文件路径')
    parser.add_argument('--num_registers', type=int, default=1000, help='寄存器数量（当没有CSV文件时）')
    parser.add_argument('--K', type=int, default=1380, help='预筛选寄存器数量（默认1380，匹配贪婪算法）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')

    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"sac_moea_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(log_dir=str(log_dir), experiment_name="sac_moea_experiment")

    try:
        config = load_config(args.config)
        gc.collect()
        
        csv_file_path = args.csv_file
        
        if not csv_file_path:
            csv_file_path = config.get('moea', {}).get('csv_file')
        
        if csv_file_path and Path(csv_file_path).exists():
            simulator = ICDesignSimulator(csv_file_path=csv_file_path, K=args.K, config=config)
        else:
            simulator = ICDesignSimulator(csv_file_path=None, num_registers=args.num_registers, K=args.K, config=config)
        
        gc.collect()
        
        reg_table = create_reg_table(simulator, config, logger)
        gc.collect()
        
        if not validate_reg_table(reg_table, logger):
            return
        
        # ======= 启动训练 =======
        agent, env, training_logger, convergence_history, all_params_df = train_sac(config, reg_table, log_dir, simulator)
        
        print("\n" + "="*80)
        completed_episodes = int(config.get('training', {}).get('max_episodes', 50))
        print(f"全部{completed_episodes}个回合训练完成 - 清理内存并生成分析图")
        print("="*80)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            from utils.plot_parameters import (
                plot_parameter_history, plot_parameter_vs_performance, plot_parameter_statistics
            )
            plot_parameter_history(log_dir, save_fig=True)
            plot_parameter_vs_performance(log_dir, save_fig=True)
            plot_parameter_statistics(log_dir, save_fig=True)
        except Exception as e:
            pass
        
    except MemoryError as e:
        print("\n" + "="*80)
        print("内存错误!")
        print("="*80)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    finally:
        try:
            if 'env' in locals() and env is not None:
                if hasattr(env, 'close'):
                    env.close()
        except Exception:
            pass
        if 'logger' in locals():
            logger.close()
        print("\n程序正常安全结束")

if __name__ == "__main__":
    main()
