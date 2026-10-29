# utils/logger.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
import csv
from pathlib import Path

class Logger:
    """
    功能完善的日志记录器
    支持：控制台输出、CSV文件、JSON文件、图表保存
    """
    
    def __init__(self, log_dir: Optional[str] = None, 
                 experiment_name: Optional[str] = None,
                 use_tensorboard: bool = False):
        """
        初始化日志记录器
        
        参数:
        ----------
        log_dir : str, optional
            日志目录，如果为None则使用默认目录
        experiment_name : str, optional
            实验名称，如果为None则使用时间戳
        use_tensorboard : bool
            是否使用TensorBoard
        """
        # 创建日志目录
        if log_dir is None:
            log_dir = "logs"
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.figures_dir = self.log_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        self.data_dir = self.log_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # 初始化数据存储
        self.metrics = {}  # 存储各种指标
        self.history = {}  # 存储历史记录
        self.config = {}   # 存储配置
        
        # 初始化CSV文件
        self.csv_file = self.log_dir / "metrics.csv"
        self.csv_initialized = False
        
        # 初始化TensorBoard（如果可用）
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
                self.log(f"TensorBoard日志保存到: {self.log_dir / 'tensorboard'}")
            except ImportError:
                self.log("TensorBoard不可用，继续使用CSV日志")
                self.use_tensorboard = False
        
        # 记录开始时间
        self.start_time = datetime.now()
        self.log(f"实验开始: {experiment_name}")
        self.log(f"日志目录: {self.log_dir}")
    
    def log(self, msg: str, level: str = "INFO"):
        """
        基础日志方法
        
        参数:
        ----------
        msg : str
            日志消息
        level : str
            日志级别（INFO, WARNING, ERROR, DEBUG）
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {msg}"
        print(log_msg)
        
        # 同时写入日志文件
        log_file = self.log_dir / "experiment.log"
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def log_scalar(self, name: str, value: float, step: int):
        """
        记录标量值
        
        参数:
        ----------
        name : str
            指标名称
        value : float
            指标值
        step : int
            步数/代数
        """
        # 存储到内存
        if name not in self.metrics:
            self.metrics[name] = {'steps': [], 'values': []}
        
        self.metrics[name]['steps'].append(step)
        self.metrics[name]['values'].append(value)
        
        # 写入CSV
        self._write_to_csv(name, value, step)
        
        # 写入TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_scalar(name, value, step)
        
        # 每100步记录一次到控制台
        if step % 100 == 0:
            self.log(f"Step {step}: {name} = {value:.4f}")
    
    def log_dict(self, name: str, data_dict: Dict[str, Any], step: int):
        """
        记录字典数据
        
        参数:
        ----------
        name : str
            前缀名称
        data_dict : Dict[str, Any]
            数据字典
        step : int
            步数/代数
        """
        for key, value in data_dict.items():
            if isinstance(value, (int, float)):
                full_name = f"{name}/{key}"
                self.log_scalar(full_name, value, step)
            elif isinstance(value, dict):
                self.log_dict(f"{name}/{key}", value, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        记录多个指标
        
        参数:
        ----------
        metrics : Dict[str, float]
            指标字典
        step : int
            步数/代数
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, step)
    
    def log_pareto_front(self, pareto_front: np.ndarray, 
                        filename: str = "pareto_front.npy"):
        """
        保存Pareto前沿
        
        参数:
        ----------
        pareto_front : np.ndarray
            Pareto前沿数据
        filename : str
            保存文件名
        """
        if pareto_front is not None:
            filepath = self.data_dir / filename
            np.save(filepath, pareto_front)
            self.log(f"Pareto前沿保存到: {filepath}")
    
    def log_population(self, population: List[Any], 
                      filename: str = "population.npy"):
        """
        保存种群数据
        
        参数:
        ----------
        population : List[Any]
            种群数据
        filename : str
            保存文件名
        """
        if population:
            # 尝试提取基因数据
            genes = []
            for ind in population:
                if hasattr(ind, 'gene'):
                    genes.append(ind.gene)
                elif isinstance(ind, np.ndarray):
                    genes.append(ind)
            
            if genes:
                filepath = self.data_dir / filename
                np.save(filepath, np.array(genes))
                self.log(f"种群数据保存到: {filepath}")
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """
        保存参数配置
        
        参数:
        ----------
        parameters : Dict[str, Any]
            参数字典
        """
        self.config.update(parameters)
        
        # 保存为JSON
        param_file = self.log_dir / "parameters.json"
        with open(param_file, 'w') as f:
            json.dump(parameters, f, indent=2, default=str)
        
        self.log(f"参数保存到: {param_file}")
        
        # 同时保存为易读的文本格式
        text_file = self.log_dir / "parameters.txt"
        with open(text_file, 'w') as f:
            f.write("实验参数:\n")
            f.write("=" * 50 + "\n")
            for key, value in parameters.items():
                f.write(f"{key}: {value}\n")
    
    def log_action_history(self, actions: List[np.ndarray] | np.ndarray,
                      filename: str = "actions_history.npy"):
        """
        保存动作历史
        
        参数:
        ----------
        actions : List[np.ndarray] 或 np.ndarray
            动作历史列表或数组
        filename : str
            保存文件名
        """
        # 处理 numpy 数组和列表两种情况
        if isinstance(actions, np.ndarray):
            # 如果是 numpy 数组，检查其第一个维度
            if actions.shape[0] > 0:
                action_list = actions
            else:
                self.log("动作历史为空，跳过保存", "WARNING")
                return
        else:
            # 如果是列表
            if len(actions) > 0:
                action_list = np.array(actions)
            else:
                self.log("动作历史为空，跳过保存", "WARNING")
                return
        
        # 保存数据
        filepath = self.data_dir / filename
        np.save(filepath, action_list)
        
        # 同时保存为CSV便于查看
        csv_file = self.data_dir / "actions_history.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'epsilon', 'tau', 'delta', 'beta', 'pm', 'pc'])
            for i, action in enumerate(action_list):
                writer.writerow([i] + list(action))
        
        self.log(f"动作历史保存到: {filepath}")
    
    def save_checkpoint(self, agent: Any, filename: str = "checkpoint.pth"):
        """
        保存检查点
        
        参数:
        ----------
        agent : Any
            SAC代理对象
        filename : str
            保存文件名
        """
        try:
            if hasattr(agent, 'save_checkpoint'):
                agent.save_checkpoint(self.checkpoint_dir / filename)
                self.log(f"检查点保存到: {self.checkpoint_dir / filename}")
            else:
                self.log("代理没有save_checkpoint方法，跳过保存", "WARNING")
        except Exception as e:
            self.log(f"保存检查点时出错: {e}", "ERROR")
    
    def plot_metrics(self, metric_names: Optional[List[str]] = None,
                    save: bool = True, show: bool = False):
        """
        绘制指标图表
        
        参数:
        ----------
        metric_names : List[str], optional
            要绘制的指标名称列表，如果为None则绘制所有指标
        save : bool
            是否保存图表
        show : bool
            是否显示图表
        """
        if not self.metrics:
            self.log("没有可用的指标数据", "WARNING")
            return
        
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        
        n_metrics = len(metric_names)
        if n_metrics == 0:
            return
        
        # 创建子图
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for i, name in enumerate(metric_names):
            if name in self.metrics:
                data = self.metrics[name]
                axes[i].plot(data['steps'], data['values'], 'b-', linewidth=2)
                axes[i].set_title(name, fontsize=14)
                axes[i].set_xlabel('Step', fontsize=12)
                axes[i].set_ylabel('Value', fontsize=12)
                axes[i].grid(True, alpha=0.3)
                
                # 添加统计信息
                if data['values']:
                    last_value = data['values'][-1]
                    max_value = np.max(data['values'])
                    min_value = np.min(data['values'])
                    mean_value = np.mean(data['values'])
                    
                    stats_text = f"当前: {last_value:.4f} | 最大: {max_value:.4f} | 最小: {min_value:.4f} | 平均: {mean_value:.4f}"
                    axes[i].text(0.02, 0.95, stats_text, transform=axes[i].transAxes,
                                fontsize=10, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                axes[i].text(0.5, 0.5, f"无数据: {name}", 
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=axes[i].transAxes)
                axes[i].set_title(name, fontsize=14)
        
        plt.tight_layout()
        
        if save:
            plot_file = self.figures_dir / "metrics_plot.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.log(f"指标图表保存到: {plot_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(self, metric_pairs: Dict[str, List[str]], 
                       save: bool = True, show: bool = False):
        """
        绘制指标对比图
        
        参数:
        ----------
        metric_pairs : Dict[str, List[str]]
            要对比的指标对，键为图表标题，值为指标名称列表
        save : bool
            是否保存图表
        show : bool
            是否显示图表
        """
        n_pairs = len(metric_pairs)
        if n_pairs == 0:
            return
        
        fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 4 * n_pairs))
        if n_pairs == 1:
            axes = [axes]
        
        for i, (title, metrics) in enumerate(metric_pairs.items()):
            ax = axes[i]
            
            for metric_name in metrics:
                if metric_name in self.metrics:
                    data = self.metrics[metric_name]
                    ax.plot(data['steps'], data['values'], label=metric_name, linewidth=2)
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_file = self.figures_dir / "comparison_plot.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.log(f"对比图表保存到: {plot_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def summary(self):
        """
        生成实验摘要
        """
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = {
            "experiment_start": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_end": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration.total_seconds(),
            "duration_human": str(duration),
            "log_directory": str(self.log_dir),
            "metrics_summary": {}
        }
        
        # 计算各个指标的摘要统计
        for name, data in self.metrics.items():
            if data['values']:
                values = np.array(data['values'])
                summary["metrics_summary"][name] = {
                    "final": float(values[-1]),
                    "max": float(np.max(values)),
                    "min": float(np.min(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "total_steps": len(values)
                }
        
        # 保存摘要
        summary_file = self.log_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # 打印摘要
        self.log("=" * 50)
        self.log("实验摘要")
        self.log("=" * 50)
        self.log(f"开始时间: {summary['experiment_start']}")
        self.log(f"结束时间: {summary['experiment_end']}")
        self.log(f"持续时间: {summary['duration_human']}")
        self.log(f"日志目录: {summary['log_directory']}")
        
        for metric_name, stats in summary["metrics_summary"].items():
            self.log(f"{metric_name}:")
            self.log(f"  最终值: {stats['final']:.4f}")
            self.log(f"  最大值: {stats['max']:.4f}")
            self.log(f"  最小值: {stats['min']:.4f}")
            self.log(f"  平均值: {stats['mean']:.4f}")
            self.log(f"  标准差: {stats['std']:.4f}")
            self.log(f"  总步数: {stats['total_steps']}")
        
        self.log("=" * 50)
    
    def _write_to_csv(self, name: str, value: float, step: int):
        """
        将指标写入CSV文件
        
        参数:
        ----------
        name : str
            指标名称
        value : float
            指标值
        step : int
            步数/代数
        """
        # 检查CSV文件是否存在，如果不存在则创建并写入表头
        if not self.csv_initialized:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'metric', 'value'])
            self.csv_initialized = True
        
        # 追加数据
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, name, value])
    
    def close(self):
        """
        关闭日志记录器
        """
        # 生成摘要
        self.summary()
        
        # 绘制所有指标
        self.plot_metrics(save=True, show=False)
        
        # 关闭TensorBoard写入器
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()
        
        # 记录结束消息
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.log(f"实验结束，总耗时: {duration}")
        self.log(f"所有日志和数据保存在: {self.log_dir}")


# 简化版的日志记录器，用于快速使用
class SimpleLogger:
    """
    简化版日志记录器，保持与原来接口的兼容性
    """
    
    def __init__(self):
        pass
    
    def log(self, msg: str):
        """简单的日志记录"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")
    
    def info(self, msg: str):
        """信息级别日志"""
        self.log(f"[INFO] {msg}")
    
    def warning(self, msg: str):
        """警告级别日志"""
        self.log(f"[WARNING] {msg}")
    
    def error(self, msg: str):
        """错误级别日志"""
        self.log(f"[ERROR] {msg}")
    
    def debug(self, msg: str):
        """调试级别日志"""
        self.log(f"[DEBUG] {msg}")


# 全局日志记录器实例
_logger_instance = None

def get_logger(log_dir: Optional[str] = None, 
               experiment_name: Optional[str] = None,
               use_tensorboard: bool = False) -> Logger:
    """
    获取或创建全局日志记录器实例
    
    参数:
    ----------
    log_dir : str, optional
        日志目录
    experiment_name : str, optional
        实验名称
    use_tensorboard : bool
        是否使用TensorBoard
    
    返回:
    ----------
    Logger: 日志记录器实例
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger(log_dir, experiment_name, use_tensorboard)
    return _logger_instance

def get_simple_logger() -> SimpleLogger:
    """
    获取简化版日志记录器
    
    返回:
    ----------
    SimpleLogger: 简化版日志记录器
    """
    return SimpleLogger()