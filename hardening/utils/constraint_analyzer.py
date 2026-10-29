# utils/constraint_analyzer.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from pathlib import Path


class ConstraintAnalyzer:
    """约束违例分析器：统计并可视化面积/功耗/slack/敏感度的约束违例情况。"""

    @staticmethod
    def _get_original_values(simulator: Any) -> Dict[str, float]:
        """
        从模拟器/模型对象中尽量鲁棒地获取原始PPA基准值。
        支持多种字段名（兼容 PPAModel, 旧版 simulator 等）。
        """

        def _first_attr(obj: Any, names, default: float) -> float:
            for n in names:
                if hasattr(obj, n):
                    return float(getattr(obj, n))
            return float(default)

        area_orig = _first_attr(
            simulator,
            ['Area_orig', 'area_orig', 'design_area', 'base_area'],
            100000.0,
        )
        power_orig = _first_attr(
            simulator,
            ['Power_orig', 'power_orig', 'design_power', 'base_power'],
            50.0,
        )
        slack_orig = _first_attr(
            simulator,
            ['WNS_orig', 'slack_orig', 'design_slack', 'base_slack'],
            0.05,
        )
        # 【修复】：sens_orig 的优先级应该与 CSV 中 design_sensitivity_before 一致
        # CSV 第2行第4列是 design_sensitivity_before，对应 simulator.SER_orig
        # 而不是 simulator.sens_orig（那是完全不同的物理量）
        sens_orig = _first_attr(
            simulator,
            ['SER_orig', 'design_sensitivity', 'sens_orig', 'base_sens'],
            1e-5,
        )

        return {
            'area_orig': area_orig,
            'power_orig': power_orig,
            'slack_orig': slack_orig,
            'sens_orig': sens_orig,
        }

    @staticmethod
    def analyze_population_constraints(
        population,
        simulator: Any,
        config: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """分析种群约束违例，返回统计信息（均值/方差/最大最小/违例比例）。"""
        if not population:
            return {}

        # 获取原始值（兼容 PPAModel / 旧模拟器 等不同实现）
        orig_vals = ConstraintAnalyzer._get_original_values(simulator)
        area_orig = orig_vals['area_orig']
        power_orig = orig_vals['power_orig']
        slack_orig = orig_vals['slack_orig']
        sens_orig = orig_vals['sens_orig']
        
        # 【调试】打印获取的原始值
        # print(f"[DEBUG ConstraintAnalyzer] 获取的原始值:")
        # print(f"  area_orig = {area_orig}")
        # print(f"  power_orig = {power_orig}")
        # print(f"  slack_orig = {slack_orig}")
        # print(f"  sens_orig = {sens_orig}")
        
        # 获取约束阈值
        thresholds = config.get('constraint_thresholds', {})
        delta_area_max = thresholds.get('delta_area_max', 0.2)
        delta_power_max = thresholds.get('delta_power_max', 0.15)
        slack_min = thresholds.get('slack_min', slack_orig - 0.05)  # slack最小值（更直观）
        delta_sens_min = thresholds.get('delta_sens_min', 0.5)
        # Ensure delta_sens_min is a sensible ratio in (0,1)
        try:
            delta_sens_min = float(delta_sens_min)
            if not (0.0 < delta_sens_min < 1.0):
                print(f"[WARN] constraint_thresholds.delta_sens_min={delta_sens_min} 不在 (0,1) 范围内，自动调整为 0.5")
                delta_sens_min = 0.5
        except Exception:
            print(f"[WARN] 无法解析 constraint_thresholds.delta_sens_min 的值，使用默认 0.5")
            delta_sens_min = 0.5
        
        # 收集数据
        violations_data = {
            'area': [], 'power': [], 'slack': [], 'sens': [],
            'delta_area': [], 'delta_power': [], 'delta_slack': [], 'delta_sens': [],
            # 额外的用于可视化对比的数据
            'raw_wns': [],            # 每个个体的真实 worst_slack（可能为负，单位 ns）
            'adjusted_wns': [],       # 基于 frequency_after_tmr 的调整后 worst_slack（单位 ns），若无法计算则与 raw_wns 相同
            'frequency_after_hz': []  # 计算得到的 frequency_after_tmr（Hz）；供调试/过滤使用
        }

        # 如果 simulator 提供了敏感性权重与 base_total_failure_rate，则收集单寄存器失效率分布
        try:
            if hasattr(simulator, 'sensitivity_weights') and hasattr(simulator, 'base_total_failure_rate'):
                p_vals = (np.array(simulator.sensitivity_weights, dtype=np.float64) * float(simulator.base_total_failure_rate)).tolist()
            else:
                p_vals = []
        except Exception:
            p_vals = []
        violations_data['p_values'] = p_vals
        violations_data['base_total_failure_rate'] = [float(getattr(simulator, 'base_total_failure_rate', np.nan))]
        
        # 使用evaluate_with_details获取真实的整体PPA指标（而不是objectives中的归一化值）
        model_obj = simulator.model if hasattr(simulator, 'model') else simulator
        use_real_metrics = hasattr(model_obj, 'evaluate_with_details')
        
        for ind in population:
            if ind.objectives is None:
                continue
            
            # 尝试使用真实的整体PPA指标
            processed = False
            if use_real_metrics:
                try:
                    genes = ind.gene.flatten() if hasattr(ind, 'gene') else None
                    if genes is not None:
                        # 确保genes长度正确
                        if hasattr(model_obj, 'K'):
                            if len(genes) != model_obj.K:
                                if len(genes) > model_obj.K:
                                    genes = genes[:model_obj.K]
                                else:
                                    genes = np.pad(genes, (0, model_obj.K - len(genes)), 'constant')
                        
                        # 使用正确的 model_obj 调用评估
                        results = model_obj.evaluate_with_details(genes)
                        if results is not None and len(results) >= 4:
                            area_total = float(results[0])
                            power_total = float(results[1])
                            
                            # 重新计算真实的worst_slack（可能为负）
                            worst_slack_real = None
                            if hasattr(model_obj, 'slack') and hasattr(model_obj, 'voter_delay'):
                                slack_after = model_obj.slack.copy()
                                tmr_indices = np.where(np.array(genes) == 1)[0]
                                if len(tmr_indices) > 0:
                                    slack_after[tmr_indices] -= model_obj.voter_delay
                                worst_slack_real = float(np.min(slack_after))  # 真实值，可能为负
                            else:
                                worst_slack_real = float(results[2])
                            
                            remaining_sens = float(results[3])
                            
                            # 【修复】：remaining_sens 是绝对敏感性值（约0.6），需要转换为比例值
                            # sens_ratio = 绝对敏感性 / 原始敏感性 = remaining_sens / sens_orig
                            sens_ratio = remaining_sens / sens_orig if sens_orig > 0 else 0
                            
                            # 【调试】打印敏感性计算
                            # print(f"[DEBUG ConstraintAnalyzer] evaluate_with_details 结果:")
                            # print(f"  remaining_sens = {remaining_sens}")
                            # print(f"  sens_orig = {sens_orig}")
                            # print(f"  sens_ratio = {sens_ratio}")
                            
                            # 计算Δ值（基于真实值）
                            delta_area = (area_total - area_orig) / area_orig
                            delta_power = (power_total - power_orig) / power_orig
                            delta_slack = slack_orig - worst_slack_real  # slack的减少量
                            
                            # 计算违例
                            area_viol = max(0, delta_area - delta_area_max)
                            power_viol = max(0, delta_power - delta_power_max)
                            # Slack约束：slack必须 >= slack_min（使用真实值）
                            slack_viol = max(0, slack_min - worst_slack_real)  # 如果worst_slack_real < slack_min，则违例
                            # 【修复】：sens_ratio 是比例值(0~1)，直接与阈值比较
                            # 约束要求：sens_ratio <= delta_sens_min
                            sens_viol = max(0, sens_ratio - delta_sens_min)

                            processed = True
                            # 记录 raw WNS 以便绘图
                            violations_data['raw_wns'].append(worst_slack_real)

                            # 计算 frequency_after_tmr（尽量兼容不同 simulator 实现）
                            try:
                                # 尝试从 simulator 或 simulator.model 中获取频率与 voter_delay
                                if hasattr(simulator, 'model'):
                                    model_obj = simulator.model
                                else:
                                    model_obj = simulator

                                freq0 = float(getattr(model_obj, 'frequency', getattr(simulator, 'frequency', 200e6)))
                                voter = float(getattr(model_obj, 'voter_delay', getattr(simulator, 'voter_delay', 0.0)))

                                # 构造 slack_after 数组（以 ns 为单位）
                                if hasattr(model_obj, 'slack'):
                                    slack_arr = np.array(model_obj.slack, dtype=np.float64).copy()
                                    tmr_indices = np.where(np.array(genes) == 1)[0]
                                    if tmr_indices.size > 0:
                                        slack_arr[tmr_indices] -= voter
                                else:
                                    # 无路径信息时无法计算调整后的值，退回到 worst_slack_real
                                    slack_arr = None

                                if slack_arr is not None:
                                    period_ns = 1e9 / freq0
                                    path_delays = period_ns - slack_arr
                                    # 注意：此处在静态方法中没有 self，可使用外部已解析的 slack_min 或传入的 config
                                    try:
                                        slack_min_ns = float(slack_min)
                                    except Exception:
                                        # 退回为从 config 中读取（向后兼容）
                                        slack_min_ns = float(config.get('constraint_thresholds', {}).get('slack_min', 0.0))
                                    denom = slack_min_ns + path_delays + voter
                                    with np.errstate(divide='ignore', invalid='ignore'):
                                        freq_bounds_hz = np.where(denom > 0, 1e9 / denom, np.inf)
                                    frequency_after_hz = float(np.min(freq_bounds_hz)) if freq_bounds_hz.size > 0 else freq0
                                    if np.isfinite(frequency_after_hz):
                                        frequency_after_hz = min(frequency_after_hz, freq0)
                                    else:
                                        frequency_after_hz = freq0

                                    # 计算调整后 worst_slack： worst_slack + (period_new - period_old)
                                    period_new_ns = 1e9 / frequency_after_hz
                                    adjusted_worst = worst_slack_real + (period_new_ns - period_ns)
                                else:
                                    frequency_after_hz = freq0
                                    adjusted_worst = worst_slack_real

                            except Exception:
                                frequency_after_hz = float(getattr(simulator, 'frequency', 200e6))
                                adjusted_worst = worst_slack_real

                            violations_data['adjusted_wns'].append(float(adjusted_worst))
                            violations_data['frequency_after_hz'].append(float(frequency_after_hz))
                except Exception as e:
                    # 如果evaluate_with_details失败，回退到objectives
                    pass
            
            if not processed:
                # RealMoeaWrapper.evaluate 返回的格式: [area_ratio, power_ratio, -WNS, sens_ratio, violation]
                # ParallelEvaluator._calculate_objectives_and_violation 返回的格式: [area_ratio, power_ratio, wns_objective, ser_ratio]
                # area_ratio 和 power_ratio 是比例值，wns_objective 是 WNS 目标值，ser_ratio 是敏感性比例
                
                area_ratio = float(ind.objectives[0])
                power_ratio = float(ind.objectives[1])
                wns_objective = float(ind.objectives[2])
                sens_ratio = float(ind.objectives[3]) if len(ind.objectives) > 3 else 0.5
                
                # 【调试】打印 fallback 路径数据
                # print(f"[DEBUG ConstraintAnalyzer] fallback 路径:")
                # print(f"  objectives格式: [area_ratio={area_ratio:.4f}, power_ratio={power_ratio:.4f}, wns_obj={wns_objective:.4f}, sens={sens_ratio:.4f}]")
                # print(f"  原始基准值: area_orig={area_orig:.2f}, power_orig={power_orig:.2f}, slack_orig={slack_orig:.4f}")
                
                # 转换为绝对值
                area_abs = area_ratio * area_orig
                power_abs = power_ratio * power_orig
                
                # WNS 转换为真实 slack
                # wns_objective = 0.0 if wns < 0 else -wns
                # 如果 wns < 0（负 slack），wns_objective = 0
                # 如果 wns >= 0（正 slack），wns_objective = -wns <= 0
                if wns_objective <= 0:
                    slack_new = -wns_objective  # 恢复原始 slack（如果 slack >= 0）
                else:
                    slack_new = -wns_objective  # 这应该不会出现
                
                print(f"  还原绝对值: area_abs={area_abs:.2f}, power_abs={power_abs:.2f}, slack_new={slack_new:.4f}")
                
                # 计算真实的增量百分比 (用于违例判断和画图)
                delta_area = area_ratio - 1.0  # ratio - 1 = delta 百分比
                delta_power = power_ratio - 1.0
                delta_slack = slack_orig - slack_new  # slack的减少量
                
                # 计算违例 (超过允许阈值的部分)
                area_viol = max(0.0, delta_area - float(delta_area_max))
                power_viol = max(0.0, delta_power - float(delta_power_max))
                # Slack约束：slack必须 >= slack_min (门限)
                slack_viol = max(0.0, float(slack_min) - slack_new)
                # 【修复】：敏感性约束 - sens_ratio 已经是比例(0~1)，直接与阈值比较
                # 约束要求：sens_ratio <= delta_sens_min
                sens_viol = max(0.0, sens_ratio - float(delta_sens_min))
            
            violations_data['area'].append(area_viol)
            violations_data['power'].append(power_viol)
            violations_data['slack'].append(slack_viol)
            violations_data['sens'].append(sens_viol)
            violations_data['delta_area'].append(delta_area)
            violations_data['delta_power'].append(delta_power)
            violations_data['delta_slack'].append(delta_slack)
            # 【修复】：sens_ratio 已经是敏感性比例（0~1），不需要再除以 sens_orig
            violations_data['delta_sens'].append(sens_ratio)
            # 记录 sens_ratio 单独数组，并归一化违例 (ratio - threshold)
            violations_data.setdefault('sens_ratio', []).append(sens_ratio)
            violations_data.setdefault('sens_norm_viol', []).append(max(0.0, sens_ratio - float(delta_sens_min)))
        # 在结束时，如果尚未收集到 p_values 的样本且 simulator 支持，保证其在 violations_data 中存在
        if (not violations_data.get('p_values')) and hasattr(simulator, 'sensitivity_weights'):
            try:
                violations_data['p_values'] = (np.array(simulator.sensitivity_weights, dtype=np.float64) * float(getattr(simulator, 'base_total_failure_rate', np.nan))).tolist()
            except Exception:
                pass
        
        # 计算统计信息
        stats = {}
        for key in violations_data:
            if violations_data[key]:
                stats[f'{key}_mean'] = np.mean(violations_data[key])
                stats[f'{key}_std'] = np.std(violations_data[key])
                stats[f'{key}_max'] = np.max(violations_data[key])
                stats[f'{key}_min'] = np.min(violations_data[key])
                # 对于 sens_ratio，违例定义为 sens_ratio > delta_sens_min
                if key == 'sens_ratio':
                    stats[f'{key}_viol_ratio'] = np.sum(np.array(violations_data[key]) > float(delta_sens_min)) / len(violations_data[key])
                else:
                    stats[f'{key}_viol_ratio'] = np.sum(np.array(violations_data[key]) > 0) / len(violations_data[key])
        
        # 绘制约束违例分布图
        if save_path:
            ConstraintAnalyzer._plot_constraint_violations(violations_data, thresholds, save_path, slack_orig, sens_orig)
        
        return stats
    
    @staticmethod
    def _plot_constraint_violations(violations_data, thresholds, save_path, slack_orig=None, sens_orig=None):
        """绘制约束违例分布图"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        metrics = ['area', 'power', 'slack', 'sens']
        
        for i, metric in enumerate(metrics):
            # 第一排：物理值/增量分布
            ax_delta = axes[0, i]
            
            if metric == 'slack':
                # ===================================================
                # 【核心修复】：绘制真正的绝对 WNS，而不是荒谬的 ΔSlack
                # ===================================================
                wns_data = violations_data.get('raw_wns', violations_data.get('delta_slack', []))
                ax_delta.hist(wns_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                
                # 画出红线（绝对最小值阈值）
                slack_min_val = float(thresholds.get('slack_min', 0.0))
                ax_delta.axvline(slack_min_val, color='red', linestyle='dashed', linewidth=2, 
                                 label=f'Min Slack ({slack_min_val}ns)')
                
                # 标红违例区域：对于时序，WNS 小于红线（左侧）才是违例的！
                x_min, x_max = ax_delta.get_xlim()
                # 确保视图能展示红线左侧的阴影区
                fill_left = min(x_min, slack_min_val - 0.05) 
                ax_delta.axvspan(fill_left, slack_min_val, color='red', alpha=0.15, label='Violation Area')
                ax_delta.set_xlim(left=fill_left)
                
                ax_delta.set_title('Absolute WNS (Timing) Distribution')
                ax_delta.set_xlabel('Worst Negative Slack (ns)')
                ax_delta.set_ylabel('Count')
                ax_delta.legend()
                
            else:
                delta_data = violations_data[f'delta_{metric}']
                ax_delta.hist(delta_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                
                # 对 sens 使用更直观的标签（敏感性比例而非Δ）
                if metric == 'sens':
                    ax_delta.set_title('Sensitivity Ratio Distribution')
                    ax_delta.set_xlabel('Sensitivity_after / Sens_orig')
                else:
                    ax_delta.set_title(f'Δ{metric.capitalize()} Distribution')
                    ax_delta.set_xlabel(f'Δ{metric}')
                ax_delta.set_ylabel('Count')
                
                # 添加阈值线
                if metric == 'area':
                    threshold = thresholds.get('delta_area_max', 0.2)
                    ax_delta.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
                elif metric == 'power':
                    threshold = thresholds.get('delta_power_max', 0.15)
                    ax_delta.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
                elif metric == 'sens':
                    delta_sens_min = thresholds.get('delta_sens_min', 0.5)
                    ax_delta.axvline(delta_sens_min, color='green', linestyle='--', 
                                   label=f'Max Ratio Threshold: {delta_sens_min:.2f}')
                    ax_delta.text(0.02, 0.95, 'Sens_after <= delta_sens_min × Sens_orig', 
                                transform=ax_delta.transAxes, fontsize=9,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

                # 只在有数据时显示图例
                handles, labels = ax_delta.get_legend_handles_labels()
                if handles:
                    ax_delta.legend()
            
            # ===================================================
            # 第二排：违例值分布 (保留原样，这部分的数学逻辑本来就是正确的)
            # ===================================================
            ax_viol = axes[1, i]
            viol_data = violations_data[metric]
            
            if metric == 'sens' and 'sens_norm_viol' in violations_data:
                viol_plot_data = violations_data.get('sens_norm_viol', viol_data)
                ax_viol.hist(viol_plot_data, bins=30, alpha=0.7, color='salmon', edgecolor='black')
                ax_viol.set_title('Sens Normalized Violation Distribution')
                ax_viol.set_xlabel('Sens Ratio - Threshold')
            else:
                ax_viol.hist(viol_data, bins=30, alpha=0.7, color='salmon', edgecolor='black')
                ax_viol.set_title(f'{metric.capitalize()} Violation Distribution')
                ax_viol.set_xlabel(f'{metric} Violation')
            ax_viol.set_ylabel('Count')
            
            # 显示违例比例
            if metric == 'sens' and 'sens_ratio' in violations_data:
                delta_sens_min = thresholds.get('delta_sens_min', 0.5)
                viol_ratio = np.sum(np.array(violations_data['sens_ratio']) > delta_sens_min) / len(violations_data['sens_ratio'])
            else:
                viol_ratio = np.sum(np.array(viol_data) > 0) / len(viol_data) if viol_data else 0
                
            ax_viol.text(0.05, 0.95, f'Violation Ratio: {viol_ratio:.2%}', 
                        transform=ax_viol.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 确保目录存在
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"约束违例分析图保存到: {save_path}")
        
        # 额外绘制 slack 对比图
        try:
            if 'raw_wns' in violations_data and len(violations_data['raw_wns']) > 0:
                raw = np.array(violations_data['raw_wns'], dtype=np.float64)
                adjusted = np.array(violations_data.get('adjusted_wns', raw), dtype=np.float64)
                delta = np.array(violations_data.get('delta_slack', np.zeros_like(raw)), dtype=np.float64)

                order = np.argsort(raw)
                x = np.arange(len(raw))

                fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
                ax2.plot(x, raw[order], marker='o', linestyle='-', color='tab:blue', label='Raw WNS (ns)')
                ax2.plot(x, adjusted[order], marker='s', linestyle='--', color='tab:green', label='Freq-adjusted WNS (ns)')
                ax2.plot(x, delta[order], marker='^', linestyle='-.', color='tab:orange', label='ΔSlack (slack_orig - WNS) (ns)')
                ax2.set_xlabel('Individuals (sorted by raw WNS)')
                ax2.set_ylabel('ns')
                ax2.set_title('Slack Comparison: Raw WNS, ΔSlack and Frequency-adjusted WNS')
                handles, labels = ax2.get_legend_handles_labels()
                if handles:
                    ax2.legend()
                ax2.grid(True, linestyle=':', alpha=0.6)

                comp_path = save_path.with_name(save_path.stem + '_slack_comparison.png')
                fig2.tight_layout()
                fig2.savefig(comp_path, dpi=300, bbox_inches='tight')
                plt.close(fig2)
                print(f"Slack comparison 图保存到: {comp_path}")
        except Exception:
            pass
