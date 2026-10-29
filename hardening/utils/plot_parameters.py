# utils/plot_parameters.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _search_control_specs(df):
    if 'mutation_flips_spea2' not in df.columns:
        return None
    return [
        ('mutation_flips_spea2', 'SPEA2期望变异位数'),
        ('pc_spea2', 'SPEA2交叉概率'),
        ('mutation_flips_nsga2', 'NSGA-II期望变异位数'),
        ('pc_nsga2', 'NSGA-II交叉概率'),
        ('spea2_budget_share', 'SPEA2计算预算占比'),
        ('elite_injection_ratio', '精英注入比例'),
        ('local_search_strength', '局部搜索强度'),
        ('restart_ratio', '受控重启比例'),
    ]

def plot_parameter_history(log_dir: Path, save_fig: bool = True):
    """
    绘制参数变化历史（归一化）

    参数:
    ----------
    log_dir : Path
        日志目录
    save_fig : bool
        是否保存图像
    """
    # 读取所有参数历史
    params_path = log_dir / 'all_parameters_history.csv'
    if not params_path.exists():
        print(f"错误: 找不到参数历史文件: {params_path}")
        return

    df = pd.read_csv(params_path)

    if len(df) == 0:
        print("错误: 参数历史文件为空")
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    search_specs = _search_control_specs(df)
    if search_specs:
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        all_parameters = [
            (name, title, axes.flat[i]) for i, (name, title) in enumerate(search_specs)
        ]
    else:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        all_parameters = [
            ('epsilon', 'ε (支配放宽)', axes[0, 0]),
            ('tau', 'τ (接纳阈值)', axes[0, 1]),
            ('delta', 'δ (多样性阈值)', axes[1, 0]),
            ('beta', 'β (违例罚因子)', axes[1, 1]),
            ('pm', 'pm (变异概率)', axes[2, 0]),
            ('pc', 'pc (交叉概率)', axes[2, 1]),
        ]
    fig.suptitle('SAC参数变化历史（原始数值）', fontsize=16, fontweight='bold')

    # 如果列中包含后缀，则创建平均列以便绘制
    needs_averaging = False
    for col in df.columns:
        if col.endswith('_spea2') or col.endswith('_nsga2'):
            needs_averaging = True
            break

    if needs_averaging:
        # 为每个参数创建平均列
        param_names = ['epsilon', 'tau', 'delta', 'beta', 'pm', 'pc']
        for param in param_names:
            spea2_col = f'{param}_spea2'
            nsga2_col = f'{param}_nsga2'
            if spea2_col in df.columns and nsga2_col in df.columns:
                df[param] = (df[spea2_col] + df[nsga2_col]) / 2
            elif spea2_col in df.columns:
                df[param] = df[spea2_col]
            elif nsga2_col in df.columns:
                df[param] = df[nsga2_col]

    episodes_all = df['episode'].unique()
    max_ep = float(np.max(episodes_all)) if len(episodes_all) > 0 else 1.0

    for param_name, param_title, ax in all_parameters:
        if param_name in df.columns:
            # 按Episode分组并绘制
            episodes = np.sort(df['episode'].unique())

            for episode in episodes:
                episode_data = df[df['episode'] == episode]
                x = episode_data['sac_step'].values
                y = episode_data[param_name].values

                # 使用渐变色（归一化到 [0,1]）
                try:
                    color = plt.cm.viridis(float(episode) / max_ep)
                except Exception:
                    color = plt.cm.viridis(0.5)

                ax.plot(x, y, label=f'Episode {int(episode)}', color=color, alpha=0.7, linewidth=1.5)

            ax.set_title(param_title, fontsize=12, fontweight='bold')
            ax.set_xlabel('SAC步数')
            ax.set_ylabel('参数值')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8, ncol=2)

            # 添加统计信息
            stats_text = f'范围: [{y.min():.4f}, {y.max():.4f}]'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_fig:
        fig_path = log_dir / 'parameter_history.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  参数历史图保存到: {fig_path}")

    # 仅在支持交互的后端下显示图像
    backend = plt.get_backend().lower()
    if not any(k in backend for k in ('agg', 'pdf', 'svg')):
        plt.show()

    # ============== 新增：归一化对比图 ==============
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
    fig2.suptitle('SAC参数变化历史（归一化到[0,1]）', fontsize=16, fontweight='bold')

    if search_specs:
        colors = plt.cm.tab10(np.linspace(0, 1, len(search_specs)))
        param_configs = [
            (name, title, colors[i]) for i, (name, title) in enumerate(search_specs)
        ]
    else:
        param_configs = [
            ('epsilon', 'ε', 'tab:blue'),
            ('tau', 'τ', 'tab:orange'),
            ('delta', 'δ', 'tab:green'),
            ('beta', 'β', 'tab:red'),
            ('pm', 'pm', 'tab:purple'),
            ('pc', 'pc', 'tab:brown')
        ]

    for param_name, param_label, color in param_configs:
        if param_name in df.columns:
            # 归一化到[0,1]
            y = df[param_name].values
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                y_norm = (y - y_min) / (y_max - y_min)
            else:
                y_norm = np.zeros_like(y)

            x = range(len(y))
            ax2.plot(x, y_norm, label=f'{param_label} ({param_name})', color=color, linewidth=2, alpha=0.8)

    ax2.set_xlabel('训练步数', fontsize=12)
    ax2.set_ylabel('归一化参数值', fontsize=12)
    ax2.set_title('所有参数的归一化变化趋势对比', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        fig_path2 = log_dir / 'parameter_history_normalized.png'
        plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
        print(f"  归一化参数历史图保存到: {fig_path2}")

def plot_parameter_vs_performance(log_dir: Path, save_fig: bool = True):
    """
    绘制参数与性能的关系
    
    参数:
    ----------
    log_dir : Path
        日志目录
    save_fig : bool
        是否保存图像
    """
    # 读取所有参数历史
    params_path = log_dir / 'all_parameters_history.csv'
    if not params_path.exists():
        print(f"错误: 找不到参数历史文件: {params_path}")
        return
    
    df = pd.read_csv(params_path)
    
    if len(df) == 0:
        print("错误: 参数历史文件为空")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    search_specs = _search_control_specs(df)
    if search_specs:
        fig, axes = plt.subplots(2, 4, figsize=(21, 9))
        columns_per_row = 4
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        columns_per_row = 3
    performance_col = 'action_quality' if 'action_quality' in df.columns else 'reward'
    performance_title = '动作归因质量增量' if performance_col == 'action_quality' else '奖励'
    fig.suptitle(
        f'参数与{performance_title}关系图（联合动作诊断，非单参数因果效应）',
        fontsize=16,
        fontweight='bold',
    )

    # 参数列表（只绘制存在的列）
    if search_specs:
        all_params = [item[0] for item in search_specs]
        all_titles = [item[1] for item in search_specs]
    else:
        all_params = ['epsilon', 'tau', 'delta', 'beta', 'pm', 'pc']
        all_titles = ['ε (支配放宽)', 'τ (接纳阈值)', 'δ (多样性阈值)',
                      'β (违例罚因子)', 'pm (变异概率)', 'pc (交叉概率)']
    
    # 如果列中包含后缀，则创建平均列
    needs_averaging = False
    if not search_specs:
        for col in df.columns:
            if col.endswith('_spea2') or col.endswith('_nsga2'):
                needs_averaging = True
                break
    
    if needs_averaging:
        # 前四维是两个算法真正共享的参数；pm/pc保留两种算法的独立曲线。
        for param in all_params[:4]:
            spea2_col = f'{param}_spea2'
            nsga2_col = f'{param}_nsga2'
            if spea2_col in df.columns and nsga2_col in df.columns:
                df[param] = (df[spea2_col] + df[nsga2_col]) / 2
            elif spea2_col in df.columns:
                df[param] = df[spea2_col]
            elif nsga2_col in df.columns:
                df[param] = df[nsga2_col]

    existing_params = []
    existing_titles = []
    for param, title in zip(all_params, all_titles):
        has_shared = param in df.columns
        has_specific = any(f'{param}_{suffix}' in df.columns for suffix in ('spea2', 'nsga2'))
        if has_shared or has_specific:
            existing_params.append(param)
            existing_titles.append(title)

    for i, param_name in enumerate(existing_params):
        param_title = existing_titles[i]
        ax = axes[i // columns_per_row, i % columns_per_row]

        if performance_col in df.columns:
            if param_name in ('pm', 'pc') and needs_averaging:
                series = [
                    (f'{param_name}_spea2', 'SPEA2', '#2f78b7'),
                    (f'{param_name}_nsga2', 'NSGA-II', '#d07a00'),
                ]
            else:
                series = [(param_name, '共享参数', '#2f78b7')]

            scatter = None
            for column, label, color in series:
                if column not in df.columns:
                    continue
                valid = df[[column, performance_col]].dropna()
                if valid.empty:
                    continue
                cvals = df.loc[valid.index, 'global_step'] if 'global_step' in df.columns else None
                scatter_kwargs = {'alpha': 0.55, 's': 42, 'label': label}
                if cvals is not None and len(series) == 1:
                    scatter_kwargs.update({'c': cvals, 'cmap': 'viridis'})
                else:
                    scatter_kwargs.update({'color': color})
                scatter = ax.scatter(valid[column], valid[performance_col], **scatter_kwargs)

                # 分位箱中位数比全样本直线拟合更能抵抗训练阶段和联合动作噪声。
                try:
                    q = min(5, max(2, int(np.sqrt(len(valid)))))
                    bins = pd.qcut(valid[column], q=q, duplicates='drop')
                    med = valid.assign(_bin=bins).groupby('_bin', observed=True).median(numeric_only=True)
                    centers = med.index.map(lambda interval: float(interval.mid)).to_numpy()
                    ax.plot(centers, med[performance_col].to_numpy(), color=color, linewidth=2.2, label=f'{label}分箱中位数')
                except Exception:
                    pass

            ax.axhline(0.0, color='#777777', linewidth=1.0, alpha=0.6)
            ax.set_title(f'{param_title} vs {performance_title}', fontsize=12, fontweight='bold')
            ax.set_xlabel(param_title)
            ax.set_ylabel(performance_title)
            ax.grid(True, alpha=0.3)
            
            # 只在有数据时显示图例
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()

            # 添加颜色条（如果有）
            if i == 0 and scatter is not None and 'global_step' in df.columns:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('全局SAC步')
    
    plt.tight_layout()

    if save_fig:
        fig_path = log_dir / 'parameter_performance.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  参数性能关系图保存到: {fig_path}")

    backend = plt.get_backend().lower()
    if not any(k in backend for k in ('agg', 'pdf', 'svg')):
        plt.show()

def plot_parameter_statistics(log_dir: Path, save_fig: bool = True):
    """
    绘制参数统计信息
    
    参数:
    ----------
    log_dir : Path
        日志目录
    save_fig : bool
        是否保存图像
    """
    # 读取所有参数历史
    params_path = log_dir / 'all_parameters_history.csv'
    if not params_path.exists():
        print(f"错误: 找不到参数历史文件: {params_path}")
        return
    
    df = pd.read_csv(params_path)
    
    if len(df) == 0:
        print("错误: 参数历史文件为空")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算统计信息
    search_specs = _search_control_specs(df)
    if search_specs:
        all_params = [item[0] for item in search_specs]
        all_titles = [item[1] for item in search_specs]
    else:
        all_params = ['epsilon', 'tau', 'delta', 'beta', 'pm', 'pc']
        all_titles = ['ε (支配放宽)', 'τ (接纳阈值)', 'δ (多样性阈值)',
                      'β (违例罚因子)', 'pm (变异概率)', 'pc (交叉概率)']
    
    # 如果列中包含后缀，则创建平均列
    needs_averaging = False
    if not search_specs:
        for col in df.columns:
            if col.endswith('_spea2') or col.endswith('_nsga2'):
                needs_averaging = True
                break
    
    if needs_averaging:
        # 为每个参数创建平均列
        for param in all_params:
            spea2_col = f'{param}_spea2'
            nsga2_col = f'{param}_nsga2'
            if spea2_col in df.columns and nsga2_col in df.columns:
                df[param] = (df[spea2_col] + df[nsga2_col]) / 2
            elif spea2_col in df.columns:
                df[param] = df[spea2_col]
            elif nsga2_col in df.columns:
                df[param] = df[nsga2_col]

    existing_params = [p for p in all_params if p in df.columns]
    existing_titles = [t for p, t in zip(all_params, all_titles) if p in df.columns]

    stats_data = []
    for param_name, param_title in zip(existing_params, existing_titles):
        stats_data.append({
            '参数': param_title,
            '均值': df[param_name].mean(),
            '标准差': df[param_name].std(),
            '最小值': df[param_name].min(),
            '最大值': df[param_name].max(),
            '中位数': df[param_name].median()
        })

    stats_df = pd.DataFrame(stats_data)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('参数统计信息', fontsize=16, fontweight='bold')

    # 1. 箱线图
    ax1 = axes[0, 0]
    box_data_all = [df[param].dropna() for param in existing_params]
    # 仅保留非空数据列，同时同步标签，避免维度不匹配
    nonempty_idx = [i for i, b in enumerate(box_data_all) if len(b) > 0]
    box_data = [box_data_all[i] for i in nonempty_idx]
    labels = [existing_titles[i] for i in nonempty_idx]
    if box_data:
        ax1.boxplot(box_data, tick_labels=labels)
    else:
        ax1.text(0.5, 0.5, '无有效参数数据用于箱线图', ha='center', va='center')
    ax1.set_title('参数分布箱线图', fontsize=12, fontweight='bold')
    ax1.set_ylabel('参数值')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # 2. 均值条形图
    ax2 = axes[0, 1]
    x = np.arange(len(stats_df))
    if not stats_df.empty:
        ax2.bar(x, stats_df['均值'], yerr=stats_df['标准差'],
                capsize=5, alpha=0.7, color='skyblue')
        ax2.set_title('参数均值与标准差', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stats_df['参数'], rotation=45, ha='right')
        ax2.set_ylabel('均值 ± 标准差')
        ax2.grid(True, alpha=0.3, axis='y')

    # 3. 范围图
    ax3 = axes[1, 0]
    for i, row in stats_df.iterrows():
        ax3.plot([i, i], [row['最小值'], row['最大值']],
                marker='o', linewidth=3, label=row['参数'])
    ax3.set_title('参数范围', fontsize=12, fontweight='bold')
    ax3.set_xlabel('参数')
    ax3.set_ylabel('参数值范围')
    ax3.set_xticks(range(len(stats_df)))
    if not stats_df.empty:
        ax3.set_xticklabels(stats_df['参数'].values, rotation=45, ha='right')
    else:
        ax3.set_xticklabels([])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)

    # 4. 相关性热图（仅当有至少2个参数时绘制）
    ax4 = axes[1, 1]
    if len(existing_params) >= 2:
        corr_matrix = df[existing_params].corr()
        im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title('参数相关性热图', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(existing_params)))
        ax4.set_yticks(range(len(existing_params)))
        ax4.set_xticklabels(existing_titles, rotation=45, ha='right')
        ax4.set_yticklabels(existing_titles)

        # 添加相关性数值
        for i in range(len(existing_params)):
            for j in range(len(existing_params)):
                text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)

        # 添加颜色条
        plt.colorbar(im, ax=ax4)
    else:
        ax4.text(0.5, 0.5, '参数数量不足，无法绘制相关性热图', ha='center', va='center')
    
    plt.tight_layout()
    
    if save_fig:
        fig_path = log_dir / 'parameter_statistics.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  参数统计图保存到: {fig_path}")
        
        # 保存统计表格
        stats_path = log_dir / 'parameter_statistics.csv'
        stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
        print(f"  参数统计表格保存到: {stats_path}")
    
    # 仅在支持交互的后端下显示图像
    backend = plt.get_backend().lower()
    if not any(k in backend for k in ('agg', 'pdf', 'svg')):
        plt.show()

# 使用示例
if __name__ == "__main__":
    log_dir = Path("logs/sac_moea_20240101_120000")
    plot_parameter_history(log_dir, save_fig=True)
    plot_parameter_vs_performance(log_dir, save_fig=True)
    plot_parameter_statistics(log_dir, save_fig=True)
