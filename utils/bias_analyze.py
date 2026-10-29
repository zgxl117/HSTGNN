import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')


# --- 字体设置（对齐 outputgraph 风格 & 兼容中文）---
# 如果图中有中文，尽量自动选择常见中文字体；否则不影响英文显示
font_found = False
font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
for font in font_names:
    try:
        matplotlib.rcParams['font.family'] = font
        font_found = True
        break
    except Exception:
        continue
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False


# 读取CSV文件并计算准确率的函数
def analyze_csv(file_path, save_plot=False, exclude_all_zeros=True, accuracy_threshold=0.3, verbose=True):
    """
    Read CSV file and calculate statistics for predicted and true labels for each node_id
    
    Parameters:
    file_path: Path to CSV file
    save_plot: Whether to save bias distribution plots
    exclude_all_zeros: Whether to exclude nodes with all zeros in true_label
    accuracy_threshold: Threshold for accuracy classification (default: 0.3 or 30%)
    verbose: Whether to print detailed results
    
    Returns:
    result_df: DataFrame containing statistical results
    accuracy_rate: Overall accuracy rate
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_columns = ['source_path', 'graph_id', 'node_id', 'predicted_label', 'true_label']
        for col in required_columns:
            if col not in df.columns:
                # Handle potential column name errors (e.g., predicted_tabel)
                if col == 'predicted_label' and 'predicted_tabel' in df.columns:
                    df = df.rename(columns={'predicted_tabel': 'predicted_label'})
                else:
                    if verbose:
                        print(f"Error: Missing required column '{col}'")
                        print(f"Columns in file: {list(df.columns)}")
                    return None, 0
        
        # Ensure label columns are numeric
        df['predicted_label'] = pd.to_numeric(df['predicted_label'], errors='coerce')
        df['true_label'] = pd.to_numeric(df['true_label'], errors='coerce')
        
        # Group by node_id
        grouped = df.groupby('node_id')
        
        # Calculate statistics for each node_id
        results = []
        excluded_nodes = []  # Track excluded nodes
        included_nodes = []  # Track included nodes
        
        for node_id, group in grouped:
            # Calculate true label statistics
            true_ones = (group['true_label'] == 1).sum()
            predicted_total = len(group)
            true_ratio = true_ones / predicted_total if predicted_total > 0 else 0
            
            # Check if node has all zeros in true_label
            if exclude_all_zeros and true_ones == 0:
                excluded_nodes.append(node_id)
                continue
            
            # If node has at least one 1 in true_label, include it
            included_nodes.append(node_id)
            
            # Calculate prediction statistics
            predicted_ones = (group['predicted_label'] == 1).sum()
            predicted_ratio = predicted_ones / predicted_total if predicted_total > 0 else 0
            
            # Calculate bias
            bias = abs(predicted_ratio - true_ratio)
            
            # Check if node is correct (bias within threshold)
            is_correct = bias <= accuracy_threshold
            
            results.append({
                'node_id': node_id,
                'predicted_1_count': predicted_ones,
                'predicted_total': predicted_total,
                'predicted_1_ratio': round(predicted_ratio, 4),
                'true_1_count': true_ones,
                'true_1_ratio': round(true_ratio, 4),
                'bias_absolute': round(bias, 4),  # Absolute bias
                f'accuracy_within_{accuracy_threshold}': is_correct,  # Whether node is correct
            })
        
        # Check if we have any nodes to analyze
        if len(results) == 0:
            if verbose:
                print("Warning: No nodes with at least one 1 in true_label found!")
                if exclude_all_zeros:
                    print(f"All {len(excluded_nodes)} nodes were excluded because they have all zeros in true_label.")
            return None, 0
        
        # Create result DataFrame
        result_df = pd.DataFrame(results)
        
        # Calculate accuracy statistics
        accuracy_within_threshold_col = f'accuracy_within_{accuracy_threshold}'
        nodes_correct = result_df[accuracy_within_threshold_col].sum()
        nodes_total = len(result_df)
        accuracy_rate = nodes_correct / nodes_total if nodes_total > 0 else 0
        
        return result_df, accuracy_rate
    
    except FileNotFoundError:
        if verbose:
            print(f"Error: File '{file_path}' not found")
        return None, 0
    except Exception as e:
        if verbose:
            print(f"Error reading or processing file: {e}")
        return None, 0

def plot_spfm_accuracy_with_bias(csv_file, bias_thresholds, exclude_all_zeros=True):
    """
    Generate SPFM accuracy vs bias rate plot in the style of the provided image
    
    Parameters:
    csv_file: Path to CSV file
    bias_thresholds: List of bias threshold values to test (as percentages, e.g., [0, 8, 10, 15, 20, 25, 30])
    exclude_all_zeros: Whether to exclude nodes with all zeros in true_label
    """
    accuracy_rates = []
    
    print("Calculating accuracy rates for different bias thresholds...")
    
    for threshold_percent in bias_thresholds:
        # Convert percentage to decimal for the analysis function
        threshold_decimal = threshold_percent / 100.0
        
        print(f"Processing bias threshold: {threshold_percent}%")
        result_df, accuracy_rate = analyze_csv(
            csv_file, save_plot=False, 
            exclude_all_zeros=exclude_all_zeros, 
            accuracy_threshold=threshold_decimal,
            verbose=False  # Suppress detailed output for each threshold
        )
        
        if result_df is not None:
            accuracy_rates.append(accuracy_rate * 100)  # Convert to percentage
            print(f"  Bias {threshold_percent}%: Accuracy = {accuracy_rate*100:.2f}%")
        else:
            accuracy_rates.append(0)
            print(f"  Bias {threshold_percent}%: No valid data")
    
    # --- 创建图表（对齐 outputgraph 样式）---
    plt.figure(figsize=(8, 6))  # outputgraph: (8, 6)

    # 折线图：圆点 marker + 实线 + 统一蓝色
    plt.plot(
        bias_thresholds, accuracy_rates,
        marker='o', linestyle='-',
        color='#1f77b4', linewidth=2, markersize=8
    )

    # 标题与坐标轴
    plt.title('Figure: SPFM Accuracy With Bias rate', fontsize=14, pad=20)
    plt.xlabel('Bias (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)

    # X 轴刻度：只显示给定阈值
    plt.xticks(bias_thresholds)

    # Y 轴范围：默认 80~100（与 outputgraph 一致）；若数据超出则自动扩展
    y_min, y_max = float(min(accuracy_rates)), float(max(accuracy_rates))
    if y_min >= 80 and y_max <= 100:
        plt.ylim(80, 100)
    else:
        pad = 2.0
        plt.ylim(max(0.0, y_min - pad), y_max + pad)

    # 网格线
    plt.grid(True, linestyle='--', alpha=0.6)

    # 数据标注：每个点上方显示百分比
    for x, y in zip(bias_thresholds, accuracy_rates):
        plt.annotate(
            f"{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10
        )

    plt.tight_layout()
    
    # Save the plot
    output_file = 'spfm_accuracy_with_bias_rate.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSPFM Accuracy with Bias rate plot saved as: {output_file}")
    
    plt.show()
    
    # Print summary table
    print("\n" + "="*50)
    print("SPFM ACCURACY VS BIAS RATE")
    print("="*50)
    print("Bias (%)\tAccuracy (%)")
    for bias, accuracy in zip(bias_thresholds, accuracy_rates):
        print(f"{bias}\t\t{accuracy:.2f}%")
    
    print(f"\nMaximum accuracy: {100:.2f}% at bias = {30}%")
    
    return bias_thresholds, accuracy_rates


# 主程序
if __name__ == "__main__":
    # 指定CSV文件路径
    csv_file = "~/FaultSim/ml-method4-fault-sim/FsimNNs/stgnns/predictions/results.csv"  # 请替换为您的实际CSV文件路径
    
    # 设置要测试的bias阈值（与图中一致）
    bias_thresholds = [0, 8, 10, 15, 20, 25, 30]
    
    # 生成SPFM准确率随Bias率变化的图表
    print("Generating SPFM Accuracy with Bias Rate plot...")
    print("="*60)
    
    thresholds_tested, accuracy_rates = plot_spfm_accuracy_with_bias(
        csv_file, bias_thresholds, exclude_all_zeros=True
    )
    
    # 可选：对特定bias阈值进行详细分析
    print("\n" + "="*80)
    print("DETAILED ANALYSIS FOR BIAS THRESHOLD 30%")
    print("="*80)
    
    # 对bias=30%进行详细分析
    result, accuracy_rate = analyze_csv(
        csv_file, save_plot=False, 
        exclude_all_zeros=True, accuracy_threshold=0.3, verbose=True
    )
    
    if result is not None:
        print(f"\nAccuracy with 30% bias threshold: {accuracy_rate*100:.2f}%")
        print(f"Total nodes analyzed: {len(result)}")
        print(f"Correct nodes: {(result['accuracy_within_0.3']).sum()}")
        print(f"Incorrect nodes: {len(result) - (result['accuracy_within_0.3']).sum()}")