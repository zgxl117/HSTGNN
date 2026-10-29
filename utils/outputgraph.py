import matplotlib.pyplot as plt
import matplotlib

# --- 设置中文字体 ---
# 为了在图表中正常显示中文，需要设置支持中文的字体。
# 根据您的操作系统，可能需要调整以下代码块。
# 尝试寻找系统中常用的中文字体，如 SimHei (Windows), Heiti TC (macOS), Droid Sans Fallback (Linux)
font_found = False
font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
for font in font_names:
    try:
        matplotlib.rcParams['font.family'] = font
        font_found = True
        # print(f"已设置字体为: {font}")
        break
    except:
        continue

# 如果没有找到常用字体，Matplotlib可能会使用默认字体，中文可能会显示为方框。
# 此时需要手动指定一个您系统中存在的字体的路径。
if not font_found:
    print("未自动找到常用中文字体，图表中的中文可能无法正常显示。请手动配置字体。")
    # 解决负号'-'显示为方块的问题
    matplotlib.rcParams['axes.unicode_minus'] = False

# --- 数据准备 ---
# 横坐标：偏差阈值 (%)，为了图表符合直觉，按从小到大排列
deviations = [0, 8, 10, 15, 20, 30]
# 纵坐标：对应的准确率 (%)
accuracies = [83.20, 83.20, 84.28, 90.96, 94.34, 97.60]

# --- 创建图表 ---
plt.figure(figsize=(8, 6)) # 设置画布大小

# 绘制折线图
# marker='o' 表示在数据点处标记圆点，linestyle='-' 表示实线连接，linewidth=2 设置线宽
plt.plot(deviations, accuracies, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

# --- 添加图表元素 ---
# 设置标题和坐标轴标签
plt.title('Figure: SPFM Accuracy With Bias rate', fontsize=14, pad=20)
plt.xlabel('Bias (%)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)

# 设置坐标轴刻度，确保X轴只显示10, 20, 30
plt.xticks(deviations)
# 设置Y轴范围，让图表更美观
plt.ylim(80, 100)

# 添加网格线，增加可读性
plt.grid(True, linestyle='--', alpha=0.6)

# --- 添加数据标注 ---
# 在每个数据点旁边标出具体的数值
for x, y in zip(deviations, accuracies):
    label = f"{y:.2f}%"
    # xytext=(0, 10) 表示文本在数据点上方10个单位的位置
    plt.annotate(label, # 文本内容
                 (x, y), # 数据点坐标
                 textcoords="offset points", # 坐标系偏移
                 xytext=(0, 10), # 距离数据点的偏移量
                 ha='center', # 水平居中
                 fontsize=10)

# --- 显示图表 ---
plt.tight_layout() # 自动调整布局，防止标签重叠
plt.show()