import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 配置参数
DATA_TYPES = {
    "torque": {
        "pattern": r"Output Torque: \[([-\d\., ]+)\]",
        "label": "Torque (Nm)"
    },
    "action": {
        "pattern": r"Actions\[0 ~ 11\] --> joint_target: \[([-\d\., ]+)\]",
        "label": "Action"
    }
}

def parse_log_data(log_file, data_type):
    """解析指定类型的数据（Torque/Action）"""
    step_pattern = re.compile(r"Step (\d+):")
    data_pattern = re.compile(DATA_TYPES[data_type]["pattern"])
    
    steps, data_matrix = [], []
    current_step = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # 捕获步骤编号
            if step_match := step_pattern.search(line):
                current_step = int(step_match.group(1))
            
            # 捕获目标数据
            if (data_match := data_pattern.search(line)) and current_step is not None:
                steps.append(current_step)
                values = [float(x) for x in data_match.group(1).split(", ")]
                data_matrix.append(values)
    
    return steps, np.array(data_matrix)

def calculate_stats(data_matrix):
    """计算统计指标"""
    return {
        "max_abs": np.max(np.abs(data_matrix), axis=0).tolist(),
        "mean": np.mean(data_matrix, axis=0).tolist(),
        "mean_abs": np.mean(np.abs(data_matrix), axis=0).tolist(),
        "std": np.std(data_matrix, axis=0).tolist()
    }

def print_statistics(stats, data_type):
    """打印统计结果"""
    label = DATA_TYPES[data_type]["label"]
    print(f"\n[ {label} 统计结果 ]")
    print(f"{'关节':<4} | {'最大绝对值':<6} | {'实际平均值':<6} | {'绝对平均值':<6} | {'标准差':<6}")
    print("-" * 65)
    
    for idx in range(len(stats["max_abs"])):
        print(
            f"{idx:<5} | "
            f"{stats['max_abs'][idx]:<12.6f} | "
            f"{stats['mean'][idx]:<12.6f} | "
            f"{stats['mean_abs'][idx]:<12.6f} | "
            f"{stats['std'][idx]:<12.6f}"
        )

def plot_data_curves(steps, data_matrix, stats, data_type):
    """绘制数据曲线"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    label = DATA_TYPES[data_type]["label"]
    
    # 绘制所有关节曲线
    for joint in range(data_matrix.shape[1]):
        plt.plot(
            steps, 
            data_matrix[:, joint], 
            alpha=0.5, 
            label=f'Joint {joint}'
        )
        
        # 添加统计标注线
        plt.axhline(
            stats["max_abs"][joint], 
            color='r', linestyle='--', linewidth=0.8
        )
        plt.axhline(
            stats["mean_abs"][joint], 
            color='b', linestyle='-.', linewidth=0.8
        )
    
    # 图表装饰
    plt.title(
        f"{label} Analysis\n"
        "(Red Dashed: Max Absolute | Blue Dash-dot: Mean Absolute)",
        fontsize=12
    )
    plt.xlabel("Step Number", fontsize=10)
    plt.ylabel(label, fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{data_type}_analysis.png", 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()
    print(f"{label} 图表已保存至：{output_dir.resolve()}/{data_type}_analysis.png")

def main(log_file):
    # 处理 Torque 数据
    torque_steps, torque_data = parse_log_data(log_file, "torque")
    if torque_data.size > 0:
        torque_stats = calculate_stats(torque_data)
        print_statistics(torque_stats, "torque")
        plot_data_curves(torque_steps, torque_data, torque_stats, "torque")
    else:
        print("警告：未找到 Torque 数据")
    
    # 处理 Action 数据
    action_steps, action_data = parse_log_data(log_file, "action")
    if action_data.size > 0:
        action_stats = calculate_stats(action_data)
        print_statistics(action_stats, "action")
        plot_data_curves(action_steps, action_data, action_stats, "action")
    else:
        print("警告：未找到 Action 数据")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python data_analyzer.py <log_file>")
        sys.exit(1)
    
    main(sys.argv[1])
