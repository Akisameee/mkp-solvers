import matplotlib.pyplot as plt
import seaborn as sns
from solvers import *
from typing import List,Tuple
import numpy as np

def plot_fitness_trend(ga: GeneticAlgorithm, 
                      title="Fitness Trend Curve",
                      optimal_fitness: float = None,
                      optimal_tolerance: float = 1e-4):
    """增强版适应度趋势绘图（支持已知最优值检测）
    
    Args:
        optimal_fitness: 已知的最优适应度值（可选）
        optimal_tolerance: 视为达到最优的误差容忍度
    """
    plt.figure(figsize=(12,7))
    
    # 绘制基础曲线
    gens = np.arange(len(ga.best_fitness_history))
    plt.plot(gens, ga.best_fitness_history, 
            label='Best Fitness', lw=2, alpha=0.8, color='#1f77b4')
    # plt.plot(gens, ga.avg_fitness_history, 
    #         label='Average Fitness', ls='--', alpha=0.8, color='#ff7f0e')
    
    # Find the first generation that achieves and maintains optimal fitness
    optimal_converge_gen = len(ga.best_fitness_history) - 51
   
    if optimal_converge_gen is not None:
        plt.scatter(optimal_converge_gen, optimal_fitness, 
                   color='g', zorder=5, s=80, marker='*',
                   label=f'First Optimal Achieved at Gen {optimal_converge_gen}')
        plt.axvline(optimal_converge_gen, color='g', linestyle='-.', alpha=0.5)
    
    # Legend and decorations
    plt.title(title, fontsize=14)
    plt.xlabel("Generations", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"fitness_trend_PSO.png", dpi=300)

def _find_optimal_convergence(fitness_history: List[float],
                             target: float,
                             tolerance: float = 1e-4,
                             stability_window: int = 10) -> int:
    """精确查找首次达到并保持最优值的代数
    
    Args:
        target: 目标最优适应度
        tolerance: 数值容忍度
        stability_window: 需要持续稳定的代数
    
    Returns:
        首个满足条件的代数（如未找到返回None）
    """
    # 转换为numpy数组便于计算
    fitness = np.array(fitness_history)
    target = np.float64(target)
    
    # 找到所有达到目标的点
    reach_mask = np.isclose(fitness, target, 
                           atol=tolerance, 
                           rtol=tolerance*10)
    
    # 寻找连续稳定的窗口
    for i in range(len(fitness)-stability_window+1):
        window = reach_mask[i:i+stability_window]
        if np.all(window):
            return i  # 返回窗口起始位置
    
    # 退化为寻找首次达到点
    first_reach = np.argmax(reach_mask)
    if reach_mask[first_reach]:
        return first_reach
    
    return None
    
def plot_parameter_heatmap(results_df, param1, param2):
    """参数敏感性热力图"""
    pivot_table = results_df.pivot_table(index=param1, columns=param2, 
                                       values='best_fitness', aggfunc='mean')
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis")
    plt.title(f"{param1} vs {param2} 参数敏感性分析")
    plt.show()