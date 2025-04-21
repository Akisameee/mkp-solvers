import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.insert(0, '.')

from solvers import *
from utils import *
from mkp_instance import *

def plot_lines(ys, labels, save_path):

    # 创建画布和坐标轴
    plt.figure(figsize=(8, 5))

    # 绘制折线图
    for y, label in zip(ys, labels):
        x = np.arange(0, len(y))
        plt.plot(x, y, label=label)

    # 添加标题和标签
    plt.title("Parameter Influence - Decay", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.savefig(f'{save_path}/decay_influence.png', dpi=300, bbox_inches='tight')

aco_params = {
    'max_iter': 1000,
    'num_ants': 10,
    'decay': 0.1,
    'alpha': 1.0,
    'beta': 0.25,
    'n_early_stop': 200
}

if __name__ == '__main__':

    data_path = './datas/mknap1.txt'
    instance = read_mkp_file(data_path)[-1]

    mkp_solver = AntColonyOptimizer(**aco_params)

    decay_grid = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]

    ys = []
    for decay in decay_grid:
        offset_params = aco_params.copy()
        offset_params['decay'] = decay

        mkp_solver = AntColonyOptimizer(**offset_params)
        solution, value, stats = mkp_solver.run(instance)
        ys.append(stats['best_fitnesses'])
    
    plot_lines(
        ys,
        labels = decay_grid,
        save_path = './aco_plots/'
    )
