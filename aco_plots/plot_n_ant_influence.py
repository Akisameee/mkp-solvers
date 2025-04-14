import matplotlib.pyplot as plt
import numpy as np
import os
import json

def get_n_ant_search_res(dir_path):

    res = []
    for filename in os.listdir(dir_path):
        if 'n_ant' in filename and filename.endswith('.json'):  # 假设是JSON文件
            with open(os.path.join(dir_path, filename), 'r') as f:
                res.append(json.load(f))
    return res

def plot_line(x, y, save_path):

    # 创建画布和坐标轴
    plt.figure(figsize=(8, 5))

    # 绘制折线图
    plt.plot(x, y)

    # 添加标题和标签
    plt.title("Parameter Influence - n_ant", fontsize=14)
    plt.xlabel("n_ant", fontsize=12)
    plt.ylabel("Best Fitness", fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.2)
    plt.savefig(f'{save_path}/n_ant_influence.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    search_res = get_n_ant_search_res('./grid_search_res/')
    search_res = sorted(search_res, key = lambda r:r['params']['num_ants'])
    x = np.array([res['params']['num_ants'] for res in search_res])
    y = np.array([sum(res['values']) / len(res['values']) for res in search_res])

    plot_line(x, y, './aco_plots/')
