import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.colors import Normalize

import sys
sys.path.insert(0, '.')

from solvers import *
from utils import *
from mkp_instance import *

def plot_pheromones(pheromones: np.ndarray, save_path):

    time_steps = pheromones.shape[0]
    global_min = np.min(pheromones)
    global_max = np.max(pheromones)

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 初始化热力图
    im = ax.imshow(pheromones[0, :, :], cmap='hot', interpolation='nearest', vmin=global_min, vmax=global_max)
    cbar = plt.colorbar(im, ax=ax, label='Pheromone')
    ax.set_title('Iteration: 0')

    # 更新函数，用于动画的每一帧
    def update(frame):
        current_data = pheromones[frame, :, :]
        im.set_array(current_data)
        ax.set_title(f'Iteration: {frame + 1}')
        return [im]

    # 创建动画
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=time_steps,
        interval=50,
        blit=True
    )

    # 保存为GIF
    writer = PillowWriter(fps=20)
    ani.save(f'{save_path}/pheromones.gif', writer=writer)

    plt.close()

aco_params = {
    'max_iter': 1000,
    'num_ants': 10,
    # 'decay': 0.025,
    'decay': 0.1,
    'alpha': 1.0,
    'beta': 0.25,
    'n_early_stop': 200
}

def get_param_grid(base_param: dict, steps: tuple, coordinate: tuple):

    decay_mean = base_param['decay']
    beta_mean = base_param['beta'] / base_param['alpha']

    offset_params = aco_params.copy()
    offset_params['decay'] = decay_mean + coordinate[0] * steps[0]
    offset_params['beta'] = beta_mean + coordinate[1] * steps[1]

    return offset_params

if __name__ == '__main__':

    data_path = './datas/mknap1.txt'
    instance = read_mkp_file(data_path)[-1]

    mkp_solver = AntColonyOptimizer(**aco_params)

    solution, value, stats = mkp_solver.run(instance)
    plot_pheromones(stats['pheromones'], './aco_plots/')