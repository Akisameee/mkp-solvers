import json
import os

import sys
sys.path.insert(0, '.')

from solvers import *
from utils import *
from mkp_instance import *

aco_params = {
    'max_iter': 1000,
    'num_ants': 10,
    'decay': 0.5,
    'alpha': 1.0,
    'beta': 1.0,
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

    if not os.path.exists('./grid_search_res'):
        os.mkdir('./grid_search_res')

    data_name = ('mknapcb5', 0)
    data_path = f'./datas/{data_name[0]}.txt'
    instances = read_mkp_file(data_path)
    instance = instances[data_name[1]]

    n_episode = 10
    num_ant_grid = [1, 5, 10, 20, 50]
    num_ant_grid = [3, 15, 50]
    for num_ant in num_ant_grid:
        offset_params = aco_params.copy()
        offset_params['num_ants'] = num_ant

        mkp_solver = AntColonyOptimizer(**offset_params)
        best_values = []
        best_solution_iters = []
        for episode in range(n_episode):
            solution, value, stats = mkp_solver.run(instance)
            best_values.append(value)
            best_solution_iters.append(stats['best_solution_iter'])

        avg_value = sum(best_values) / n_episode

        print(f'num_ant: {num_ant}')
        print(f'Average Value: {avg_value:.4g}')

        result = {
            'params': offset_params,
            'values': best_values,
            'best_solution_iters': best_solution_iters
        }

        with open(
            f'./grid_search_res/{data_name[0]}_{data_name[1]}_n_ant_{num_ant}.json', 'w'
        ) as f:
            json.dump(result, f, indent = 4)

