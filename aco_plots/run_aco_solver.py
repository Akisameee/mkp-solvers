import json

import sys
sys.path.insert(0, '.')

from solvers import *
from utils import *
from mkp_instance import *

aco_params = {
    'max_iter': 1000,
    'num_ants': 20,
    'decay': 0.025,
    'alpha': 1.0,
    'beta': 16.0,
    'n_early_stop': 200
}

if __name__ == '__main__':

    res = []
    for idx in range(1, 10):

        data_name = (f'mknapcb{idx}', 0)
        data_path = f'./datas/{data_name[0]}.txt'
        instances = read_mkp_file(data_path)
        instance = instances[data_name[1]]

        n_episode = 5
        best_values = []
        best_solution_iters = []
        elapsed_times = []
        for episode in range(n_episode):
            mkp_solver = AntColonyOptimizer(**aco_params)
            solution, value, stats = mkp_solver.run(instance)
            best_values.append(value)
            best_solution_iters.append(stats['best_solution_iter'])
            elapsed_times.append(stats['elapsed_time'])

        avg_value = sum(best_values) / n_episode
        avg_iter = sum(best_solution_iters) / n_episode
        avg_elapsed_time = sum(elapsed_times) / n_episode

        res.append({
            'data_name': data_name[0],
            'data_idx': data_name[1],
            'avg_value': avg_value,
            'avg_iter': avg_iter,
            'avg_elapsed_time': avg_elapsed_time
        })

    with open('./aco_plots/aco_results.json', 'w') as f:
        json.dump(res, f)