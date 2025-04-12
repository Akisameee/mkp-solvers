from solvers import *
from utils import *
from mkp_instance import *

aco_params = {
    'max_iter': 1000,
    'num_ants': 10,
    'decay': 0.1,
    'alpha': 1.0,
    'beta': 2.0,
    'initial_pheromone': 1.0,
    'n_early_stop': 100
}

def get_param_grid(base_param: dict, steps: tuple, coordinate: tuple):

    decay_mean = base_param['decay']
    beta_mean = base_param['beta'] / base_param['alpha']

    offset_params = aco_params.copy()
    offset_params['decay'] = decay_mean + coordinate[0] * steps[0]
    offset_params['beta'] = beta_mean + coordinate[1] * steps[1]

    return offset_params

if __name__ == '__main__':

    data_path = './datas/mknapcb5.txt'
    instances = read_mkp_file(data_path)
    instance = instances[0]

    n_episode = 2
    decay_grid = [0.05, 0.1, 0.2, 0.4]
    beta_grid = [1.0, 2.0, 3.0, 5.0, 7.0]
    for decay in decay_grid:
        for beta in beta_grid:
            offset_params = aco_params.copy()
            offset_params['decay'] = decay
            offset_params['beta'] = beta

            mkp_solver = AntColonyOptimizer(**offset_params)
            best_values = []
            for episode in range(n_episode):
                solution, value, stats = mkp_solver.run(instance)
                best_values.append(value)

            avg_value = sum(best_values) / n_episode

            ratio = beta / offset_params['alpha']
            print(f'decay: {decay}, beta / alpha: {ratio}')
            print(f'Average Value: {avg_value:.4g}')