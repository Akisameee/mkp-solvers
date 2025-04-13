from solvers import *
from utils import *
from mkp_instance import *

aco_params = {
    'max_iter': 1000,
    'num_ants': 10,
    'decay': 0.1,
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

    data_path = './datas/mknap1.txt'
    instances = read_mkp_file(data_path)

    mkp_solver = AntColonyOptimizer(**aco_params)

    for idx, instance in enumerate(instances):
        solution, value, stats = mkp_solver.run(instance)
        print(f'Problem: {idx + 1}\nSolution: {solution}\nValue: {value}\nOptimal: {instance.optimal}')
        # print(f'Stats: {stats}\n')