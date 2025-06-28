from solvers import *
from utils import *
from mkp_instance import *
from tqdm import tqdm
from visualization import plot_fitness_trend

aco_params = {
    'max_iter': 1000,
    'num_ants': 10,
    'decay': 0.1,
    'alpha': 1.0,
    'beta': 2.0,
    'initial_pheromone': 1.0,
    'n_early_stop': 200
}

pso_params = {
    'max_iter': 1000,
    'num_particles': 30,
    'w': 0.8,        # 惯性权重
    'c1': 2.0,       # 个体学习因子
    'c2': 2.0,       # 社会学习因子
    'v_max': 6.0,    # 最大速度限制
    'n_early_stop': 200
}

def get_param_grid(base_param: dict, steps: tuple, coordinate: tuple, algorithm='aco'):
    if algorithm == 'aco':
        decay_mean = base_param['decay']
        beta_mean = base_param['beta'] / base_param['alpha']

        offset_params = base_param.copy()
        offset_params['decay'] = decay_mean + coordinate[0] * steps[0]
        offset_params['beta'] = beta_mean + coordinate[1] * steps[1]
        
        return offset_params
    
    elif algorithm == 'pso':
        w_mean = base_param['w']
        c_ratio_mean = base_param['c1'] / base_param['c2']  # 个体学习与社会学习的比例
        
        offset_params = base_param.copy()
        offset_params['w'] = w_mean + coordinate[0] * steps[0]
        
        # 调整c1和c2，但保持它们的总和不变(通常为4)
        c_ratio = c_ratio_mean + coordinate[1] * steps[1]
        c_sum = base_param['c1'] + base_param['c2']
        offset_params['c1'] = (c_sum * c_ratio) / (1 + c_ratio)
        offset_params['c2'] = c_sum - offset_params['c1']
        
        return offset_params
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

if __name__ == '__main__':

    values = []
    epochs = []
    times = []
    data_path = './datas/mknapcb5.txt'
    instances = read_mkp_file(data_path)

    # mkp_solver = DynamicProgramming()
    # mkp_solver = SimulatedAnnealing(
    #     max_iter = 1000,
    #     initial_temp = 1000,
    #     cooling_rate = 0.95,
    #     temp_iter = 100
    # )
    # mkp_solver = GeneticAlgorithm()
    # for _ in tqdm(range(10)):
        
    mkp_solver = ParticleSwarmOptimizer()
    # 使用参数配置创建PSO求解器的示例:
    # mkp_solver = ParticleSwarmOptimizer(**pso_params)
    # mkp_solver = AntColonyOptimizer()

    # for idx, instance in enumerate(instances):
    instance = instances[0]
    solution, value, stats = mkp_solver.run(instance)
    print(f'Problem: 1\nSolution: {solution}\nValue: {value}\nOptimal: {instance.optimal}')
    print(f'Stats: {stats}\n')
    values.append(value)
    epochs.append(stats['iter'])
    times.append(stats['run_time'])
    
    plot_fitness_trend(mkp_solver)