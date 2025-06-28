from solvers import *
from utils import *
from mkp_instance import *
from ga_visualization import *
from typing import List, Tuple
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  # 进度条库，可选



# 设置全局字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def run_parameter_analysis(instance,params_grid, n_runs=3):
    """参数敏感性分析主函数"""
    param_names = list(params_grid.keys())
    param_values = list(params_grid.values())
    combinations = list(product(*param_values))
    
    results = []
    
    for combo in combinations:
        current_params = dict(zip(param_names, combo))
        fitness_history = []
        
        for _ in range(n_runs):
            # 直接使用类的默认参数 + 当前测试参数覆盖
            ga = GeneticAlgorithm(**current_params)
            ga.run(instance)
            fitness_history.append(ga.best_fitness_history)
        
        results.append({
            'params': current_params,
            'mean_curve': np.mean(fitness_history, axis=0),
            'std_curve': np.std(fitness_history, axis=0),
            'final_values': [fh[-1] for fh in fitness_history]
        })
    
    return results

def plot_analysis(results, param_name):
    """可视化分析结果"""
    plt.figure(figsize=(12, 5))
    
    # 收敛曲线子图
    plt.subplot(1, 2, 1)
    for res in results:
        param_val = res['params'][param_name]
        plt.plot(res['mean_curve'], 
                label=f"{param_name}={param_val}")
        plt.fill_between(range(len(res['mean_curve'])),
                        res['mean_curve'] - res['std_curve'],
                        res['mean_curve'] + res['std_curve'],
                        alpha=0.1)
    plt.title(f"收敛曲线 ({param_name})")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    
    # 最终结果分布子图
    plt.subplot(1, 2, 2)
    final_values = [res['final_values'] for res in results]
    labels = [str(res['params'][param_name]) for res in results]
    plt.boxplot(final_values, labels=labels)
    plt.title(f"最终结果分布 ({param_name})")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    data_path = './datas/mknapcb9.txt'
    instances = read_mkp_file(data_path)

    # mkp_solver = SimulatedAnnealing(
    #     max_iter = 1000,
    #     initial_temp = 1000,
    #     cooling_rate = 0.95,
    #     temp_iter = 100
    # )
    
    # solver = ParticleSwarmOptimizer()
    # solver = AntColonyOptimizer()

    for idx, instance in enumerate(instances):
        GA_solver = GeneticAlgorithm()
        solution, value = GA_solver.run(instance)
        print(f'Problem: {idx + 1}\nSolution: {solution}\nValue: {value}\nOptimal: {instance.optimal}\nTime: {GA_solver.runtime}s')

        # 绘制适应度曲线
        plot_fitness_trend(GA_solver, optimal_fitness=value, title="基础实验适应度变化")

        # # 测试交叉概率（只需指定需要覆盖的参数）
        # pc_results = run_parameter_analysis(
        #     instance=instance,
        #     #params_grid={'crossover_rate': [0.7, 0.8, 0.9]},  # 其他参数自动使用类默认值
        #     params_grid={'population_size': [20, 50, 100]},
        #     #params_grid={'mutation_rate': [0.01, 0.03, 0.05]},
        #     n_runs=5
        # )
        # #plot_analysis(pc_results, 'crossover_rate')
        # plot_analysis(pc_results, 'population_size')
        # #plot_analysis(pc_results,'mutation_rate')

        
        

        
