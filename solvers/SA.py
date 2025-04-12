from .solver import *

class SimulatedAnnealing(BaseSolver):
    '''模拟退火算法'''
    def __init__(
        self,
        max_iter=1000,
        initial_temp=1000,
        cooling_rate=0.95,
        temp_iter=100
    ):
        super().__init__(max_iter)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.temp_iter = temp_iter
    
    def run(self, problem: MKPInstance):

        # 初始化全零解
        current_solution = np.zeros(problem.n, dtype=int)
        current_fitness = problem.evaluate(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        T = self.initial_temp
        
        for n_iter in range(self.max_iter):
            for __ in range(self.temp_iter):
                # 生成邻域解：随机翻转一个物品的状态
                neighbor_solution = current_solution.copy()
                idx = np.random.randint(problem.n)
                neighbor_solution[idx] = 1 - neighbor_solution[idx]
                
                neighbor_fitness = problem.evaluate(neighbor_solution)
                
                # 计算能量差
                delta_e = neighbor_fitness - current_fitness
                
                # 决定是否接受邻域解
                if delta_e > 0 or (np.exp(delta_e / T) > np.random.rand()):
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
                    
                    # 更新全局最优解
                    if current_fitness > best_fitness:
                        best_solution = current_solution.copy()
                        best_fitness = current_fitness
            
            # 降低温度
            T *= self.cooling_rate

        return best_solution, best_fitness, {
            'n_iter': n_iter + 1
        }