import random
from tqdm import tqdm
from .solver import *

class AntColonyOptimizer_Altered(BaseSolver):
    '''蚁群算法'''
    def __init__(
        self,
        max_iter = 100,
        num_ants = 30,
        decay = 0.1,
        alpha = 1.0,
        beta = 2.0,
        initial_pheromone = 1.0,
        n_early_stop = 50
    ):
        super().__init__(max_iter)
        self.num_ants = num_ants
        self.decay = decay  # 信息素挥发率
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta    # 启发式信息重要程度
        self.initial_pheromone = initial_pheromone
        self.n_early_stop = n_early_stop

    def _calculate_heuristic(self, problem: MKPInstance):  
        '''计算启发式信息 (eta)'''  
        # 使用物品的 "价值密度" 作为启发式信息
        # 避免除以零，为分母添加一个小的 epsilon
        epsilon = 1e-10
        # 启发式信息: 收益 / (所有维度资源消耗的加权和)  
        # 权重可以是 1/资源上限，表示资源的稀缺性  
        # eta = problem.p / (np.sum(problem.r / (problem.b[:, np.newaxis] + epsilon), axis=0) + epsilon)  

        # 更简单的启发式：收益 / 总资源消耗 (未加权)
        total_resource_consumption = np.sum(problem.r, axis=0)
        eta = problem.p / (total_resource_consumption + epsilon)

        # 处理收益为0或负数，或资源消耗为0的情况
        eta[problem.p <= 0] = epsilon # 收益低则不吸引
        eta[total_resource_consumption <= 0] = problem.p[total_resource_consumption <= 0] # 若不耗资源且有收益，则非常吸引

        return eta

    def _construct_solution(self, problem: MKPInstance, pheromone: np.ndarray, eta: np.ndarray):
        '''单个蚂蚁构建一个解'''
        n = problem.n
        solution = np.zeros(n, dtype=int)
        current_capacity = problem.b.copy()
        available_items = list(range(n)) # 尚未考虑的物品列表
        random.shuffle(available_items) # 引入随机性

        while available_items:
            candidate_items = []
            feasibility_flags = []

            # 检查剩余可选物品是否可行
            for item_idx in available_items:
                if np.all(problem.r[:, item_idx] <= current_capacity):
                    candidate_items.append(item_idx)
                    feasibility_flags.append(True)
                else:
                    feasibility_flags.append(False)

            # 如果没有可行的物品可选，则停止构建  
            if not candidate_items:
                break

            # 计算可行物品的选择概率  
            probs = []  
            total_prob_measure = 0.0  
            epsilon = 1e-10 # 避免数值问题  

            for item_idx in candidate_items:  
                # 确保信息素和启发值是正数，防止计算错误  
                tau = max(pheromone[item_idx], epsilon)  
                heur = max(eta[item_idx], epsilon)  
                prob_measure = (tau ** self.alpha) * (heur ** self.beta)  
                probs.append(prob_measure)  
                total_prob_measure += prob_measure  

            # 选择下一个物品  
            selected_item = -1  
            if total_prob_measure < epsilon:  
                 # 如果概率和接近零（例如所有候选项信息素和启发值都极小），随机选一个可行的  
                 selected_item = random.choice(candidate_items)
            else:
                probabilities = np.array(probs) / total_prob_measure
                # 使用轮盘赌选择
                selected_item = np.random.choice(candidate_items, p=probabilities)

            # 更新解和状态  
            solution[selected_item] = 1  
            current_capacity -= problem.r[:, selected_item]  
            available_items.remove(selected_item) # 从可用列表中移除已选物品  

            # 从 available_items 中移除不再可行的项目 (优化，避免下次迭代无效检查)  
            items_to_remove = []  
            for item_idx in available_items:  
                if not np.all(problem.r[:, item_idx] <= current_capacity):  
                     items_to_remove.append(item_idx)  
            for item in items_to_remove:  
                 available_items.remove(item)  

        return solution  

    def run(self, problem: MKPInstance):
        
        # 1. 初始化信息素
        pheromone = np.full(problem.n, self.initial_pheromone)

        # 2. 计算启发式信息
        eta = self._calculate_heuristic(problem)

        # 初始化全局最优解
        best_solution = np.zeros(problem.n, dtype=int)
        self.best_fitness = problem.evaluate(best_solution) # 可能为0或负无穷
        self.best_fitness_history = []

        pbar = tqdm(
            range(self.max_iter),
            desc = "ACO Iterations"
        )
        for iter in pbar:
            ant_solutions = []
            ant_fitnesses = []

            # 3. 每只蚂蚁构建解  
            for _ in range(self.num_ants):
                solution = self._construct_solution(problem, pheromone, eta)
                fitness = problem.evaluate(solution)
                ant_solutions.append(solution)
                ant_fitnesses.append(fitness)

            # 4. 更新全局最优解  
            iter_best_idx = np.argmax(ant_fitnesses)
            iter_best_fitness = ant_fitnesses[iter_best_idx]
            iter_best_solution = ant_solutions[iter_best_idx]

            if iter_best_fitness > self.best_fitness:
                self.best_fitness = iter_best_fitness
                best_solution = iter_best_solution.copy()
                best_solution_iter = iter

            self.best_fitness_history.append(self.best_fitness)

            pbar.set_postfix({
                "Best Fitness": f"{self.best_fitness:.2f}",
                "Best Solution Iter": best_solution_iter
            })

            # 5. 更新信息素  
            # 5.1 信息素挥发  
            pheromone *= (1 - self.decay)  

            # 5.2 信息素增加 (这里使用全局最优解来增强)  
            # 可以在此选择不同策略：迭代最优增强、全局最优增强、所有蚂蚁按适应度增强等  
            # 全局最优增强策略:  
            if self.best_fitness > -np.inf and np.sum(best_solution) > 0: # 确保找到有效解  
                # delta_pheromone = self.best_fitness # 简单的增量方式  
                delta_pheromone = 1.0 # 或者固定增量  
                # 或者基于质量的增量，例如 Q / (Optimal - Fitness) 或 Q * Fitness  
                # 这里使用与适应度成正比的简单方式，如果适应度范围大可能需要归一化  
                pheromone[best_solution == 1] += delta_pheromone  
                # 可以考虑加入最大最小信息素限制 (Pheromone Clamping) 来避免过早收敛或停滞  
                # pheromone = np.clip(pheromone, min_pheromone, max_pheromone)  

            # 迭代最优增强策略 (替代上面的全局最优增强):  
            # if iter_best_fitness > -np.inf and np.sum(iter_best_solution) > 0:  
            #     delta_pheromone = iter_best_fitness # or 1.0  
            #     pheromone[iter_best_solution == 1] += delta_pheromone  
                
            # 6. Check for Early Stopping  
            if self.n_early_stop is not None and \
                iter - best_solution_iter >= self.n_early_stop:
                pbar.close()
                print(f"Early stopped")  
                break

        # 返回找到的最优解和适应度  
        return best_solution, self.best_fitness, {
            'iter': best_solution_iter,
            'pheromone': pheromone
        }