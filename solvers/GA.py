from .solver import *
import numpy as np
from typing import List,Tuple
import time

class GeneticAlgorithm:
    """遗传算法求解多维背包问题"""
    def __init__(self, 
                 population_size: int = 50,
                 max_iter: int = 10000,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.01,
                 tournament_size: int = 5,
                 elitism: bool = True):
        """
        参数初始化
        :param population_size: 种群规模
        :param max_iter: 最大迭代次数
        :param crossover_rate: 交叉概率 
        :param mutation_rate: 变异概率
        :param tournament_size: 锦标赛选择规模
        :param elitism: 是否使用精英保留
        """
        self.pop_size = population_size
        self.max_iter = max_iter
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.tour_size = tournament_size
        self.elitism = elitism
        self.best_fitness_history = []  # 每代最优适应度
        self.avg_fitness_history = []   # 每代平均适应度
        self.runtime = 0.0  # 新增运行时间记录

    def run(self, instance: MKPInstance) -> Tuple[List[int], float]:
        start_time = time.time()  # 记录开始时间
        """执行算法主流程"""
        n = instance.n  # 物品数量
        best_sol, best_val = None, -np.inf  # 全局最优解
        
        # 初始化种群
        population = self._init_population(n)
        
        for _ in range(self.max_iter):
            # 评估种群适应度
            fitness = [self._evaluate(ind, instance) for ind in population]
            #print(fitness)
            
            # 更新全局最优
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_val:
                best_val = fitness[current_best_idx]
                best_sol = population[current_best_idx].copy()
            
            # 选择操作（锦标赛）
            parents = self._tournament_selection(population, fitness)
            
            # 交叉操作（均匀交叉）
            offspring = self._uniform_crossover(parents, n)
            
            # 变异操作（位翻转）
            offspring = self._bit_flip_mutation(offspring, n)
            
            # 环境选择（精英保留）
            population = self._environmental_selection(
                population, offspring, instance)
            
            # 记录本代数据
            current_best_val = fitness[current_best_idx]
            self.best_fitness_history.append(current_best_val)
            self.avg_fitness_history.append(np.mean(fitness))
            #print(current_best_val, np.mean(fitness))

        end_time = time.time()  # 记录结束时间
        self.runtime = end_time - start_time  # 计算运行时间
        # 验证最终解的可行性
        if best_val <= 0 or not instance.is_feasible(best_sol):
            return [], 0
        return best_sol, instance.evaluate(best_sol)

    def _init_population(self, n: int) -> List[np.ndarray]:
        """生成随机初始种群"""
        return [np.random.randint(0, 2, n) for _ in range(self.pop_size)]

    def _evaluate(self, solution: np.ndarray, instance: MKPInstance) -> float:
        """带惩罚项的适应度评估"""
        if instance.is_feasible(solution):
            return float(instance.evaluate(solution))
        
        violation = np.sum(np.maximum(np.dot(instance.r, solution) - instance.b, 0))
        return -violation  # 不可行解的惩罚值为负的总违反量

    def _tournament_selection(self, 
                            population: List[np.ndarray],
                            fitness: List[float]) -> List[np.ndarray]:
        """锦标赛选择"""
        selected = []
        for _ in range(self.pop_size):
            # 随机选择参赛个体
            candidates = np.random.choice(len(population), self.tour_size)
            # 选择适应度最高的
            winner = max(candidates, key=lambda x: fitness[x])
            selected.append(population[winner].copy())
        return selected

    def _uniform_crossover(self, 
                         parents: List[np.ndarray],
                         n: int) -> List[np.ndarray]:
        """均匀交叉"""
       
        offspring = []
        shuffled_parents = parents.copy()
        np.random.shuffle(shuffled_parents)  # 打乱父代顺序增加多样性
        
        # 确保处理偶数数量父代
        num_parents = len(shuffled_parents)
        if num_parents % 2 != 0:
            shuffled_parents = shuffled_parents[:-1]  # 丢弃最后一个父代保持偶数
        
        for i in range(0, len(shuffled_parents), 2):
            p1, p2 = shuffled_parents[i], shuffled_parents[i+1]
            
            # 决定是否进行交叉
            if np.random.rand() > self.cr:
                offspring.extend([p1.copy(), p2.copy()])
                continue
                
            # 执行均匀交叉
            mask = np.random.randint(0, 2, n)
            c1 = np.where(mask, p1, p2)
            c2 = np.where(mask, p2, p1)
            
            offspring.extend([c1, c2])
        
        return offspring

    def _bit_flip_mutation(self, 
                         offspring: List[np.ndarray],
                         n: int) -> List[np.ndarray]:
        """位翻转变异"""
        for i in range(len(offspring)):
            for j in range(n):
                if np.random.rand() < self.mr:
                    offspring[i][j] = 1 - offspring[i][j]
        return offspring

    def _environmental_selection(self,
                               parents: List[np.ndarray],
                               offspring: List[np.ndarray],
                               instance: MKPInstance) -> List[np.ndarray]:
        """环境选择（精英保留策略）"""
        combined = parents + offspring
        # 计算适应度排序
        fitness = [self._evaluate(ind, instance) for ind in combined]
        sorted_idx = np.argsort(fitness)[::-1]  # 降序排列
        
        # 精英保留
        if self.elitism:
            # 保留合并后的全局最优，并确保种群大小正确
            elite = combined[np.argmax(fitness)]
            new_pop = [combined[i] for i in sorted_idx[:self.pop_size-1]]
            new_pop.append(elite.copy())
        else:
            new_pop = [combined[i] for i in sorted_idx[:self.pop_size]]
        return new_pop