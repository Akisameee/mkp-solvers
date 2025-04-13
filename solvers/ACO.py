import random
import copy
from tqdm import tqdm
from .solver import *

class Ant():

    def __init__(
        self,
        alpha = 1.0,
        beta = 1.0,
    ) -> None:
        
        self.alpha = alpha
        self.beta = beta

        self.items_visited = []
        self.solution = None
        self.fitness = 0

    def get_visited_edges(self):

        if len(self.items_visited) < 2:
            return []
        
        visited_edges = []
        for item_idx in range(len(self.items_visited) - 1):
            visited_edges.append((
                self.items_visited[item_idx],
                self.items_visited[item_idx + 1]
            ))
        return visited_edges

    def run(self, problem: MKPInstance, pheromone: np.ndarray, eta: np.ndarray):

        self.items_visited.append(0)
        capacity_current = problem.b.copy()

        while True:
            item_current = self.items_visited[-1]

            items_available = []
            for item in range(problem.n):
                if item in self.items_visited:
                    continue
                if np.all(capacity_current >= problem.r[:, item]):
                    items_available.append(item)

            if len(items_available) > 0:

                tau = pheromone[item_current, items_available]
                heur = eta[item_current, items_available]
                probs = (tau ** self.alpha) * (heur ** self.beta)
                probs_sum = probs.sum()
                
                if probs_sum < 1e-10:
                    item_selected = random.choice(items_available)
                else:
                    probs /= probs_sum
                    item_selected = np.random.choice(items_available, p = probs)
                self.items_visited.append(item_selected)
                capacity_current -= problem.r[:, item_selected]

            else:
                self.solution = np.zeros(problem.n)
                for item in self.items_visited:
                    self.solution[item] = 1

                self.fitness = problem.evaluate(self.solution)
                return

class AntColonyOptimizer(BaseSolver):
    '''蚁群算法'''
    def __init__(
        self,
        max_iter = 100,
        num_ants = 30,
        decay = 0.1,
        alpha = 1.0,
        beta = 1.0,
        n_early_stop = 50
    ):
        super().__init__(max_iter)
        self.num_ants = num_ants
        self.decay = decay  # 信息素挥发率
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta    # 启发式信息重要程度
        self.n_early_stop = n_early_stop

    def _initialize_problem(self, problem: MKPInstance):

        problem_init = copy.deepcopy(problem)
        problem_init.n = problem.n + 1

        problem_init.p = np.zeros(problem.n + 1)
        problem_init.p[1:] = problem.p

        problem_init.r = np.zeros((problem.m, problem.n + 1))
        problem_init.r[:, 1:] = problem.r

        return problem_init

    def _calculate_heuristic(self, problem: MKPInstance):
        
        eta = np.zeros((problem.n, problem.n))
        for item in range(problem.n):
            weight = (problem.r[:, item] / problem.b).mean()
            eta[:, item].fill(problem.p[item] * weight)
            eta[item, item] = 0

        return eta

    def run(self, problem: MKPInstance):
        
        problem = self._initialize_problem(problem)

        # 初始化信息素
        pheromone = np.full((problem.n, problem.n), 1.0)

        # 计算启发式信息
        eta = self._calculate_heuristic(problem)

        # 初始化全局最优解
        best_ant = Ant(self.alpha, self.beta)

        pbar = tqdm(
            range(self.max_iter),
            desc = "ACO Iterations"
        )
        for iter in pbar:
            ants = []

            # 每只蚂蚁构建解
            for _ in range(self.num_ants):
                ant = Ant(self.alpha, self.beta)
                ant.run(problem, pheromone, eta)
                ants.append(ant)

            # 更新全局最优解  
            local_best_idx = np.argmax(np.array([ant.fitness for ant in ants]))
            local_best_ant: Ant = ants[local_best_idx]

            if local_best_ant.fitness > best_ant.fitness:
                best_ant = local_best_ant
                best_solution_iter = iter

            pbar.set_postfix({
                "Best Fitness": f"{best_ant.fitness:.2f}",
                "Best Solution Iter": best_solution_iter
            })

            # 信息素挥发  
            pheromone *= (1 - self.decay)

            # 信息素增加  
            if best_ant.fitness > 0:
                edges = set(best_ant.get_visited_edges() + local_best_ant.get_visited_edges())
                for edge in edges:
                    pheromone[edge[0], edge[1]] += 1.0
            
            if self.n_early_stop is not None and \
                iter - best_solution_iter >= self.n_early_stop:
                pbar.close()
                print(f"Early stopped")  
                break
        
        return best_ant.solution, best_ant.fitness, {
            'best_solution_iter': best_solution_iter,
            'pheromone': pheromone
        }