from .solver import *

class GeneticAlgorithm(BaseSolver):
    '''遗传算法'''
    def __init__(self, max_iter=100, pop_size=50,
                 crossover_rate=0.8, mutation_rate=0.1):
        super().__init__(max_iter)
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def run(self, problem: MKPInstance):
        pass