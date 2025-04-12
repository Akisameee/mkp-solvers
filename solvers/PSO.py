from .solver import *

class ParticleSwarmOptimizer(BaseSolver):
    '''粒子群算法'''
    def __init__(self, max_iter=100, num_particles=30,
                 w=0.8, c1=2.0, c2=2.0):
        super().__init__(max_iter)
        self.num_particles = num_particles
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子

    def run(self, problem: MKPInstance):
        pass