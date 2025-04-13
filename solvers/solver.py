import numpy as np
from abc import ABC, abstractmethod
from mkp_instance import MKPInstance

class BaseSolver(ABC):
    '''优化算法基类'''
    def __init__(self, max_iter: int):
        '''
        参数:
        max_iter: 最大迭代次数
        '''
        self.max_iter = max_iter
        self.best_solution = None
        self.best_fitness = -np.inf
    
    @abstractmethod
    def run(self, problem: MKPInstance):
        '''运行算法'''
        pass