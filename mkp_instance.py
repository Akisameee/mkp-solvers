import numpy as np

class MKPInstance:
    '''多维背包问题实例'''
    def __init__(self, n, m, p, r, b, optimal=0):
        '''
        参数:
        n: 物品数量
        m: 约束数量
        p: 收益数组 (n,)
        r: 资源消耗矩阵 (m, n)
        b: 资源上限数组 (m,)
        optimal: 已知最优解
        '''
        self.n = n
        self.m = m
        self.p = np.array(p, dtype = np.float64)
        self.r = np.array(r, dtype = np.float64)
        self.b = np.array(b, dtype = np.float64)
        self.optimal = optimal

    def evaluate(self, solution):
        '''评估解的适应度，返回收益值'''
        if not self.is_feasible(solution):
            return -np.inf  # 不可行解返回负无穷
        return np.dot(self.p, solution)
    
    def is_feasible(self, solution):
        '''检查解是否满足所有约束'''
        resource_usage = np.dot(self.r, solution)
        return np.all(resource_usage <= self.b)
