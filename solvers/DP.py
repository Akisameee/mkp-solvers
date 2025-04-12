import numpy as np
from .solver import *

class DynamicProgramming(BaseSolver):
    '''动态规划'''
    def __init__(self):
        super().__init__(max_iter = None)

    def run(self, problem: MKPInstance):

        # 初始状态：所有资源都未使用，收益为0
        dp = {tuple(problem.b.tolist()): 0}
        
        for i in range(problem.n):
            current_p = problem.p[i]
            current_r = problem.r[:, i].tolist()  # 当前物品在各维度的资源消耗
            
            temp_dp = {}
            for resources, profit in dp.items():
                # 不选当前物品的情况
                if resources in temp_dp:
                    if profit > temp_dp[resources]:
                        temp_dp[resources] = profit
                else:
                    temp_dp[resources] = profit
                
                # 选当前物品的情况，检查是否所有维度的资源都足够
                feasible = True
                for j in range(problem.m):
                    if resources[j] < current_r[j]:
                        feasible = False
                        break
                if feasible:
                    # 计算新资源状态
                    new_resources = list(resources)
                    for j in range(problem.m):
                        new_resources[j] -= current_r[j]
                    new_resources_tuple = tuple(new_resources)
                    new_profit = profit + current_p
                    
                    # 更新临时DP表
                    if new_resources_tuple in temp_dp:
                        if new_profit > temp_dp[new_resources_tuple]:
                            temp_dp[new_resources_tuple] = new_profit
                    else:
                        temp_dp[new_resources_tuple] = new_profit
            
            dp = temp_dp  # 更新为处理完当前物品后的状态
        
        # 找出所有可行状态中的最大收益
        max_profit = max(dp.values()) if dp else 0

        return None, max_profit, {}
