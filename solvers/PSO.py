from .solver import *
import random
from tqdm import tqdm
import numpy as np
import time  # 添加time模块导入

class ParticleSwarmOptimizer(BaseSolver):
    '''粒子群算法'''
    def __init__(self, max_iter=10000, num_particles=30,
                 w_start=0.9, w_end=0.4, c1_start=2.5, c1_end=0.5, c2_start=0.5, c2_end=2.5, 
                 v_max=6.0, n_early_stop=50, disturb_prob=0.2, local_search_freq=5):
        super().__init__(max_iter)
        self.num_particles = num_particles
        # 动态惯性权重参数
        self.w_start = w_start  # 初始惯性权重
        self.w_end = w_end      # 最终惯性权重
        # 动态学习因子参数
        self.c1_start = c1_start  # 初始个体学习因子
        self.c1_end = c1_end      # 最终个体学习因子
        self.c2_start = c2_start  # 初始社会学习因子
        self.c2_end = c2_end      # 最终社会学习因子
        
        self.v_max = v_max  # 最大速度限制
        self.n_early_stop = n_early_stop  # 提前停止的迭代次数
        self.disturb_prob = disturb_prob  # 粒子扰动概率
        self.local_search_freq = local_search_freq  # 局部搜索频率
    
    def _initialize_particles(self, problem: MKPInstance):
        '''初始化粒子群，使用价值密度启发式生成部分高质量解'''
        n = problem.n  # 物品数量
        particles = []
        
        # 计算每个物品的价值密度（平均每单位资源的价值）
        value_density = np.zeros(n)
        for i in range(n):
            # 计算物品i消耗的总资源
            total_resource = np.sum(problem.r[:, i])
            if total_resource > 0:
                value_density[i] = problem.p[i] / total_resource
            else:
                value_density[i] = float('inf')  # 防止除零错误
        
        for i in range(self.num_particles):
            # 随机初始化粒子位置（0-1向量）
            position = np.zeros(n, dtype=int)
            
            # 使用不同的初始化策略
            if i < self.num_particles // 3:  # 1/3的粒子使用价值密度贪心策略
                # 按价值密度排序物品
                sorted_indices = np.argsort(-value_density)
                current_capacity = problem.b.copy()
                
                for idx in sorted_indices:
                    # 检查添加物品是否可行
                    if np.all(problem.r[:, idx] <= current_capacity):
                        position[idx] = 1
                        current_capacity -= problem.r[:, idx]
            else:  # 2/3的粒子使用随机策略
                available_items = list(range(n))
                random.shuffle(available_items)
                
                current_capacity = problem.b.copy()
                for item_idx in available_items:
                    # 检查添加该物品是否可行
                    if np.all(problem.r[:, item_idx] <= current_capacity):
                        position[item_idx] = 1
                        current_capacity -= problem.r[:, item_idx]
            
            # 随机初始化速度（实数向量）
            velocity = np.random.uniform(-self.v_max/2, self.v_max/2, n)
            
            # 计算初始适应度
            fitness = problem.evaluate(position)
            
            particles.append({
                'position': position,
                'velocity': velocity,
                'fitness': fitness,
                'pbest_position': position.copy(),
                'pbest_fitness': fitness
            })
        
        return particles
    
    def _adjust_position(self, position, velocity, problem: MKPInstance):
        '''调整粒子位置使其成为可行解，考虑价值密度'''
        n = problem.n
        new_position = np.zeros(n, dtype=int)
        
        # 计算选择物品的概率（基于sigmoid函数）
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        probabilities = sigmoid(velocity)
        
        # 计算每个物品的价值密度
        value_density = np.zeros(n)
        for i in range(n):
            # 计算物品i消耗的总资源
            total_resource = np.sum(problem.r[:, i])
            if total_resource > 0:
                value_density[i] = problem.p[i] / total_resource
            else:
                value_density[i] = float('inf')
        
        # 结合概率和价值密度创建综合评分
        combined_score = probabilities * 0.7 + value_density / np.max(value_density) * 0.3
        
        # 按综合评分从高到低排序物品索引
        sorted_indices = np.argsort(-combined_score)
        
        # 贪心策略：按评分高低顺序尝试放入物品
        current_capacity = problem.b.copy()
        
        for idx in sorted_indices:
            # 检查添加物品是否可行
            if np.all(problem.r[:, idx] <= current_capacity):
                new_position[idx] = 1
                current_capacity -= problem.r[:, idx]
        
        return new_position
    
    def _local_search(self, position, problem: MKPInstance):
        '''对当前解进行局部搜索优化'''
        n = problem.n
        best_position = position.copy()
        best_fitness = problem.evaluate(position)
        
        # 计算当前剩余容量
        current_capacity = problem.b.copy() - np.sum(problem.r[:, position == 1], axis=1)
        
        # 尝试1-0交换（移除一个已选物品）
        for i in range(n):
            if position[i] == 1:
                new_position = position.copy()
                new_position[i] = 0
                
                # 腾出空间后，尝试添加多个未选物品
                new_capacity = current_capacity + problem.r[:, i]
                
                # 按价值排序未选物品
                unselected_items = [j for j in range(n) if new_position[j] == 0]
                unselected_items.sort(key=lambda j: problem.p[j], reverse=True)
                
                for j in unselected_items:
                    if np.all(problem.r[:, j] <= new_capacity):
                        new_position[j] = 1
                        new_capacity -= problem.r[:, j]
                
                new_fitness = problem.evaluate(new_position)
                if new_fitness > best_fitness:
                    best_position = new_position.copy()
                    best_fitness = new_fitness
        
        # 尝试1-1交换
        for i in range(n):
            if position[i] == 1:
                for j in range(n):
                    if position[j] == 0:
                        # 检查交换是否可行
                        new_capacity = current_capacity + problem.r[:, i] - problem.r[:, j]
                        if np.all(new_capacity >= 0):
                            new_position = position.copy()
                            new_position[i] = 0
                            new_position[j] = 1
                            
                            new_fitness = problem.evaluate(new_position)
                            if new_fitness > best_fitness:
                                best_position = new_position.copy()
                                best_fitness = new_fitness
        
        return best_position, best_fitness
    
    def _disturb_particle(self, particle, problem: MKPInstance):
        '''扰动粒子避免早熟收敛'''
        position = particle['position'].copy()
        n = len(position)
        
        # 随机选择约30%的位置进行翻转
        num_flip = max(1, int(0.3 * n))
        flip_indices = np.random.choice(n, num_flip, replace=False)
        
        # 创建临时解进行扰动
        temp_position = position.copy()
        temp_position[flip_indices] = 1 - temp_position[flip_indices]
        
        # 修复成为可行解
        new_position = np.zeros(n, dtype=int)
        current_capacity = problem.b.copy()
        
        # 首先尝试保留原来的1
        for i in range(n):
            if position[i] == 1 and np.all(problem.r[:, i] <= current_capacity):
                new_position[i] = 1
                current_capacity -= problem.r[:, i]
        
        # 然后考虑扰动后新增的1
        for i in flip_indices:
            if temp_position[i] == 1 and position[i] == 0 and np.all(problem.r[:, i] <= current_capacity):
                new_position[i] = 1
                current_capacity -= problem.r[:, i]
        
        # 最后考虑其他潜在可添加的物品
        remaining_items = [i for i in range(n) if new_position[i] == 0]
        random.shuffle(remaining_items)
        
        for i in remaining_items:
            if np.all(problem.r[:, i] <= current_capacity):
                new_position[i] = 1
                current_capacity -= problem.r[:, i]
        
        return new_position
    
    def run(self, problem: MKPInstance):
        # 记录开始时间
        start_time = time.time()
        
        # 初始化粒子群
        particles = self._initialize_particles(problem)
        
        # 初始化全局最优解
        gbest_position = np.zeros(problem.n, dtype=int)
        gbest_fitness = float('-inf')
        self.best_fitness_history = []
        
        # 找到初始的全局最优解
        for particle in particles:
            if particle['fitness'] > gbest_fitness:
                gbest_fitness = particle['fitness']
                gbest_position = particle['position'].copy()
        
        best_solution_iter = 0
        no_improve_count = 0
        
        # 主循环
        pbar = tqdm(
            range(self.max_iter),
            desc="PSO Iterations"
        )
        
        for iter in pbar:
            # 计算当前迭代的动态参数
            progress = iter / self.max_iter
            # 线性递减的惯性权重
            w = self.w_start - (self.w_start - self.w_end) * progress
            # 动态调整学习因子
            c1 = self.c1_start - (self.c1_start - self.c1_end) * progress
            c2 = self.c2_start + (self.c2_end - self.c2_start) * progress
            
            # 更新每个粒子的位置和速度
            for i, particle in enumerate(particles):
                # 生成随机因子
                r1 = np.random.random(problem.n)
                r2 = np.random.random(problem.n)
                
                # 更新速度
                particle['velocity'] = (w * particle['velocity'] + 
                                      c1 * r1 * (particle['pbest_position'] - particle['position']) +
                                      c2 * r2 * (gbest_position - particle['position']))
                
                # 限制速度范围
                particle['velocity'] = np.clip(particle['velocity'], -self.v_max, self.v_max)
                
                # 扰动机制：当算法陷入局部最优时，随机扰动一些粒子
                if no_improve_count > 10 and np.random.random() < self.disturb_prob:
                    particle['position'] = self._disturb_particle(particle, problem)
                else:
                    # 调整位置使其成为可行解
                    particle['position'] = self._adjust_position(
                        particle['position'], particle['velocity'], problem
                    )
                
                # 计算新适应度
                particle['fitness'] = problem.evaluate(particle['position'])
                
                # 更新个体最优解
                if particle['fitness'] > particle['pbest_fitness']:
                    particle['pbest_fitness'] = particle['fitness']
                    particle['pbest_position'] = particle['position'].copy()
                
                # 更新全局最优解
                if particle['fitness'] > gbest_fitness:
                    gbest_fitness = particle['fitness']
                    gbest_position = particle['position'].copy()
                    best_solution_iter = iter
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # 局部搜索：定期对全局最优解进行局部搜索
            if iter % self.local_search_freq == 0:
                improved_position, improved_fitness = self._local_search(gbest_position, problem)
                if improved_fitness > gbest_fitness:
                    gbest_position = improved_position.copy()
                    gbest_fitness = improved_fitness
                    best_solution_iter = iter
                    no_improve_count = 0
            
            # 记录历史最优值
            self.best_fitness_history.append(gbest_fitness)
            
            # 更新进度条显示
            pbar.set_postfix({
                "Best Fitness": f"{gbest_fitness:.2f}",
                "Best Solution Iter": best_solution_iter,
                "No Improve": no_improve_count
            })
            
            # 检查是否提前停止
            if self.n_early_stop is not None and \
               iter - best_solution_iter >= self.n_early_stop:
                pbar.close()
                print(f"Early stopped")
                break
        
        # 计算总运行时间
        run_time = time.time() - start_time
        
        # 返回找到的最优解、适应度和额外信息
        return gbest_position, gbest_fitness, {
            'iter': best_solution_iter,
            'fitness_history': self.best_fitness_history,
            'run_time': run_time  # 添加运行时间信息
        }