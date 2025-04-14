import matplotlib.pyplot as plt
import numpy as np
import os
import json

def get_decay_ratio_search_res(dir_path):

    res = []
    for filename in os.listdir(dir_path):
        if 'decay_beta' in filename and filename.endswith('.json'):  # 假设是JSON文件
            with open(os.path.join(dir_path, filename), 'r') as f:
                res.append(json.load(f))
    return res

def plot_heatmap(decay_grid, ratio_grid, values, save_path):
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 创建热力图
    plt.imshow(
        values, cmap='hot', interpolation='nearest', 
        extent=[min(ratio_grid), max(ratio_grid), max(decay_grid), min(decay_grid)],
        aspect='auto'
    )
    
    # 添加颜色条
    plt.colorbar(label='Performance Value')
    
    # 设置坐标轴
    plt.xlabel('Beta/Alpha Ratio')
    plt.ylabel('Decay Rate')
    plt.title("Decay Ratio and Beta/Alpha Grid Search", fontsize=14)
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 保存图像
    plt.savefig(f'{save_path}/decay_ratio_search_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':

    search_res = get_decay_ratio_search_res('./grid_search_res/')
    
    # 获取唯一的decay和ratio值
    decay_grid = sorted(list(set(res['params']['decay'] for res in search_res)))
    ratio_grid = sorted(list(set(res['params']['beta'] / res['params']['alpha'] for res in search_res)))
    
    # 初始化值矩阵
    values = np.zeros((len(decay_grid), len(ratio_grid)))
    
    # 填充值矩阵
    for res in search_res:
        decay = res['params']['decay']
        ratio = res['params']['beta'] / res['params']['alpha']
        value = sum(res['values']) / len(res['values'])
        
        # 修正索引赋值
        i = decay_grid.index(decay)
        j = ratio_grid.index(ratio)
        values[i, j] = value
    
    # 绘制热力图
    plot_heatmap(decay_grid, ratio_grid, values, './aco_plots/')
