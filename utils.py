from mkp_instance import MKPInstance

def read_mkp_file(file_path) -> list[MKPInstance]:
    '''
    读取OR-Library格式的多维背包问题文件
    返回包含MKPInstance对象的列表
    '''
    instances = []
    
    with open(file_path, 'r') as f:
        # 读取所有数据并转换为浮点数列表
        data = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # 跳过空行和注释
            # 处理可能存在的混合整数和浮点数
            converted = []
            for num in line.split():
                try:
                    converted.append(eval(num))
                except:
                    continue
            data.extend(converted)
        
        ptr = 0
        total_length = len(data)
        
        try:
            # 读取测试问题数量
            num_problems = int(data[ptr])
            ptr += 1
            
            for _ in range(num_problems):
                # 读取问题基本信息
                n = int(data[ptr])
                m = int(data[ptr + 1])
                optimal = data[ptr + 2]
                ptr += 3
                
                # 读取收益数组
                p = []
                for _ in range(n):
                    p.append(data[ptr])
                    ptr += 1
                
                # 读取资源消耗矩阵
                r = []
                for _ in range(m):
                    row = []
                    for _ in range(n):
                        row.append(data[ptr])
                        ptr += 1
                    r.append(row)
                
                # 读取资源上限
                b = []
                for _ in range(m):
                    b.append(data[ptr])
                    ptr += 1
                
                # 创建实例（将数值转换为整数）
                instance = MKPInstance(
                    n = n, m = m, p = p, r = r, b = b,
                    optimal = optimal if optimal > 0 else None
                )
                instances.append(instance)
        
        except IndexError:
            raise ValueError('Unexpected end of file while parsing')
    
    return instances


if __name__ == '__main__':
    # 读取测试文件
    instances = read_mkp_file('./datas/mknap1.txt')
    
    # 打印问题实例信息
    for idx, instance in enumerate(instances):
        print(f'\nProblem {idx+1}:')
        print(f'Variables (n): {instance.n}')
        print(f'Constraints (m): {instance.m}')
        print(f'Optimal value: {instance.optimal}')
        print(f'Profit array (p) sample: {instance.p[:5]}...')
        print(f'Resource matrix (r) first row: {instance.r[0][:5]}...')
        print(f'Resource limits (b) sample: {instance.b[:5]}...')
