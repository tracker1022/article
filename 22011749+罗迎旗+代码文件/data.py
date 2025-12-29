import numpy as np
import pandas as pd
import os

# 四类用户24小时初始用电数据
USER_INITIAL_DEMAND = np.array([
    [160, 300, 100, 350],
    [155, 350, 100, 340],
    [145, 200, 50, 330],
    [155, 250, 50, 220],
    [170, 150, 50, 200],
    [135, 180, 50, 200],
    [142, 170, 100, 200],
    [150, 300, 150, 380],
    [155, 420, 200, 580],
    [175, 400, 190, 400],
    [137, 410, 200, 420],
    [139, 420, 270, 450],
    [155, 415, 260, 410],
    [110, 410, 210, 400],
    [75, 400, 220, 580],
    [90, 300, 190, 400],
    [82, 300, 195, 400],
    [95, 500, 200, 500],
    [80, 520, 210, 610],
    [85, 600, 300, 640],
    [115, 500, 290, 620],
    [125, 650, 295, 610],
    [130, 630, 310, 630],
    [160, 400, 220, 590]
])

# 可再生能源初始发电量
RENEWABLE_INITIAL_POWER = {
    'solar': np.array([0, 0, 0, 0, 0, 0, 0, 0, 100, 200, 420, 950, 1000, 850, 600, 500, 100, 20, 0, 0, 0, 0, 0, 0]),
    'wind': np.array([0, 0, 0, 0, 0, 0, 0, 0, 100, 200, 450, 970, 1050, 900, 650, 400, 180, 20, 0, 0, 0, 0, 0, 0])
}

# 传统发电机初始发电量
TRADITIONAL_INITIAL_POWER = {
    'generator1': np.array([1050, 950, 950, 950, 900, 1050, 1100, 0, 0, 0, 0, 500, 500, 0, 0, 0, 0, 0, 100, 580, 1000, 1050, 1000, 1050]),
    'generator2': np.array([0, 0, 0, 0, 0, 0, 0, 500, 510, 550, 510, 450, 600, 1200, 1100, 0, 500, 500, 490, 550, 0, 0, 550, 540]),
    'generator3': np.array([0, 0, 0, 0, 0, 0, 0, 500, 510, 550, 510, 450, 600, 1200, 1100, 0, 500, 500, 490, 550, 0, 0, 550, 540])  # 内燃机发电机，与generator2参数相同
}

# 市场电价
MARKET_PRICES = {
    'sell': np.array([0.325] * 8 + [0.92] * 4 + [0.622] * 5 + [0.92] * 4 + [0.622] * 3),  # 售电价
    'buy': np.array([0.26] * 8 + [0.736] * 4 + [0.4976] * 5 + [0.736] * 4 + [0.4976] * 3)  # 购电价
}

# 电动汽车类别参数
EV_CLASSES = {
    'class1': {
        'name': '第一类电动汽车',
        'max_capacity': 12.5,  # 最大充电容量（kW）
        'min_capacity': -12.5,  # 最大放电容量（kW）
        'initial_soc': 0.5,  # 初始SOC（比例0-1）
        'max_soc': 62.5,  # 电池容量（kWh），SOC上限为1.0
        'min_soc': 0.0,  # SOC下限（比例0-1）
        'count': 20,  # 车辆数量
        'usage_price': 0.4  # 用电价格
    },
    'class2': {
        'name': '第二类电动汽车',
        'max_capacity': 15.75,
        'min_capacity': -15.75,
        'initial_soc': 0.5,  # 初始SOC（比例0-1）
        'max_soc': 78.75,  # 电池容量（kWh），SOC上限为1.0
        'min_soc': 0.0,  # SOC下限（比例0-1）
        'count': 20,
        'usage_price': 0.45
    },
    'class3': {
        'name': '第三类电动汽车',
        'max_capacity': 9.98,
        'min_capacity': -9.98,
        'initial_soc': 0.4,  # 初始SOC（比例0-1）
        'max_soc': 49.9,  # 电池容量（kWh），SOC上限为1.0
        'min_soc': 0.0,  # SOC下限（比例0-1）
        'count': 20,
        'usage_price': 0.5
    }
}

# 用户参数
USER_PARAMS = {
    'class1': {
        'max_demand': 400,
        'min_demand': 50,
        'utility_params': [0.004,0]  # 效用函数参数
        # 效用函数参数
    },
    'class2': {
        'max_demand': 700,
        'min_demand': 100,
        'utility_params': [0.002,0]
    },
    'class3': {
        'max_demand': 350,
        'min_demand': 30,
        'utility_params': [0.004,0]
    },
    'class4': {
        'max_demand': 700,
        'min_demand': 150,
        'utility_params': [0.002,0]
    }
}

# 发电机参数
GENERATOR_PARAMS = {
    'traditional': {  # 两台传统发电机使用相同参数
        'max_capacity': 1200,
        'min_capacity': 0,
        'cost_params': [0.0005, 0.0175, 105]  # 二次成本函数参数 [a, b, c]
    },
    'solar': {  # 光能发电
        'max_capacity': 1000,
        'min_capacity': 0,
        'cost_params': [0.01, 0]  # 线性成本函数参数 [a, b]
    },
    'wind': {  # 风能发电
        'max_capacity': 1000,
        'min_capacity': 0,
        'cost_params': [0.01, 0]  # 线性成本函数参数 [a, b]
    }
}

# 决策变量定义域
DECISION_BOUNDS = {
    'ev_charging': {
        'class1': {'min': -62.5, 'max': 62.5},
        'class2': {'min': -78.75, 'max': 78.75},
        'class3': {'min': -49.9, 'max': 49.9}
    },
    'user_demand': {
        'class1': {'min': 50, 'max': 200},
        'class2': {'min': 100, 'max': 700},
        'class3': {'min': 30, 'max': 350},
        'class4': {'min': 150, 'max': 700}
    },
    'generator_output': {
        'traditional': {'min': 0, 'max': 1200},  # 两台传统发电机相同
        'solar': {'min': 0, 'max': 1000},
        'wind': {'min': 0, 'max': 1000}
    }
}

def read_ev_data_from_excel(file_path, verify=False):
    """
    从Excel读取电动汽车数据
    Args:
        file_path: Excel文件路径
        verify: 是否验证数据读取
    Returns:
        dict: 电动汽车数据，格式为 {class_name: {vehicle_name: {charging: [], usage: []}}}
    """
    ev_data = {}
    # 读取6张表
    sheets = ['Sheet1', 'Sheet2', 
             'Sheet3', 'Sheet4',
             'Sheet5', 'Sheet6']
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
            
        print(f"\n开始读取Excel文件: {file_path}")
        
        # 检查Excel文件的基本信息
        try:
            excel_file = pd.ExcelFile(file_path)
            print(f"Excel文件包含的表格: {excel_file.sheet_names}")
            for sheet_name in excel_file.sheet_names:
                df_info = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
                print(f"  {sheet_name}: 形状={df_info.shape}, 数据类型={df_info.dtypes.tolist()}")
        except Exception as e:
            print(f"检查Excel文件信息时出错: {str(e)}")
            
        for sheet in sheets:
            print(f"\n正在读取sheet: {sheet}")
            try:
                # 读取数据，不使用表头，不使用索引
                df = pd.read_excel(
                    file_path, 
                    sheet_name=sheet,
                    header=None,  # 不使用表头
                    index_col=None,  # 不使用索引
                )
                
                # 检查数据是否为空
                if df.empty:
                    print(f"警告: {sheet} 是空表")
                    continue
                
                # 检查是否有NaN值
                if df.isna().any().any():
                    print(f"警告: {sheet} 包含空值，将用0填充")
                    df = df.fillna(0)
                
                # 确定类别和数据类型
                class_id = (sheets.index(sheet) // 2) + 1
                class_name = f"class{class_id}"
                data_type = 'charging' if sheets.index(sheet) % 2 == 0 else 'usage'
                
                print(f"当前处理: {class_name} - {data_type}")
                print(f"数据形状: {df.shape}")
                
                # 初始化类别数据结构
                if class_name not in ev_data:
                    ev_data[class_name] = {}
                
                # 将数据转换为numpy数组并分配给每辆车
                data = df.values.astype(np.float64)  # 确保数据是float64类型
                
                # 添加原始数据检查
                print(f"  原始数据统计: 最小值={np.min(data):.4f}, 最大值={np.max(data):.4f}, 平均值={np.mean(data):.4f}")
                print(f"  原始数据样本: {data[0][:5] if data.size > 0 else '空数据'}")
                
                # 检查数据维度
                if data.shape[1] != 24:
                    print(f"警告: {sheet} 的数据列数不是24，实际列数为 {data.shape[1]}")
                    # 如果列数不足24，用0填充
                    if data.shape[1] < 24:
                        padding = np.zeros((data.shape[0], 24 - data.shape[1]))
                        data = np.hstack((data, padding))
                        print(f"  已用0填充列数，填充后数据样本: {data[0][:5]}")
                    # 如果列数超过24，只取前24列
                    else:
                        data = data[:, :24]
                        print(f"  已截取前24列，截取后数据样本: {data[0][:5]}")
                
                # 检查行数
                if data.shape[0] < 200:
                    print(f"警告: {sheet} 的数据行数不足200，实际行数为 {data.shape[0]}")
                    # 用0填充缺失的行
                    padding = np.zeros((200 - data.shape[0], 24))
                    data = np.vstack((data, padding))
                    print(f"  已用0填充行数，填充后数据样本: {data[0][:5]}")
                elif data.shape[0] > 200:
                    print(f"警告: {sheet} 的数据行数超过200，将只使用前200行")
                    data = data[:200]
                    print(f"  已截取前200行，截取后数据样本: {data[0][:5]}")
                
                print(f"处理后的数据形状: {data.shape}")
                
                # 检查数据范围 - 暂时不进行范围限制，查看原始数据
                print(f"  {data_type} 数据范围检查: 最小值={np.min(data):.4f}, 最大值={np.max(data):.4f}")
                
                # 检查是否有异常大的数值
                if np.max(np.abs(data)) > 1000:
                    print(f"  警告: {data_type} 数据包含异常大的数值 (>1000)")
                    print(f"  异常值位置: {np.where(np.abs(data) > 1000)}")
                
                # 暂时不进行范围限制，保持原始数据
                # if data_type == 'charging':
                #     if class_name == 'class1':
                #         data = np.clip(data, -62.5, 62.5)
                #     elif class_name == 'class2':
                #         data = np.clip(data, -78.75, 78.75)
                #     elif class_name == 'class3':
                #         data = np.clip(data, -49.9, 49.9)
                # elif data_type == 'usage':
                #     data = np.clip(data, 0, np.max(data))
                
                for i in range(200):
                    vehicle_name = f"{class_name}_EV_{i}"
                    if vehicle_name not in ev_data[class_name]:
                        ev_data[class_name][vehicle_name] = {}
                    ev_data[class_name][vehicle_name][data_type] = data[i]
                    
                    # 打印前5辆车的数据样本
                    if i < 5:
                        print(f"  {vehicle_name} {data_type} 数据样本: {data[i][:5]}...")
                
            except Exception as e:
                print(f"读取 {sheet} 时出错: {str(e)}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                continue
        
        if verify:
            verify_ev_data(ev_data)

        # 新增：输出每类车前5辆的charging和usage数据样本
        for class_name in ev_data:
            print(f"\n调试输出: {class_name} 共{len(ev_data[class_name])}辆车")
            for vname in list(ev_data[class_name].keys())[:5]:
                charging_sample = ev_data[class_name][vname].get('charging', None)
                usage_sample = ev_data[class_name][vname].get('usage', None)
                print(f"  {vname} charging样本: {charging_sample[:5] if charging_sample is not None else '无'}")
                print(f"  {vname} usage样本: {usage_sample[:5] if usage_sample is not None else '无'}")

        return ev_data
    except Exception as e:
        print(f"读取Excel文件时出错: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return {}

def verify_ev_data(ev_data):
    """
    验证电动汽车数据
    Args:
        ev_data: 电动汽车数据
    """
    print("\n=== 数据验证开始 ===")
    
    # 验证类别数量
    expected_classes = ['class1', 'class2', 'class3']
    if not all(cls in ev_data for cls in expected_classes):
        print("错误: 缺少某些类别")
        return False
    
    # 验证每类车辆数量
    for class_name in ev_data:
        if len(ev_data[class_name]) != 200:
            print(f"错误: {class_name} 车辆数量不是200")
            return False
    
    # 验证每辆车的数据
    for class_name in ev_data:
        print(f"\n{class_name} 数据验证:")
        for i, (vehicle_name, vehicle_data) in enumerate(list(ev_data[class_name].items())[:5]):  # 只显示前5辆车
            # 验证数据完整性
            if 'charging' not in vehicle_data or 'usage' not in vehicle_data:
                print(f"错误: {vehicle_name} 缺少充电或用电数据")
                return False
            
            # 验证数据长度
            if len(vehicle_data['charging']) != 24 or len(vehicle_data['usage']) != 24:
                print(f"错误: {vehicle_name} 数据长度不是24小时")
                return False
            
            # 显示数据样本
            print(f"  {vehicle_name}:")
            print(f"    充电量: {vehicle_data['charging'][:5]} ...")
            print(f"    用电量: {vehicle_data['usage'][:5]} ...")
    
    print("\n=== 数据验证完成 ===")
    return True
