import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
from models import TraditionalGenerator, RenewableGenerator, User, ElectricVehicle
from data import MARKET_PRICES, TRADITIONAL_INITIAL_POWER, RENEWABLE_INITIAL_POWER, USER_PARAMS, USER_INITIAL_DEMAND, EV_CLASSES, read_ev_data_from_excel


def calculate_adjustment_capability():
    """
    计算每个主体的调节能力（不包括电动汽车）
    返回包含所有主体调节能力的字典
    """
    adjustment_capabilities = {
        'traditional_generators': {},
        'users': {}
    }
    
    # 1. 计算传统发电机的调节能力
    print("\n=== 计算传统发电机调节能力 ===")
    for i, gen_name in enumerate(['generator1', 'generator2', 'generator3']):
        gen = TraditionalGenerator(
            gen_name, 
            1200,  # max_capacity
            0,     # min_capacity
            [0.0005, 0.15, 105] if i == 0 else ([0.0005, 0.12, 105] if i == 1 else [0.0005, 0.18, 105]),
            TRADITIONAL_INITIAL_POWER[gen_name]
        )
        
        current_power = gen.generate_power()  # 当前发电量
        max_power = np.full(24, gen.max_capacity)
        min_power = np.full(24, gen.min_capacity)
        
        # 向上调节能力：可以增加的发电量
        upward_capability = max_power - current_power
        # 向下调节能力：可以减少的发电量
        downward_capability = current_power - min_power
        # 总调节能力
        total_capability = upward_capability + downward_capability
        
        adjustment_capabilities['traditional_generators'][gen_name] = {
            'current_power': current_power,
            'upward': upward_capability,
            'downward': downward_capability,
            'total': total_capability,
            'max_capacity': max_power,
            'min_capacity': min_power
        }
        
        print(f"{gen_name}:")
        print(f"  平均向上调节能力: {np.mean(upward_capability):.2f} kWh")
        print(f"  平均向下调节能力: {np.mean(downward_capability):.2f} kWh")
        print(f"  平均总调节能力: {np.mean(total_capability):.2f} kWh")
    
    # 2. 计算用户的调节能力
    print("\n=== 计算用户调节能力 ===")
    for i, (user_class, params) in enumerate(USER_PARAMS.items()):
        user = User(
            f"{user_class}_user",
            params['max_demand'],
            params['min_demand'],
            params['utility_params'],
            USER_INITIAL_DEMAND[:, i]
        )
        
        current_demand = user.generate_demand()  # 当前用电量
        max_demand = np.full(24, user.max_demand)
        min_demand = np.full(24, user.min_demand)
        
        # 向上调节能力：可以增加的用电量（增加负荷）
        upward_capability = max_demand - current_demand
        # 向下调节能力：可以减少的用电量（减少负荷）
        downward_capability = current_demand - min_demand
        # 总调节能力
        total_capability = upward_capability + downward_capability
        
        adjustment_capabilities['users'][user_class] = {
            'current_demand': current_demand,
            'upward': upward_capability,
            'downward': downward_capability,
            'total': total_capability,
            'max_demand': max_demand,
            'min_demand': min_demand
        }
        
        print(f"{user_class}:")
        print(f"  平均向上调节能力: {np.mean(upward_capability):.2f} kWh")
        print(f"  平均向下调节能力: {np.mean(downward_capability):.2f} kWh")
        print(f"  平均总调节能力: {np.mean(total_capability):.2f} kWh")
    
    return adjustment_capabilities


def calculate_system_adjustment_capability(adjustment_capabilities):
    """
    计算系统总调节能力
    发电侧的上调 + 用电侧的下调
    发电侧的下调 + 用电侧的上调
    """
    hours = 24
    
    # 计算发电侧总调节能力（只包括传统发电机，不包括可再生能源）
    gen_upward_total = np.zeros(hours)
    gen_downward_total = np.zeros(hours)
    
    for gen_name, data in adjustment_capabilities['traditional_generators'].items():
        gen_upward_total += data['upward']
        gen_downward_total += data['downward']
    
    # 计算用电侧总调节能力
    user_upward_total = np.zeros(hours)
    user_downward_total = np.zeros(hours)
    
    for user_class, data in adjustment_capabilities['users'].items():
        user_upward_total += data['upward']
        user_downward_total += data['downward']
    
    # 系统总调节能力
    # 场景1：发电侧上调 + 用电侧下调（增加发电，减少用电）
    system_scenario1 = gen_upward_total + user_downward_total
    
    # 场景2：发电侧下调 + 用电侧上调（减少发电，增加用电）
    system_scenario2 = gen_downward_total + user_upward_total
    
    return {
        'gen_upward': gen_upward_total,
        'gen_downward': gen_downward_total,
        'user_upward': user_upward_total,
        'user_downward': user_downward_total,
        'scenario1': system_scenario1,  # 发电侧上调 + 用电侧下调
        'scenario2': system_scenario2   # 发电侧下调 + 用电侧上调
    }


def calculate_adjustment_proportion(adjustment_capabilities, system_capability):
    """
    计算每个主体平均调节能力占总调节能力的比例
    特别注意：发电设施的上调 vs 用户的下调（场景1）
    """
    proportions = {
        'traditional_generators': {},
        'users': {},
        'scenario1_proportion': {}  # 场景1中各主体的比例
    }
    
    # 计算场景1的总调节能力（用于计算比例）
    scenario1_total = np.sum(system_capability['scenario1'])
    scenario2_total = np.sum(system_capability['scenario2'])
    
    # 计算传统发电机的平均调节能力及其比例
    for gen_name, data in adjustment_capabilities['traditional_generators'].items():
        avg_upward = np.mean(data['upward'])
        avg_downward = np.mean(data['downward'])
        avg_total = np.mean(data['total'])
        
        # 在场景1中，发电侧的上调能力占比
        gen_upward_sum = np.sum(data['upward'])
        proportion_scenario1 = (gen_upward_sum / scenario1_total * 100) if scenario1_total > 0 else 0
        
        proportions['traditional_generators'][gen_name] = {
            'avg_upward': avg_upward,
            'avg_downward': avg_downward,
            'avg_total': avg_total,
            'proportion_scenario1_upward': proportion_scenario1
        }
    
    # 计算用户的平均调节能力及其比例
    for user_class, data in adjustment_capabilities['users'].items():
        avg_upward = np.mean(data['upward'])
        avg_downward = np.mean(data['downward'])
        avg_total = np.mean(data['total'])
        
        # 在场景1中，用户的下调能力占比
        user_downward_sum = np.sum(data['downward'])
        proportion_scenario1 = (user_downward_sum / scenario1_total * 100) if scenario1_total > 0 else 0
        
        proportions['users'][user_class] = {
            'avg_upward': avg_upward,
            'avg_downward': avg_downward,
            'avg_total': avg_total,
            'proportion_scenario1_downward': proportion_scenario1
        }
    
    # 计算场景1中发电侧总上调能力和用户侧总下调能力的比例
    gen_upward_sum = np.sum(system_capability['gen_upward'])
    user_downward_sum = np.sum(system_capability['user_downward'])
    
    proportions['scenario1_proportion'] = {
        'gen_upward_proportion': (gen_upward_sum / scenario1_total * 100) if scenario1_total > 0 else 0,
        'user_downward_proportion': (user_downward_sum / scenario1_total * 100) if scenario1_total > 0 else 0,
        'scenario1_total': scenario1_total
    }
    
    return proportions


def calculate_bound_based_adjustment_capability():
    """
    计算基于上下界的调节能力
    调节能力 = 上界 - 下界
    包括：发电设施、用户、汽车
    """
    print("\n" + "=" * 60)
    print("计算基于上下界的调节能力")
    print("=" * 60)
    
    bound_capabilities = {
        'traditional_generators': {},
        'users': {},
        'electric_vehicles': {}
    }
    
    # 1. 计算传统发电机的调节能力（上界 - 下界）
    print("\n=== 传统发电机调节能力（上界-下界） ===")
    for i, gen_name in enumerate(['generator1', 'generator2', 'generator3']):
        max_capacity = 1200  # 上界
        min_capacity = 0     # 下界
        adjustment_capability = max_capacity - min_capacity  # 上界 - 下界
        
        bound_capabilities['traditional_generators'][gen_name] = {
            'max_bound': max_capacity,
            'min_bound': min_capacity,
            'adjustment_capability': adjustment_capability
        }
        
        print(f"{gen_name}:")
        print(f"  上界: {max_capacity:.2f} kWh")
        print(f"  下界: {min_capacity:.2f} kWh")
        print(f"  调节能力（上界-下界）: {adjustment_capability:.2f} kWh")
    
    # 2. 计算用户的调节能力（上界 - 下界）
    print("\n=== 用户调节能力（上界-下界） ===")
    for user_class, params in USER_PARAMS.items():
        max_demand = params['max_demand']  # 上界
        min_demand = params['min_demand']  # 下界
        adjustment_capability = max_demand - min_demand  # 上界 - 下界
        
        bound_capabilities['users'][user_class] = {
            'max_bound': max_demand,
            'min_bound': min_demand,
            'adjustment_capability': adjustment_capability
        }
        
        print(f"{user_class}:")
        print(f"  上界: {max_demand:.2f} kWh")
        print(f"  下界: {min_demand:.2f} kWh")
        print(f"  调节能力（上界-下界）: {adjustment_capability:.2f} kWh")
    
    # 3. 计算电动汽车的调节能力（上界 - 下界）
    # 每类只考虑前20个
    print("\n=== 电动汽车调节能力（上界-下界，每类前20个） ===")
    ev_count_per_class = 20
    
    for class_name, class_params in EV_CLASSES.items():
        max_capacity = class_params['max_capacity']  # 上界（最大充电容量）
        min_capacity = class_params['min_capacity']  # 下界（最大放电容量，为负值）
        # 调节能力 = 上界 - 下界 = max_capacity - min_capacity
        # 例如：62.5 - (-62.5) = 125
        adjustment_capability_per_vehicle = max_capacity - min_capacity
        # 每类前20个的总调节能力
        total_adjustment_capability = adjustment_capability_per_vehicle * ev_count_per_class
        
        bound_capabilities['electric_vehicles'][class_name] = {
            'max_bound': max_capacity,
            'min_bound': min_capacity,
            'adjustment_capability_per_vehicle': adjustment_capability_per_vehicle,
            'vehicle_count': ev_count_per_class,
            'total_adjustment_capability': total_adjustment_capability
        }
        
        print(f"{class_name} (前{ev_count_per_class}个):")
        print(f"  上界（最大充电容量）: {max_capacity:.2f} kWh")
        print(f"  下界（最大放电容量）: {min_capacity:.2f} kWh")
        print(f"  单辆车调节能力（上界-下界）: {adjustment_capability_per_vehicle:.2f} kWh")
        print(f"  总调节能力（{ev_count_per_class}辆车）: {total_adjustment_capability:.2f} kWh")
    
    return bound_capabilities


def calculate_bound_based_proportion(bound_capabilities):
    """
    计算基于上下界的调节能力占比
    """
    # 计算总调节能力
    total_capability = 0.0
    
    # 传统发电机总调节能力
    gen_total = 0.0
    for gen_name, data in bound_capabilities['traditional_generators'].items():
        gen_total += data['adjustment_capability']
    
    # 用户总调节能力
    user_total = 0.0
    for user_class, data in bound_capabilities['users'].items():
        user_total += data['adjustment_capability']
    
    # 电动汽车总调节能力
    ev_total = 0.0
    for ev_class, data in bound_capabilities['electric_vehicles'].items():
        ev_total += data['total_adjustment_capability']
    
    total_capability = gen_total + user_total + ev_total
    
    # 计算各主体占比
    proportions = {
        'traditional_generators': {},
        'users': {},
        'electric_vehicles': {},
        'summary': {
            'gen_total': gen_total,
            'user_total': user_total,
            'ev_total': ev_total,
            'total_capability': total_capability
        }
    }
    
    # 传统发电机占比
    for gen_name, data in bound_capabilities['traditional_generators'].items():
        proportion = (data['adjustment_capability'] / total_capability * 100) if total_capability > 0 else 0
        proportions['traditional_generators'][gen_name] = {
            'capability': data['adjustment_capability'],
            'proportion': proportion
        }
    
    # 用户占比
    for user_class, data in bound_capabilities['users'].items():
        proportion = (data['adjustment_capability'] / total_capability * 100) if total_capability > 0 else 0
        proportions['users'][user_class] = {
            'capability': data['adjustment_capability'],
            'proportion': proportion
        }
    
    # 电动汽车占比
    for ev_class, data in bound_capabilities['electric_vehicles'].items():
        proportion = (data['total_adjustment_capability'] / total_capability * 100) if total_capability > 0 else 0
        proportions['electric_vehicles'][ev_class] = {
            'capability': data['total_adjustment_capability'],
            'proportion': proportion
        }
    
    return proportions


def visualize_bound_based_capability(bound_capabilities, proportions):
    """
    可视化基于上下界的调节能力和占比
    """
    # 1. 各主体调节能力柱状图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('基于上下界的调节能力分析', fontsize=16, fontweight='bold')
    
    # 1.1 传统发电机调节能力
    ax1 = axes[0, 0]
    gen_names = []
    gen_capabilities = []
    for gen_name, data in bound_capabilities['traditional_generators'].items():
        gen_names.append(gen_name)
        gen_capabilities.append(data['adjustment_capability'])
    
    bars1 = ax1.bar(gen_names, gen_capabilities, color='blue', alpha=0.7)
    ax1.set_title('传统发电机调节能力（上界-下界）', fontsize=12, fontweight='bold')
    ax1.set_xlabel('发电机', fontsize=10)
    ax1.set_ylabel('调节能力 (kWh)', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, gen_capabilities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 1.2 用户调节能力
    ax2 = axes[0, 1]
    user_names = []
    user_capabilities = []
    for user_class, data in bound_capabilities['users'].items():
        user_names.append(user_class)
        user_capabilities.append(data['adjustment_capability'])
    
    bars2 = ax2.bar(user_names, user_capabilities, color='green', alpha=0.7)
    ax2.set_title('用户调节能力（上界-下界）', fontsize=12, fontweight='bold')
    ax2.set_xlabel('用户类别', fontsize=10)
    ax2.set_ylabel('调节能力 (kWh)', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, user_capabilities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 1.3 电动汽车调节能力
    ax3 = axes[1, 0]
    ev_names = []
    ev_capabilities = []
    for ev_class, data in bound_capabilities['electric_vehicles'].items():
        ev_names.append(ev_class)
        ev_capabilities.append(data['total_adjustment_capability'])
    
    bars3 = ax3.bar(ev_names, ev_capabilities, color='orange', alpha=0.7)
    ax3.set_title('电动汽车调节能力（上界-下界，每类前20个）', fontsize=12, fontweight='bold')
    ax3.set_xlabel('电动汽车类别', fontsize=10)
    ax3.set_ylabel('总调节能力 (kWh)', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, ev_capabilities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 1.4 总调节能力对比
    ax4 = axes[1, 1]
    summary = proportions['summary']
    categories = ['传统发电机', '用户', '电动汽车']
    values = [summary['gen_total'], summary['user_total'], summary['ev_total']]
    colors = ['blue', 'green', 'orange']
    bars4 = ax4.bar(categories, values, color=colors, alpha=0.7)
    ax4.set_title('各类主体总调节能力对比', fontsize=12, fontweight='bold')
    ax4.set_ylabel('总调节能力 (kWh)', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars4, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('基于上下界的调节能力.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 各主体占比可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('基于上下界的调节能力占比分析', fontsize=16, fontweight='bold')
    
    # 2.1 传统发电机占比
    ax1 = axes[0, 0]
    gen_proportions = [proportions['traditional_generators'][name]['proportion'] 
                      for name in gen_names]
    bars1 = ax1.bar(gen_names, gen_proportions, color='blue', alpha=0.7)
    ax1.set_title('传统发电机调节能力占比', fontsize=12, fontweight='bold')
    ax1.set_xlabel('发电机', fontsize=10)
    ax1.set_ylabel('占比 (%)', fontsize=10)
    ax1.set_ylim([0, max(gen_proportions) * 1.2 if gen_proportions else 100])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, gen_proportions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 2.2 用户占比
    ax2 = axes[0, 1]
    user_proportions = [proportions['users'][name]['proportion'] 
                       for name in user_names]
    bars2 = ax2.bar(user_names, user_proportions, color='green', alpha=0.7)
    ax2.set_title('用户调节能力占比', fontsize=12, fontweight='bold')
    ax2.set_xlabel('用户类别', fontsize=10)
    ax2.set_ylabel('占比 (%)', fontsize=10)
    ax2.set_ylim([0, max(user_proportions) * 1.2 if user_proportions else 100])
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, user_proportions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 2.3 电动汽车占比
    ax3 = axes[1, 0]
    ev_proportions = [proportions['electric_vehicles'][name]['proportion'] 
                     for name in ev_names]
    bars3 = ax3.bar(ev_names, ev_proportions, color='orange', alpha=0.7)
    ax3.set_title('电动汽车调节能力占比', fontsize=12, fontweight='bold')
    ax3.set_xlabel('电动汽车类别', fontsize=10)
    ax3.set_ylabel('占比 (%)', fontsize=10)
    ax3.set_ylim([0, max(ev_proportions) * 1.2 if ev_proportions else 100])
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, ev_proportions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 2.4 总体占比饼图
    ax4 = axes[1, 1]
    summary = proportions['summary']
    categories = ['传统发电机', '用户', '电动汽车']
    values = [summary['gen_total'], summary['user_total'], summary['ev_total']]
    colors = ['blue', 'green', 'orange']
    
    # 只显示非零值
    non_zero_data = [(cat, val, col) for cat, val, col in zip(categories, values, colors) if val > 0]
    if non_zero_data:
        cats, vals, cols = zip(*non_zero_data)
        wedges, texts, autotexts = ax4.pie(vals, labels=cats, colors=cols, autopct='%1.2f%%', 
                                          startangle=90, textprops={'fontsize': 10})
        ax4.set_title('各类主体总调节能力占比', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('基于上下界的调节能力占比.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 所有主体占比汇总
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    all_names = gen_names + user_names + ev_names
    all_proportions = gen_proportions + user_proportions + ev_proportions
    all_colors = ['blue'] * len(gen_names) + ['green'] * len(user_names) + ['orange'] * len(ev_names)
    
    bars = ax.bar(range(len(all_names)), all_proportions, color=all_colors, alpha=0.7)
    ax.set_title('所有主体调节能力占比汇总', fontsize=14, fontweight='bold')
    ax.set_xlabel('主体', fontsize=12)
    ax.set_ylabel('占比 (%)', fontsize=12)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, max(all_proportions) * 1.2 if all_proportions else 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, all_proportions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='传统发电机'),
        Patch(facecolor='green', alpha=0.7, label='用户'),
        Patch(facecolor='orange', alpha=0.7, label='电动汽车')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('所有主体调节能力占比汇总.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_adjustment_task_penalty(adjustment_instruction, initial_grid_exchange, bound_proportions):
    """
    计算调节任务惩罚函数
    
    参数：
    - adjustment_instruction: 调节指令（24小时数组）
    - initial_grid_exchange: 初始购售电（24小时数组）
    - bound_proportions: 基于上下界的调节能力占比
    
    返回：
    - 包含各主体调节任务和惩罚函数的字典
    """
    print("\n" + "=" * 60)
    print("计算调节任务惩罚函数")
    print("=" * 60)
    
    # 计算调节指令与初始购售电的差额
    adjustment_difference = adjustment_instruction - initial_grid_exchange
    
    penalty_results = {
        'traditional_generators': {},
        'users': {},
        'electric_vehicles': {},
        'adjustment_difference': adjustment_difference
    }
    
    penalty_weight = 0.1  # 惩罚函数权重
    
    # 1. 计算传统发电机的调节任务和惩罚函数
    print("\n=== 传统发电机调节任务和惩罚函数 ===")
    for gen_name, prop_data in bound_proportions['traditional_generators'].items():
        proportion = prop_data['proportion'] / 100.0  # 转换为小数比例
        
        # 调节任务 = (调节指令 - 初始购售电) × 比例
        adjustment_task = adjustment_difference * proportion
        
        # 初始化发电机以获取初始发电量
        gen_idx = ['generator1', 'generator2', 'generator3'].index(gen_name)
        gen = TraditionalGenerator(
            gen_name,
            1200, 0,
            [0.0005, 0.15, 105] if gen_idx == 0 else ([0.0005, 0.12, 105] if gen_idx == 1 else [0.0005, 0.18, 105]),
            TRADITIONAL_INITIAL_POWER[gen_name]
        )
        initial_power = gen.generate_power()
        
        # 假设最终发电量（这里用初始发电量作为示例，实际应该用优化后的发电量）
        # 发电变化量 = 最终发电量 - 初始发电量
        # 这里假设没有变化，实际应该从优化结果中获取
        power_change = np.zeros(24)  # 实际应该从优化结果获取
        
        # 惩罚函数 = |发电变化量 - 调节任务| × 0.1
        penalty = np.abs(power_change - adjustment_task) * penalty_weight
        
        penalty_results['traditional_generators'][gen_name] = {
            'proportion': proportion,
            'adjustment_task': adjustment_task,
            'initial_power': initial_power,
            'power_change': power_change,
            'penalty': penalty,
            'total_penalty': np.sum(penalty)
        }
        
        print(f"{gen_name}:")
        print(f"  占比: {proportion*100:.2f}%")
        print(f"  平均调节任务: {np.mean(adjustment_task):.2f} kWh")
        print(f"  总惩罚值: {np.sum(penalty):.2f}")
    
    # 2. 计算用户的调节任务和惩罚函数
    print("\n=== 用户调节任务和惩罚函数 ===")
    for user_class, prop_data in bound_proportions['users'].items():
        proportion = prop_data['proportion'] / 100.0  # 转换为小数比例
        
        # 调节任务 = -(调节指令 - 初始购售电) × 比例（负数）
        adjustment_task = -adjustment_difference * proportion
        
        # 获取用户初始用电量
        user_idx = list(USER_PARAMS.keys()).index(user_class)
        initial_demand = USER_INITIAL_DEMAND[:, user_idx]
        
        # 假设最终用电量（这里用初始用电量作为示例）
        # 用电变化量 = 最终用电量 - 初始用电量
        demand_change = np.zeros(24)  # 实际应该从优化结果获取
        
        # 惩罚函数 = |用电变化量 - 调节任务| × 0.1
        penalty = np.abs(demand_change - adjustment_task) * penalty_weight
        
        penalty_results['users'][user_class] = {
            'proportion': proportion,
            'adjustment_task': adjustment_task,
            'initial_demand': initial_demand,
            'demand_change': demand_change,
            'penalty': penalty,
            'total_penalty': np.sum(penalty)
        }
        
        print(f"{user_class}:")
        print(f"  占比: {proportion*100:.2f}%")
        print(f"  平均调节任务: {np.mean(adjustment_task):.2f} kWh")
        print(f"  总惩罚值: {np.sum(penalty):.2f}")
    
    # 3. 计算电动汽车的调节任务和惩罚函数
    print("\n=== 电动汽车调节任务和惩罚函数 ===")
    ev_count_per_class = 20
    
    for ev_class, prop_data in bound_proportions['electric_vehicles'].items():
        proportion = prop_data['proportion'] / 100.0  # 转换为小数比例
        
        # 调节任务 = (调节指令 - 初始购售电) × 比例
        adjustment_task = adjustment_difference * proportion
        
        # 根据差额确定充放电任务
        # 当差额 > 0，就放电，减去任务（即 charging = -任务）
        # 当差额 < 0，就充电，减去任务的负数（即 charging = 任务）
        ev_charging_task = np.zeros(24)
        for h in range(24):
            if adjustment_difference[h] > 0:
                # 差额 > 0，放电
                ev_charging_task[h] = -adjustment_task[h]
            else:
                # 差额 < 0，充电
                ev_charging_task[h] = adjustment_task[h]
        
        # 读取初始充电数据
        ev_data = read_ev_data_from_excel('充电和行驶初始数据.xlsx', verify=False)
        initial_charging_total = np.zeros(24)
        for i in range(min(ev_count_per_class, EV_CLASSES[ev_class]['count'])):
            vehicle_name = f"{ev_class}_EV_{i}"
            if ev_class in ev_data and vehicle_name in ev_data[ev_class]:
                vehicle_data = ev_data[ev_class][vehicle_name]
                initial_charging_total += vehicle_data.get('charging', np.zeros(24))
        
        # 假设最终充电量（这里用初始充电量作为示例）
        # 充电/放电变化量 = 最终充电量 - 初始充电量
        charging_change = np.zeros(24)  # 实际应该从优化结果获取
        
        # 惩罚函数 = |充电/放电变化量 - 调节任务| × 0.1
        penalty = np.abs(charging_change - ev_charging_task) * penalty_weight
        
        penalty_results['electric_vehicles'][ev_class] = {
            'proportion': proportion,
            'adjustment_task': adjustment_task,
            'ev_charging_task': ev_charging_task,
            'initial_charging': initial_charging_total,
            'charging_change': charging_change,
            'penalty': penalty,
            'total_penalty': np.sum(penalty)
        }
        
        print(f"{ev_class} (前{ev_count_per_class}个):")
        print(f"  占比: {proportion*100:.2f}%")
        print(f"  平均调节任务: {np.mean(adjustment_task):.2f} kWh")
        print(f"  平均充放电任务: {np.mean(ev_charging_task):.2f} kWh")
        print(f"  总惩罚值: {np.sum(penalty):.2f}")
    
    return penalty_results


def visualize_adjustment_task_penalty(penalty_results, adjustment_instruction, initial_grid_exchange):
    """
    可视化调节任务和惩罚函数
    """
    hours = np.arange(1, 25)
    
    # 1. 调节任务可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('各主体调节任务分析', fontsize=16, fontweight='bold')
    
    # 1.1 传统发电机调节任务
    ax1 = axes[0, 0]
    for gen_name, data in penalty_results['traditional_generators'].items():
        ax1.plot(hours, data['adjustment_task'], label=f'{gen_name}调节任务', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.set_title('传统发电机调节任务', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时段', fontsize=10)
    ax1.set_ylabel('调节任务 (kWh)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 1.2 用户调节任务
    ax2 = axes[0, 1]
    for user_class, data in penalty_results['users'].items():
        ax2.plot(hours, data['adjustment_task'], label=f'{user_class}调节任务', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_title('用户调节任务', fontsize=12, fontweight='bold')
    ax2.set_xlabel('时段', fontsize=10)
    ax2.set_ylabel('调节任务 (kWh)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 1.3 电动汽车调节任务
    ax3 = axes[1, 0]
    for ev_class, data in penalty_results['electric_vehicles'].items():
        ax3.plot(hours, data['adjustment_task'], label=f'{ev_class}调节任务', linewidth=1.5, linestyle='--')
        ax3.plot(hours, data['ev_charging_task'], label=f'{ev_class}充放电任务', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_title('电动汽车调节任务和充放电任务', fontsize=12, fontweight='bold')
    ax3.set_xlabel('时段', fontsize=10)
    ax3.set_ylabel('任务 (kWh)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 1.4 调节指令与初始购售电差额
    ax4 = axes[1, 1]
    ax4.plot(hours, initial_grid_exchange, 'b-', label='初始购售电', linewidth=1.5)
    ax4.plot(hours, adjustment_instruction, 'r--', label='调节指令', linewidth=1.5)
    ax4.plot(hours, penalty_results['adjustment_difference'], 'g-', label='差额（调节指令-初始购售电）', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax4.set_title('调节指令与初始购售电差额', fontsize=12, fontweight='bold')
    ax4.set_xlabel('时段', fontsize=10)
    ax4.set_ylabel('电量 (kWh)', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('各主体调节任务分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 惩罚函数可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('各主体调节任务惩罚函数分析', fontsize=16, fontweight='bold')
    
    # 2.1 传统发电机惩罚函数
    ax1 = axes[0, 0]
    for gen_name, data in penalty_results['traditional_generators'].items():
        ax1.plot(hours, data['penalty'], label=f'{gen_name}惩罚', linewidth=1.5, marker='o', markersize=3)
    ax1.set_title('传统发电机惩罚函数', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时段', fontsize=10)
    ax1.set_ylabel('惩罚值', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2.2 用户惩罚函数
    ax2 = axes[0, 1]
    for user_class, data in penalty_results['users'].items():
        ax2.plot(hours, data['penalty'], label=f'{user_class}惩罚', linewidth=1.5, marker='s', markersize=3)
    ax2.set_title('用户惩罚函数', fontsize=12, fontweight='bold')
    ax2.set_xlabel('时段', fontsize=10)
    ax2.set_ylabel('惩罚值', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 2.3 电动汽车惩罚函数
    ax3 = axes[1, 0]
    for ev_class, data in penalty_results['electric_vehicles'].items():
        ax3.plot(hours, data['penalty'], label=f'{ev_class}惩罚', linewidth=1.5, marker='^', markersize=3)
    ax3.set_title('电动汽车惩罚函数', fontsize=12, fontweight='bold')
    ax3.set_xlabel('时段', fontsize=10)
    ax3.set_ylabel('惩罚值', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 2.4 总惩罚值统计
    ax4 = axes[1, 1]
    categories = []
    total_penalties = []
    colors_list = []
    
    for gen_name, data in penalty_results['traditional_generators'].items():
        categories.append(gen_name)
        total_penalties.append(data['total_penalty'])
        colors_list.append('blue')
    
    for user_class, data in penalty_results['users'].items():
        categories.append(user_class)
        total_penalties.append(data['total_penalty'])
        colors_list.append('green')
    
    for ev_class, data in penalty_results['electric_vehicles'].items():
        categories.append(ev_class)
        total_penalties.append(data['total_penalty'])
        colors_list.append('orange')
    
    bars = ax4.bar(range(len(categories)), total_penalties, color=colors_list, alpha=0.7)
    ax4.set_title('各主体总惩罚值统计', fontsize=12, fontweight='bold')
    ax4.set_xlabel('主体', fontsize=10)
    ax4.set_ylabel('总惩罚值', fontsize=10)
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, total_penalties):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('各主体调节任务惩罚函数分析.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_system_power_exchange():
    """
    计算系统总的购售电情况
    汽车的用电不考虑计算进入，但充电算
    每组汽车只读取前20个
    """
    # 初始化发电机
    trad_gen1 = TraditionalGenerator('generator1', 1200, 0, [0.0005, 0.15, 105], TRADITIONAL_INITIAL_POWER['generator1'])
    trad_gen2 = TraditionalGenerator('generator2', 1200, 0, [0.0005, 0.12, 105], TRADITIONAL_INITIAL_POWER['generator2'])
    trad_gen3 = TraditionalGenerator('generator3', 1200, 0, [0.0005, 0.18, 105], TRADITIONAL_INITIAL_POWER['generator3'])
    solar_gen = RenewableGenerator('solar', 1000, 0, [0.01, 0], RENEWABLE_INITIAL_POWER['solar'])
    wind_gen = RenewableGenerator('wind', 1000, 0, [0.01, 0], RENEWABLE_INITIAL_POWER['wind'])
    
    # 初始化用户
    users = {}
    for i, (user_class, params) in enumerate(USER_PARAMS.items()):
        user = User(f"{user_class}_user",
                   params['max_demand'],
                   params['min_demand'],
                   params['utility_params'],
                   USER_INITIAL_DEMAND[:, i])
        users[user_class] = user
    
    # 读取电动汽车数据（只读取前20个）
    ev_data = read_ev_data_from_excel('充电和行驶初始数据.xlsx', verify=False)
    
    # 初始化电动汽车（每组只读取前20个）
    evs = {}
    ev_count_per_class = 20  # 每组只读取20个
    
    for class_name, class_params in EV_CLASSES.items():
        evs[class_name] = []
        for i in range(min(ev_count_per_class, class_params['count'])):
            vehicle_name = f"{class_name}_EV_{i}"
            ev = ElectricVehicle(
                class_name=class_name,
                params={
                    'name': vehicle_name,
                    'max_capacity': class_params['max_capacity'],
                    'min_capacity': class_params['min_capacity'],
                    'initial_soc': class_params['initial_soc'],
                    'max_soc': class_params['max_soc'],
                    'min_soc': class_params['min_soc'],
                    'count': class_params['count']
                }
            )
            
            # 从Excel数据中获取初始充电和用电数据
            if class_name in ev_data and vehicle_name in ev_data[class_name]:
                vehicle_data = ev_data[class_name][vehicle_name]
                ev.charging = vehicle_data.get('charging', np.zeros(24))
                ev.usage = vehicle_data.get('usage', np.zeros(24))
            else:
                ev.charging = np.zeros(24)
                ev.usage = np.zeros(24)
            
            evs[class_name].append(ev)
    
    # 计算总发电量（包括传统发电机和可再生能源）
    total_generation = np.zeros(24)
    trad_gen1_power = trad_gen1.generate_power()
    trad_gen2_power = trad_gen2.generate_power()
    trad_gen3_power = trad_gen3.generate_power()
    solar_power = solar_gen.generate_power()
    wind_power = wind_gen.generate_power()
    
    total_generation += trad_gen1_power
    total_generation += trad_gen2_power
    total_generation += trad_gen3_power
    total_generation += solar_power
    total_generation += wind_power
    
    # 计算总用电量（不包括电动汽车的用电）
    total_consumption = np.zeros(24)
    for user_class, user in users.items():
        total_consumption += user.generate_demand()
    # 注意：不包括电动汽车的usage
    
    # 计算电动汽车总充电量（计入系统）
    total_ev_charging = np.zeros(24)
    for ev_class, ev_list in evs.items():
        for ev in ev_list:
            total_ev_charging += ev.charging
    
    # 计算系统购售电情况
    # 购售电 = 发电量（包括可再生能源） - 用电量 - 电动汽车充电量
    # 正值表示向电网售电，负值表示从电网购电
    grid_exchange = total_generation - total_consumption - total_ev_charging
    
    return {
        'total_generation': total_generation,
        'traditional_generation': trad_gen1_power + trad_gen2_power + trad_gen3_power,
        'renewable_generation': solar_power + wind_power,
        'solar_generation': solar_power,
        'wind_generation': wind_power,
        'total_consumption': total_consumption,
        'total_ev_charging': total_ev_charging,
        'grid_exchange': grid_exchange,
        'ev_count': ev_count_per_class
    }


def visualize_adjustment_capability(adjustment_capabilities, system_capability, system_power, proportions):
    """
    可视化每个主体的调节能力和系统总调节能力、购售电情况
    """
    hours = np.arange(1, 25)
    
    # 1. 传统发电机调节能力可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('传统发电机调节能力', fontsize=16, fontweight='bold')
    
    for idx, (gen_name, data) in enumerate(adjustment_capabilities['traditional_generators'].items()):
        row = idx // 2
        col = idx % 2
        if row < 2 and col < 2:
            ax = axes[row, col]
            ax.plot(hours, data['current_power'], 'b-', label='当前发电量', linewidth=2)
            ax.plot(hours, data['max_capacity'], 'g--', label='最大容量', linewidth=1.5, alpha=0.7)
            ax.plot(hours, data['min_capacity'], 'r--', label='最小容量', linewidth=1.5, alpha=0.7)
            ax.fill_between(hours, data['current_power'], data['max_capacity'], 
                          alpha=0.3, color='green', label='向上调节能力')
            ax.fill_between(hours, data['min_capacity'], data['current_power'], 
                          alpha=0.3, color='red', label='向下调节能力')
            ax.set_title(f'{gen_name} 调节能力', fontsize=12, fontweight='bold')
            ax.set_xlabel('时段', fontsize=10)
            ax.set_ylabel('电量 (kWh)', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('传统发电机调节能力.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 用户调节能力可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('用户调节能力', fontsize=16, fontweight='bold')
    
    for idx, (user_class, data) in enumerate(adjustment_capabilities['users'].items()):
        row = idx // 2
        col = idx % 2
        if row < 2 and col < 2:
            ax = axes[row, col]
            ax.plot(hours, data['current_demand'], 'b-', label='当前用电量', linewidth=2)
            ax.plot(hours, data['max_demand'], 'g--', label='最大需求', linewidth=1.5, alpha=0.7)
            ax.plot(hours, data['min_demand'], 'r--', label='最小需求', linewidth=1.5, alpha=0.7)
            ax.fill_between(hours, data['current_demand'], data['max_demand'], 
                          alpha=0.3, color='green', label='向上调节能力（增加负荷）')
            ax.fill_between(hours, data['min_demand'], data['current_demand'], 
                          alpha=0.3, color='red', label='向下调节能力（减少负荷）')
            ax.set_title(f'{user_class} 用户调节能力', fontsize=12, fontweight='bold')
            ax.set_xlabel('时段', fontsize=10)
            ax.set_ylabel('电量 (kWh)', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('用户调节能力.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 系统总调节能力可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('系统总调节能力', fontsize=16, fontweight='bold')
    
    # 4.1 发电侧调节能力
    ax1 = axes[0, 0]
    ax1.plot(hours, system_capability['gen_upward'], 'g-', label='发电侧向上调节能力', linewidth=2, marker='o')
    ax1.plot(hours, system_capability['gen_downward'], 'r-', label='发电侧向下调节能力', linewidth=2, marker='s')
    ax1.set_title('发电侧总调节能力', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时段', fontsize=10)
    ax1.set_ylabel('电量 (kWh)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 4.2 用电侧调节能力
    ax2 = axes[0, 1]
    ax2.plot(hours, system_capability['user_upward'], 'g-', label='用电侧向上调节能力', linewidth=2, marker='o')
    ax2.plot(hours, system_capability['user_downward'], 'r-', label='用电侧向下调节能力', linewidth=2, marker='s')
    ax2.set_title('用电侧总调节能力', fontsize=12, fontweight='bold')
    ax2.set_xlabel('时段', fontsize=10)
    ax2.set_ylabel('电量 (kWh)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 4.3 场景1：发电侧上调 + 用电侧下调
    ax3 = axes[1, 0]
    ax3.plot(hours, system_capability['scenario1'], 'b-', label='发电侧上调+用电侧下调', linewidth=2, marker='o')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_title('系统调节能力场景1：发电侧上调 + 用电侧下调', fontsize=12, fontweight='bold')
    ax3.set_xlabel('时段', fontsize=10)
    ax3.set_ylabel('电量 (kWh)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4.4 场景2：发电侧下调 + 用电侧上调
    ax4 = axes[1, 1]
    ax4.plot(hours, system_capability['scenario2'], 'm-', label='发电侧下调+用电侧上调', linewidth=2, marker='s')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax4.set_title('系统调节能力场景2：发电侧下调 + 用电侧上调', fontsize=12, fontweight='bold')
    ax4.set_xlabel('时段', fontsize=10)
    ax4.set_ylabel('电量 (kWh)', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('系统总调节能力.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 系统购售电情况可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('系统购售电情况分析', fontsize=16, fontweight='bold')
    
    # 5.1 发电量和用电量对比（包括可再生能源）
    ax1 = axes[0, 0]
    ax1.plot(hours, system_power['total_generation'], 'g-', label='总发电量（含可再生能源）', linewidth=2, marker='o')
    ax1.plot(hours, system_power['traditional_generation'], 'g--', label='传统发电量', linewidth=1.5, alpha=0.7)
    ax1.plot(hours, system_power['renewable_generation'], 'g:', label='可再生能源发电量', linewidth=1.5, alpha=0.7)
    ax1.plot(hours, system_power['total_consumption'], 'r-', label='总用电量（不含EV用电）', linewidth=2, marker='s')
    ax1.plot(hours, system_power['total_ev_charging'], 'b-', label='电动汽车总充电量', linewidth=2, marker='^')
    ax1.set_title('发电量与用电量对比（包括可再生能源）', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时段', fontsize=10)
    ax1.set_ylabel('电量 (kWh)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 5.2 系统购售电曲线
    ax2 = axes[0, 1]
    ax2.plot(hours, system_power['grid_exchange'], 'purple', label='系统购售电量', linewidth=2, marker='o')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax2.fill_between(hours, 0, system_power['grid_exchange'], 
                     where=(system_power['grid_exchange'] >= 0), 
                     alpha=0.3, color='green', label='向电网售电')
    ax2.fill_between(hours, 0, system_power['grid_exchange'], 
                     where=(system_power['grid_exchange'] < 0), 
                     alpha=0.3, color='red', label='从电网购电')
    ax2.set_title('系统购售电曲线', fontsize=12, fontweight='bold')
    ax2.set_xlabel('时段', fontsize=10)
    ax2.set_ylabel('电量 (kWh)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 5.3 购售电统计
    ax3 = axes[1, 0]
    buy_hours = np.sum(system_power['grid_exchange'] < 0)
    sell_hours = np.sum(system_power['grid_exchange'] > 0)
    zero_hours = np.sum(system_power['grid_exchange'] == 0)
    total_buy = np.sum(np.abs(system_power['grid_exchange'][system_power['grid_exchange'] < 0]))
    total_sell = np.sum(system_power['grid_exchange'][system_power['grid_exchange'] > 0])
    
    categories = ['购电时段', '售电时段', '平衡时段']
    values = [buy_hours, sell_hours, zero_hours]
    colors = ['red', 'green', 'gray']
    bars = ax3.bar(categories, values, color=colors, alpha=0.7)
    ax3.set_title('购售电时段统计', fontsize=12, fontweight='bold')
    ax3.set_ylabel('时段数', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=10)
    
    # 5.4 购售电量统计
    ax4 = axes[1, 1]
    categories2 = ['总购电量', '总售电量']
    values2 = [total_buy, total_sell]
    colors2 = ['red', 'green']
    bars2 = ax4.bar(categories2, values2, color=colors2, alpha=0.7)
    ax4.set_title('购售电量统计', fontsize=12, fontweight='bold')
    ax4.set_ylabel('电量 (kWh)', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, values2):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('系统购售电情况.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 各主体调节能力比例可视化
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('各主体调节能力占比分析（场景1：发电侧上调+用电侧下调）', fontsize=16, fontweight='bold')
    
    # 5.1 传统发电机上调能力占比
    ax1 = axes[0, 0]
    gen_names = []
    gen_proportions = []
    for gen_name, prop_data in proportions['traditional_generators'].items():
        gen_names.append(gen_name)
        gen_proportions.append(prop_data['proportion_scenario1_upward'])
    
    bars1 = ax1.bar(gen_names, gen_proportions, color='green', alpha=0.7)
    ax1.set_title('传统发电机上调能力在场景1中的占比', fontsize=12, fontweight='bold')
    ax1.set_xlabel('发电机', fontsize=10)
    ax1.set_ylabel('占比 (%)', fontsize=10)
    ax1.set_ylim([0, max(gen_proportions) * 1.2 if gen_proportions else 100])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, gen_proportions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 5.2 用户下调能力占比
    ax2 = axes[0, 1]
    user_names = []
    user_proportions = []
    for user_class, prop_data in proportions['users'].items():
        user_names.append(user_class)
        user_proportions.append(prop_data['proportion_scenario1_downward'])
    
    bars2 = ax2.bar(user_names, user_proportions, color='red', alpha=0.7)
    ax2.set_title('用户下调能力在场景1中的占比', fontsize=12, fontweight='bold')
    ax2.set_xlabel('用户类别', fontsize=10)
    ax2.set_ylabel('占比 (%)', fontsize=10)
    ax2.set_ylim([0, max(user_proportions) * 1.2 if user_proportions else 100])
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, user_proportions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 5.3 场景1总体比例：发电侧上调 vs 用电侧下调
    ax3 = axes[1, 0]
    scenario1_prop = proportions['scenario1_proportion']
    categories_scenario1 = ['发电侧上调', '用电侧下调']
    values_scenario1 = [scenario1_prop['gen_upward_proportion'], 
                       scenario1_prop['user_downward_proportion']]
    colors_scenario1 = ['green', 'red']
    bars3 = ax3.bar(categories_scenario1, values_scenario1, color=colors_scenario1, alpha=0.7)
    ax3.set_title('场景1总调节能力占比：发电侧上调 vs 用电侧下调', fontsize=12, fontweight='bold')
    ax3.set_ylabel('占比 (%)', fontsize=10)
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, values_scenario1):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 5.4 所有主体平均调节能力占比汇总
    ax4 = axes[1, 1]
    all_names = gen_names + user_names
    all_proportions = gen_proportions + user_proportions
    all_colors = ['green'] * len(gen_names) + ['red'] * len(user_names)
    
    bars4 = ax4.bar(range(len(all_names)), all_proportions, color=all_colors, alpha=0.7)
    ax4.set_title('所有主体在总调节能力中的占比', fontsize=12, fontweight='bold')
    ax4.set_xlabel('主体', fontsize=10)
    ax4.set_ylabel('占比 (%)', fontsize=10)
    ax4.set_xticks(range(len(all_names)))
    ax4.set_xticklabels(all_names, rotation=45, ha='right', fontsize=9)
    ax4.set_ylim([0, max(all_proportions) * 1.2 if all_proportions else 100])
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars4, all_proportions):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('各主体调节能力占比分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 综合对比图：所有主体的总调节能力
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('各主体调节能力综合对比', fontsize=16, fontweight='bold')
    
    # 6.1 向上调节能力对比（发电侧）
    ax1 = axes[0, 0]
    for gen_name, data in adjustment_capabilities['traditional_generators'].items():
        ax1.plot(hours, data['upward'], label=f'{gen_name}向上', linewidth=1.5)
    ax1.set_title('向上调节能力对比（发电侧）', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时段', fontsize=10)
    ax1.set_ylabel('电量 (kWh)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 6.2 向下调节能力对比（发电侧）
    ax2 = axes[0, 1]
    for gen_name, data in adjustment_capabilities['traditional_generators'].items():
        ax2.plot(hours, data['downward'], label=f'{gen_name}向下', linewidth=1.5)
    ax2.set_title('向下调节能力对比（发电侧）', fontsize=12, fontweight='bold')
    ax2.set_xlabel('时段', fontsize=10)
    ax2.set_ylabel('电量 (kWh)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 6.3 向上调节能力对比（用电侧）
    ax3 = axes[1, 0]
    for user_class, data in adjustment_capabilities['users'].items():
        ax3.plot(hours, data['upward'], label=f'{user_class}向上', linewidth=1.5)
    ax3.set_title('向上调节能力对比（用电侧）', fontsize=12, fontweight='bold')
    ax3.set_xlabel('时段', fontsize=10)
    ax3.set_ylabel('电量 (kWh)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 6.4 向下调节能力对比（用电侧）
    ax4 = axes[1, 1]
    for user_class, data in adjustment_capabilities['users'].items():
        ax4.plot(hours, data['downward'], label=f'{user_class}向下', linewidth=1.5)
    ax4.set_title('向下调节能力对比（用电侧）', fontsize=12, fontweight='bold')
    ax4.set_xlabel('时段', fontsize=10)
    ax4.set_ylabel('电量 (kWh)', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('各主体调节能力综合对比.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 总调节能力统计柱状图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 7.1 平均向上调节能力
    ax1 = axes[0]
    categories = []
    values_upward = []
    for gen_name, data in adjustment_capabilities['traditional_generators'].items():
        categories.append(gen_name)
        values_upward.append(np.mean(data['upward']))
    for user_class, data in adjustment_capabilities['users'].items():
        categories.append(f'{user_class}_user')
        values_upward.append(np.mean(data['upward']))
    
    bars1 = ax1.bar(range(len(categories)), values_upward, color='green', alpha=0.7)
    ax1.set_title('平均向上调节能力', fontsize=12, fontweight='bold')
    ax1.set_xlabel('主体', fontsize=10)
    ax1.set_ylabel('电量 (kWh)', fontsize=10)
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, values_upward)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 7.2 平均向下调节能力
    ax2 = axes[1]
    values_downward = []
    for gen_name, data in adjustment_capabilities['traditional_generators'].items():
        values_downward.append(np.mean(data['downward']))
    for user_class, data in adjustment_capabilities['users'].items():
        values_downward.append(np.mean(data['downward']))
    
    bars2 = ax2.bar(range(len(categories)), values_downward, color='red', alpha=0.7)
    ax2.set_title('平均向下调节能力', fontsize=12, fontweight='bold')
    ax2.set_xlabel('主体', fontsize=10)
    ax2.set_ylabel('电量 (kWh)', fontsize=10)
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars2, values_downward)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('调节能力统计对比.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. 调节指令与系统购售电对比可视化
    if 'adjustment_instruction' in system_power:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('调节指令与系统购售电对比分析', fontsize=16, fontweight='bold')
        
        # 8.1 调节指令与系统购售电曲线对比
        ax1 = axes[0, 0]
        ax1.plot(hours, system_power['grid_exchange'], 'b-', label='系统购售电量', linewidth=2, marker='o')
        ax1.plot(hours, system_power['adjustment_instruction'], 'r--', label='调节指令（购售电+10%）', linewidth=2, marker='s')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax1.fill_between(hours, 0, system_power['grid_exchange'], 
                        where=(system_power['grid_exchange'] >= 0), 
                        alpha=0.2, color='green', label='原售电区域')
        ax1.fill_between(hours, 0, system_power['grid_exchange'], 
                        where=(system_power['grid_exchange'] < 0), 
                        alpha=0.2, color='red', label='原购电区域')
        ax1.set_title('调节指令与系统购售电曲线对比', fontsize=12, fontweight='bold')
        ax1.set_xlabel('时段', fontsize=10)
        ax1.set_ylabel('电量 (kWh)', fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 8.2 调节指令变化量
        ax2 = axes[0, 1]
        adjustment_change = system_power['adjustment_instruction'] - system_power['grid_exchange']
        ax2.plot(hours, adjustment_change, 'purple', label='调节指令变化量', linewidth=2, marker='^')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax2.fill_between(hours, 0, adjustment_change, 
                        where=(adjustment_change >= 0), 
                        alpha=0.3, color='green', label='增加')
        ax2.fill_between(hours, 0, adjustment_change, 
                        where=(adjustment_change < 0), 
                        alpha=0.3, color='red', label='减少')
        ax2.set_title('调节指令变化量（调节指令 - 原购售电量）', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时段', fontsize=10)
        ax2.set_ylabel('电量变化 (kWh)', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 8.3 购售电量对比统计
        ax3 = axes[1, 0]
        original_buy = np.sum(np.abs(system_power['grid_exchange'][system_power['grid_exchange'] < 0]))
        original_sell = np.sum(system_power['grid_exchange'][system_power['grid_exchange'] > 0])
        instruction_buy = np.sum(np.abs(system_power['adjustment_instruction'][system_power['adjustment_instruction'] < 0]))
        instruction_sell = np.sum(system_power['adjustment_instruction'][system_power['adjustment_instruction'] > 0])
        
        x = np.arange(2)
        width = 0.35
        bars1 = ax3.bar(x - width/2, [original_buy, original_sell], width, label='原系统', color=['red', 'green'], alpha=0.7)
        bars2 = ax3.bar(x + width/2, [instruction_buy, instruction_sell], width, label='调节指令', color=['darkred', 'darkgreen'], alpha=0.7)
        ax3.set_title('购售电量对比：原系统 vs 调节指令', fontsize=12, fontweight='bold')
        ax3.set_ylabel('电量 (kWh)', fontsize=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['购电量', '售电量'])
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 8.4 购售电量变化百分比
        ax4 = axes[1, 1]
        buy_change_pct = ((instruction_buy - original_buy) / original_buy * 100) if original_buy > 0 else 0
        sell_change_pct = ((instruction_sell - original_sell) / original_sell * 100) if original_sell > 0 else 0
        
        categories = ['购电量变化', '售电量变化']
        values = [buy_change_pct, sell_change_pct]
        colors = ['red', 'green']
        bars = ax4.bar(categories, values, color=colors, alpha=0.7)
        ax4.set_title('购售电量变化百分比（调节指令相对原系统）', fontsize=12, fontweight='bold')
        ax4.set_ylabel('变化百分比 (%)', fontsize=10)
        ax4.axhline(y=10, color='black', linestyle='--', linewidth=1, alpha=0.5, label='目标变化（10%）')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend(fontsize=9)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('调节指令与系统购售电对比.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    主函数：计算并可视化每个主体的调节能力和系统购售电情况
    """
    print("=" * 60)
    print("开始计算各主体调节能力（不包括电动汽车）")
    print("=" * 60)
    
    # 计算调节能力（不包括电动汽车）
    adjustment_capabilities = calculate_adjustment_capability()
    
    print("\n" + "=" * 60)
    print("开始计算系统总调节能力")
    print("=" * 60)
    
    # 计算系统总调节能力
    system_capability = calculate_system_adjustment_capability(adjustment_capabilities)
    
    print("\n=== 系统总调节能力统计 ===")
    print(f"场景1（发电侧上调+用电侧下调）平均: {np.mean(system_capability['scenario1']):.2f} kWh")
    print(f"场景2（发电侧下调+用电侧上调）平均: {np.mean(system_capability['scenario2']):.2f} kWh")
    
    print("\n" + "=" * 60)
    print("开始计算系统购售电情况")
    print("=" * 60)
    
    # 计算系统购售电情况
    system_power = calculate_system_power_exchange()
    
    print(f"\n=== 系统购售电情况统计（包括可再生能源） ===")
    print(f"电动汽车数量（每组）: {system_power['ev_count']} 辆")
    print(f"总发电量: {np.sum(system_power['total_generation']):.2f} kWh")
    print(f"  其中传统发电量: {np.sum(system_power['traditional_generation']):.2f} kWh")
    print(f"  其中可再生能源发电量: {np.sum(system_power['renewable_generation']):.2f} kWh")
    print(f"    其中太阳能发电量: {np.sum(system_power['solar_generation']):.2f} kWh")
    print(f"    其中风能发电量: {np.sum(system_power['wind_generation']):.2f} kWh")
    print(f"总用电量（不含EV用电）: {np.sum(system_power['total_consumption']):.2f} kWh")
    print(f"电动汽车总充电量: {np.sum(system_power['total_ev_charging']):.2f} kWh")
    print(f"系统总购电量: {np.sum(np.abs(system_power['grid_exchange'][system_power['grid_exchange'] < 0])):.2f} kWh")
    print(f"系统总售电量: {np.sum(system_power['grid_exchange'][system_power['grid_exchange'] > 0]):.2f} kWh")
    
    print("\n" + "=" * 60)
    print("开始计算各主体调节能力占比")
    print("=" * 60)
    
    # 计算各主体调节能力占比
    proportions = calculate_adjustment_proportion(adjustment_capabilities, system_capability)
    
    print("\n=== 各主体调节能力占比统计（场景1：发电侧上调+用电侧下调） ===")
    print("\n【传统发电机上调能力占比】")
    for gen_name, prop_data in proportions['traditional_generators'].items():
        print(f"  {gen_name}: {prop_data['proportion_scenario1_upward']:.2f}%")
    
    print("\n【用户下调能力占比】")
    for user_class, prop_data in proportions['users'].items():
        print(f"  {user_class}: {prop_data['proportion_scenario1_downward']:.2f}%")
    
    scenario1_prop = proportions['scenario1_proportion']
    print(f"\n【场景1总体占比】")
    print(f"  发电侧上调占比: {scenario1_prop['gen_upward_proportion']:.2f}%")
    print(f"  用电侧下调占比: {scenario1_prop['user_downward_proportion']:.2f}%")
    
    # 输出用户1（class1）每个小时的上下调能力
    print("\n" + "=" * 60)
    print("用户1（class1）每小时调节能力数值")
    print("=" * 60)
    if 'class1' in adjustment_capabilities['users']:
        user1_data = adjustment_capabilities['users']['class1']
        print("\n时段\t向上调节能力(kWh)\t向下调节能力(kWh)")
        print("-" * 50)
        for h in range(24):
            print(f"{h+1:2d}\t{user1_data['upward'][h]:12.2f}\t\t{user1_data['downward'][h]:12.2f}")
        print(f"\n用户1（class1）平均向上调节能力: {np.mean(user1_data['upward']):.2f} kWh")
        print(f"用户1（class1）平均向下调节能力: {np.mean(user1_data['downward']):.2f} kWh")
        print(f"用户1（class1）总向上调节能力: {np.sum(user1_data['upward']):.2f} kWh")
        print(f"用户1（class1）总向下调节能力: {np.sum(user1_data['downward']):.2f} kWh")
    
    # 计算调节指令：系统购电和售电均增加10%
    # 增加10%意味着：原值 + 原值的10% = 原值 * 1.1
    print("\n" + "=" * 60)
    print("计算调节指令（系统购电和售电均增加10%）")
    print("=" * 60)
    print("说明：增加10% = 原值 × 1.1")
    print("例如：原购售电量100 kWh → 调节指令110 kWh（增加10 kWh）")
    print("例如：原购售电量-100 kWh → 调节指令-110 kWh（绝对值增加10 kWh）")
    
    adjustment_instruction = np.zeros(24)
    for h in range(24):
        original_value = system_power['grid_exchange'][h]
        # 增加10%：原值 × 1.1
        adjustment_instruction[h] = original_value * 1.1
    
    # 单独输出调节指令
    print("\n" + "=" * 60)
    print("调节指令数值（24小时）")
    print("=" * 60)
    print("\n时段\t调节指令(kWh)")
    print("-" * 30)
    for h in range(24):
        print(f"{h+1:2d}\t{adjustment_instruction[h]:12.2f}")
    
    print(f"\n调节指令统计：")
    print(f"  总购电量: {np.sum(np.abs(adjustment_instruction[adjustment_instruction < 0])):.2f} kWh")
    print(f"  总售电量: {np.sum(adjustment_instruction[adjustment_instruction > 0]):.2f} kWh")
    print(f"  平均购售电量: {np.mean(adjustment_instruction):.2f} kWh")
    print(f"  最大购售电量: {np.max(adjustment_instruction):.2f} kWh")
    print(f"  最小购售电量: {np.min(adjustment_instruction):.2f} kWh")
    
    # 输出调节指令与原购售电量的对比
    print("\n" + "=" * 60)
    print("调节指令与原购售电量对比")
    print("=" * 60)
    print("\n时段\t原购售电量(kWh)\t调节指令(kWh)\t变化量(kWh)\t变化百分比(%)")
    print("-" * 85)
    for h in range(24):
        original = system_power['grid_exchange'][h]
        instruction = adjustment_instruction[h]
        change = instruction - original
        # 计算变化百分比（避免除以0）
        if abs(original) > 1e-6:
            change_pct = (change / abs(original)) * 100
        else:
            change_pct = 0.0
        print(f"{h+1:2d}\t{original:12.2f}\t\t{instruction:12.2f}\t\t{change:12.2f}\t\t{change_pct:6.2f}%")
    
    print(f"\n原系统总购电量: {np.sum(np.abs(system_power['grid_exchange'][system_power['grid_exchange'] < 0])):.2f} kWh")
    print(f"原系统总售电量: {np.sum(system_power['grid_exchange'][system_power['grid_exchange'] > 0]):.2f} kWh")
    print(f"调节指令总购电量: {np.sum(np.abs(adjustment_instruction[adjustment_instruction < 0])):.2f} kWh")
    print(f"调节指令总售电量: {np.sum(adjustment_instruction[adjustment_instruction > 0]):.2f} kWh")
    
    # 将调节指令添加到system_power中
    system_power['adjustment_instruction'] = adjustment_instruction
    
    # 计算基于上下界的调节能力
    print("\n" + "=" * 60)
    print("开始计算基于上下界的调节能力（上界-下界）")
    print("=" * 60)
    
    bound_capabilities = calculate_bound_based_adjustment_capability()
    
    print("\n" + "=" * 60)
    print("开始计算基于上下界的调节能力占比")
    print("=" * 60)
    
    bound_proportions = calculate_bound_based_proportion(bound_capabilities)
    
    print("\n=== 基于上下界的调节能力占比统计 ===")
    print("\n【传统发电机调节能力占比】")
    for gen_name, prop_data in bound_proportions['traditional_generators'].items():
        print(f"  {gen_name}: {prop_data['capability']:.2f} kWh ({prop_data['proportion']:.2f}%)")
    
    print("\n【用户调节能力占比】")
    for user_class, prop_data in bound_proportions['users'].items():
        print(f"  {user_class}: {prop_data['capability']:.2f} kWh ({prop_data['proportion']:.2f}%)")
    
    print("\n【电动汽车调节能力占比】")
    for ev_class, prop_data in bound_proportions['electric_vehicles'].items():
        print(f"  {ev_class}: {prop_data['capability']:.2f} kWh ({prop_data['proportion']:.2f}%)")
    
    summary = bound_proportions['summary']
    print(f"\n【总体统计】")
    print(f"  传统发电机总调节能力: {summary['gen_total']:.2f} kWh ({summary['gen_total']/summary['total_capability']*100:.2f}%)")
    print(f"  用户总调节能力: {summary['user_total']:.2f} kWh ({summary['user_total']/summary['total_capability']*100:.2f}%)")
    print(f"  电动汽车总调节能力: {summary['ev_total']:.2f} kWh ({summary['ev_total']/summary['total_capability']*100:.2f}%)")
    print(f"  系统总调节能力: {summary['total_capability']:.2f} kWh")
    
    print("\n" + "=" * 60)
    print("开始可视化调节能力和购售电情况")
    print("=" * 60)
    
    # 可视化调节能力和购售电情况
    visualize_adjustment_capability(adjustment_capabilities, system_capability, system_power, proportions)
    
    # 可视化基于上下界的调节能力
    print("\n" + "=" * 60)
    print("开始可视化基于上下界的调节能力")
    print("=" * 60)
    
    visualize_bound_based_capability(bound_capabilities, bound_proportions)
    
    # 计算调节任务惩罚函数
    print("\n" + "=" * 60)
    print("开始计算调节任务惩罚函数")
    print("=" * 60)
    
    # 初始购售电就是system_power['grid_exchange']
    initial_grid_exchange = system_power['grid_exchange'].copy()
    
    penalty_results = calculate_adjustment_task_penalty(
        adjustment_instruction, 
        initial_grid_exchange, 
        bound_proportions
    )
    
    print("\n" + "=" * 60)
    print("开始可视化调节任务惩罚函数")
    print("=" * 60)
    
    visualize_adjustment_task_penalty(penalty_results, adjustment_instruction, initial_grid_exchange)
    
    print("\n" + "=" * 60)
    print("调节能力计算和可视化完成！")
    print("=" * 60)
    
    # 输出汇总统计
    print("\n=== 调节能力汇总统计 ===")
    print("\n【发电侧（传统发电机）】")
    total_gen_upward = 0
    total_gen_downward = 0
    for gen_name, data in adjustment_capabilities['traditional_generators'].items():
        total_gen_upward += np.sum(data['upward'])
        total_gen_downward += np.sum(data['downward'])
    print(f"  总向上调节能力: {total_gen_upward:.2f} kWh")
    print(f"  总向下调节能力: {total_gen_downward:.2f} kWh")
    
    print("\n【用电侧】")
    total_user_upward = 0
    total_user_downward = 0
    for user_class, data in adjustment_capabilities['users'].items():
        total_user_upward += np.sum(data['upward'])
        total_user_downward += np.sum(data['downward'])
    print(f"  总向上调节能力: {total_user_upward:.2f} kWh")
    print(f"  总向下调节能力: {total_user_downward:.2f} kWh")
    
    print(f"\n【系统总调节能力（场景1：发电侧上调+用电侧下调）】")
    print(f"  总调节能力: {np.sum(system_capability['scenario1']):.2f} kWh")
    print(f"  平均调节能力: {np.mean(system_capability['scenario1']):.2f} kWh")
    
    print(f"\n【系统总调节能力（场景2：发电侧下调+用电侧上调）】")
    print(f"  总调节能力: {np.sum(system_capability['scenario2']):.2f} kWh")
    print(f"  平均调节能力: {np.mean(system_capability['scenario2']):.2f} kWh")


if __name__ == '__main__':
    main()

