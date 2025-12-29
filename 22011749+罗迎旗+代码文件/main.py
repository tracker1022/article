import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
from models import VirtualPowerPlant, TraditionalGenerator, RenewableGenerator, User, ElectricVehicle
from data import MARKET_PRICES, TRADITIONAL_INITIAL_POWER, RENEWABLE_INITIAL_POWER, USER_PARAMS, USER_INITIAL_DEMAND, EV_CLASSES, read_ev_data_from_excel

def main():
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
    
    # 读取电动汽车数据
    ev_data = read_ev_data_from_excel('充电和行驶初始数据.xlsx', verify=True)
    
    # 初始化电动汽车（每类只考虑前20个）
    evs = {}
    initial_ev_charging_dict = {}
    initial_ev_usage_dict = {}
    ev_count_per_class = 20  # 每类只考虑前20个汽车
    for class_name, class_params in EV_CLASSES.items():
        evs[class_name] = []
        # 只考虑前20个汽车
        for i in range(min(ev_count_per_class, class_params['count'])):
            vehicle_name = f"{class_name}_EV_{i}"
            ev = ElectricVehicle(
                class_name=class_name,
                params={
                    'name': vehicle_name,
                    'max_capacity': class_params['max_capacity'],
                    'min_capacity': class_params['min_capacity'],
                    'initial_soc': class_params['initial_soc'],  # SOC比例0-1
                    'max_soc': class_params['max_soc'],  # 电池容量（kWh），在ElectricVehicle中会转换为battery_capacity
                    'min_soc': class_params['min_soc'],  # SOC下限，比例0-1
                    'count': class_params['count']
                }
            )
            
            # 从Excel数据中获取初始充电和用电数据
            if class_name in ev_data and vehicle_name in ev_data[class_name]:
                vehicle_data = ev_data[class_name][vehicle_name]
                ev.charging = vehicle_data.get('charging', np.zeros(24))
                ev.usage = vehicle_data.get('usage', np.zeros(24))
            else:
                print(f"警告: 未找到 {vehicle_name} 的数据，使用默认值")
                ev.charging = np.zeros(24)
                ev.usage = np.zeros(24)
            
            # 保存初始数据副本
            initial_ev_charging_dict[ev.name] = ev.charging.copy()
            initial_ev_usage_dict[ev.name] = ev.usage.copy()
            
            evs[class_name].append(ev)
    
    # 调试输出：检查EV初始数据
    print(f"\n=== EV初始数据检查（每类只考虑前{ev_count_per_class}个） ===")
    for ev_class, ev_list in evs.items():
        if len(ev_list) > 0:
            ev = ev_list[0]  # 每类取第一个
            print(f"{ev.name} (共{len(ev_list)}辆车):")
            print(f"  充电数据样本: {ev.charging[:5]}...")
            print(f"  用电数据样本: {ev.usage[:5]}...")
            print(f"  充电数据范围: [{np.min(ev.charging):.2f}, {np.max(ev.charging):.2f}]")
            print(f"  用电数据范围: [{np.min(ev.usage):.2f}, {np.max(ev.usage):.2f}]")
    
    # 初始化虚拟电厂
    vpp = VirtualPowerPlant("VPP1")
    vpp.add_traditional_generator(trad_gen1)
    vpp.add_traditional_generator(trad_gen2)
    vpp.add_traditional_generator(trad_gen3)
    vpp.add_renewable_generator('solar', solar_gen)
    vpp.add_renewable_generator('wind', wind_gen)
    for user_class, user in users.items():
        vpp.add_user(user_class, user)
    for ev_class, ev_list in evs.items():
        for ev in ev_list:
            vpp.add_ev(ev_class, ev)
    
    # ----------- 设置调节指令为初始购售电曲线 -----------
    # 计算初始发电量
    initial_generation = np.zeros(24)
    for g in vpp.traditional_generators:
        initial_generation += g.generate_power()
    for g in vpp.renewable_generators.values():
        if g is not None:
            initial_generation += g.generate_power()
    # 计算初始用电量
    initial_consumption = np.zeros(24)
    for user_list in vpp.users.values():
        for user in user_list:
            initial_consumption += user.generate_demand()
    for ev_list in vpp.evs.values():
        for ev in ev_list:
            initial_consumption += ev.usage
    # 计算初始购售电曲线
    initial_grid_exchange = initial_generation - initial_consumption
    # 设置为调节指令
    vpp.set_adjustment_instruction(initial_grid_exchange)
    vpp.set_penalty_weight(1.1)  # 可根据需求调整权重
    
    # 记录PSO搜索开始时间
    start_time = time.time()
    # 只用PSO算法全局优化VPP价格，传递verbose参数加速
    pso_result = vpp.global_price_optimize(MARKET_PRICES, verbose=False)
    end_time = time.time()
    print(f"PSO搜索耗时: {end_time - start_time:.2f} 秒")

    best_prices = pso_result['best_prices']
    best_revenue = pso_result['best_revenue']
    decision_vars = pso_result['decision_vars']

    # 输出最优价格、决策变量、收益等分析结果
    print('最优24小时VPP价格:')
    for h, p in enumerate(best_prices):
        print(f"时段{h+1}: buy={p['buy']:.4f}, sell={p['sell']:.4f}")
    print('\n最优VPP总收益: {:.4f}'.format(best_revenue))
    print('\n各发电机24小时最优发电量:')
    for k, v in decision_vars['generator_powers'].items():
        print(f"{k}: {v}")
    print('\n各用户24小时最优用电量:')
    for k, v in decision_vars['user_demands'].items():
        print(f"{k}: {v}")
    print('\n各电动汽车24小时充电量（初始数据保持不变）:')
    for k, v in decision_vars['ev_charging'].items():
        print(f"{k}: {v}")

    

    # 输出初始EV数据统计
    print('\n=== 初始电动汽车数据统计 ===')
    for ev_class, ev_list in evs.items():
        class_usage = np.zeros(24)
        class_charging = np.zeros(24)
        for ev in ev_list:
            class_usage += ev.usage
            class_charging += ev.charging
        
        total_usage = np.sum(class_usage)
        total_charging = np.sum(class_charging)
        net_energy = total_charging - total_usage
        
        print(f"{ev_class}:")
        print(f"  总用电量: {total_usage:.2f} kWh")
        print(f"  总充放电量: {total_charging:.2f} kWh")
        print(f"  净能量变化: {net_energy:.2f} kWh")
        print(f"  平均每小时用电量: {np.mean(class_usage):.2f} kWh")
        print(f"  平均每小时充放电量: {np.mean(class_charging):.2f} kWh")
    


    # ----------- 可视化部分 -----------
    hours = np.arange(1, 25)
    buy_prices = [p['buy'] for p in best_prices]
    sell_prices = [p['sell'] for p in best_prices]
    # 用户总用电量
    user_total_demand = np.zeros(24)
    for v in decision_vars['user_demands'].values():
        user_total_demand += np.array(v)
    # EV总充电量
    ev_total_charging = np.zeros(24)
    for v in decision_vars['ev_charging'].values():
        ev_total_charging += np.array(v['charging']) if isinstance(v, dict) and 'charging' in v else np.array(v)
    # 发电侧总发电量
    gen_total_power = np.zeros(24)
    for v in decision_vars['generator_powers'].values():
        gen_total_power += np.array(v)

    # --- 新增：市场电价、调节指令和实际购售电情况可视化 ---
    plt.figure(figsize=(14, 10))
    
    # 市场电价
    plt.subplot(2, 2, 1)
    market_buy_prices = MARKET_PRICES['buy']
    market_sell_prices = MARKET_PRICES['sell']
    plt.plot(hours, market_buy_prices, label='市场购电价', marker='o')
    plt.plot(hours, market_sell_prices, label='市场售电价', marker='s')
    plt.title('市场24小时购售电价')
    plt.xlabel('时段')
    plt.ylabel('元/kWh')
    plt.legend()
    plt.grid(True)

    # 调节指令
    plt.subplot(2, 2, 2)
    plt.plot(hours, initial_grid_exchange, label='调节指令', marker='o', color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('VPP调节指令（初始购售电曲线）')
    plt.xlabel('时段')
    plt.ylabel('kWh (正值送电，负值购电)')
    plt.legend()
    plt.grid(True)
    
    # 实际购售电情况
    plt.subplot(2, 2, 3)
    actual_grid_exchange = gen_total_power - user_total_demand - ev_total_charging
    plt.plot(hours, actual_grid_exchange, label='实际购售电', marker='s', color='blue')
    plt.plot(hours, initial_grid_exchange, label='调节指令', marker='o', color='red', linestyle='--')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('实际购售电与调节指令对比')
    plt.xlabel('时段')
    plt.ylabel('kWh (正值送电，负值购电)')
    plt.legend()
    plt.grid(True)
    
    # 偏差分析
    plt.subplot(2, 2, 4)
    deviation = actual_grid_exchange - initial_grid_exchange
    plt.plot(hours, deviation, label='偏差', marker='^', color='green')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('购售电偏差分析')
    plt.xlabel('时段')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 新增：初始EV总用电量和充放电量可视化
    plt.figure(figsize=(15, 10))
    
    # 计算初始EV总用电量
    plt.subplot(2, 2, 1)
    initial_ev_usage = np.zeros(24)
    for usage in initial_ev_usage_dict.values():
        initial_ev_usage += usage
    plt.plot(hours, initial_ev_usage, label='所有EV初始总用电量', marker='o', markersize=3)
    plt.title('各类电动汽车初始总用电量')
    plt.xlabel('时段')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True)
    
    # 计算初始EV总充放电量
    plt.subplot(2, 2, 2)
    initial_ev_charging = np.zeros(24)
    for charging in initial_ev_charging_dict.values():
        initial_ev_charging += charging
    plt.plot(hours, initial_ev_charging, label='所有EV初始总充放电量', marker='o', markersize=3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='充放电平衡线')
    plt.title('各类电动汽车初始总充放电量')
    plt.xlabel('时段')
    plt.ylabel('kWh (正值充电，负值放电)')
    plt.legend()
    plt.grid(True)
    
    # 所有EV总用电量和充放电量对比
    plt.subplot(2, 2, 3)
    plt.plot(hours, initial_ev_usage, label='所有EV总用电量', marker='o', color='blue')
    plt.plot(hours, initial_ev_charging, label='所有EV总充放电量', marker='s', color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('所有电动汽车初始总用电量和充放电量对比')
    plt.xlabel('时段')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True)
    
    # EV总充放电量对比
    plt.subplot(2, 2, 4)
    final_ev_charging = np.zeros(24)
    for v in decision_vars['ev_charging'].values():
        if isinstance(v, dict) and 'charging' in v:
            final_ev_charging += np.array(v['charging'])
    plt.plot(hours, initial_ev_charging, label='初始总充放电量', marker='o', color='blue')
    plt.plot(hours, final_ev_charging, label='最终总充放电量', marker='s', color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('电动汽车总充放电量对比')
    plt.xlabel('时段')
    plt.ylabel('kWh (正值充电，负值放电)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # --- 新增：VPP电价初始/最终对比 ---
    plt.figure(figsize=(10, 5))
    initial_buy_prices = [p['buy'] for p in pso_result['initial_prices']] if 'initial_prices' in pso_result else buy_prices
    initial_sell_prices = [p['sell'] for p in pso_result['initial_prices']] if 'initial_prices' in pso_result else sell_prices
    
    plt.plot(hours, buy_prices, label='最终VPP购电价', marker='o')
    plt.plot(hours, sell_prices, label='最终VPP售电价', marker='s')
    plt.title('VPP 24小时购售最终电价')
    plt.xlabel('时段')
    plt.ylabel('元/kWh')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # --- 新增：用户用电量和发电机发电量初始/最终对比 ---
    plt.figure(figsize=(15, 12))
    
    # 几类用户的初始和最终用电量
    plt.subplot(3, 2, 1)
    for i, (user_class, params) in enumerate(USER_PARAMS.items()):
        plt.plot(hours, USER_INITIAL_DEMAND[:, i], '--', label=f'{user_class}初始', alpha=0.7)
        key = f'{user_class}_0'
        if key in decision_vars['user_demands']:
            plt.plot(hours, decision_vars['user_demands'][key], label=f'{user_class}最终')
    plt.title('四类用户初始/最终用电量')
    plt.xlabel('时段')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True)
    
    # 用户总用电量对比
    plt.subplot(3, 2, 2)
    initial_user_total = np.sum(USER_INITIAL_DEMAND, axis=1)
    final_user_total = np.zeros(24)
    for v in decision_vars['user_demands'].values():
        final_user_total += np.array(v)
    plt.plot(hours, initial_user_total, '--', label='用户总用电量初始', marker='o')
    plt.plot(hours, final_user_total, label='用户总用电量最终', marker='s')
    plt.title('用户总用电量初始/最终对比')
    plt.xlabel('时段')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True)
    
    # 可再生能源发电的初始和最终发电量
    plt.subplot(3, 2, 3)
    for gen_type in ['solar', 'wind']:
        initial_power = RENEWABLE_INITIAL_POWER[gen_type]
        plt.plot(hours, initial_power, '--', label=f'{gen_type}初始', alpha=0.7)
        if gen_type in decision_vars['generator_powers']:
            plt.plot(hours, decision_vars['generator_powers'][gen_type], label=f'{gen_type}最终')
    plt.title('可再生能源发电初始/最终发电量')
    plt.xlabel('时段')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True)
    
    # 可再生能源总发电量对比
    plt.subplot(3, 2, 4)
    initial_renewable_total = RENEWABLE_INITIAL_POWER['solar'] + RENEWABLE_INITIAL_POWER['wind']
    final_renewable_total = np.zeros(24)
    for gen_type in ['solar', 'wind']:
        if gen_type in decision_vars['generator_powers']:
            final_renewable_total += decision_vars['generator_powers'][gen_type]
    plt.plot(hours, initial_renewable_total, '--', label='可再生能源总发电量初始', marker='o')
    plt.plot(hours, final_renewable_total, label='可再生能源总发电量最终', marker='s')
    plt.title('可再生能源总发电量初始/最终对比')
    plt.xlabel('时段')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True)
    
    # 传统发电的初始和最终发电量
    plt.subplot(3, 2, 5)
    for i, gen_name in enumerate(['traditional_0', 'traditional_1', 'traditional_2']):
        initial_power = TRADITIONAL_INITIAL_POWER[f'generator{i+1}']
        plt.plot(hours, initial_power, '--', label=f'{gen_name}初始', alpha=0.7)
        if gen_name in decision_vars['generator_powers']:
            plt.plot(hours, decision_vars['generator_powers'][gen_name], label=f'{gen_name}最终')
    plt.title('传统发电初始/最终发电量')
    plt.xlabel('时段')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True)
    
    # 传统发电总发电量对比
    plt.subplot(3, 2, 6)
    initial_traditional_total = np.zeros(24)
    for i in range(3):
        initial_traditional_total += TRADITIONAL_INITIAL_POWER[f'generator{i+1}']
    final_traditional_total = np.zeros(24)
    for gen_name in ['traditional_0', 'traditional_1', 'traditional_2']:
        if gen_name in decision_vars['generator_powers']:
            final_traditional_total += decision_vars['generator_powers'][gen_name]
    plt.plot(hours, initial_traditional_total, '--', label='传统发电总发电量初始', marker='o')
    plt.plot(hours, final_traditional_total, label='传统发电总发电量最终', marker='s')
    plt.title('传统发电总发电量初始/最终对比')
    plt.xlabel('时段')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # --- 用户用电量：每类单独一张图 ---
    for i, (user_class, params) in enumerate(USER_PARAMS.items()):
        plt.figure(figsize=(8, 5))
        plt.plot(hours, USER_INITIAL_DEMAND[:, i], '--', label=f'{user_class}初始', alpha=0.7)
        key = f'{user_class}_0'
        if key in decision_vars['user_demands']:
            plt.plot(hours, decision_vars['user_demands'][key], label=f'{user_class}最终')
        plt.title(f'{user_class}用户初始/最终用电量')
        plt.xlabel('时段')
        plt.ylabel('kWh')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- EV总充放电量对比：每类单独一张图（只考虑前20个） ---
    for ev_class in EV_CLASSES.keys():
        plt.figure(figsize=(8, 5))
        # 初始（只考虑前20个）
        initial_ev_charging = np.zeros(24)
        for i in range(min(ev_count_per_class, EV_CLASSES[ev_class]['count'])):
            vehicle_name = f"{ev_class}_EV_{i}"
            if vehicle_name in initial_ev_charging_dict:
                initial_ev_charging += initial_ev_charging_dict[vehicle_name]
        # 最终（只考虑前20个）
        final_ev_charging = np.zeros(24)
        for i in range(min(ev_count_per_class, EV_CLASSES[ev_class]['count'])):
            key = f"{ev_class}_{i}"
            if key in decision_vars['ev_charging']:
                v = decision_vars['ev_charging'][key]
                if isinstance(v, dict) and 'charging' in v:
                    final_ev_charging += np.array(v['charging'])
                elif isinstance(v, np.ndarray):
                    final_ev_charging += v
        plt.plot(hours, initial_ev_charging, label=f'{ev_class}初始总充放电量', marker='o')
        plt.plot(hours, final_ev_charging, label=f'{ev_class}最终总充放电量', marker='s')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title(f'{ev_class}电动汽车总充放电量对比')
        plt.xlabel('时段')
        plt.ylabel('kWh (正值充电，负值放电)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
if __name__ == '__main__':
    main()   
    