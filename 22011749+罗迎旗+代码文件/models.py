import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, LpStatusOptimal, value, PULP_CBC_CMD
import time
from scipy.optimize import differential_evolution, linprog
import matplotlib.pyplot as plt
from pyswarm import pso  # 需在requirements.txt中添加pyswarm

class MarketParticipant:
    def __init__(self, name):
        self.name = name

class TraditionalGenerator(MarketParticipant):
    def __init__(self, name, max_capacity, min_capacity, cost_params, initial_power=None):
        super().__init__(name)
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.cost_params = cost_params  # [a, b, c] for quadratic cost function
        self.initial_power = initial_power
        
    def calculate_cost(self, power):
        """计算发电成本（二次成本函数）"""
        return self.cost_params[0] * power**2 + self.cost_params[1] * power + self.cost_params[2]
    
    def calculate_revenue(self, power, vpp_buy_price):
        """计算与虚拟电厂交易的24小时总收益
        Args:
            power: 24小时发电量数组（kWh）
            vpp_buy_price: 虚拟电厂24小时购电价格数组（元/kWh）
        Returns:
            float: 24小时总收益（元）
        """
        # 确保输入是24小时数组
        if isinstance(power, (int, float)):
            power = np.full(24, power)
        if isinstance(vpp_buy_price, (int, float)):
            vpp_buy_price = np.full(24, vpp_buy_price)
        
        # 计算24小时总收益
        total_revenue = 0.0
        for h in range(24):
            # 交易收入（元）
            transaction_revenue = power[h] * vpp_buy_price[h]
            # 发电成本（元）
            generation_cost = self.calculate_cost(power[h])
            # 时段收益（传统发电机没有弃电成本）
            hourly_revenue = transaction_revenue - generation_cost
            total_revenue += hourly_revenue
            
        return total_revenue
        
    def generate_power(self, hours=24):
        """生成发电量
        Args:
            hours: 时间范围（小时）
        Returns:
            numpy.ndarray: 24小时发电量数组
        """
        if self.initial_power is not None:
            return self.initial_power
        return np.random.uniform(self.min_capacity, self.max_capacity, hours)
        
    def optimize(self, vpp_buy_price, hours=24):
        """每小时均衡解析解，严格用每小时电价，结果不被覆盖"""
        try:
            if isinstance(vpp_buy_price, (list, tuple, np.ndarray)):
                if len(vpp_buy_price) == 24:
                    price_array = np.array(vpp_buy_price)
                else:
                    price_array = np.full(24, vpp_buy_price[0])
            else:
                price_array = np.full(24, vpp_buy_price)
            optimal_power = np.zeros(hours)
            if hasattr(self, 'cost_params') and len(self.cost_params) == 3:
                a, b, c = self.cost_params
                for h in range(hours):
                    if a == 0:
                        # 线性成本函数情况
                        if price_array[h] > b:
                            P_star = self.max_capacity
                        else:
                            P_star = self.min_capacity
                    else:
                        # 二次成本函数情况
                        P_star = (price_array[h] - b) / (2 * a)
                        # 确保最优解在合理范围内
                        P_star = np.clip(P_star, self.min_capacity, self.max_capacity)
                    
                    # 如果最优解为0但价格高于边际成本，则发电
                    if P_star <= 0 and price_array[h] > b:
                        P_star = self.min_capacity
                    
                    optimal_power[h] = P_star
                return optimal_power
            elif hasattr(self, 'cost_params') and len(self.cost_params) == 2:
                a, b = self.cost_params
                for h in range(hours):
                    if price_array[h] - a > 0:
                        optimal_power[h] = self.actual_max_power[h]
                    else:
                        optimal_power[h] = 0
                return optimal_power
            else:
                # 如果没有成本参数，使用简单的价格比较
                for h in range(hours):
                    if price_array[h] > 0.1:  # 设置一个最小价格阈值
                        optimal_power[h] = self.max_capacity
                    else:
                        optimal_power[h] = self.min_capacity
                return optimal_power
        except Exception as e:
            print(f"传统发电机优化错误: {str(e)}")
            # 返回初始发电量而不是0
            return self.initial_power if self.initial_power is not None else np.full(hours, self.min_capacity)

class RenewableGenerator(MarketParticipant):
    def __init__(self, name, max_capacity, min_capacity, cost_params, initial_power=None):
        super().__init__(name)
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.cost_params = cost_params  # [a, b] for linear cost function (元/kWh)
        self.initial_power = initial_power
        self.actual_max_power = initial_power if initial_power is not None else np.zeros(24)  # 实际最大出力（kWh）
        self.curtailment_cost = 0.1  # 默认弃电成本（元/kWh）
        self.efficiency = 0.95  # 新增，默认0.95
        
    def set_curtailment_cost(self, cost_kwh):
        """设置弃电成本
        Args:
            cost_kwh: 弃电成本（元/kWh）
        """
        self.curtailment_cost = cost_kwh
        
    def set_actual_max_power(self, max_power):
        """设置实际最大出力
        Args:
            max_power: 24小时实际最大出力数组（kWh）
        """
        if isinstance(max_power, np.ndarray) and max_power.shape == (24,):
            self.actual_max_power = max_power
        else:
            print(f"警告: {self.name} 的实际最大出力数据格式不正确")
        
    def calculate_cost(self, power):
        """计算发电成本（线性成本函数）
        Args:
            power: 发电量（kWh）
        Returns:
            float: 发电成本（元）
        """
        # 直接使用kWh的成本参数
        return self.cost_params[0] * power + self.cost_params[1]
    
    def calculate_curtailment_cost(self, power):
        """计算弃电成本
        Args:
            power: 实际发电量（kWh）
        Returns:
            float: 弃电成本（元）
        """
        # 计算弃电量（kWh）
        curtailment = np.maximum(self.actual_max_power - power, 0)
        # 使用元/kWh的弃电成本
        return np.sum(curtailment * self.curtailment_cost)
    
    def calculate_revenue(self, power, vpp_buy_price):
        """计算与虚拟电厂交易的24小时总收益
        Args:
            power: 24小时发电量数组（kWh）
            vpp_buy_price: 虚拟电厂24小时购电价格数组（元/kWh）
        Returns:
            float: 24小时总收益（元）
        """
        # 确保输入是24小时数组
        if isinstance(power, (int, float)):
            power = np.full(24, power)
        if isinstance(vpp_buy_price, (int, float)):
            vpp_buy_price = np.full(24, vpp_buy_price)
        
        # 计算24小时总收益
        total_revenue = 0.0
        for h in range(24):
            # 交易收入（元）
            transaction_revenue = power[h] * vpp_buy_price[h]
            # 发电成本（元）
            generation_cost = self.calculate_cost(power[h])
            # 弃电成本（元）
            curtailment_cost = self.calculate_curtailment_cost(np.array([power[h]]))
            # 时段收益
            hourly_revenue = transaction_revenue - generation_cost - curtailment_cost
            total_revenue += hourly_revenue
            
        return total_revenue
        
    def generate_power(self, hours=24):
        """生成发电量
        Args:
            hours: 时间范围（小时）
        Returns:
            numpy.ndarray: 24小时发电量数组
        """
        if self.initial_power is not None:
            return self.initial_power
        return np.random.uniform(self.min_capacity, self.max_capacity, hours)
        
    def optimize(self, vpp_buy_price, hours=24):
        """全局24小时联合优化发电量，目标为总收益最大化"""
        try:
            if isinstance(vpp_buy_price, (list, tuple, np.ndarray)):
                if len(vpp_buy_price) == 24:
                    price_array = np.array(vpp_buy_price)
                else:
                    price_array = np.full(24, vpp_buy_price[0])
            else:
                price_array = np.full(24, vpp_buy_price)
            c = np.zeros(24)
            for h in range(24):
                c[h] = -(price_array[h] - self.cost_params[0])
            bounds = [(0, self.actual_max_power[h]) for h in range(24)]
            res = linprog(c, bounds=bounds, method='highs')
            if res.success:
                return res.x
            else:
                return np.zeros(24)
        except Exception as e:
            return np.zeros(24)

class ElectricVehicle:
    def __init__(self, class_name, params):
        self.class_name = class_name
        self.name = params['name']
        self.max_capacity = params['max_capacity']  # 最大充电容量
        self.min_capacity = params['min_capacity']  # 最大放电容量
        self.initial_soc = params['initial_soc']    # 初始SOC（比例0~1）
        self.max_soc = 1.0                           # SOC上限（比例0~1，固定为1.0）
        self.min_soc = params['min_soc']            # SOC下限（比例0~1）
        self.count = params['count']                # 车辆数量
        self.charging = None
        self.usage = None
        self.soc = None
        self.efficiency = params.get('efficiency', 0.95)  # 充放电效率
        # max_soc参数现在表示电池容量（kWh），而不是SOC上限
        # 注意：params['max_soc']是电池容量（kWh），不是SOC上限
        self.battery_capacity = params['max_soc']   # 电池总容量（kWh），从max_soc参数获取
        
        # 验证SOC参数范围
        if not (0.0 <= self.initial_soc <= 1.0):
            print(f"警告: {self.name} 初始SOC {self.initial_soc} 不在[0,1]范围内")
        if not (0.0 <= self.min_soc <= 1.0):
            print(f"警告: {self.name} 最小SOC {self.min_soc} 不在[0,1]范围内")
        if self.max_soc != 1.0:
            print(f"警告: {self.name} 最大SOC {self.max_soc} 应该为1.0")
        
    def calculate_revenue(self, vpp_prices):
        """计算24小时总收益
        Args:
            vpp_prices: 虚拟电厂24小时电价列表，包含buy和sell价格
        Returns:
            float: 总收益
        """
        if self.charging is None or self.usage is None:
            return 0.0
            
        total_revenue = 0.0
        for t in range(24):
            # 充电支出（正值）：充电量 * 售电价
            if self.charging[t] > 0:
                total_revenue -= self.charging[t] * vpp_prices[t]['sell']
            # 放电收入（负值）：放电量 * 购电价
            elif self.charging[t] < 0:
                total_revenue += abs(self.charging[t]) * vpp_prices[t]['buy']
            
            # 用电成本：用电量 * 虚拟电厂售电价格
            usage_cost = self.usage[t] * vpp_prices[t]['sell']
            total_revenue -= usage_cost
            
        return total_revenue

    def optimize(self, charging_data, usage_data, vpp_prices=None, optimization_method='dynamic_programming_with_balance'):
        """优化充放电计划
        Args:
            charging_data: 初始充电数据
            usage_data: 固定用电数据
            vpp_prices: VPP电价数据（可选，用于收益优化）
            optimization_method: 优化方法选择
                - 'dynamic_programming_with_balance': 支持SOC平衡的动态规划优化（默认）
                - 'efficient_dynamic_programming': 高效动态规划优化（快速计算）
                - 'constraint_validation': 仅验证约束条件
        Returns:
            dict: 包含优化结果的字典
        """
        try:
            # 数据预处理
            self.charging = self._process_data(charging_data, 'charging')
            self.usage = self._process_data(usage_data, 'usage')

            if self.charging is None or self.usage is None:
                return {'success': False, 'error': '数据预处理失败'}
            
            # 根据优化方法选择相应的优化策略
            if vpp_prices is not None:
                if optimization_method == 'dynamic_programming_with_balance':
                    result = self._optimize_with_soc_balance(vpp_prices)
                elif optimization_method == 'efficient_dynamic_programming':
                    result = self._optimize_efficient_dp(vpp_prices)
                else:
                    result = self._validate_constraints()
            else:
                result = self._validate_constraints()

            return result
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'charging': self.charging if hasattr(self, 'charging') else None,
                'usage': self.usage if hasattr(self, 'usage') else None,
                'soc': self.soc if hasattr(self, 'soc') else None
            }
    
    def _process_data(self, data, data_type):
        """处理输入数据"""
        try:
            if isinstance(data, dict):
                if 'value' in data:
                    data = data['value']
                elif 'data' in data:
                    data = data['data']
                else:
                    print(f"警告: {data_type}数据字典格式不正确")
                    return None
            
            if isinstance(data, (list, tuple)):
                return np.array(data, dtype=float)
            elif isinstance(data, np.ndarray):
                return data.astype(float)
            else:
                print(f"警告: {data_type}数据类型不正确: {type(data)}")
                return None
        except Exception as e:
            print(f"{data_type}数据转换错误: {str(e)}")
            return None
    
    def _calculate_soc_change(self, charging_power, usage_power, efficiency, battery_capacity):
        """计算SOC变化量
        Args:
            charging_power: 充放电功率（正值表示充电，负值表示放电，单位：kW）
            usage_power: 用电功率（单位：kW）
            efficiency: 充放电效率
            battery_capacity: 电池容量（单位：kWh）
        Returns:
            float: SOC变化量
        """
        # 明确区分充电和放电
        if charging_power > 0:
            # 充电：充电能量 = 充电功率 × 效率
            charge_energy = charging_power * efficiency
            discharge_energy = 0
        elif charging_power < 0:
            # 放电：放电能量 = 放电功率 / 效率（负值转正值）
            charge_energy = 0
            discharge_energy = abs(charging_power) / efficiency
        else:
            # 待机：不充不放
            charge_energy = 0
            discharge_energy = 0
        
        # 用电消耗能量 = 用电功率 / 效率
        usage_energy = usage_power / efficiency
        
        # SOC变化 = (充电能量 - 放电能量 - 用电能量) / 电池容量
        soc_change = (charge_energy - discharge_energy - usage_energy) / battery_capacity
        return soc_change
    
    def _validate_constraints(self):
        """验证约束条件 - 初始数据不需要SOC平衡"""
        # 初始化SOC数组
        self.soc = np.zeros(25)  # 24小时 + 初始状态
        self.soc[0] = self.initial_soc
        
        # 计算每个时段的SOC
        for t in range(24):
            # 使用辅助方法计算SOC变化（明确处理充电和放电）
            soc_change = self._calculate_soc_change(
                self.charging[t], 
                self.usage[t], 
                self.efficiency, 
                self.battery_capacity
            )
            self.soc[t + 1] = self.soc[t] + soc_change
            
            # 检查SOC约束
            if not (self.min_soc <= self.soc[t + 1] <= self.max_soc):
                return {
                    'success': False,
                    'error': f'时段 {t} 的SOC超出范围: {self.soc[t + 1]:.3f}',
                    'charging': self.charging,
                    'usage': self.usage,
                    'soc': self.soc
                }
        
        # 初始数据不需要SOC平衡检查，直接返回结果
        return {
            'success': True,
            'charging': self.charging,
            'usage': self.usage,
            'soc': self.soc,
            'initial_soc': self.soc[0],
            'final_soc': self.soc[-1],
            'soc_balance': self.soc[24] - self.soc[0]  # 记录不平衡量但不修正
        }
    
    def _correct_soc_balance(self, vpp_prices=None):
        """修正SOC首尾不平衡，将不平衡量平均分配到各个时间
        Args:
            vpp_prices: VPP电价数据，用于优化修正策略
        Returns:
            dict: 修正结果
        """
        try:
            # 计算SOC不平衡量
            soc_balance = self.soc[24] - self.soc[0]
            
            if abs(soc_balance) < 1e-3:
                return {
                    'success': True,
                    'charging': self.charging,
                    'usage': self.usage,
                    'soc': self.soc,
                    'initial_soc': self.soc[0],
                    'final_soc': self.soc[-1]
                }
            
            # 复制充电和用电数据用于修正
            corrected_charging = self.charging.copy()
            corrected_usage = self.usage.copy()
            
            # 计算需要调整的能量
            # 当最终SOC小于初始SOC时，需要增加充电量来平衡
            # 当最终SOC大于初始SOC时，需要减少充电量来平衡
            energy_adjustment = soc_balance * self.battery_capacity
            
            if soc_balance > 0:
                # 最终SOC大于初始SOC，需要减少充电或增加用电
                # 将不平衡量平均分配到各个时段
                energy_per_hour = energy_adjustment / 24
                
                # 优先调整充电量（减少充电）
                for h in range(24):
                    if corrected_charging[h] > 0:
                        # 减少充电
                        reduction = min(corrected_charging[h], energy_per_hour)
                        corrected_charging[h] -= reduction
                        energy_per_hour -= reduction
                        
                        if energy_per_hour <= 0:
                            break
                
                # 如果充电调整不够，再调整用电量（增加用电）
                if energy_per_hour > 0:
                    usage_increase_per_hour = energy_per_hour / 24
                    for h in range(24):
                        corrected_usage[h] += usage_increase_per_hour
                        
            else:
                # 最终SOC小于初始SOC，需要增加充电或减少用电
                # 将不平衡量平均分配到各个时段
                energy_per_hour = abs(energy_adjustment) / 24
                
                # 优先调整充电量（增加充电）
                for h in range(24):
                    if corrected_charging[h] < self.max_capacity:
                        # 增加充电
                        increase = min(self.max_capacity - corrected_charging[h], energy_per_hour)
                        corrected_charging[h] += increase
                        energy_per_hour -= increase
                        
                        if energy_per_hour <= 0:
                            break
                
                # 如果充电调整不够，再调整用电量（减少用电）
                if energy_per_hour > 0:
                    usage_decrease_per_hour = energy_per_hour / 24
                    for h in range(24):
                        corrected_usage[h] = max(0, corrected_usage[h] - usage_decrease_per_hour)
            
            # 确保数据在合理范围内
            corrected_charging = np.clip(corrected_charging, self.min_capacity, self.max_capacity)
            corrected_usage = np.clip(corrected_usage, 0, self.max_capacity)
            
            # 重新计算SOC轨迹
            corrected_soc = np.zeros(25)
            corrected_soc[0] = self.initial_soc
            
            for t in range(24):
                # 使用辅助方法计算SOC变化（明确处理充电和放电）
                soc_change = self._calculate_soc_change(
                    corrected_charging[t], 
                    corrected_usage[t], 
                    self.efficiency, 
                    self.battery_capacity
                )
                corrected_soc[t + 1] = corrected_soc[t] + soc_change
                
                # 检查SOC约束
                if not (self.min_soc <= corrected_soc[t + 1] <= self.max_soc):
                    return {
                        'success': False,
                        'error': f'修正后时段 {t} 的SOC超出范围: {corrected_soc[t + 1]:.3f}',
                        'charging': corrected_charging,
                        'usage': corrected_usage,
                        'soc': corrected_soc
                    }
            
            # 更新充电、用电数据和SOC
            self.charging = corrected_charging
            self.usage = corrected_usage
            self.soc = corrected_soc
            
            return {
                'success': True,
                'charging': self.charging,
                'usage': self.usage,
                'soc': self.soc,
                'initial_soc': self.soc[0],
                'final_soc': self.soc[-1],
                'soc_corrected': True,
                'soc_balance_before': soc_balance,
                'energy_adjustment': energy_adjustment
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'SOC修正失败: {str(e)}',
                'charging': self.charging,
                'usage': self.usage,
                'soc': self.soc
            }
    

    
    def _optimize_with_soc_balance(self, vpp_prices):
        """基于动态规划的电动汽车充放电优化（支持SOC首尾平衡约束）
        状态：(SOC水平, 时间)
        决策：充电/放电/待机
        目标：最大化24小时总收益，同时满足SOC首尾平衡
        """
        try:
            usage = np.array(self.usage)
            soc0 = self.initial_soc
            min_soc = self.min_soc
            max_soc = self.max_soc  # 应该是1.0
            battery_capacity = self.battery_capacity
            efficiency = self.efficiency
            max_charge = self.max_capacity
            max_discharge = self.max_capacity

            # 验证SOC参数
            if max_soc != 1.0:
                print(f"错误: {self.name} 动态规划中max_soc={max_soc}，应该为1.0")
            if not (0.0 <= min_soc <= 1.0):
                print(f"错误: {self.name} 动态规划中min_soc={min_soc}，应该在[0,1]范围内")
            if not (0.0 <= soc0 <= 1.0):
                print(f"错误: {self.name} 动态规划中初始SOC={soc0}，应该在[0,1]范围内")

            # 动态规划参数设置（增加搜索范围）
            soc_discretization = 100  # 更高精度
            soc_states = np.linspace(min_soc, max_soc, soc_discretization)
            
            # 充放电决策离散化（更细）
            charge_actions = np.linspace(0, max_charge, 25 )  # 24等分
            discharge_actions = np.linspace(0, max_discharge, 25)
            
            # 初始化动态规划表
            # V[t][s] 表示在时刻t，SOC为s时的最大累积收益
            V = np.full((25, soc_discretization), -np.inf)
            # policy[t][s] 记录最优决策
            policy = np.full((24, soc_discretization), None, dtype=object)
            
            # 边界条件：初始SOC的收益为0
            initial_soc_idx = np.argmin(np.abs(soc_states - soc0))
            V[0, initial_soc_idx] = 0.0
            
            # 动态规划前向递推
            for t in range(24):
                for s_idx, current_soc in enumerate(soc_states):
                    if V[t, s_idx] == -np.inf:
                        continue
                    # 尝试所有可能的充放电决策
                    best_value = -np.inf
                    best_action = None
                    # 充电决策
                    for charge in charge_actions:
                        # 计算下一时刻的SOC
                        charge_energy = charge * efficiency
                        usage_energy = usage[t] / efficiency
                        soc_change = (charge_energy - usage_energy) / battery_capacity
                        next_soc = current_soc + soc_change
                        # 检查SOC约束
                        if min_soc <= next_soc <= max_soc:
                            # 计算当前时刻的收益
                            current_revenue = -charge * vpp_prices[t]['sell'] - usage[t] * vpp_prices[t]['sell']
                            # 找到下一时刻SOC对应的状态索引
                            next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
                            # 总收益 = 当前收益 + 未来收益
                            total_value = current_revenue + V[t, s_idx]
                            if total_value > best_value:
                                best_value = total_value
                                best_action = {'type': 'charge', 'power': charge}
                    # 放电决策
                    for discharge in discharge_actions:
                        # 计算下一时刻的SOC
                        usage_energy = usage[t] / efficiency
                        discharge_energy = discharge / efficiency
                        soc_change = (-usage_energy - discharge_energy) / battery_capacity
                        next_soc = current_soc + soc_change
                        # 检查SOC约束
                        if min_soc <= next_soc <= max_soc:
                            # 计算当前时刻的收益
                            current_revenue = discharge * vpp_prices[t]['buy'] - usage[t] * vpp_prices[t]['sell']
                            # 找到下一时刻SOC对应的状态索引
                            next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
                            # 总收益 = 当前收益 + 未来收益
                            total_value = current_revenue + V[t, s_idx]
                            if total_value > best_value:
                                best_value = total_value
                                best_action = {'type': 'discharge', 'power': discharge}
                    # 待机决策（不充不放）
                    usage_energy = usage[t] / efficiency
                    soc_change = -usage_energy / battery_capacity
                    next_soc = current_soc + soc_change
                    if min_soc <= next_soc <= max_soc:
                        current_revenue = -usage[t] * vpp_prices[t]['sell']
                        total_value = current_revenue + V[t, s_idx]
                        if total_value > best_value:
                            best_value = total_value
                            best_action = {'type': 'idle', 'power': 0}
                    # 更新动态规划表
                    if best_action is not None:
                        next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
                        # 更新V表：取最大值（多个路径可能到达同一状态）
                        V[t+1, next_soc_idx] = max(V[t+1, next_soc_idx], best_value)
                        # 更新policy：记录从当前状态出发的最优决策
                        # best_action已经是从当前状态出发的最优决策
                        policy[t, s_idx] = best_action
            
            # 回溯最优路径：从初始状态开始，根据policy前向回溯
            optimal_charging = np.zeros(24)
            optimal_soc = np.zeros(25)
            optimal_soc[0] = soc0
            
            current_soc_idx = initial_soc_idx
            
            for t in range(24):
                # 如果当前状态没有policy，尝试找到最近的可达状态
                if policy[t, current_soc_idx] is None:
                    # 找到所有可达的下一状态
                    reachable_next_indices = np.where(V[t+1, :] != -np.inf)[0]
                    if len(reachable_next_indices) > 0:
                        # 计算从当前SOC出发，执行各种决策后可能到达的状态
                        current_soc = optimal_soc[t]
                        best_action = None
                        best_next_soc_idx = None
                        best_next_value = -np.inf
                        
                        # 尝试所有可能的决策
                        for charge in charge_actions:
                            charge_energy = charge * efficiency
                            usage_energy = usage[t] / efficiency
                            soc_change = (charge_energy - usage_energy) / battery_capacity
                            next_soc = current_soc + soc_change
                            if min_soc <= next_soc <= max_soc:
                                next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
                                if next_soc_idx in reachable_next_indices and V[t+1, next_soc_idx] > best_next_value:
                                    best_next_value = V[t+1, next_soc_idx]
                                    best_next_soc_idx = next_soc_idx
                                    best_action = {'type': 'charge', 'power': charge}
                        
                        for discharge in discharge_actions:
                            usage_energy = usage[t] / efficiency
                            discharge_energy = discharge / efficiency
                            soc_change = (-usage_energy - discharge_energy) / battery_capacity
                            next_soc = current_soc + soc_change
                            if min_soc <= next_soc <= max_soc:
                                next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
                                if next_soc_idx in reachable_next_indices and V[t+1, next_soc_idx] > best_next_value:
                                    best_next_value = V[t+1, next_soc_idx]
                                    best_next_soc_idx = next_soc_idx
                                    best_action = {'type': 'discharge', 'power': discharge}
                        
                        # 待机决策
                        usage_energy = usage[t] / efficiency
                        soc_change = -usage_energy / battery_capacity
                        next_soc = current_soc + soc_change
                        if min_soc <= next_soc <= max_soc:
                            next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
                            if next_soc_idx in reachable_next_indices and V[t+1, next_soc_idx] > best_next_value:
                                best_next_value = V[t+1, next_soc_idx]
                                best_next_soc_idx = next_soc_idx
                                best_action = {'type': 'idle', 'power': 0}
                        
                        if best_action is not None:
                            if best_action['type'] == 'charge':
                                optimal_charging[t] = best_action['power']
                            elif best_action['type'] == 'discharge':
                                optimal_charging[t] = -best_action['power']
                            else:
                                optimal_charging[t] = 0
                            current_soc_idx = best_next_soc_idx
                        else:
                            # 如果仍然找不到，使用原始数据
                            optimal_charging[t] = self.charging[t]
                    else:
                        # 如果没有可达状态，使用原始数据
                        optimal_charging[t] = self.charging[t]
                else:
                    # 正常情况：使用policy中的最优决策
                    action = policy[t, current_soc_idx]
                    if action['type'] == 'charge':
                        optimal_charging[t] = action['power']  # 正值表示充电
                    elif action['type'] == 'discharge':
                        optimal_charging[t] = -action['power']  # 负值表示放电
                    else:  # idle
                        optimal_charging[t] = 0
                
                # 使用辅助方法更新SOC（明确处理充电和放电）
                soc_change = self._calculate_soc_change(
                    optimal_charging[t], 
                    usage[t], 
                    efficiency, 
                    battery_capacity
                )
                optimal_soc[t+1] = optimal_soc[t] + soc_change
                
                # 更新当前SOC索引（使用最近的可达状态）
                next_soc_idx = np.argmin(np.abs(soc_states - optimal_soc[t+1]))
                # 确保找到的状态是可达的
                if V[t+1, next_soc_idx] != -np.inf:
                    current_soc_idx = next_soc_idx
                else:
                    # 如果不可达，找到最近的可达状态
                    reachable_indices = np.where(V[t+1, :] != -np.inf)[0]
                    if len(reachable_indices) > 0:
                        distances = np.abs(soc_states[reachable_indices] - optimal_soc[t+1])
                        current_soc_idx = reachable_indices[np.argmin(distances)]
                    else:
                        # 如果完全没有可达状态，使用最近的状态
                        current_soc_idx = next_soc_idx
            
            # 如果SOC不平衡，进行微调
            soc_balance = optimal_soc[24] - optimal_soc[0]
            if abs(soc_balance) > 0.01:  # 如果不平衡超过1%
                # 计算需要调整的能量
                energy_adjustment = soc_balance * battery_capacity
                
                # 在电价较低的时候调整充电量
                if soc_balance > 0:  # 最终SOC大于初始SOC，需要减少充电
                    # 找到电价最高的时段减少充电
                    price_indices = np.argsort([vpp_prices[t]['sell'] for t in range(24)])[::-1]
                    for idx in price_indices:
                        if optimal_charging[idx] > 0 and energy_adjustment > 0:
                            reduction = min(optimal_charging[idx], energy_adjustment / efficiency)
                            optimal_charging[idx] -= reduction
                            energy_adjustment -= reduction * efficiency
                            if energy_adjustment <= 0:
                                break
                else:  # 最终SOC小于初始SOC，需要增加充电
                    # 找到电价最低的时段增加充电
                    price_indices = np.argsort([vpp_prices[t]['sell'] for t in range(24)])
                    for idx in price_indices:
                        if optimal_charging[idx] < max_charge and energy_adjustment < 0:
                            increase = min(max_charge - optimal_charging[idx], abs(energy_adjustment) / efficiency)
                            optimal_charging[idx] += increase
                            energy_adjustment += increase * efficiency
                            if energy_adjustment >= 0:
                                break
                
                # 重新计算SOC轨迹
                optimal_soc[0] = soc0
                for t in range(24):
                    # 使用辅助方法计算SOC变化（明确处理充电和放电）
                    soc_change = self._calculate_soc_change(
                        optimal_charging[t], 
                        usage[t], 
                        efficiency, 
                        battery_capacity
                    )
                    optimal_soc[t+1] = optimal_soc[t] + soc_change

            # 更新电动汽车状态
            self.charging = optimal_charging
            self.soc = optimal_soc

            # 计算最终收益
            revenue = self.calculate_revenue(vpp_prices)

            return {
                'success': True,
                'charging': self.charging,
                'usage': self.usage,
                'soc': self.soc,
                'revenue': revenue,
                'initial_soc': self.soc[0],
                'final_soc': self.soc[-1],
                'soc_balance': self.soc[-1] - self.soc[0],
                'optimization_method': 'dynamic_programming_with_balance'
            }
            
        except Exception as e:
            print(f"SOC平衡动态规划优化失败: {str(e)}")
            return self._validate_constraints()  # 回退到约束验证
    
    def _optimize_efficient_dp(self, vpp_prices):
        """高效动态规划优化方法（使用状态压缩和启发式剪枝）
        适用于大规模优化问题，计算效率更高
        """
        try:
            usage = np.array(self.usage)
            soc0 = self.initial_soc
            min_soc = self.min_soc
            max_soc = self.max_soc
            battery_capacity = self.battery_capacity
            efficiency = self.efficiency
            max_charge = self.max_capacity
            max_discharge = self.max_capacity
            
            # 使用更粗的离散化以提高效率
            soc_discretization = 30  # 减少状态数量
            soc_states = np.linspace(min_soc, max_soc, soc_discretization)
            
            # 减少决策空间
            charge_actions = np.linspace(0, max_charge, 5)  # 5个离散点
            discharge_actions = np.linspace(0, max_discharge, 5)
            
            # 初始化
            V = np.full((25, soc_discretization), -np.inf)
            policy = np.full((24, soc_discretization), None, dtype=object)
            
            # 边界条件
            initial_soc_idx = np.argmin(np.abs(soc_states - soc0))
            V[0, initial_soc_idx] = 0.0
            
            # 前向递推（添加启发式剪枝）
            for t in range(24):
                # 只考虑当前可达的状态
                reachable_states = np.where(V[t, :] > -np.inf)[0]
                
                for s_idx in reachable_states:
                    current_soc = soc_states[s_idx]
                    current_value = V[t, s_idx]
                    
                    best_value = -np.inf
                    best_action = None
                    
                    # 充电决策（添加价格启发式）
                    if t < 23:  # 不是最后一个时段
                        # 优先考虑电价较低的时段充电
                        if vpp_prices[t]['sell'] < np.mean([vpp_prices[h]['sell'] for h in range(24)]):
                            charge_priority = charge_actions
                        else:
                            charge_priority = charge_actions[::-1]  # 电价高时减少充电
                    else:
                        charge_priority = charge_actions
                    
                    for charge in charge_priority:
                        charge_energy = charge * efficiency
                        usage_energy = usage[t] / efficiency
                        soc_change = (charge_energy - usage_energy) / battery_capacity
                        next_soc = current_soc + soc_change
                        
                        if min_soc <= next_soc <= max_soc:
                            current_revenue = -charge * vpp_prices[t]['sell'] - usage[t] * vpp_prices[t]['sell']
                            next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
                            total_value = current_revenue + current_value
                            
                            if total_value > best_value:
                                best_value = total_value
                                best_action = {'type': 'charge', 'power': charge}
                    
                    # 放电决策（添加价格启发式）
                    if t < 23:  # 不是最后一个时段
                        # 优先考虑电价较高的时段放电
                        if vpp_prices[t]['buy'] > np.mean([vpp_prices[h]['buy'] for h in range(24)]):
                            discharge_priority = discharge_actions
                        else:
                            discharge_priority = discharge_actions[::-1]  # 电价低时减少放电
                    else:
                        discharge_priority = discharge_actions
                    
                    for discharge in discharge_priority:
                        usage_energy = usage[t] / efficiency
                        discharge_energy = discharge / efficiency
                        soc_change = (-usage_energy - discharge_energy) / battery_capacity
                        next_soc = current_soc + soc_change
                        
                        if min_soc <= next_soc <= max_soc:
                            current_revenue = discharge * vpp_prices[t]['buy'] - usage[t] * vpp_prices[t]['sell']
                            next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
                            total_value = current_revenue + current_value
                            
                            if total_value > best_value:
                                best_value = total_value
                                best_action = {'type': 'discharge', 'power': discharge}
                    
                    # 待机决策
                    usage_energy = usage[t] / efficiency
                    soc_change = -usage_energy / battery_capacity
                    next_soc = current_soc + soc_change
                    
                    if min_soc <= next_soc <= max_soc:
                        current_revenue = -usage[t] * vpp_prices[t]['sell']
                        total_value = current_revenue + current_value
                        
                        if total_value > best_value:
                            best_value = total_value
                            best_action = {'type': 'idle', 'power': 0}
                    
                    # 更新状态
                    if best_action is not None:
                        next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
                        V[t+1, next_soc_idx] = max(V[t+1, next_soc_idx], best_value)
                        policy[t, s_idx] = best_action
            
            # 回溯最优路径
            optimal_charging = np.zeros(24)
            optimal_soc = np.zeros(25)
            optimal_soc[0] = soc0
            
            current_soc_idx = initial_soc_idx
            
            for t in range(24):
                if policy[t, current_soc_idx] is None:
                    optimal_charging[t] = self.charging[t]
                else:
                    action = policy[t, current_soc_idx]
                    if action['type'] == 'charge':
                        optimal_charging[t] = action['power']  # 正值表示充电
                    elif action['type'] == 'discharge':
                        optimal_charging[t] = -action['power']  # 负值表示放电
                    else:
                        optimal_charging[t] = 0
                
                # 使用辅助方法更新SOC（明确处理充电和放电）
                soc_change = self._calculate_soc_change(
                    optimal_charging[t], 
                    usage[t], 
                    efficiency, 
                    battery_capacity
                )
                optimal_soc[t+1] = optimal_soc[t] + soc_change
                
                current_soc_idx = np.argmin(np.abs(soc_states - optimal_soc[t+1]))
            
            # 更新状态
            self.charging = optimal_charging
            self.soc = optimal_soc
            
            revenue = self.calculate_revenue(vpp_prices)
            
            return {
                'success': True,
                'charging': self.charging,
                'usage': self.usage,
                'soc': self.soc,
                'revenue': revenue,
                'initial_soc': self.soc[0],
                'final_soc': self.soc[-1],
                'soc_balance': self.soc[-1] - self.soc[0],
                'optimization_method': 'efficient_dynamic_programming'
            }
            
        except Exception as e:
            print(f"高效动态规划优化失败: {str(e)}")
            return self._validate_constraints()

class User(MarketParticipant):
    def __init__(self, name, max_demand, min_demand, utility_params, initial_demand=None):
        super().__init__(name)
        self.max_demand = max_demand
        self.min_demand = min_demand
        self.utility_params = utility_params  # 对数效用函数参数
        self.initial_demand = initial_demand
        self.class_name = name.split('_')[0]  # 从name中提取类别名称
        self.last_optimization_time = None  # 添加时间戳记录
        
    def generate_demand(self, hours=24):
        """生成初始用电需求"""
        if self.initial_demand is not None:
            return self.initial_demand
        return np.random.uniform(self.min_demand, self.max_demand, hours)
    
    def calculate_utility(self, consumption):
        """计算用电效用（改进的对数效用函数）
        U = 1/a * ln(a*C + a) + b
        其中 utility_params = [a, b]
        """
        a, b = self.utility_params
        return (1/a) * np.log(a * consumption + a) + b
    
    def calculate_revenue(self, demand, vpp_sell_price):
        """计算用户24小时总收益
        Args:
            demand: 24小时用电量数组
            vpp_sell_price: 虚拟电厂24小时售电价格数组
        Returns:
            float: 24小时总收益（效用收益 - 用电成本）
        """
        # 确保输入是24小时数组
        if isinstance(demand, (int, float)):
            demand = np.full(24, demand)
        if isinstance(vpp_sell_price, (int, float)):
            vpp_sell_price = np.full(24, vpp_sell_price)
        
        # 计算24小时总收益
        total_revenue = 0.0
        for h in range(24):
            # 效用收益
            utility_revenue = self.calculate_utility(demand[h])
            # 用电成本
            usage_cost = demand[h] * vpp_sell_price[h]
            # 时段收益
            hourly_revenue = utility_revenue - usage_cost
            total_revenue += hourly_revenue
            
        return total_revenue
        
    def optimize(self, vpp_sell_price, hours=24):
        """每小时均衡解析解，严格用每小时电价，结果不被覆盖"""
        try:
            if isinstance(vpp_sell_price, (list, tuple, np.ndarray)):
                if len(vpp_sell_price) == 24:
                    price_array = np.array(vpp_sell_price)
                else:
                    price_array = np.full(24, vpp_sell_price[0])
            else:
                price_array = np.full(24, vpp_sell_price)
            optimal_demand = np.zeros(hours)
            a, b = self.utility_params
            for h in range(hours):
                theoretical_optimal = 1 / (a * price_array[h]) - 1
                optimal_demand[h] = np.clip(theoretical_optimal, self.min_demand, self.max_demand)
            return optimal_demand
        except Exception as e:
            return np.full(hours, self.min_demand)

class DEOptimizer:
    def __init__(self, n_population=100, max_iter=200, F=0.5, CR=0.7):
        """
        差分进化优化器
        Args:
            n_population: 种群大小
            max_iter: 最大迭代次数
            F: 缩放因子
            CR: 交叉概率
        """
        self.n_population = n_population
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        
    def optimize(self, objective_func, bounds, n_vars):
        """
        执行差分进化优化
        Args:
            objective_func: 目标函数
            bounds: 变量边界 [(min1, max1), (min2, max2), ...]
            n_vars: 变量数量
        Returns:
            tuple: (最优解, 最优值)
        """
        # 初始化种群
        population = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.n_population, n_vars)
        )
        
        # 计算初始适应度
        fitness = np.array([objective_func(p) for p in population])
        best_idx = np.argmax(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # 迭代优化
        for _ in range(self.max_iter):
            for i in range(self.n_population):
                # 选择三个不同的个体
                idxs = [idx for idx in range(self.n_population) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # 变异
                mutant = a + self.F * (b - c)
                
                # 边界处理
                for j in range(n_vars):
                    mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
                
                # 交叉
                cross_points = np.random.rand(n_vars) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, n_vars)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # 选择
                try:
                    f = objective_func(trial)
                    if f > fitness[i]:
                        fitness[i] = f
                        population[i] = trial
                        if f > best_fitness:
                            best_fitness = f
                            best_solution = trial.copy()
                except Exception as e:
                    continue
        
        return best_solution, best_fitness

class VirtualPowerPlant:
    def __init__(self, name):
        self.name = name
        self.traditional_generators = []  # 传统发电机
        self.renewable_generators = {'solar': None, 'wind': None}  # 可再生能源发电机
        self.users = {}  # 按类别存储用户
        self.evs = {}  # 按类别存储电动汽车
        self.grid = None  # 主网对象
        
        # 交易价格
        self.generator_prices = {}  # 发电侧交易价格（虚拟电厂购电价格）
        self.user_prices = {}      # 用电侧交易价格（虚拟电厂售电价格）
        self.market_prices = None  # 市场电价
        
        # 迭代参数
        self.max_iterations = 100  # 最大迭代次数
        self.price_tolerance = 0.001  # 电价波动容忍度（0.1%）
        
        # 差分进化优化器
        self.optimizer = DEOptimizer(
            n_population=100,  # 减小种群大小以提高计算效率
            max_iter=100,      # 减小迭代次数以提高计算效率
            F=0.7,            # 缩放因子
            CR=0.9            # 交叉概率
        )
        
        # 设置随机种子
        np.random.seed(int(time.time()))
        
    def set_max_iterations(self, max_iterations):
        """设置最大迭代次数
        Args:
            max_iterations: 最大迭代次数
        """
        if max_iterations <= 0:
            raise ValueError("最大迭代次数必须大于0")
        self.max_iterations = max_iterations
        self.optimizer.max_iter = max_iterations
        
    def set_price_tolerance(self, tolerance):
        """设置电价波动容忍度
        Args:
            tolerance: 电价波动容忍度（例如：0.01表示1%）
        """
        if tolerance <= 0:
            raise ValueError("电价波动容忍度必须大于0")
        self.price_tolerance = tolerance
        
    def set_grid(self, grid):
        """设置主网对象"""
        self.grid = grid
        
    def set_market_prices(self, prices):
        """设置市场电价作为初始电价"""
        self.market_prices = prices
        
    def set_generator_prices(self, prices):
        """设置发电侧交易价格（虚拟电厂购电价格，在市场电价±40%范围内）"""
        for name, price in prices.items():
            if self.market_prices is not None:
                market_price = self.market_prices[name]
                min_price = market_price * 0.6  # 下限：市场电价-40%
                max_price = market_price * 1.4  # 上限：市场电价+40%
                self.generator_prices[name] = max(min_price, min(max_price, price))
        
    def set_user_prices(self, prices):
        """设置用电侧交易价格（虚拟电厂售电价格，在市场电价±40%范围内）"""
        for name, price in prices.items():
            if self.market_prices is not None:
                market_price = self.market_prices[name]
                min_price = market_price * 0.6  # 下限：市场电价-40%
                max_price = market_price * 1.4  # 上限：市场电价+40%
                self.user_prices[name] = max(min_price, min(max_price, price))
        
    def add_traditional_generator(self, generator):
        """添加传统发电机"""
        self.traditional_generators.append(generator)
        
    def add_renewable_generator(self, type_name, generator):
        """添加可再生能源发电机"""
        self.renewable_generators[type_name] = generator
        
    def add_user(self, user_class, user):
        """添加用户"""
        if user_class not in self.users:
            self.users[user_class] = []
        self.users[user_class].append(user)
        
    def add_ev(self, ev_class, ev):
        """添加电动汽车"""
        if ev_class not in self.evs:
            self.evs[ev_class] = []
        self.evs[ev_class].append(ev)
        
    def calculate_grid_power(self, hours=24):
        """计算虚拟电厂的购电量和售电量"""
        total_generation = np.zeros(hours)  # 初始化为24小时的零数组
        total_consumption = np.zeros(hours)  # 初始化为24小时的零数组
        ev_charging = np.zeros(hours)  # 初始化为24小时的零数组
        
        # 计算总发电量
        for g in self.traditional_generators:
            total_generation += g.generate_power(hours)
        for g in self.renewable_generators.values():
            if g is not None:
                total_generation += g.generate_power(hours)
        
        # 计算总用电量
        for user_list in self.users.values():
            for user in user_list:
                total_consumption += user.generate_demand(hours)
        
        # 计算电动汽车充放电量
        for ev_list in self.evs.values():
            for ev in ev_list:
                ev_charging += ev.charging
                total_consumption += ev.usage  # 电动汽车用电量为固定值
        
        # 计算购电量和售电量
        grid_buy = np.maximum(0, total_consumption + ev_charging - total_generation)
        grid_sell = np.maximum(0, total_generation - total_consumption - ev_charging)
        
        return grid_buy, grid_sell
        
    def calculate_vpp_prices(self, grid_prices, total_generation, total_consumption):
        """
        计算虚拟电厂内部电价
        Args:
            grid_prices: 主网电价列表，每个元素为包含'buy'和'sell'的字典
            total_generation: 总发电量
            total_consumption: 总用电量
        Returns:
            vpp_prices: 虚拟电厂内部电价列表
        """
        vpp_prices = []
        for h in range(24):
            # 直接使用主网电价作为VPP初始电价
            current_buy_price = float(grid_prices[h]['buy'])
            current_sell_price = float(grid_prices[h]['sell'])
            # 保证售电价大于购电价
            if current_sell_price <= current_buy_price:
                current_sell_price = current_buy_price + 0.01  # 保证大于购电价
            vpp_prices.append({
                'buy': current_buy_price,
                'sell': current_sell_price
            })
        return vpp_prices
        
    def calculate_vpp_objective(self, final_generation, final_consumption, grid_prices, 
                               adjustment_instruction=None, penalty_weight=None):
        """
        计算VPPO目标函数
        Args:
            final_generation: 最终发电量（24小时数组）
            final_consumption: 最终用电量（24小时数组）
            grid_prices: 主网电价（24小时列表，每个元素包含'buy'和'sell'）
            adjustment_instruction: 调节指令（24小时数组，可选）
            penalty_weight: 惩罚函数权重
        Returns:
            dict: 包含各项收益和目标函数值的字典
        """
        total_objective = 0.0
        grid_trading_revenue = 0.0  # 与主网交易收益
        internal_trading_revenue = 0.0  # 与用电侧和发电侧交易收益净值
        adjustment_penalty = 0.0  # 调节指令偏差惩罚
        
        # 使用默认惩罚权重（如果未设置）
        if penalty_weight is None:
            penalty_weight = getattr(self, 'penalty_weight', 1.0)
        
        # 使用设置的调节指令（如果未提供参数）
        if adjustment_instruction is None:
            adjustment_instruction = getattr(self, 'adjustment_instruction', None)
        
        for h in range(24):
            # 1. 计算与主网交易收益
            grid_exchange = final_generation[h] - final_consumption[h]
            if grid_exchange >= 0:  # 向主网售电
                grid_trading_revenue += grid_exchange * grid_prices[h]['sell'] * 0.9  # 考虑网损
            else:  # 从主网购电
                grid_trading_revenue += grid_exchange * grid_prices[h]['buy'] * 1.1  # 考虑网损
            
            # 2. 计算与用电侧和发电侧交易收益净值
            # VPP向发电侧购电成本
            generation_cost = final_generation[h] * grid_prices[h]['buy']
            
            # VPP向用电侧售电收益
            consumption_revenue = final_consumption[h] * grid_prices[h]['sell']
            
            # 内部交易净值 = 售电收益 - 购电成本
            internal_trading_revenue += consumption_revenue - generation_cost
            
            # 3. 计算调节指令偏差惩罚（如果提供了调节指令）
            if adjustment_instruction is not None:
                deviation = abs(grid_exchange - adjustment_instruction[h])
                adjustment_penalty += penalty_weight * deviation * deviation  # 二次惩罚函数
        
        # 计算总目标函数值
        total_objective = grid_trading_revenue + internal_trading_revenue - adjustment_penalty
        
        return {
            'total_objective': total_objective,
            'grid_trading_revenue': grid_trading_revenue,
            'internal_trading_revenue': internal_trading_revenue,
            'adjustment_penalty': adjustment_penalty,
            'grid_exchange': final_generation - final_consumption
        }
        
    def optimize(self, grid_prices):
        """
        优化虚拟电厂运行
        Args:
            grid_prices: 主网电价
        Returns:
            dict: 优化结果
        """
        try:
            # 初始化结果
            result = {
                'grid_interaction': np.zeros(24),
                'revenue': 0.0,
                'vpp_prices': [],
                'initial_prices': [],
                'final_prices': [],
                'generator_powers': {},
                'user_demands': {},
                'ev_charging': {}
            }
            
            # 检查电价数据格式
            if isinstance(grid_prices, (list, tuple)):
                if all(isinstance(p, dict) for p in grid_prices):
                    pass
                else:
                    grid_prices = [{'buy': float(p), 'sell': float(p)} for p in grid_prices]
            elif isinstance(grid_prices, dict):
                if 'buy' in grid_prices and 'sell' in grid_prices:
                    grid_prices = [{'buy': float(grid_prices['buy'][h]), 
                                  'sell': float(grid_prices['sell'][h])} 
                                 for h in range(24)]
                else:
                    return None
            else:
                return None
            
            # 初始化迭代变量
            iteration = 0
            price_converged = False
            current_prices = None
            prev_prices = None
            
            # 预计算初始发电量和用电量
            total_generation = np.zeros(24)
            total_consumption = np.zeros(24)
            
            # 记录初始发电量
            for i, gen in enumerate(self.traditional_generators):
                gen_power = gen.generate_power()
                total_generation += gen_power
                result['generator_powers'][f'traditional_{i}'] = gen_power.copy()
            
            for gen_type, gen in self.renewable_generators.items():
                if gen is not None:
                    gen_power = gen.generate_power()
                    total_generation += gen_power
                    result['generator_powers'][gen_type] = gen_power.copy()
            
            # 记录初始用电量
            for user_class, user_list in self.users.items():
                for i, user in enumerate(user_list):
                    user_demand = user.generate_demand()
                    total_consumption += user_demand
                    result['user_demands'][f'{user_class}_{i}'] = user_demand.copy()
            
            # 记录初始电动汽车充放电
            for ev_class, ev_list in self.evs.items():
                for i, ev in enumerate(ev_list):
                    if hasattr(ev, 'usage') and isinstance(ev.usage, np.ndarray):
                        total_consumption += ev.usage
                        result['ev_charging'][f'{ev_class}_{i}'] = {
                            'charging': ev.charging.copy() if ev.charging is not None else np.zeros(24),
                            'usage': ev.usage.copy()
                        }
            
            # 计算初始价格
            current_prices = self.calculate_vpp_prices(grid_prices, total_generation, total_consumption)
            result['initial_prices'] = current_prices.copy()
            
            # 开始迭代优化
            while not price_converged and iteration < self.max_iterations:
                iteration += 1
                # 保存上一次的价格
                prev_prices = current_prices.copy()
                
                # 更新发电量和用电量
                final_generation = np.zeros(24)
                final_consumption = np.zeros(24)
                
                # 1. 优化传统发电机
                for i, gen in enumerate(self.traditional_generators):
                    gen_power = np.zeros(24)
                    for h in range(24):
                        power = gen.optimize([grid_prices[h]['buy']], 1)[0]
                        gen_power[h] = power
                    final_generation += gen_power
                    result['generator_powers'][f'traditional_{i}'] = gen_power.copy()
                
                # 2. 优化可再生能源发电机
                for gen_type, gen in self.renewable_generators.items():
                    if gen is not None:
                        gen_power = np.zeros(24)
                        for h in range(24):
                            power = gen.optimize([grid_prices[h]['buy']], 1)[0]
                            gen_power[h] = power
                            final_generation += gen_power
                            result['generator_powers'][gen_type] = gen_power.copy()
                
                # 3. 优化用户需求
                for user_class, user_list in self.users.items():
                    for i, user in enumerate(user_list):
                        demand = np.zeros(24)
                        for h in range(24):
                            demand[h] = user.optimize([grid_prices[h]['sell']], 1)[0]
                        final_consumption += demand
                        result['user_demands'][f'{user_class}_{i}'] = demand.copy()
            
                # 4. 更新价格
                current_prices = self.calculate_vpp_prices(grid_prices, final_generation, final_consumption)
                
                # 5. 检查价格收敛性
                price_diff = 0.0
                for h in range(24):
                    price_diff += abs(current_prices[h]['buy'] - prev_prices[h]['buy'])
                    price_diff += abs(current_prices[h]['sell'] - prev_prices[h]['sell'])
                price_diff /= (24 * 2)  # 平均价格差异
                
                # 检查是否收敛
                if price_diff < self.price_tolerance:
                    price_converged = True
                elif iteration >= self.max_iterations:
                    break
            
            # 计算VPPO目标函数
            objective_result = self.calculate_vpp_objective(
                final_generation, final_consumption, grid_prices,
                adjustment_instruction=getattr(self, 'adjustment_instruction', None),
                penalty_weight=getattr(self, 'penalty_weight', 1.0)
            )
            
            # 更新结果
            result['grid_interaction'] = objective_result['grid_exchange']
            result['revenue'] = objective_result['total_objective']
            result['grid_trading_revenue'] = objective_result['grid_trading_revenue']
            result['internal_trading_revenue'] = objective_result['internal_trading_revenue']
            result['adjustment_penalty'] = objective_result['adjustment_penalty']
            result['final_prices'] = current_prices
            result['vpp_prices'] = current_prices
            
            return result
                
        except Exception as e:
            return None

    def set_adjustment_instruction(self, instruction):
        """设置调节指令
        Args:
            instruction: 24小时调节指令数组
        """
        if isinstance(instruction, np.ndarray) and instruction.shape == (24,):
            self.adjustment_instruction = instruction
        else:
            self.adjustment_instruction = None

    def set_penalty_weight(self, weight):
        """设置惩罚函数权重
        Args:
            weight: 惩罚权重（正数）
        """
        if weight > 0:
            self.penalty_weight = weight
        else:
            print("警告: 惩罚权重必须为正数")

    def global_price_optimize(self, market_prices, price_bounds=None, verbose=False):
        """
        基于粒子群优化（PSO）全局优化VPP价格，最大化VPP收益。
        Args:
            market_prices: 市场电价（dict，含buy/sell，24小时）
            price_bounds: [(min_buy, max_buy, min_sell, max_sell) for each hour]，如None则±40%
            verbose: 是否打印调试信息
        Returns:
            dict: {'best_prices': vpp_prices, 'best_revenue': max_revenue, 'decision_vars': {...}}
        """
        if price_bounds is None:
            price_bounds = []
            for h in range(24):
                min_buy = market_prices['buy'][h] * 0.6
                max_buy = market_prices['buy'][h] * 1.4
                min_sell = max(market_prices['sell'][h], min_buy + 0.01)
                max_sell = max(market_prices['sell'][h] * 1.4, min_sell + 0.01)
                price_bounds.append((min_buy, max_buy, min_sell, max_sell))
        # 拼接为粒子向量上下界
        lb = []
        ub = []
        for b in price_bounds:
            lb.extend([b[0], b[2]])  # buy, sell
            ub.extend([b[1], b[3]])
        lb = np.array(lb)
        ub = np.array(ub)

        best_decision_vars = None
        best_prices = None
        best_revenue = -np.inf

        def pso_objective(price_vector):
            # price_vector: [buy0, sell0, buy1, sell1, ..., buy23, sell23]
            vpp_prices = []
            for h in range(24):
                buy = price_vector[2*h]
                sell = price_vector[2*h+1]
                # 保证sell>buy
                if sell <= buy:
                    sell = buy + 0.01
                vpp_prices.append({'buy': buy, 'sell': sell})
            # 用该价格分别优化所有用户/发电机/EV
            generation, consumption, decision_vars = self.inner_optimize_for_prices(vpp_prices)
            vpp_obj = self.calculate_vpp_objective(generation, consumption, vpp_prices)
            revenue = vpp_obj['total_objective']
            nonlocal best_revenue, best_prices, best_decision_vars
            if revenue > best_revenue:
                best_revenue = revenue
                best_prices = vpp_prices.copy()
                best_decision_vars = decision_vars.copy()
            if verbose:
                print(f"PSO粒子价格: 收益={revenue:.2f}")
            return -revenue  # PSO最小化

        # PSO主循环
        xopt, fopt = pso(pso_objective, lb, ub, swarmsize=24, maxiter=100, debug=verbose)
        return {'best_prices': best_prices, 'best_revenue': best_revenue, 'decision_vars': best_decision_vars}

    def inner_optimize_for_prices(self, vpp_prices):
        """
        用给定VPP价格分别整体优化所有用户/发电机/EV，返回发电量、用电量、所有决策变量。
        Returns:
            generation: 24小时总发电量
            consumption: 24小时总用电量
            decision_vars: dict，包含各类决策变量
        """
        generation = np.zeros(24)
        consumption = np.zeros(24)
        decision_vars = {'generator_powers': {}, 'user_demands': {}, 'ev_charging': {}}
        # 传统发电机
        for i, gen in enumerate(self.traditional_generators):
            power = gen.optimize([p['buy'] for p in vpp_prices], 24)
            generation += power
            decision_vars['generator_powers'][f'traditional_{i}'] = power.copy()
        # 可再生能源
        for gen_type, gen in self.renewable_generators.items():
            if gen is not None:
                power = gen.optimize([p['buy'] for p in vpp_prices], 24)
                generation += power
                decision_vars['generator_powers'][gen_type] = power.copy()
        # 用户
        for user_class, user_list in self.users.items():
            for i, user in enumerate(user_list):
                demand = user.optimize([p['sell'] for p in vpp_prices], 24)
                consumption += demand
                decision_vars['user_demands'][f'{user_class}_{i}'] = demand.copy()
        # 电动汽车 - 动态规划优化（支持SOC首尾平衡约束，增加搜索范围）
        for ev_class, ev_list in self.evs.items():
            for i, ev in enumerate(ev_list):
                ev_result = ev.optimize(ev.charging, ev.usage, vpp_prices, optimization_method='dynamic_programming_with_balance')
                if isinstance(ev_result, dict) and ev_result.get('success', False):
                    charging = np.array(ev_result['charging'])
                    usage = np.array(ev_result['usage'])
                    consumption += usage
                    decision_vars['ev_charging'][f'{ev_class}_{i}'] = {
                        'charging': charging, 
                        'usage': usage,
                        'soc': ev_result.get('soc', None),
                        'initial_soc': ev_result.get('initial_soc', None),
                        'final_soc': ev_result.get('final_soc', None),
                        'soc_corrected': ev_result.get('soc_corrected', False),
                        'energy_adjustment': ev_result.get('energy_adjustment', 0)
                    }
        

        
        return generation, consumption, decision_vars

class Grid:
    def __init__(self, name, base_prices):
        self.name = name
        self.base_prices = base_prices  # 市场电价，包含'sell'和'buy'的字典
        
    def get_prices(self, hours=24):
        """获取市场电价
        Args:
            hours: 时间范围（小时）
        Returns:
            list: 24小时电价列表，每个元素包含'buy'和'sell'价格
        """
        prices = []
        for h in range(hours):
            # 直接返回基础电价，不进行动态调整
            prices.append({
                'buy': self.base_prices['buy'][h],
                'sell': self.base_prices['sell'][h]
            })
        return prices 