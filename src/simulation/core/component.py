"""
组件基类

基于 AirFogSim 架构设计文档的 Component 设计。
使用 AirFogSim 的 Component 基类。

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../src'))

from airfogsim.core.component import Component as BaseComponent
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class Component(BaseComponent):
    """
    组件基类

    基于 AirFogSim Component 基类实现，增强了性能指标计算和状态监听能力。
    支持：
    1. 状态变化监听和响应
    2. 性能指标计算和聚合
    3. 功耗管理
    4. 任务执行支持
    """

    # AirFogSim 要求的类属性 - 子类应该重写这个属性
    PRODUCED_METRICS = ['status', 'power_consumption']
    MONITORED_STATES = []

    def __init__(self, agent, component_name: str = None, **kwargs):
        """
        初始化组件

        Args:
            agent: 所属智能体
            component_name: 组件名称
            **kwargs: 其他配置参数
        """
        super().__init__(agent.env, agent, component_name or self.__class__.__name__)

        # 组件配置
        self.config = kwargs
        self.enabled = kwargs.get('enabled', True)

        # 性能指标
        self.metrics = {}
        self.power_consumption = 0.0  # 当前功耗 W
        self.base_power = kwargs.get('base_power', 0.0)  # 基础功耗 W
    
    def update_power_consumption(self, base_power: float = None, activity_power: float = 0.0):
        """
        更新功耗
        
        Args:
            base_power: 基础功耗 W
            activity_power: 活动功耗 W
        """
        if base_power is not None:
            self.base_power = base_power
        
        self.power_consumption = self.base_power + activity_power
        
        # 更新智能体的对应功耗状态
        power_state_key = f'{self.name.lower()}_power_consumption'
        if hasattr(self.agent, 'set_state'):
            self.agent.set_state(power_state_key, self.power_consumption)
    
class EnergyComponent(Component):
    """
    能源管理组件

    负责统一管理所有组件的功耗，计算总功耗和电池消耗。
    """

    # AirFogSim 要求的类属性
    PRODUCED_METRICS = ['energy_consumed_wh', 'power_consumption_w', 'battery_level', 'energy_efficiency']

    def __init__(self, agent, **kwargs):
        """初始化能源组件"""
        super().__init__(agent, 'Energy', **kwargs)
        
        # 能源配置
        self.battery_capacity = kwargs.get('battery_capacity', 5000.0)  # Wh
        self.discharge_efficiency = kwargs.get('discharge_efficiency', 0.95)
        self.charge_efficiency = kwargs.get('charge_efficiency', 0.90)
        
        # 监听所有功耗相关状态
        self.power_states = [
            'mobility_power_consumption',
            'computing_power_consumption', 
            'relay_power_consumption',
            'logistics_power_consumption',
            'inspection_power_consumption',
            'communication_power_consumption'
        ]
        
        self.monitored_states = self.power_states + ['battery_level']
    
    def calculate_metrics(self, dt: float):
        """计算能源指标"""
        # 聚合所有组件功耗
        total_power = 0.0
        for power_state in self.power_states:
            power = self.agent.get_state(power_state) or 0.0
            total_power += power
        
        # 计算电池消耗
        if total_power > 0:
            # 功耗转换为电池消耗 (Wh)
            energy_consumed = (total_power * dt / 3600.0) / self.discharge_efficiency
            
            # 更新电池电量
            current_battery = self.agent.get_state('battery_level') or 100.0
            battery_capacity_wh = self.agent.get_state('battery_capacity') or self.battery_capacity
            
            # 计算电量百分比变化
            battery_percentage_consumed = (energy_consumed / battery_capacity_wh) * 100.0
            new_battery_level = max(0.0, current_battery - battery_percentage_consumed)
            
            self.agent.set_state('battery_level', new_battery_level)
        
        # 更新指标
        self.metrics.update({
            'total_power_consumption': total_power,
            'battery_level': self.agent.get_state('battery_level'),
            'energy_consumed_wh': self.metrics.get('energy_consumed_wh', 0.0) + 
                                 (total_power * dt / 3600.0) if total_power > 0 else 0.0,
            'estimated_flight_time': self._estimate_flight_time(total_power)
        })
        
        # 更新自身功耗（能源管理的基础功耗）
        self.update_power_consumption(activity_power=0.5 if total_power > 0 else 0.1)
    
    def _estimate_flight_time(self, current_power: float) -> float:
        """估算剩余飞行时间（小时）"""
        if current_power <= 0:
            return float('inf')
        
        battery_level = self.agent.get_state('battery_level') or 0.0
        battery_capacity_wh = self.agent.get_state('battery_capacity') or self.battery_capacity
        
        remaining_energy = (battery_level / 100.0) * battery_capacity_wh * self.discharge_efficiency
        return remaining_energy / current_power if current_power > 0 else float('inf')
    
    def on_state_changed(self, state_key: str, old_value: Any, new_value: Any):
        """响应状态变化"""
        if state_key == 'battery_level':
            # 检查电池电量警告
            if new_value < 20.0:
                self.trigger_event('low_battery_warning', {
                    'battery_level': new_value,
                    'estimated_flight_time': self.metrics.get('estimated_flight_time', 0)
                })
            
            if new_value < 5.0:
                self.trigger_event('critical_battery_warning', {
                    'battery_level': new_value,
                    'action_required': 'immediate_landing_or_charging'
                })
    
    def charge_battery(self, charge_power: float, duration: float):
        """
        充电
        
        Args:
            charge_power: 充电功率 W
            duration: 充电时长 s
        """
        energy_added = (charge_power * duration / 3600.0) * self.charge_efficiency
        battery_capacity_wh = self.agent.get_state('battery_capacity') or self.battery_capacity
        
        current_battery = self.agent.get_state('battery_level') or 0.0
        battery_percentage_added = (energy_added / battery_capacity_wh) * 100.0
        new_battery_level = min(100.0, current_battery + battery_percentage_added)
        
        self.agent.set_state('battery_level', new_battery_level)
        
        # 增加充电次数
        charge_cycles = self.agent.get_state('charge_cycles') or 0
        self.agent.set_state('charge_cycles', charge_cycles + 1)
        
        logger.info(f"智能体 {self.agent.id} 充电完成: {current_battery:.1f}% -> {new_battery_level:.1f}%")
