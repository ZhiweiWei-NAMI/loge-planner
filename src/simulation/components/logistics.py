"""
物流处理组件

严格按照设计文档实现的 LogisticsComponent。

@author: TCCN Auto Planner
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../src'))

from typing import Dict, Any, List
from ..core.component import Component
import logging

logger = logging.getLogger(__name__)

class LogisticsComponent(Component):
    """物流处理组件"""
    
    # 严格按照设计文档的 PRODUCED_METRICS
    PRODUCED_METRICS = [
        'pickup_processing_time',   # s - 取件处理时间
        'handover_processing_time', # s - 交接处理时间
        'logistics_base_power'      # W - 货物处理功耗
    ]

    MONITORED_STATES = [
        'attached_items',         # 这会影响货物处理的能力
        'current_payload_weight'  # 载重影响处理效率
    ]
    
    def __init__(self, agent, **kwargs):
        """初始化物流组件"""
        super().__init__(agent, 'Logistics', **kwargs)
        
        # 硬件规格参数
        self.payload_capacity = kwargs.get('payload_capacity', 5.0)  # kg
        self.pickup_speed = kwargs.get('pickup_speed', 2.0)  # 每分钟取件数
        self.handover_speed = kwargs.get('handover_speed', 1.5)  # 每分钟交付数
        self.base_power = kwargs.get('base_power', 20.0)  # W
        
        # 处理能力参数
        self.processing_efficiency = 1.0  # 基础处理效率
    
    def _calculate_performance_metrics(self):
        """计算物流处理性能指标"""
        # 获取监听的状态
        attached_items = self.agent.get_state('attached_items') or []
        payload_weight = self.agent.get_state('current_payload_weight') or 0.0
        
        # 计算取件处理时间
        pickup_processing_time = self._calculate_pickup_processing_time(attached_items, payload_weight)
        
        # 计算交接处理时间
        handover_processing_time = self._calculate_handover_processing_time(attached_items, payload_weight)
        
        # 计算物流基础功耗
        logistics_base_power = self._calculate_logistics_power(attached_items, payload_weight)
        
        return {
            'pickup_processing_time': pickup_processing_time,
            'handover_processing_time': handover_processing_time,
            'logistics_base_power': logistics_base_power
        }
    
    def _calculate_pickup_processing_time(self, attached_items: List, payload_weight: float) -> float:
        """计算取件处理时间"""
        # 基础处理时间（每个物品）
        base_time_per_item = 60.0 / self.pickup_speed  # 秒
        
        # 当前载重影响处理效率
        weight_factor = 1.0 + (payload_weight / self.payload_capacity) * 0.5
        
        # 已有物品数量影响（空间占用）
        item_count = len(attached_items)
        item_factor = 1.0 + item_count * 0.1  # 每个已有物品增加10%处理时间
        
        # 处理效率影响
        efficiency_factor = 1.0 / self.processing_efficiency
        
        processing_time = base_time_per_item * weight_factor * item_factor * efficiency_factor
        
        return max(5.0, processing_time)  # 最少5秒
    
    def _calculate_handover_processing_time(self, attached_items: List, payload_weight: float) -> float:
        """计算交接处理时间"""
        # 基础处理时间（每个物品）
        base_time_per_item = 60.0 / self.handover_speed  # 秒
        
        # 当前载重影响处理效率（交接时需要更仔细）
        weight_factor = 1.0 + (payload_weight / self.payload_capacity) * 0.3
        
        # 物品数量影响
        item_count = len(attached_items)
        item_factor = 1.0 + item_count * 0.15  # 每个物品增加15%处理时间
        
        # 处理效率影响
        efficiency_factor = 1.0 / self.processing_efficiency
        
        processing_time = base_time_per_item * weight_factor * item_factor * efficiency_factor
        
        return max(8.0, processing_time)  # 最少8秒（交接比取件更复杂）
    
    def _calculate_logistics_power(self, attached_items: List, payload_weight: float) -> float:
        """计算物流处理功耗"""
        base_power = self.base_power
        
        # 载重影响功耗（需要更多功率处理重物）
        weight_factor = 1.0 + (payload_weight / self.payload_capacity) * 0.4
        
        # 物品数量影响功耗（处理多个物品需要更多操作）
        item_count = len(attached_items)
        item_factor = 1.0 + item_count * 0.05
        
        # 处理活动影响功耗
        activity_factor = 1.0
        if item_count > 0:
            activity_factor = 1.5  # 有物品时功耗增加50%
        
        return base_power * weight_factor * item_factor * activity_factor
    
    def can_pickup_item(self, item_weight: float) -> bool:
        """检查是否可以取件"""
        current_weight = self.agent.get_state('current_payload_weight') or 0.0
        return (current_weight + item_weight) <= self.payload_capacity
    