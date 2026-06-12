"""
Workflow生成器

支持动态生成workflow，可配置：
- 泊松分布的到达率
- 不同workflow类型的概率分布
- workflow参数的随机生成

@author: TCCN Auto Planner
"""

import random
import logging
import numpy as np
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class WorkflowGenerator:
    """Workflow动态生成器
    
    根据泊松分布动态生成workflow
    """
    
    def __init__(self, env, workflow_configs: List[Dict], arrival_rate: float,
                 region_bounds: Dict, duration: float = None):
        """初始化workflow生成器
        
        Args:
            env: AirFogSim环境对象
            workflow_configs: workflow类型配置列表
            arrival_rate: 泊松分布的lambda参数（每秒平均到达数）
            region_bounds: 区域边界，包含x_min, x_max, y_min, y_max, z_min, z_max
            duration: 生成持续时间（秒），None表示持续生成
        """
        self.env = env
        self.workflow_configs = workflow_configs
        self.arrival_rate = arrival_rate
        self.region_bounds = region_bounds
        self.duration = duration
        
        # 计算workflow类型的累积概率
        self._prepare_workflow_probabilities()
        
        # 生成的workflow计数器
        self.workflow_counter = 0
        self.generated_workflows = []
        
        # 启动生成进程
        self.generation_process = env.process(self._workflow_generation_process())
        
        logger.info(f"Workflow生成器初始化完成，到达率: {arrival_rate}/秒")
    
    def _prepare_workflow_probabilities(self):
        """准备workflow类型的概率分布"""
        self.workflow_types = []
        self.cumulative_probs = []
        
        cumsum = 0.0
        for config in self.workflow_configs:
            self.workflow_types.append(config['type'])
            cumsum += config['probability']
            self.cumulative_probs.append(cumsum)
        
        # 归一化
        total = self.cumulative_probs[-1]
        self.cumulative_probs = [p / total for p in self.cumulative_probs]
        
        logger.debug(f"Workflow类型概率分布: {dict(zip(self.workflow_types, self.cumulative_probs))}")
    
    def _workflow_generation_process(self):
        """Workflow生成进程"""
        try:
            start_time = self.env.now
            
            while True:
                # 根据泊松分布计算下一个到达时间
                inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
                
                # 等待到达
                yield self.env.timeout(inter_arrival_time)
                
                # 检查是否超过持续时间
                if self.duration and (self.env.now - start_time) > self.duration:
                    logger.info(f"Workflow生成器已达到持续时间限制 ({self.duration}秒)，停止生成")
                    break
                
                # 生成workflow
                workflow_info = self._generate_workflow()
                if workflow_info:
                    self.generated_workflows.append(workflow_info)
                    logger.debug(f"时间 {self.env.now:.2f}: 生成workflow {workflow_info['workflow_id']}")
                    
        except Exception as e:
            logger.error(f"Workflow生成进程出错: {e}")
    
    def _generate_workflow(self) -> Dict[str, Any]:
        """生成单个workflow
        
        Returns:
            workflow信息字典
        """
        try:
            # 根据概率分布选择workflow类型
            workflow_type = self._select_workflow_type()
            
            # 生成workflow ID
            self.workflow_counter += 1
            workflow_id = f"{workflow_type}_wf_{self.workflow_counter}"
            
            # 根据类型生成workflow参数
            workflow_info = {
                'workflow_id': workflow_id,
                'workflow_type': workflow_type,
                'arrival_time': self.env.now,
            }
            
            # 使用.lower()和in来匹配，支持'logistics'或'LogisticsWorkflow'
            wf_type_lower = workflow_type.lower()
            if 'logistics' in wf_type_lower:
                workflow_info.update(self._generate_logistics_params())
            elif 'inspection' in wf_type_lower:
                workflow_info.update(self._generate_inspection_params())
            else:
                logger.warning(f"未知的workflow类型: {workflow_type}")
                return None
            
            return workflow_info
            
        except Exception as e:
            logger.error(f"生成workflow时出错: {e}")
            return None
    
    def _select_workflow_type(self) -> str:
        """根据概率分布选择workflow类型
        
        Returns:
            选中的workflow类型
        """
        rand = random.random()
        for i, cum_prob in enumerate(self.cumulative_probs):
            if rand <= cum_prob:
                return self.workflow_types[i]
        return self.workflow_types[-1]
    
    def _generate_logistics_params(self) -> Dict[str, Any]:
        """生成物流workflow参数
        
        Returns:
            物流workflow参数字典
        """
        # 随机生成pickup和delivery位置
        pickup_location = self._random_position()
        delivery_location = self._random_position()
        
        # 随机生成payload重量（1-10kg）
        payload_weight = random.uniform(1.0, 10.0)
        
        return {
            'pickup_location': pickup_location,
            'delivery_location': delivery_location,
            'payload_weight': payload_weight,
            'payload_id': f"payload_{self.workflow_counter}"
        }
    
    def _generate_inspection_params(self) -> Dict[str, Any]:
        """生成巡检workflow参数
        
        Returns:
            巡检workflow参数字典
        """
        # 随机生成巡检点数量（3-8个点）
        num_points = random.randint(3, 8)
        
        # 生成巡检点位置
        inspection_points = [self._random_position() for _ in range(num_points)]
        
        return {
            'inspection_points': inspection_points,
            'num_inspection_points': num_points
        }
    
    def _random_position(self) -> Tuple[float, float, float]:
        """生成随机位置
        
        Returns:
            (x, y, z)位置元组
        """
        x = random.uniform(self.region_bounds['x_min'], self.region_bounds['x_max'])
        y = random.uniform(self.region_bounds['y_min'], self.region_bounds['y_max'])
        z = random.uniform(self.region_bounds['z_min'], self.region_bounds['z_max'])
        return (x, y, z)
    
    def get_generated_workflows(self) -> List[Dict[str, Any]]:
        """获取已生成的workflow列表
        
        Returns:
            workflow信息列表
        """
        return self.generated_workflows.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取生成统计
        
        Returns:
            统计信息字典
        """
        workflow_type_counts = {}
        for wf in self.generated_workflows:
            wf_type = wf['workflow_type']
            if wf_type not in workflow_type_counts:
                workflow_type_counts[wf_type] = 0
            workflow_type_counts[wf_type] += 1
        
        return {
            'total_generated': len(self.generated_workflows),
            'workflow_type_counts': workflow_type_counts,
            'arrival_rate': self.arrival_rate,
            'simulation_time': self.env.now
        }
    
    def print_statistics(self):
        """打印生成统计"""
        stats = self.get_statistics()
        
        logger.info("\n" + "="*60)
        logger.info("Workflow生成统计")
        logger.info("="*60)
        logger.info(f"总生成数量: {stats['total_generated']}")
        logger.info(f"到达率: {stats['arrival_rate']}/秒")
        logger.info(f"仿真时长: {stats['simulation_time']:.2f} 秒")
        logger.info("各类型workflow数量:")
        for wf_type, count in stats['workflow_type_counts'].items():
            logger.info(f"  {wf_type}: {count}")
        logger.info("="*60)
