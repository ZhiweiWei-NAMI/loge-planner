"""配置加载器

提供配置文件加载和验证功能。

@author: TCCN Auto Planner
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../..')
sys.path.append(project_root)

import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，默认为项目config目录
        """
        if config_dir is None:
            # 默认使用项目的config目录
            self.config_dir = Path(__file__).parent.parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)
        
    def load_logistics_config(self, config_file: str = "logistics_config.yaml") -> Dict[str, Any]:
        """加载物流配置文件
        
        Args:
            config_file: 配置文件名
            
        Returns:
            配置字典
        """
        return self.load_config(config_file)
    
    def load_inspection_config(self, config_file: str = "inspection_config.yaml") -> Dict[str, Any]:
        """加载巡检配置文件
        
        Args:
            config_file: 配置文件名
            
        Returns:
            配置字典
        """
        return self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """通用配置文件加载方法

        Args:
            config_file: 配置文件名

        Returns:
            配置字典
        """
        config_path = self.config_dir / config_file

        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 检查是否引用了公共配置
            if 'common_config' in config:
                common_config_file = config['common_config']
                common_config = self.load_common_config(common_config_file)

                # 合并配置，场景特定配置优先
                merged_config = common_config.copy()
                merged_config.update(config)

                # 删除common_config引用
                if 'common_config' in merged_config:
                    del merged_config['common_config']

                config = merged_config

            logger.info(f"成功加载配置文件: {config_path}")
            return config

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def load_common_config(self, common_config_file: str = "common_config.yaml") -> Dict[str, Any]:
        """加载公共配置文件

        Args:
            common_config_file: 公共配置文件名

        Returns:
            公共配置字典
        """
        common_config_path = self.config_dir / common_config_file

        if not common_config_path.exists():
            logger.warning(f"公共配置文件不存在: {common_config_path}")
            return {}

        try:
            with open(common_config_path, 'r', encoding='utf-8') as f:
                common_config = yaml.safe_load(f)

            logger.info(f"成功加载公共配置文件: {common_config_path}")
            return common_config

        except Exception as e:
            logger.error(f"加载公共配置文件失败: {e}")
            return {}
    
    def get_agents_config(self, config: Dict[str, Any]) -> list:
        """获取智能体配置
        
        Args:
            config: 完整配置字典
            
        Returns:
            智能体配置列表
        """
        return config.get('agents', [])
    
    def get_workflows_config(self, config: Dict[str, Any]) -> list:
        """获取工作流配置
        
        Args:
            config: 完整配置字典
            
        Returns:
            工作流配置列表
        """
        return config.get('workflows', [])
    
    def get_simulation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """获取仿真配置
        
        Args:
            config: 完整配置字典
            
        Returns:
            仿真配置字典
        """
        return config.get('simulation', {})
    
    def get_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """获取环境配置
        
        Args:
            config: 完整配置字典
            
        Returns:
            环境配置字典
        """
        return config.get('environment', {})
    
    def validate_agent_config(self, agent_config: Dict[str, Any]) -> bool:
        """验证单个智能体配置
        
        Args:
            agent_config: 智能体配置字典
            
        Returns:
            是否有效
        """
        required_fields = ['id', 'type', 'properties', 'initial_position']
        
        for field in required_fields:
            if field not in agent_config:
                logger.error(f"智能体配置缺少必需字段: {field}")
                return False
                
        # 验证位置配置
        position = agent_config['initial_position']
        # 支持两种位置格式：x/y/z 或 latitude/longitude/altitude
        has_xyz = all(key in position for key in ['x', 'y', 'z'])
        has_lat_lon_alt = all(key in position for key in ['latitude', 'longitude', 'altitude'])
        
        if not (has_xyz or has_lat_lon_alt):
            logger.error("智能体初始位置配置不完整，需要包含x/y/z坐标或latitude/longitude/altitude坐标")
            return False
            
        # 验证属性配置
        properties = agent_config['properties']
        required_props = ['battery_level', 'battery_capacity', 'payload_capacity', 'weight']
        
        for prop in required_props:
            if prop not in properties:
                logger.error(f"智能体属性配置缺少必需字段: {prop}")
                return False
                
        return True
    
    def validate_workflow_config(self, workflow_config: Dict[str, Any]) -> bool:
        """验证单个工作流配置
        
        Args:
            workflow_config: 工作流配置字典
            
        Returns:
            是否有效
        """
        required_fields = ['id', 'name', 'type', 'agent_id', 'properties']
        
        for field in required_fields:
            if field not in workflow_config:
                logger.error(f"工作流配置缺少必需字段: {field}")
                return False
        
        workflow_type = workflow_config['type']
        properties = workflow_config['properties']
        
        # 根据工作流类型验证属性
        if workflow_type == 'logistics':
            required_props = ['pickup_location', 'delivery_location', 'payload_id', 'payload_weight']
            
            for prop in required_props:
                if prop not in properties:
                    logger.error(f"物流工作流属性配置缺少必需字段: {prop}")
                    return False
                    
            # 验证位置配置
            for location_key in ['pickup_location', 'delivery_location']:
                location = properties[location_key]
                if not all(key in location for key in ['x', 'y', 'z']):
                    logger.error(f"工作流{location_key}配置不完整，需要包含x, y, z坐标")
                    return False
                    
        elif workflow_type == 'inspection':
            required_props = ['inspection_points']
            
            for prop in required_props:
                if prop not in properties:
                    logger.error(f"巡检工作流属性配置缺少必需字段: {prop}")
                    return False
            
            # 验证巡检点配置
            inspection_points = properties['inspection_points']
            if not isinstance(inspection_points, list) or len(inspection_points) == 0:
                logger.error("巡检点配置必须是非空列表")
                return False
                
            for i, point in enumerate(inspection_points):
                if isinstance(point, dict):
                    if not all(key in point for key in ['x', 'y', 'z']):
                        logger.error(f"巡检点{i}配置不完整，需要包含x, y, z坐标")
                        return False
                elif isinstance(point, (list, tuple)):
                    if len(point) != 3:
                        logger.error(f"巡检点{i}配置不完整，需要包含3个坐标值")
                        return False
                else:
                    logger.error(f"巡检点{i}配置格式错误")
                    return False
                # --- 新增: 验证 communication 工作流 ---
        elif workflow_type == 'communication':
            required_props = ['data_id', 'data_size_mb', 'target_location']
            
            for prop in required_props:
                if prop not in properties:
                    logger.error(f"通信工作流属性配置缺少必需字段: {prop}")
                    return False
            
            # 验证数据大小是否为数字
            if not isinstance(properties['data_size_mb'], (int, float)):
                logger.error("通信工作流属性 'data_size_mb' 必须是数字")
                return False

            # 验证目标位置配置
            location = properties['target_location']
            if not isinstance(location, (list, tuple)) or len(location) != 3:
                logger.error("通信工作流 'target_location' 配置必须是包含3个坐标值的列表或元组")
                return False
            if not all(isinstance(val, (int, float)) for val in location):
                logger.error("通信工作流 'target_location' 的坐标值必须是数字")
                return False
        # --- 新增: 验证 computation 工作流 ---
        elif workflow_type == 'computation':
            required_props = ['job_id', 'required_gflops', 'workload_profile']

            for prop in required_props:
                if prop not in properties:
                    logger.error(f"计算工作流属性配置缺少必需字段: {prop}")
                    return False

            # 验证计算量是否为数字
            if not isinstance(properties['required_gflops'], (int, float)):
                logger.error("计算工作流属性 'required_gflops' 必须是数字")
                return False

            # 验证工作负载配置
            workload_profile = properties['workload_profile']
            if not isinstance(workload_profile, dict):
                logger.error("计算工作流属性 'workload_profile' 必须是字典")
                return False

            required_workload_props = ['cpu_intensity', 'memory_intensity']
            for prop in required_workload_props:
                if prop not in workload_profile:
                    logger.error(f"计算工作流 'workload_profile' 缺少必需字段: {prop}")
                    return False
                if not isinstance(workload_profile[prop], (int, float)):
                    logger.error(f"计算工作流 'workload_profile.{prop}' 必须是数字")
                    return False

            # 验证计算位置（如果存在）
            if 'compute_location' in properties:
                location = properties['compute_location']
                if not isinstance(location, (list, tuple)) or len(location) != 3:
                    logger.error("计算工作流 'compute_location' 配置必须是包含3个坐标值的列表或元组")
                    return False
                if not all(isinstance(val, (int, float)) for val in location):
                    logger.error("计算工作流 'compute_location' 的坐标值必须是数字")
                    return False
        # ------------------------------------

        else:
            logger.warning(f"未知的工作流类型: {workflow_type}，跳过属性验证")

        return True

    def discover_scenario_configs(self) -> List[str]:
        """发现所有场景配置文件

        Returns:
            场景配置文件名列表
        """
        scenario_configs = []

        # 查找所有yaml文件，排除公共配置文件
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.name not in ["common_config.yaml"]:
                scenario_configs.append(config_file.name)

        logger.info(f"发现 {len(scenario_configs)} 个场景配置文件: {scenario_configs}")
        return scenario_configs

    def load_all_scenario_configs(self) -> Dict[str, Dict[str, Any]]:
        """加载所有场景配置文件

        Returns:
            场景配置字典，键为配置文件名，值为配置内容
        """
        scenario_configs = self.discover_scenario_configs()
        all_configs = {}

        for config_file in scenario_configs:
            try:
                config = self.load_config(config_file)
                scenario_name = config.get('scenario_name', None)
                if scenario_name:
                    all_configs[scenario_name] = config
                    logger.info(f"成功加载场景配置: {scenario_name}")
            except Exception as e:
                logger.error(f"加载场景配置失败 {config_file}: {e}")
                continue

        return all_configs
