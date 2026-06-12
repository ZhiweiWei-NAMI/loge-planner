"""
天气预测工具

基于WeatherController实现天气预测功能，为工作流规划提供未来时间窗口内的天气信息。

@author: TCCN Auto Planner
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

logger = logging.getLogger(__name__)


class WeatherForecastTool:
    """天气预测工具类"""
    
    def __init__(self):
        """初始化天气预测工具"""
        pass
    
    def get_weather_forecast(self, env, current_time: float, forecast_window: float, 
                           forecast_steps: int = 5, positions: List[Tuple[float, float, float]] = None) -> Dict[str, Any]:
        """
        获取天气预测信息
        
        Args:
            env: AirFogSim环境实例
            current_time: 当前仿真时间
            forecast_window: 预测时间窗口（秒）
            forecast_steps: 预测时间点数量
            positions: 需要预测的位置列表，如果为None则只预测全局天气
            
        Returns:
            包含当前天气和预测信息的字典
        """
        try:
            # 通过env.get_data_provider("weather")获取weather_controller
            weather_controller = env.get_data_provider("weather")
            
            if not weather_controller:
                logger.warning("未找到天气控制器，返回默认天气信息")
                return self._get_default_weather_info(current_time, forecast_window, forecast_steps)
            
            # 获取当前天气信息
            current_weather = self._get_current_weather_info(weather_controller, positions)
            
            # 生成未来天气预测
            forecast_data = self._generate_weather_forecast(
                weather_controller, current_time, forecast_window, forecast_steps, positions
            )
            
            return {
                'current': current_weather,
                'forecast': forecast_data,
                'forecast_window': forecast_window,
                'forecast_steps': forecast_steps,
                'positions': positions
            }
            
        except Exception as e:
            logger.error(f"获取天气预测信息时出错: {str(e)}")
            return self._get_default_weather_info(current_time, forecast_window, forecast_steps)
    
    def _get_current_weather_info(self, weather_controller, positions: List[Tuple[float, float, float]] = None) -> Dict[str, Any]:
        """获取当前天气信息"""
        current_weather = {
            'global': self._weather_condition_to_dict(weather_controller.default_weather),
            'zones': []
        }
        
        # 添加活跃天气区域信息
        for zone in weather_controller.weather_zones:
            if zone.active:
                zone_info = {
                    'zone_id': zone.zone_id,
                    'center': zone.center,
                    'radius': zone.radius,
                    'condition': self._weather_condition_to_dict(zone.condition),
                    'start_time': zone.start_time,
                    'duration': zone.duration
                }
                current_weather['zones'].append(zone_info)
        
        # 如果指定了位置，获取各位置的具体天气
        if positions:
            current_weather['position_weather'] = {}
            for i, pos in enumerate(positions):
                pos_weather = weather_controller.get_weather_at_position(pos)
                current_weather['position_weather'][f'pos_{i}'] = {
                    'position': pos,
                    'weather': self._weather_condition_to_dict(pos_weather)
                }
        
        return current_weather
    
    def _generate_weather_forecast(self, weather_controller, current_time: float, 
                                 forecast_window: float, forecast_steps: int,
                                 positions: List[Tuple[float, float, float]] = None) -> List[Dict[str, Any]]:
        """生成天气预测数据"""
        forecast_data = []
        step_interval = forecast_window / forecast_steps
        
        for i in range(1, forecast_steps + 1):
            forecast_time = current_time + i * step_interval
            
            # 预测全局天气变化
            predicted_global = self._predict_global_weather(weather_controller, forecast_time)
            
            # 预测天气区域变化
            predicted_zones = self._predict_weather_zones(weather_controller, forecast_time)
            
            forecast_point = {
                'time': forecast_time,
                'global': predicted_global,
                'zones': predicted_zones
            }
            
            # 如果指定了位置，预测各位置的天气
            if positions:
                forecast_point['position_weather'] = {}
                for j, pos in enumerate(positions):
                    # 基于预测的全局天气和区域天气计算位置天气
                    pos_weather = self._predict_position_weather(
                        predicted_global, predicted_zones, pos
                    )
                    forecast_point['position_weather'][f'pos_{j}'] = {
                        'position': pos,
                        'weather': pos_weather
                    }
            
            forecast_data.append(forecast_point)
        
        return forecast_data
    
    def _predict_global_weather(self, weather_controller, forecast_time: float) -> Dict[str, Any]:
        """预测全局天气变化"""
        current_weather = weather_controller.default_weather
        time_delta = forecast_time - current_weather.timestamp
        
        # 基于当前天气和时间差进行简单的线性预测
        # 实际应用中可以使用更复杂的天气模型
        
        # 风速变化：假设有周期性变化
        wind_variation = 2.0 * (time_delta / 3600.0) % 1.0  # 每小时周期
        predicted_wind_speed = max(0, current_weather.wind_speed + wind_variation - 1.0)
        
        # 温度变化：假设有日周期变化
        temp_variation = 3.0 * ((time_delta / 86400.0) % 1.0)  # 每日周期
        predicted_temperature = current_weather.temperature + temp_variation - 1.5
        
        # 能见度变化：基于风速和湿度
        visibility_factor = 1.0 - (predicted_wind_speed / 30.0) * 0.2
        predicted_visibility = max(100, current_weather.visibility * visibility_factor)
        
        # 湿度变化：相对稳定，小幅波动
        humidity_variation = 5.0 * ((time_delta / 7200.0) % 1.0)  # 2小时周期
        predicted_humidity = max(20, min(100, current_weather.humidity + humidity_variation - 2.5))
        
        return {
            'wind_speed': predicted_wind_speed,
            'wind_direction': current_weather.wind_direction,  # 风向变化较慢
            'visibility': predicted_visibility,
            'temperature': predicted_temperature,
            'humidity': predicted_humidity,
            'precipitation': current_weather.precipitation,  # 降水需要更复杂的预测
            'pressure': current_weather.pressure,  # 气压变化较慢
            'timestamp': forecast_time
        }
    
    def _predict_weather_zones(self, weather_controller, forecast_time: float) -> List[Dict[str, Any]]:
        """预测天气区域变化"""
        predicted_zones = []
        
        for zone in weather_controller.weather_zones:
            if not zone.active:
                continue
                
            # 检查区域是否在预测时间内仍然活跃
            if zone.duration > 0:
                elapsed_time = forecast_time - zone.start_time
                if elapsed_time >= zone.duration:
                    continue  # 区域已过期
            
            # 预测区域内天气条件的变化
            predicted_condition = self._predict_zone_weather_condition(zone, forecast_time)
            
            predicted_zones.append({
                'zone_id': zone.zone_id,
                'center': zone.center,
                'radius': zone.radius,
                'condition': predicted_condition,
                'start_time': zone.start_time,
                'duration': zone.duration,
                'remaining_time': max(0, zone.duration - (forecast_time - zone.start_time)) if zone.duration > 0 else -1
            })
        
        return predicted_zones
    
    def _predict_zone_weather_condition(self, zone, forecast_time: float) -> Dict[str, Any]:
        """预测天气区域内的天气条件"""
        current_condition = zone.condition
        time_delta = forecast_time - current_condition.timestamp
        
        # 基于区域类型和时间进行预测
        # 这里使用简化的预测逻辑，实际应用中可以根据区域类型使用不同的模型
        
        return {
            'wind_speed': max(0, current_condition.wind_speed * (1.0 - time_delta / 7200.0)),  # 逐渐减弱
            'wind_direction': current_condition.wind_direction,
            'visibility': min(15000, current_condition.visibility * (1.0 + time_delta / 3600.0)),  # 逐渐改善
            'temperature': current_condition.temperature,  # 区域温度相对稳定
            'humidity': max(20, current_condition.humidity * (1.0 - time_delta / 10800.0)),  # 逐渐降低
            'precipitation': max(0, current_condition.precipitation * (1.0 - time_delta / 1800.0)),  # 逐渐减少
            'pressure': current_condition.pressure,
            'timestamp': forecast_time
        }
    
    def _predict_position_weather(self, global_weather: Dict[str, Any], 
                                zones: List[Dict[str, Any]], 
                                position: Tuple[float, float, float]) -> Dict[str, Any]:
        """预测指定位置的天气"""
        # 检查位置是否在任何天气区域内
        for zone in zones:
            distance = self._calculate_distance(position, zone['center'])
            if distance <= zone['radius']:
                return zone['condition']
        
        # 如果不在任何区域内，返回全局天气
        return global_weather
    
    def _weather_condition_to_dict(self, weather_condition) -> Dict[str, Any]:
        """将WeatherCondition对象转换为字典"""
        if hasattr(weather_condition, '__dict__'):
            return asdict(weather_condition)
        else:
            # 如果不是dataclass，手动提取属性
            return {
                'wind_speed': getattr(weather_condition, 'wind_speed', 0),
                'wind_direction': getattr(weather_condition, 'wind_direction', 0),
                'visibility': getattr(weather_condition, 'visibility', 1000),
                'temperature': getattr(weather_condition, 'temperature', 20),
                'humidity': getattr(weather_condition, 'humidity', 50),
                'precipitation': getattr(weather_condition, 'precipitation', 0),
                'pressure': getattr(weather_condition, 'pressure', 1013.25),
                'timestamp': getattr(weather_condition, 'timestamp', 0)
            }
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """计算两点之间的3D距离"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5
    
    def _get_default_weather_info(self, current_time: float, forecast_window: float, forecast_steps: int) -> Dict[str, Any]:
        """获取默认天气信息（当无法获取真实天气数据时）"""
        default_weather = {
            'wind_speed': 5.0,
            'wind_direction': 0.0,
            'visibility': 10000.0,
            'temperature': 20.0,
            'humidity': 60.0,
            'precipitation': 0.0,
            'pressure': 1013.25,
            'timestamp': current_time
        }
        
        forecast_data = []
        step_interval = forecast_window / forecast_steps
        
        for i in range(1, forecast_steps + 1):
            forecast_time = current_time + i * step_interval
            forecast_data.append({
                'time': forecast_time,
                'global': {**default_weather, 'timestamp': forecast_time},
                'zones': []
            })
        
        return {
            'current': {'global': default_weather, 'zones': []},
            'forecast': forecast_data,
            'forecast_window': forecast_window,
            'forecast_steps': forecast_steps,
            'positions': None
        }


# 全局实例
weather_forecast_tool = WeatherForecastTool()


def get_weather_forecast(env, current_time: float, forecast_window: float, 
                        forecast_steps: int = 5, positions: List[Tuple[float, float, float]] = None) -> Dict[str, Any]:
    """
    获取天气预测信息的便捷函数
    
    Args:
        env: AirFogSim环境实例
        current_time: 当前仿真时间
        forecast_window: 预测时间窗口（秒）
        forecast_steps: 预测时间点数量
        positions: 需要预测的位置列表
        
    Returns:
        天气预测信息字典
    """
    return weather_forecast_tool.get_weather_forecast(env, current_time, forecast_window, forecast_steps, positions)
