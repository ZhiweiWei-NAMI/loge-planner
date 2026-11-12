"""
仿真环境控制器模块

提供气象环境控制器和不明飞行物生成器，用于模拟突发事件对仿真系统的影响。

本包采用惰性导入（PEP 562）：`TopologyManager` / `GlobalMonitor` 等不依赖外部
`airfogsim` 仿真器的组件，可在纯 PDDL / 离线规划路径下直接使用；只有真正需要
气象、UFO 或事件调度时才导入对应子模块，并在此处给出明确的依赖缺失提示。

@author: TCCN Auto Planner
"""

from importlib import import_module
from typing import Any

__all__ = [
    'WeatherController',
    'WeatherCondition',
    'UFOGenerator',
    'UFOObject',
    'EventScheduler',
    'TopologyManager',
    'Node',
    'Edge',
    'GlobalMonitor',
    'ReplanTrigger'
]

# attribute -> defining submodule (relative to this package)
_LAZY_ATTRS = {
    'WeatherController': '.weather_controller',
    'WeatherCondition': '.weather_controller',
    'UFOGenerator': '.ufo_generator',
    'UFOObject': '.ufo_generator',
    'EventScheduler': '.event_scheduler',
    'TopologyManager': '.topology_manager',
    'Node': '.topology_manager',
    'Edge': '.topology_manager',
    'GlobalMonitor': '.global_monitor',
    'ReplanTrigger': '.global_monitor',
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(submodule, __name__)
    except ImportError as exc:
        raise ImportError(
            f"{name} 需要外部 airfogsim 仿真器包（pip install -e ../airfogsim-project）；"
            f"纯 PDDL / 离线规划路径不需要该依赖。原始错误: {exc}"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value
