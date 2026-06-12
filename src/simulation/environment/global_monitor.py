"""
Global Monitor - 全局环境监控与Replan触发检测

监控环境异常、任务失败、deadline违规，生成replan_triggers.csv日志。
采用混合模式：Polling (边/workflow状态) + Event Subscription (terminal异常上报)。

超速事故概率模型（基于现实逻辑）：
- 正常飞行事故率很低 (0.1%)
- 超速会显著增加事故概率，呈指数增长
- 超速10%: ~3.6% 事故概率
- 超速20%: ~12% 事故概率
- 超速50%: ~68% 事故概率
"""

import csv
import logging
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ReplanTrigger:
    """Replan触发记录"""
    timestamp: float
    trigger_type: str  # 'edge_blocked', 'terminal_anomaly', 'deadline_violation', 'battery_critical', 'speed_violation'
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    position: Optional[tuple] = None
    details: Optional[Dict] = None


class GlobalMonitor:
    """全局监控器（作为SimPy Process）"""
    
    def __init__(self, env, agents: List, topology_manager=None, output_path: str = None):
        """
        初始化监控器
        
        Args:
            env: SimPy环境
            agents: Agent列表
            topology_manager: TopologyManager实例（可选）
            output_path: 输出日志路径
        """
        self.env = env
        self.agents = agents
        self.topology = topology_manager
        self.output_path = output_path or "data/results/replan_triggers.csv"
        
        self.triggers: List[ReplanTrigger] = []
        
        # 监控配置
        self.polling_interval = 1.0  # 1秒轮询一次
        self.deadline_check_interval = 5.0  # 5秒检查一次deadline
        
        # 超速检测配置 (基于 topology_config.yaml)
        self.speed_tolerance = 1.1  # 允许超速10% (警告阈值)
        self.speed_violation_agents: Set[str] = set()  # 已因超速失效的agent，避免重复处理
        
        # 超速事故概率模型参数 (基于现实逻辑)
        # P(accident) = base_rate * (1 + severity_factor * overspeed_ratio)²
        self.accident_model_enabled = True
        self.base_accident_rate = 0.001     # 正常飞行事故率 0.1%
        self.severity_factor = 50.0          # 超速严重性系数
        
        # 超速程度阈值
        self.speed_thresholds = {
            'warning': 1.05,    # 5% 超速: 警告
            'minor': 1.10,      # 10% 超速: 轻微违规
            'moderate': 1.20,   # 20% 超速: 中度违规
            'severe': 1.50,     # 50% 超速: 严重违规 (强制停飞)
        }
        
        # 订阅terminal异常事件
        if hasattr(env, 'event_registry'):
            env.event_registry.subscribe(
                source_id='*',
                event_name='env_anomaly_report',
                listener_id='global_monitor',
                callback=self._on_anomaly_report
            )
            logger.info("GlobalMonitor subscribed to env_anomaly_report events")
        
        logger.info("GlobalMonitor initialized, output: %s", self.output_path)
    
    def _on_anomaly_report(self, source, event_name, data):
        """
        处理terminal异常上报（Event回调）
        
        Args:
            source: 事件源
            event_name: 事件名称
            data: 事件数据
        """
        trigger = ReplanTrigger(
            timestamp=self.env.now,
            trigger_type=data.get('issue_type', 'terminal_anomaly'),
            agent_id=data.get('agent_id'),
            workflow_id=data.get('workflow_id'),
            position=data.get('position'),
            details=data
        )
        
        self.triggers.append(trigger)
        
        logger.warning(
            "Replan trigger recorded: %s at t=%.1f by agent %s",
            trigger.trigger_type, trigger.timestamp, trigger.agent_id
        )
    
    def run(self):
        """监控主循环（作为SimPy Process）"""
        last_deadline_check = 0.0
        
        while True:
            # 1. Edge状态轮询（如果有TopologyManager）
            if self.topology:
                self._check_edge_traversability()
            
            # 2. Deadline检查
            if self.env.now - last_deadline_check >= self.deadline_check_interval:
                self._check_workflow_deadlines()
                last_deadline_check = self.env.now
            
            # 等待下一个轮询周期
            yield self.env.timeout(self.polling_interval)
    
    def _check_edge_traversability(self):
        """检查边的可通行性和超速违规（轮询flying agents）"""
        if not self.topology:
            return

        for agent in self.agents:
            agent_id = getattr(agent, 'id', 'unknown')
            
            # 跳过已因超速失效的agent
            if agent_id in self.speed_violation_agents:
                continue
            
            moving_status = agent.get_state('moving_status', 'idle')
            if moving_status != 'flying':
                continue

            # 获取agent位置
            position = agent.get_state('position', None)
            if not position or len(position) < 3:
                continue

            try:
                x, y, z = float(position[0]), float(position[1]), float(position[2])
            except Exception:
                continue

            # 使用最近节点近似当前所在航路段
            try:
                node_id = self.topology.get_nearest_node(x, y, z)
            except Exception as e:  # pragma: no cover - 防御性
                logger.error("GlobalMonitor: get_nearest_node failed for agent %s: %s", agent_id, e)
                continue

            if node_id is None:
                continue
            
            # === 超速检测 ===
            speed_violation_detected = self._check_speed_violation(agent, agent_id, node_id, x, y, z)
            if speed_violation_detected:
                continue  # 已触发失效，跳过后续检查

            # 检查从该节点出发的边是否阻塞或速度极低
            for succ in self.topology.graph.successors(node_id):
                available = self.topology.is_edge_available(node_id, succ, self.env.now)
                v = self.topology.get_edge_velocity(node_id, succ, self.env.now, 0.0)

                if not available or v <= 0.5:
                    trigger = ReplanTrigger(
                        timestamp=self.env.now,
                        trigger_type='edge_blocked' if not available else 'edge_slow',
                        agent_id=getattr(agent, 'id', None),
                        workflow_id=None,
                        position=(x, y, z),
                        details={
                            'edge': (node_id, succ),
                            'available': available,
                            'velocity': v,
                        },
                    )
                    self.triggers.append(trigger)
                    logger.warning(
                        "Edge traversability issue: %s on edge %s->%s (v=%.2f) for agent %s",
                        trigger.trigger_type,
                        node_id,
                        succ,
                        v,
                        getattr(agent, 'id', 'unknown'),
                    )
                    
                    # Trigger agent failure: set status to 'error' and cancel all tasks
                    if not available:
                        self._trigger_agent_failure(
                            agent,
                            reason=f"Corridor {node_id}->{succ} blocked",
                            trigger=trigger
                        )
    
    def _calculate_accident_probability(self, speed_ratio: float) -> float:
        """计算基于超速程度的事故概率
        
        基于现实逻辑的事故概率模型:
        P(accident) = base_rate * (1 + severity_factor * overspeed_ratio)²
        
        Args:
            speed_ratio: 当前速度/限速 比值 (e.g., 1.1 = 超速10%)
            
        Returns:
            事故概率 (0.0 - 1.0)
        """
        if speed_ratio <= 1.0:
            return self.base_accident_rate
        
        overspeed_ratio = speed_ratio - 1.0  # 超速比例 (e.g., 0.1 = 10%)
        probability = self.base_accident_rate * (1 + self.severity_factor * overspeed_ratio) ** 2
        return min(1.0, probability)  # 上限为100%
    
    def _get_violation_severity(self, speed_ratio: float) -> str:
        """获取超速违规严重程度
        
        Args:
            speed_ratio: 当前速度/限速 比值
            
        Returns:
            严重程度: 'normal', 'warning', 'minor', 'moderate', 'severe'
        """
        if speed_ratio >= self.speed_thresholds['severe']:
            return 'severe'
        elif speed_ratio >= self.speed_thresholds['moderate']:
            return 'moderate'
        elif speed_ratio >= self.speed_thresholds['minor']:
            return 'minor'
        elif speed_ratio >= self.speed_thresholds['warning']:
            return 'warning'
        return 'normal'

    def _check_speed_violation(self, agent, agent_id: str, node_id: str, x: float, y: float, z: float) -> bool:
        """检查agent是否超速违规（基于现实逻辑的事故概率模型）
        
        Args:
            agent: Agent实例
            agent_id: Agent ID
            node_id: 最近的拓扑节点ID
            x, y, z: Agent当前位置
            
        Returns:
            True if speed violation detected and agent failed, False otherwise
        """
        # 获取agent当前速度
        velocity = agent.get_state('velocity', None)
        if not velocity or len(velocity) < 3:
            return False
        
        try:
            vx, vy, vz = float(velocity[0]), float(velocity[1]), float(velocity[2])
            current_speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        except Exception:
            return False
        
        # 速度为0或很小时不检测
        if current_speed < 0.1:
            return False
        
        # 获取当前位置附近边的限速（取最大允许速度）
        max_speed_limit = 0.0
        relevant_edge = None
        
        for succ in self.topology.graph.successors(node_id):
            speed_limit = self.topology.get_edge_velocity(node_id, succ, self.env.now, 0.0)
            if speed_limit > max_speed_limit:
                max_speed_limit = speed_limit
                relevant_edge = (node_id, succ)
        
        # 如果没有找到有效的限速，不检测
        if max_speed_limit <= 0:
            return False
        
        # 计算速度比值和违规严重程度
        speed_ratio = current_speed / max_speed_limit
        severity = self._get_violation_severity(speed_ratio)
        
        # 计算事故概率（基于现实逻辑）
        accident_prob = self._calculate_accident_probability(speed_ratio)
        
        # 严重超速 (>=50%) 强制失效
        if severity == 'severe':
            trigger = ReplanTrigger(
                timestamp=self.env.now,
                trigger_type='speed_violation',
                agent_id=agent_id,
                workflow_id=None,
                position=(x, y, z),
                details={
                    'current_speed': round(current_speed, 2),
                    'speed_limit': round(max_speed_limit, 2),
                    'speed_ratio': round(speed_ratio, 3),
                    'severity': severity,
                    'accident_probability': round(accident_prob, 4),
                    'edge': relevant_edge,
                },
            )
            self.triggers.append(trigger)
            self.speed_violation_agents.add(agent_id)
            
            logger.error(
                "SEVERE speed violation: agent %s speed=%.1f m/s (%.0f%% over limit), forced failure",
                agent_id, current_speed, (speed_ratio - 1) * 100
            )
            self._trigger_agent_failure(
                agent,
                reason=f"Severe speed violation: {current_speed:.1f} m/s ({(speed_ratio-1)*100:.0f}% over limit)",
                trigger=trigger
            )
            return True
        
        # 超过警告阈值时，根据事故概率模型决定是否发生事故
        if severity in ('warning', 'minor', 'moderate') and self.accident_model_enabled:
            # 随机判定是否发生事故
            if random.random() < accident_prob:
                trigger = ReplanTrigger(
                    timestamp=self.env.now,
                    trigger_type='speed_violation',
                    agent_id=agent_id,
                    workflow_id=None,
                    position=(x, y, z),
                    details={
                        'current_speed': round(current_speed, 2),
                        'speed_limit': round(max_speed_limit, 2),
                        'speed_ratio': round(speed_ratio, 3),
                        'severity': severity,
                        'accident_probability': round(accident_prob, 4),
                        'accident_occurred': True,
                        'edge': relevant_edge,
                    },
                )
                self.triggers.append(trigger)
                self.speed_violation_agents.add(agent_id)
                
                logger.warning(
                    "Speed-related accident: agent %s at (%.1f, %.1f, %.1f) speed=%.1f m/s "
                    "(%.0f%% over limit, severity=%s, P=%.2f%%)",
                    agent_id, x, y, z, current_speed, (speed_ratio - 1) * 100, severity, accident_prob * 100
                )
                self._trigger_agent_failure(
                    agent,
                    reason=f"Speed-related accident: {current_speed:.1f} m/s ({severity}, P={accident_prob*100:.1f}%)",
                    trigger=trigger
                )
                return True
        
        return False
    
    def _check_workflow_deadlines(self):
        """检查workflow deadline违规（轮询active workflows）"""
        for agent in self.agents:
            # 获取agent的active workflows
            if not hasattr(agent, 'workflows'):
                continue
            
            for workflow in agent.workflows:
                if not hasattr(workflow, 'properties'):
                    continue
                
                # 检查是否超时
                timeout = workflow.properties.get('timeout', float('inf'))
                start_time = workflow.properties.get('start_time', 0.0)
                
                if self.env.now - start_time > timeout:
                    # 记录deadline违规
                    trigger = ReplanTrigger(
                        timestamp=self.env.now,
                        trigger_type='deadline_violation',
                        agent_id=agent.id,
                        workflow_id=workflow.id if hasattr(workflow, 'id') else None,
                        details={'timeout': timeout, 'elapsed': self.env.now - start_time}
                    )
                    
                    self.triggers.append(trigger)
                    
                    logger.warning(
                        "Deadline violation detected: workflow %s exceeded timeout %.1fs",
                        workflow.id if hasattr(workflow, 'id') else 'unknown',
                        timeout
                    )
    
    def _trigger_agent_failure(self, agent, reason: str, trigger: ReplanTrigger):
        """Trigger agent failure due to corridor blockage or other critical events.
        
        This method sets agent status to 'error' and cancels all pending tasks,
        effectively failing all associated workflows.
        
        Args:
            agent: The agent instance to fail
            reason: Human-readable reason for failure
            trigger: The ReplanTrigger that caused this failure
        """
        agent_id = getattr(agent, 'id', 'unknown')
        
        # Check if agent is already in error state to avoid duplicate processing
        current_status = agent.get_state('status') if hasattr(agent, 'get_state') else None
        if current_status == 'error':
            return
        
        logger.error(
            "Agent %s FAILED at t=%.1f: %s",
            agent_id, self.env.now, reason
        )
        
        # Set agent status to error - this will be checked by _check_agent_status()
        if hasattr(agent, 'update_state'):
            agent.update_state('status', 'error')
            # Also set moving_status to idle to stop movement
            agent.update_state('moving_status', 'idle')
        
        # Cancel all pending and running tasks
        if hasattr(agent, '_cancel_all_tasks'):
            agent._cancel_all_tasks()
            logger.info("Agent %s: all tasks cancelled due to corridor blockage", agent_id)
        
        # Record failure trigger with enhanced details
        failure_trigger = ReplanTrigger(
            timestamp=self.env.now,
            trigger_type='agent_failure',
            agent_id=agent_id,
            workflow_id=None,
            position=trigger.position,
            details={
                'reason': reason,
                'original_trigger': trigger.trigger_type,
                'edge': trigger.details.get('edge') if trigger.details else None,
            }
        )
        self.triggers.append(failure_trigger)

    def save_triggers(self):
        """保存触发日志到CSV"""
        output_file = Path(self.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if not self.triggers:
                # 写空文件头
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'trigger_type', 'agent_id', 'workflow_id', 'position', 'details'])
                return
            
            fieldnames = ['timestamp', 'trigger_type', 'agent_id', 'workflow_id', 'position', 'details']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for trigger in self.triggers:
                row = asdict(trigger)
                # 转换position为字符串
                if row['position']:
                    row['position'] = str(row['position'])
                # 转换details为字符串
                if row['details']:
                    row['details'] = str(row['details'])
                writer.writerow(row)
        
        logger.info("Saved %d replan triggers to: %s", len(self.triggers), output_file)

