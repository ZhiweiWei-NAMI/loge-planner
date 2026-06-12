"""
Edge Planner - 时变A*路径规划

在图拓扑上为每个moveto任务生成时变可行轨迹，
考虑拥堵和天气影响。
"""

import heapq
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RouteSegment:
    """路径段"""
    edge: Tuple[str, str]
    t_start: float
    t_end: float
    v_ref: float  # 参考速度


@dataclass
class RouteResult:
    """路径规划结果"""
    nodes: List[str]
    segments: List[RouteSegment]
    total_time: float
    total_distance: float
    congestion_index: float  # gamma_cong


@dataclass(order=True)
class AStarState:
    """A*搜索状态"""
    f_score: float
    node: str = field(compare=False)
    t: float = field(compare=False)
    g_score: float = field(compare=False)
    parent: Optional['AStarState'] = field(default=None, compare=False)


class EdgePlanner:
    """Edge侧时变A*路径规划器"""
    
    def __init__(self, topo, time_step: float = 1.0):
        """
        初始化规划器
        
        Args:
            topo: TopologyManager实例
            time_step: 时间离散步长(秒)
        """
        self.topo = topo
        self.time_step = time_step
        
        # 预计算直线距离启发值
        self._heuristic_cache: Dict[Tuple[str, str], float] = {}
    
    def _heuristic(self, node: str, goal: str) -> float:
        """
        启发函数：直线距离 / 最大速度
        
        Args:
            node: 当前节点
            goal: 目标节点
        
        Returns:
            估计时间
        """
        cache_key = (node, goal)
        if cache_key in self._heuristic_cache:
            return self._heuristic_cache[cache_key]
        
        if node not in self.topo.nodes or goal not in self.topo.nodes:
            return float('inf')
        
        n1 = self.topo.nodes[node]
        n2 = self.topo.nodes[goal]
        
        dist = np.sqrt(
            (n1.x - n2.x)**2 + 
            (n1.y - n2.y)**2 + 
            (n1.z - n2.z)**2
        )
        
        # 假设最大速度20 m/s
        max_speed = 20.0
        h = dist / max_speed
        
        self._heuristic_cache[cache_key] = h
        return h
    
    def plan_route(
        self,
        start_node: str,
        goal_node: str,
        t_start: float,
        density_fn: Optional[Callable[[str, str, float], float]] = None,
    ) -> RouteResult:
        """
        规划时变最短路径
        
        Args:
            start_node: 起始节点
            goal_node: 目标节点
            t_start: 出发时间
            density_fn: 密度查询函数 (u, v, t) -> density
        
        Returns:
            RouteResult
        """
        if start_node not in self.topo.nodes:
            logger.warning("Start node %s not found", start_node)
            return RouteResult([], [], 0.0, 0.0, 0.0)
        
        if goal_node not in self.topo.nodes:
            logger.warning("Goal node %s not found", goal_node)
            return RouteResult([], [], 0.0, 0.0, 0.0)
        
        if start_node == goal_node:
            return RouteResult([start_node], [], 0.0, 0.0, 0.0)
        
        # A*搜索
        open_set = []
        g_score_init = 0.0
        h_score_init = self._heuristic(start_node, goal_node)
        
        start_state = AStarState(
            f_score=g_score_init + h_score_init,
            node=start_node,
            t=t_start,
            g_score=g_score_init,
            parent=None
        )
        heapq.heappush(open_set, start_state)
        
        # 访问记录：(node, t_discretized) -> best g_score
        visited: Dict[Tuple[str, int], float] = {}
        
        max_iterations = 10000
        iteration = 0
        
        while open_set and iteration < max_iterations:
            iteration += 1
            current = heapq.heappop(open_set)
            
            if current.node == goal_node:
                # 重构路径
                return self._reconstruct_path(current, density_fn)
            
            # 离散化时间用于访问检查
            t_disc = int(current.t / self.time_step)
            visit_key = (current.node, t_disc)
            
            if visit_key in visited and visited[visit_key] <= current.g_score:
                continue
            visited[visit_key] = current.g_score

            # 扩展邻居
            if current.node not in self.topo.graph:
                continue

            for neighbor in self.topo.graph.successors(current.node):
                edge_key = (current.node, neighbor)

                if edge_key not in self.topo.edges:
                    continue

                # 检查边是否可用
                if not self.topo.is_edge_available(current.node, neighbor, current.t):
                    continue

                # 计算边通行时间
                edge = self.topo.edges[edge_key]
                density = density_fn(current.node, neighbor, current.t) if density_fn else 0.0
                v = self.topo.get_edge_velocity(current.node, neighbor, current.t, density)

                if v <= 0:
                    continue

                travel_time = edge.length / v
                t_next = current.t + travel_time

                # 拥堵惩罚
                congestion_penalty = density * 10.0  # 可调参数

                # 新的g_score
                g_next = current.g_score + travel_time + congestion_penalty
                h_next = self._heuristic(neighbor, goal_node)
                f_next = g_next + h_next

                next_state = AStarState(
                    f_score=f_next,
                    node=neighbor,
                    t=t_next,
                    g_score=g_next,
                    parent=current
                )

                heapq.heappush(open_set, next_state)

        # 未找到路径
        logger.warning("No path found from %s to %s", start_node, goal_node)
        return RouteResult([], [], 0.0, 0.0, 0.0)

    def _reconstruct_path(
        self,
        goal_state: AStarState,
        density_fn: Optional[Callable[[str, str, float], float]]
    ) -> RouteResult:
        """
        重构路径

        Args:
            goal_state: 目标状态
            density_fn: 密度函数

        Returns:
            RouteResult
        """
        # 回溯路径
        path_states = []
        current = goal_state
        while current is not None:
            path_states.append(current)
            current = current.parent

        path_states.reverse()

        # 构建节点列表和路径段
        nodes = [s.node for s in path_states]
        segments = []
        total_distance = 0.0
        congestion_sum = 0.0

        for i in range(len(path_states) - 1):
            s1 = path_states[i]
            s2 = path_states[i + 1]

            edge_key = (s1.node, s2.node)
            if edge_key in self.topo.edges:
                edge = self.topo.edges[edge_key]

                density = density_fn(s1.node, s2.node, s1.t) if density_fn else 0.0
                v_ref = self.topo.get_edge_velocity(s1.node, s2.node, s1.t, density)

                segment = RouteSegment(
                    edge=edge_key,
                    t_start=s1.t,
                    t_end=s2.t,
                    v_ref=v_ref
                )
                segments.append(segment)

                total_distance += edge.length
                congestion_sum += density

        total_time = goal_state.t - path_states[0].t
        avg_congestion = congestion_sum / max(len(segments), 1)

        return RouteResult(
            nodes=nodes,
            segments=segments,
            total_time=total_time,
            total_distance=total_distance,
            congestion_index=avg_congestion
        )

    def export_routes_for_agent_plan(
        self,
        agent_plan: Dict,
        scenario: Dict,
        workflow_start_times: Dict,
    ) -> Dict:
        """
        为agent plan中的所有位置生成路由

        Args:
            agent_plan: agent计划字典
            scenario: 场景字典
            workflow_start_times: workflow起始时间映射

        Returns:
            增强的agent_plan（包含edge_routes字段）
        """
        # 提取agent的home位置
        init_params = agent_plan.get("init_params", {})
        drone_locations = init_params.get("drone_locations", {})

        if not drone_locations:
            logger.warning("No drone locations found in agent plan")
            return agent_plan

        # 假设单agent
        agent_id = list(drone_locations.keys())[0]
        home_loc_label = drone_locations[agent_id]

        # 将PDDL位置标签解析为坐标，映射到最近的图节点
        def _parse_position_label(label: str) -> Optional[Tuple[float, float, float]]:
            try:
                # Support topo_<node_id> format (e.g., topo_waypoint_15)
                if label.startswith("topo_"):
                    node_id = label[5:]  # Strip "topo_" prefix
                    if node_id in self.topo.nodes:
                        node = self.topo.nodes[node_id]
                        return node.x, node.y, node.z
                    return None
                
                # Support legacy position_x_y_z format
                parts = label.split("_")
                if len(parts) == 4 and parts[0] == "position":
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    return x, y, z
                return None
            except Exception:
                return None

        # home位置映射
        home_xyz = _parse_position_label(home_loc_label)
        if home_xyz is None:
            logger.warning("Home location label %s cannot be parsed, skipping edge route export", home_loc_label)
            return agent_plan

        start_node = self.topo.get_nearest_node(home_xyz[0], home_xyz[1], home_xyz[2])
        if start_node is None:
            logger.warning("No nearest topology node found for home location %s", home_loc_label)
            return agent_plan

        # 提取cluster_meta中的位置和时间信息
        cluster_meta = agent_plan.get("cluster_meta", {})

        edge_routes = {}

        for loc_label, meta in cluster_meta.items():
            if loc_label == home_loc_label:
                continue

            # 简单实现：从home到每个目标位置规划一次
            # 实际应该根据PDDL solution的顺序规划
            earliest_start = meta.get("earliest_start_time", 0.0)

            goal_xyz = _parse_position_label(loc_label)
            if goal_xyz is None:
                logger.warning("Location label %s cannot be parsed, skipping", loc_label)
                continue

            goal_node = self.topo.get_nearest_node(goal_xyz[0], goal_xyz[1], goal_xyz[2])
            if goal_node is None:
                logger.warning("No nearest topology node found for location %s", loc_label)
                continue

            route = self.plan_route(
                start_node=start_node,
                goal_node=goal_node,
                t_start=earliest_start,
                density_fn=None  # 简化：不考虑密度
            )

            if route.nodes:
                edge_routes[f"{home_loc_label} -> {loc_label}"] = {
                    "nodes": route.nodes,
                    "segments": [
                        {
                            "edge": list(seg.edge),
                            "t_start": seg.t_start,
                            "t_end": seg.t_end,
                            "v_ref": seg.v_ref
                        }
                        for seg in route.segments
                    ],
                    "gamma_cong": route.congestion_index
                }

        agent_plan["edge_routes"] = edge_routes
        return agent_plan

