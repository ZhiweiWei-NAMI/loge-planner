"""
Topology Manager - 城市空域图拓扑管理

管理城市级低空飞行网络的拓扑结构，包括：
- 节点：vertiports, waypoints, charging stations
- 边：flight corridors with speed/capacity constraints
- 动态查询：边速度、可达性、天气影响

配置来源：
- config/topology_config.yaml: 拓扑参数、速度限制、事故模型
- graph_topology.json: 仅存统 JSON 文件 (可在仿真前固定)
"""

import json
import logging
import csv
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# 默认拓扑配置 (可被 topology_config.yaml 覆盖)
DEFAULT_TOPOLOGY_CONFIG = {
    'corridor': {
        'base_velocity': 15.0,  # m/s - 基准速度限制
        'rho_jam': 0.5,         # 拥堵密度阈值
        'capacity': 10,         # 走廊容量
    },
    'topology': {
        'n_vertiports': 40,
        'n_waypoints': 120,
        'n_charging': 10,
        'k_neighbors': 5,
        'waypoint_altitude': {'min': 40.0, 'max': 120.0},
        'vertiport_capacity': 5,
        'waypoint_capacity': 3,
        'charging_capacity': 2,
    },
}


@dataclass
class Node:
    """拓扑节点"""
    id: str
    type: str  # vertiport, waypoint, charging, hub, custom
    x: float
    y: float
    z: float
    capacity: int = 1
    meta: Optional[Dict] = None


@dataclass
class Edge:
    """拓扑边（flight corridor）"""
    u: str
    v: str
    length: float
    base_velocity: float  # m/s
    rho_jam: float = 0.5  # 拥堵密度阈值
    capacity: int = 10
    meta: Optional[Dict] = None


class TopologyManager:
    """拓扑管理器"""
    
    # 类级别配置（从 topology_config.yaml 加载）
    _config: Dict = None
    
    @classmethod
    def load_config(cls, config_path: str = None) -> Dict:
        """从 YAML 文件加载拓扑配置
        
        Args:
            config_path: 配置文件路径，默认为 config/topology_config.yaml
            
        Returns:
            配置字典
        """
        if cls._config is not None:
            return cls._config
        
        if config_path is None:
            # 默认路径: config/topology_config.yaml
            current_dir = Path(__file__).resolve().parent
            config_path = current_dir.parent.parent.parent / "config" / "topology_config.yaml"
        
        config_path = Path(config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f)
                logger.info("Loaded topology config from: %s", config_path)
            except Exception as e:
                logger.warning("Failed to load topology config: %s, using defaults", e)
                cls._config = DEFAULT_TOPOLOGY_CONFIG
        else:
            logger.info("Topology config not found, using defaults")
            cls._config = DEFAULT_TOPOLOGY_CONFIG
        
        return cls._config
    
    @classmethod
    def get_corridor_config(cls) -> Dict:
        """获取走廊配置（速度限制、容量等）"""
        config = cls.load_config()
        return config.get('corridor', DEFAULT_TOPOLOGY_CONFIG['corridor'])
    
    def __init__(self, bounds: Dict[str, float], scale_factor: float = 1.0):
        """
        初始化拓扑管理器
        
        Args:
            bounds: {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'}
            scale_factor: 坐标缩放因子
        """
        self.bounds = bounds
        self.scale_factor = scale_factor
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}
        
        # 加载配置
        self.config = self.load_config()
        self.corridor_config = self.get_corridor_config()
        
        # 天气影响（外部注入）
        self.weather_blocked_edges: set = set()
        self.weather_speed_factors: Dict[Tuple[str, str], float] = {}
        self.traffic_profiles: Dict[Tuple[str, str], List[Tuple[float, float, float, bool]]] = {}
        
        logger.info("TopologyManager initialized with bounds: %s, base_velocity: %.1f m/s", 
                   bounds, self.corridor_config.get('base_velocity', 15.0))
    
    def generate_grid_topology(
        self, 
        n_vertiports: int = 40,
        n_waypoints: int = 120,
        n_charging: int = 10,
        k_neighbors: int = 5,
        seed: int = 42
    ):
        """
        生成网格拓扑
        
        Args:
            n_vertiports: vertiport 数量
            n_waypoints: waypoint 数量
            n_charging: charging station 数量
            k_neighbors: K近邻连接数
            seed: 随机种子
        """
        np.random.seed(seed)
        
        x_min, x_max = self.bounds['x_min'], self.bounds['x_max']
        y_min, y_max = self.bounds['y_min'], self.bounds['y_max']
        z_min, z_max = self.bounds.get('z_min', 0.0), self.bounds.get('z_max', 200.0)
        
        # 生成 vertiports (地面)
        for i in range(n_vertiports):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            self.add_node(f"vertiport_{i}", x, y, 0.0, "vertiport", capacity=5)
        
        # 生成 waypoints (空中)
        for i in range(n_waypoints):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = np.random.uniform(40.0, 100.0)
            self.add_node(f"waypoint_{i}", x, y, z, "waypoint", capacity=3)
        
        # 生成 charging stations (地面)
        for i in range(n_charging):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            self.add_node(f"charging_{i}", x, y, 0.0, "charging", capacity=2)
        
        # 构建边：K近邻连接
        all_node_ids = list(self.nodes.keys())
        positions = np.array([[n.x, n.y, n.z] for n in self.nodes.values()])
        
        for i, node_id in enumerate(all_node_ids):
            # 计算到其他节点的距离
            distances = np.linalg.norm(positions - positions[i], axis=1)
            # 排序找到最近的K个邻居（排除自己）
            nearest_indices = np.argsort(distances)[1:k_neighbors+1]
            
            for j in nearest_indices:
                neighbor_id = all_node_ids[j]
                dist = distances[j]
                # 添加单向边 (使用配置中的速度限制)
                self.add_edge(
                    node_id, 
                    neighbor_id,
                    base_velocity=self.corridor_config.get('base_velocity', 15.0),
                    rho_jam=self.corridor_config.get('rho_jam', 0.5),
                    capacity=self.corridor_config.get('capacity', 10)
                )
        
        logger.info(
            "Generated topology: %d vertiports, %d waypoints, %d charging, %d edges",
            n_vertiports, n_waypoints, n_charging, self.graph.number_of_edges()
        )
    
    def add_node(
        self,
        nid: str,
        x: float,
        y: float,
        z: float,
        ntype: str,
        capacity: int = 1,
        meta: Optional[Dict] = None
    ):
        """添加节点"""
        node = Node(id=nid, type=ntype, x=x, y=y, z=z, capacity=capacity, meta=meta or {})
        self.nodes[nid] = node
        self.graph.add_node(nid, **asdict(node))

    def add_edge(
        self,
        u: str,
        v: str,
        base_velocity: float = 15.0,
        rho_jam: float = 0.5,
        capacity: int = 10,
        meta: Optional[Dict] = None
    ):
        """添加边"""
        if u not in self.nodes or v not in self.nodes:
            logger.warning("Cannot add edge %s->%s: node not found", u, v)
            return

        node_u = self.nodes[u]
        node_v = self.nodes[v]
        length = np.sqrt(
            (node_u.x - node_v.x)**2 +
            (node_u.y - node_v.y)**2 +
            (node_u.z - node_v.z)**2
        )

        edge = Edge(
            u=u, v=v, length=length,
            base_velocity=base_velocity,
            rho_jam=rho_jam,
            capacity=capacity,
            meta=meta or {}
        )
        self.edges[(u, v)] = edge
        self.graph.add_edge(u, v, **asdict(edge))

    def get_nearest_node(
        self,
        x: float,
        y: float,
        z: float,
        type_filter: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        获取最近节点（snap-to-node）

        Args:
            x, y, z: 目标坐标
            type_filter: 节点类型过滤列表，如 ['vertiport', 'charging']

        Returns:
            最近节点ID，如果没有符合条件的节点返回None
        """
        candidates = self.nodes.values()
        if type_filter:
            candidates = [n for n in candidates if n.type in type_filter]

        if not candidates:
            return None

        min_dist = float('inf')
        nearest = None

        for node in candidates:
            dist = np.sqrt(
                (node.x - x)**2 +
                (node.y - y)**2 +
                (node.z - z)**2
            )
            if dist < min_dist:
                min_dist = dist
                nearest = node.id

        return nearest

    def get_edge_velocity(
        self,
        u: str,
        v: str,
        t: float,
        current_density: float = 0.0
    ) -> float:
        """
        获取边的当前速度（考虑拥堵和天气）

        Args:
            u, v: 边的起止节点
            t: 当前时间
            current_density: 当前密度 (0-1)

        Returns:
            速度 (m/s)，如果边不存在或被阻断返回0
        """
        if (u, v) not in self.edges:
            return 0.0

        if (u, v) in self.weather_blocked_edges:
            return 0.0

        edge = self.edges[(u, v)]
        base_v = edge.base_velocity

        weather_factor = self.weather_speed_factors.get((u, v), 1.0)
        base_v *= weather_factor

        traffic_factor = 1.0
        if self.traffic_profiles:
            key = (u, v)
            profiles = self.traffic_profiles.get(key)
            if profiles:
                blocked = False
                factor_acc = []
                for t_start, t_end, speed_factor, is_blocked in profiles:
                    if t_start <= t < t_end:
                        if is_blocked:
                            blocked = True
                            break
                        factor_acc.append(speed_factor)
                if blocked:
                    return 0.0
                if factor_acc:
                    for f in factor_acc:
                        traffic_factor *= f

        base_v *= traffic_factor

        # 应用拥堵模型：v = v_free * (1 - rho/rho_jam)
        rho_jam = edge.rho_jam
        if current_density >= rho_jam:
            return 0.0

        v = base_v * (1.0 - current_density / rho_jam)
        return max(0.0, v)

    def is_edge_available(self, u: str, v: str, t: float) -> bool:
        """
        检查边在时刻t是否可用

        Args:
            u, v: 边的起止节点
            t: 当前时间

        Returns:
            是否可用
        """
        if (u, v) not in self.edges:
            return False

        if (u, v) in self.weather_blocked_edges:
            return False

        if self.traffic_profiles:
            key = (u, v)
            profiles = self.traffic_profiles.get(key)
            if profiles:
                for t_start, t_end, speed_factor, is_blocked in profiles:
                    if t_start <= t < t_end and is_blocked:
                        return False

        return True

    def save_to_file(self, path: str):
        """保存拓扑到JSON文件"""
        data = {
            "bounds": self.bounds,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
            "version": 1
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Topology saved to: %s", path)

    def load_from_file(self, path: str):
        """从JSON文件加载拓扑"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.bounds = data.get("bounds", {})

        # 清空现有数据
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()

        # 加载节点
        for n_data in data.get("nodes", []):
            node = Node(**n_data)
            self.nodes[node.id] = node
            self.graph.add_node(node.id, **asdict(node))

        # 加载边
        for e_data in data.get("edges", []):
            edge = Edge(**e_data)
            self.edges[(edge.u, edge.v)] = edge
            self.graph.add_edge(edge.u, edge.v, **asdict(edge))

        logger.info(
            "Topology loaded from %s: %d nodes, %d edges",
            path, len(self.nodes), len(self.edges)
        )

    def export_to_csv(self, nodes_path: str, edges_path: str):
        nodes_file = Path(nodes_path)
        edges_file = Path(edges_path)

        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        edges_file.parent.mkdir(parents=True, exist_ok=True)

        with nodes_file.open("w", newline="", encoding="utf-8") as f_nodes:
            writer = csv.writer(f_nodes)
            writer.writerow(["id", "type", "x", "y", "z", "capacity"])
            for n in self.nodes.values():
                writer.writerow([n.id, n.type, n.x, n.y, n.z, n.capacity])

        with edges_file.open("w", newline="", encoding="utf-8") as f_edges:
            writer = csv.writer(f_edges)
            writer.writerow(["u", "v", "length", "base_velocity", "rho_jam", "capacity"])
            for e in self.edges.values():
                writer.writerow([e.u, e.v, e.length, e.base_velocity, e.rho_jam, e.capacity])

        logger.info("Topology exported to CSV: %s, %s", nodes_file, edges_file)

    def load_traffic_from_csv(self, path: str):
        traffic_file = Path(path)
        if not traffic_file.is_file():
            return

        profiles: Dict[Tuple[str, str], List[Tuple[float, float, float, bool]]] = {}

        with traffic_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    u = str(row.get("u"))
                    v = str(row.get("v"))
                    if not u or not v:
                        continue
                    t_start = float(row.get("t_start", 0.0))
                    t_end = float(row.get("t_end", float("inf")))
                    speed_factor = float(row.get("speed_factor", 1.0))
                    blocked_val = row.get("blocked", "0")
                    is_blocked = str(blocked_val).strip().lower() in ("1", "true", "yes")
                    key = (u, v)
                    if key not in profiles:
                        profiles[key] = []
                    profiles[key].append((t_start, t_end, speed_factor, is_blocked))
                except Exception:
                    continue

        for key in profiles:
            profiles[key].sort(key=lambda x: x[0])

        self.traffic_profiles = profiles
        logger.info("Loaded traffic profiles from %s: %d edges", traffic_file, len(self.traffic_profiles))
