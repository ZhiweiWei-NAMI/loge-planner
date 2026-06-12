"""
生成时变交通配置文件 (traffic_profiles.csv)

模拟高峰时段限速场景:
- 早高峰 (0-600s): 部分走廊限速 0.3 (5 m/s)
- 正常时段 (600-1200s): 正常速度 1.0
- 晚高峰 (1200-1800s): 部分走廊限速 0.5 (7.5 m/s)

这样可以验证 topo-aware 路径规划的效果:
- 非topo-aware: 选择最短路径，但可能经过拥堵走廊
- topo-aware: 考虑时变速度，选择绕行但更快的路径
"""

import csv
import json
import random
from pathlib import Path
from typing import List, Tuple


def generate_traffic_profiles(
    topology_path: Path,
    output_path: Path,
    congestion_ratio: float = 0.3,  # 30%的边有高峰限速
    seed: int = 42
) -> None:
    """生成交通配置文件
    
    Args:
        topology_path: 拓扑JSON文件路径
        output_path: 输出CSV路径
        congestion_ratio: 有拥堵的边比例
        seed: 随机种子
    """
    random.seed(seed)
    
    # 加载拓扑
    with open(topology_path, 'r') as f:
        topo_data = json.load(f)
    
    edges = topo_data.get('edges', [])
    if not edges:
        print("No edges found in topology")
        return
    
    # 选择部分边作为拥堵走廊
    n_congested = int(len(edges) * congestion_ratio)
    congested_edges = random.sample(edges, min(n_congested, len(edges)))
    
    profiles: List[Tuple] = []
    
    for edge in congested_edges:
        u, v = edge['u'], edge['v']
        
        # 早高峰: 0-600s, 速度降到 0.3 (约5 m/s)
        profiles.append((u, v, 0.0, 600.0, 0.33, False))
        
        # 正常时段: 600-1200s, 速度恢复
        profiles.append((u, v, 600.0, 1200.0, 1.0, False))
        
        # 晚高峰: 1200-1800s, 速度降到 0.5 (约7.5 m/s)
        profiles.append((u, v, 1200.0, 1800.0, 0.5, False))
    
    # 写入CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['u', 'v', 't_start', 't_end', 'speed_factor', 'blocked'])
        for row in profiles:
            writer.writerow(row)
    
    print(f"Generated traffic profiles for {len(congested_edges)} edges -> {output_path}")
    print(f"  - Peak hours (0-600s, 1200-1800s): 30-50% speed reduction")
    print(f"  - Normal hours (600-1200s): full speed")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', type=str, required=True, help='Path to topology JSON')
    parser.add_argument('--output', type=str, required=True, help='Output traffic CSV path')
    parser.add_argument('--congestion-ratio', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    generate_traffic_profiles(
        Path(args.topology),
        Path(args.output),
        args.congestion_ratio,
        args.seed
    )


if __name__ == '__main__':
    main()
