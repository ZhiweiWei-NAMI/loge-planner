#!/usr/bin/env python3
"""
PDDLSolver - TCCN自动规划求解器

这个类提供了一个接口，让规划器可以输出init和goal，
PDDLSolver能解析并生成对应的problem对象，进而求解。

主要功能：
1. 解析规划器输出的init和goal参数
2. 构建连通图（基于距离矩阵和最小生成树）
3. 生成PDDL problem对象
4. 调用求解器求解
5. 返回解决方案
"""

import numpy as np
import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx

try:
    from unified_planning.shortcuts import *
    from unified_planning.io import PDDLReader
    UP_AVAILABLE = True
except ImportError:
    UP_AVAILABLE = False
    print("Warning: unified_planning library not found. Please install with: pip install unified-planning[tamer]")


def convert_planner_output_to_pddl_params(logger, planner_output: Dict) -> tuple:
    """将规划器输出转换为PDDLSolver所需的参数格式"""
    location_name_mapping = {}
    try:
        # 检查输入类型
        if not isinstance(planner_output, dict):
            logger.error(f"规划器输出应该是字典类型，但收到了: {type(planner_output)}")
            return None, None, location_name_mapping

        # 详细检查各个字段的类型
        locations_data = planner_output.get("locations", {})
        if not isinstance(locations_data, dict):
            logger.error(f"locations字段应该是字典类型，但收到了: {type(locations_data)}, 内容: {locations_data}")
            return None, None, location_name_mapping

        drones_data = planner_output.get("drones", {})
        if not isinstance(drones_data, dict):
            logger.error(f"drones字段应该是字典类型，但收到了: {type(drones_data)}, 内容: {drones_data}")
            return None, None, location_name_mapping

        packages_data = planner_output.get("packages", {})
        if not isinstance(packages_data, dict):
            logger.error(f"packages字段应该是字典类型，但收到了: {type(packages_data)}, 内容: {packages_data}")
            return None, None, location_name_mapping

        goals_data = planner_output.get("goals", {})
        if not isinstance(goals_data, dict):
            logger.error(f"goals字段应该是字典类型，但收到了: {type(goals_data)}, 内容: {goals_data}")
            return None, None, location_name_mapping

        # 提取候选无人机列表
        candidate_drones = planner_output.get("candidate_drones", [])
        if not isinstance(candidate_drones, list):
            logger.error(f"candidate_drones字段应该是列表类型，但收到了: {type(candidate_drones)}, 内容: {candidate_drones}")
            # 如果没有candidate_drones或格式错误，使用所有drones作为候选集合
            candidate_drones = list(drones_data.keys())
            logger.warning(f"使用所有无人机作为候选: {candidate_drones}")
        else:
            logger.info(f"使用候选无人机: {candidate_drones}")

        # 验证候选无人机是否都在drones_data中
        missing_drones = [drone for drone in candidate_drones if drone not in drones_data]
        if missing_drones:
            logger.error(f"候选无人机中有些不在drones数据中: {missing_drones}")
            # 过滤掉不存在的无人机
            candidate_drones = [drone for drone in candidate_drones if drone in drones_data]
            logger.warning(f"过滤后的候选无人机: {candidate_drones}")

        if not candidate_drones:
            logger.error("没有有效的候选无人机")
            return None, None, location_name_mapping

        # 确保所有引用的位置都在locations中
        locations_data, necessary_locations = _ensure_all_locations_exist(logger, planner_output, locations_data, candidate_drones)

        # 标准化位置名称，统一相同坐标的位置
        locations_data, location_name_mapping = _standardize_location_names(logger, locations_data)

        # 根据位置名称映射更新planner_output中的引用
        planner_output = _update_location_references_in_planner_output(logger, planner_output, location_name_mapping, candidate_drones)

        # 1. 计算距离矩阵
        locations = list(necessary_locations.keys())
        distance_matrix = {}

        for loc1 in locations:
            for loc2 in locations:
                if loc1 != loc2:
                    loc1_data = locations_data[loc1]
                    loc2_data = locations_data[loc2]

                    if not isinstance(loc1_data, dict) or not isinstance(loc2_data, dict):
                        logger.error(f"位置数据应该是字典类型: {loc1}={loc1_data}, {loc2}={loc2_data}")
                        continue

                    if 'x' not in loc1_data or 'y' not in loc1_data:
                        logger.error(f"位置 {loc1} 缺少x或y坐标: {loc1_data}")
                        continue

                    if 'x' not in loc2_data or 'y' not in loc2_data:
                        logger.error(f"位置 {loc2} 缺少x或y坐标: {loc2_data}")
                        continue

                    x1, y1 = loc1_data["x"], loc1_data["y"]
                    x2, y2 = loc2_data["x"], loc2_data["y"]
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    distance_matrix[(loc1, loc2)] = distance

        # 2. 构建init_params
        locations_dict = locations_data

        # 解析候选无人机位置（只处理候选无人机）
        drone_locations = {}
        for drone in candidate_drones:
            if drone not in drones_data:
                logger.error(f"候选无人机 {drone} 不在drones数据中")
                continue

            info = drones_data[drone]
            if not isinstance(info, dict):
                logger.error(f"无人机 {drone} 的信息应该是字典类型，但收到了: {type(info)}, 内容: {info}")
                continue

            current_location = info.get("current_location")
            resolved_location = _resolve_location_reference(logger, current_location, locations_dict)
            drone_locations[drone] = resolved_location

        # 解析包裹位置
        package_locations = {}
        for pkg, info in packages_data.items():
            if not isinstance(info, dict):
                logger.error(f"包裹 {pkg} 的信息应该是字典类型，但收到了: {type(info)}, 内容: {info}")
                continue

            current_location = info.get("current_location")
            if current_location is not None:
                resolved_location = _resolve_location_reference(logger, current_location, locations_dict)
                package_locations[pkg] = resolved_location
            else:
                package_locations[pkg] = "base"

        init_params = {
            "locations": locations,
            "distance_matrix": distance_matrix,
            "drone_locations": drone_locations,
            "package_locations": package_locations,
            "packages_on_drone": {pkg: info.get("on_drone")
                                    for pkg, info in packages_data.items()
                                    if isinstance(info, dict) and info.get("on_drone") is not None}
        }

        # 3. 构建goal_params
        tasks_data = goals_data.get("tasks", {})
        if not isinstance(tasks_data, dict):
            logger.error(f"goals.tasks字段应该是字典类型，但收到了: {type(tasks_data)}, 内容: {tasks_data}")
            tasks_data = {}

        goal_params = {
            "package_destinations": goals_data.get("deliveries", {}),
            "compute_locations": tasks_data.get("compute", []),
            "communicate_locations": tasks_data.get("communicate", []),
            "inspect_locations": tasks_data.get("inspect", [])
        }

        logger.info(f"转换完成: {len(locations)} 个地点, {len(init_params['drone_locations'])} 架候选无人机 (总共 {len(drones_data)} 架无人机)")
        logger.info(f"候选无人机: {list(init_params['drone_locations'].keys())}")
        if location_name_mapping:
            logger.info(f"位置名称标准化: {len(location_name_mapping)} 个位置被重命名")
        return init_params, goal_params, location_name_mapping

    except Exception as e:
        logger.error(f"规划器输出转换失败: {str(e)}")
        logger.error(f"规划器输出内容: {planner_output}")
        return None, None, location_name_mapping


def _parse_position_name(location_name):
    """
    解析position_x_y格式的位置名称

    Args:
        location_name: 位置名称字符串

    Returns:
        tuple: (x, y, z) 坐标，如果解析失败返回 (0, 0, 0)
    """
    if not isinstance(location_name, str):
        return (0, 0, 0)

    # 尝试解析position_x_y格式
    if location_name.startswith('position_'):
        parts = location_name[9:].split('_')  # 去掉'position_'前缀
        try:
            if len(parts) >= 2:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2]) if len(parts) >= 3 else 0
                return (x, y, z)
        except (ValueError, IndexError):
            pass

    # 如果解析失败，返回默认坐标
    return (0, 0, 0)


def _ensure_all_locations_exist(logger, planner_output, locations_data, candidate_drones):
    """
    确保所有引用的位置都在locations_data中存在

    Args:
        logger: 日志记录器
        planner_output: 规划器输出数据
        locations_data: 现有的位置数据字典
        candidate_drones: 候选无人机列表

    Returns:
        dict: 更新后的位置数据字典
    """
    # 创建locations_data的副本以避免修改原始数据
    updated_locations = locations_data.copy()
    necessary_locations = {}
    missing_locations = set()

    # 1. 收集候选无人机的current_location
    drones_data = planner_output.get("drones", {})
    for drone_id in candidate_drones:
        if drone_id in drones_data:
            drone_info = drones_data[drone_id]
            if isinstance(drone_info, dict):
                current_location = drone_info.get("current_location")
                if current_location and current_location not in updated_locations:
                    missing_locations.add(current_location)
                if current_location:
                    necessary_locations[current_location] = True

    # 2. 收集goals中tasks的所有位置
    goals_data = planner_output.get("goals", {})
    tasks_data = goals_data.get("tasks", {})
    if isinstance(tasks_data, dict):
        for task_type, locations_list in tasks_data.items():
            if isinstance(locations_list, list):
                for location in locations_list:
                    if isinstance(location, str) and location not in updated_locations:
                        missing_locations.add(location)
                    if isinstance(location, str):
                        necessary_locations[location] = True

    # 3. 收集deliveries中的目标位置
    deliveries_data = goals_data.get("deliveries", {})
    if isinstance(deliveries_data, dict):
        for package_id, destination in deliveries_data.items():
            if isinstance(destination, str) and destination not in updated_locations:
                missing_locations.add(destination)
            if isinstance(destination, str):
                necessary_locations[destination] = True

    # 4. 收集packages中的current_location
    packages_data = planner_output.get("packages", {})
    if isinstance(packages_data, dict):
        for package_id, package_info in packages_data.items():
            if isinstance(package_info, dict):
                current_location = package_info.get("current_location")
                if current_location and current_location not in updated_locations:
                    missing_locations.add(current_location)
                if current_location:
                    necessary_locations[current_location] = True

    # 5. 为缺失的位置添加坐标
    for location_name in missing_locations:
        x, y, z = _parse_position_name(location_name)
        updated_locations[location_name] = {"x": x, "y": y}
        logger.info(f"添加缺失位置: {location_name} -> ({x}, {y}, {z})")
    necessary_locations = {k: v for k, v in updated_locations.items() if k in necessary_locations}
    if missing_locations:
        logger.info(f"总共添加了 {len(missing_locations)} 个缺失位置")

    return updated_locations, necessary_locations


def _standardize_location_names(logger, locations_data):
    """
    标准化位置名称，将相同坐标的位置统一为position_x_y格式

    Args:
        logger: 日志记录器
        locations_data: 位置数据字典

    Returns:
        tuple: (标准化后的位置数据, 位置名称映射字典)
    """
    # 创建坐标到位置名称的映射
    coord_to_names = {}
    location_name_mapping = {}

    # 收集所有位置的坐标
    for location_name, location_info in locations_data.items():
        if isinstance(location_info, dict) and 'x' in location_info and 'y' in location_info and 'z' in location_info:
            x, y, z = location_info['x'], location_info['y'], location_info['z']
            coord_key = (x, y, z)

            if coord_key not in coord_to_names:
                coord_to_names[coord_key] = []
            coord_to_names[coord_key].append(location_name)

    
    standardized_locations = locations_data.copy()

    for (x, y, z), names in coord_to_names.items():
        # 生成标准化名称
        standard_name = f"position_{int(x)}_{int(y)}_{int(z)}"

        # 如果标准化名称已经在列表中，使用它；否则使用第一个名称作为标准
        if standard_name in names:
            canonical_name = standard_name
        else:
            canonical_name = names[0]
            # 如果第一个名称不是标准格式，重命名为标准格式
            if canonical_name != standard_name:
                standardized_locations[standard_name] = standardized_locations[canonical_name]
                del standardized_locations[canonical_name]
                location_name_mapping[canonical_name] = standard_name
                canonical_name = standard_name

        # 将其他名称映射到标准名称
        for name in names:
            if name != canonical_name:
                location_name_mapping[name] = canonical_name
                if name in standardized_locations:
                    del standardized_locations[name]

                logger.info(f"标准化坐标 ({x}, {y}, {z}) 的位置名称: {names} -> {canonical_name}")

    return standardized_locations, location_name_mapping


def _update_location_references_in_planner_output(logger, planner_output, location_name_mapping, candidate_drones):
    """
    根据位置名称映射更新planner_output中的位置引用

    Args:
        logger: 日志记录器
        planner_output: 规划器输出数据
        location_name_mapping: 位置名称映射字典
        candidate_drones: 候选无人机列表

    Returns:
        dict: 更新后的planner_output
    """
    if not location_name_mapping:
        return planner_output

    updated_output = planner_output.copy()

    # 更新drones中候选无人机的current_location
    drones_data = updated_output.get("drones", {})
    for drone_id in candidate_drones:
        if drone_id in drones_data:
            drone_info = drones_data[drone_id]
            if isinstance(drone_info, dict):
                current_location = drone_info.get("current_location")
                if current_location in location_name_mapping:
                    old_location = current_location
                    new_location = location_name_mapping[current_location]
                    drone_info["current_location"] = new_location
                    logger.info(f"更新无人机 {drone_id} 位置引用: {old_location} -> {new_location}")

    # 更新goals中tasks的位置引用
    goals_data = updated_output.get("goals", {})
    tasks_data = goals_data.get("tasks", {})
    if isinstance(tasks_data, dict):
        for task_type, locations_list in tasks_data.items():
            if isinstance(locations_list, list):
                for i, location in enumerate(locations_list):
                    if isinstance(location, str) and location in location_name_mapping:
                        old_location = location
                        new_location = location_name_mapping[location]
                        locations_list[i] = new_location
                        logger.info(f"更新任务 {task_type} 位置引用: {old_location} -> {new_location}")

    # 更新deliveries中的目标位置
    deliveries_data = goals_data.get("deliveries", {})
    if isinstance(deliveries_data, dict):
        for package_id, destination in deliveries_data.items():
            if isinstance(destination, str) and destination in location_name_mapping:
                old_destination = destination
                new_destination = location_name_mapping[destination]
                deliveries_data[package_id] = new_destination
                logger.info(f"更新包裹 {package_id} 目标位置: {old_destination} -> {new_destination}")

    # 更新packages中的current_location
    packages_data = updated_output.get("packages", {})
    if isinstance(packages_data, dict):
        for package_id, package_info in packages_data.items():
            if isinstance(package_info, dict):
                current_location = package_info.get("current_location")
                if current_location and current_location in location_name_mapping:
                    old_location = current_location
                    new_location = location_name_mapping[current_location]
                    package_info["current_location"] = new_location
                    logger.info(f"更新包裹 {package_id} 位置引用: {old_location} -> {new_location}")

    return updated_output


def _resolve_location_reference(logger, location_ref, locations_dict):
    """
    解析位置引用，可能是位置名称或坐标字典

    注意：'base' 在系统中表示默认的基地/起飞点位置，通常位于坐标原点附近。
    当无法确定具体位置时，系统会回退到使用 'base' 作为默认位置。

    Args:
        location_ref: 位置引用（字符串名称或坐标字典）
        locations_dict: 位置名称到坐标的映射

    Returns:
        位置名称字符串，无法解析时返回 'base'
    """
    if isinstance(location_ref, str):
        # 如果是字符串，直接返回
        return location_ref
    elif isinstance(location_ref, dict) and 'x' in location_ref and 'y' in location_ref and 'z' in location_ref:
        # 如果是坐标字典，找到最近的位置
        target_x, target_y, target_z = location_ref['x'], location_ref['y'], location_ref['z']
        min_distance = float('inf')
        closest_location = "base"  # 默认位置

        for loc_name, loc_coords in locations_dict.items():
            if isinstance(loc_coords, dict) and 'x' in loc_coords and 'y' in loc_coords and 'z' in loc_coords:
                distance = ((loc_coords['x'] - target_x) ** 2 + (loc_coords['y'] - target_y) ** 2 + (loc_coords['z'] - target_z) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_location = loc_name

        logger.info(f"将坐标 {location_ref} 映射到最近位置: {closest_location} (距离: {min_distance:.2f})")
        return closest_location
    else:
        # 其他情况返回默认位置
        logger.warning(f"无法解析位置引用: {location_ref}，使用默认位置 'base'")
        return "base"


class PDDLSolver:
    """TCCN PDDL求解器"""
    
    def __init__(self, domain_file: Optional[str] = None):
        """
        初始化PDDLSolver
        
        Args:
            domain_file: PDDL domain文件路径，如果为None则使用默认路径
        """
        if not UP_AVAILABLE:
            raise ImportError("unified_planning library not found. Please install with: pip install unified-planning[tamer]")
        
        # 禁用credits输出
        get_environment().credits_stream = None
        
        # 设置domain文件路径
        if domain_file is None:
            # 使用默认的domain文件
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            domain_file = os.path.join(current_dir, "data", "pddl", "domain", "tccn_domain.pddl")

        self.domain_file = domain_file
        if not os.path.exists(self.domain_file):
            raise FileNotFoundError(f"Domain file not found: {self.domain_file}")

        # 初始化PDDL reader
        self.reader = PDDLReader()
        self.last_problem_content = None
        self.last_solution_content = None

        print(f"✓ PDDLSolver initialized with domain: {self.domain_file}")

    def _normalize_pddl_name(self, name: str) -> str:
        """
        规范化PDDL名称，确保符合PDDL标识符规则

        Args:
            name: 原始名称

        Returns:
            规范化后的名称
        """
        import re

        # 移除或替换无效字符
        # PDDL标识符只能包含字母、数字和下划线
        normalized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

        # 确保不以数字开头
        if normalized and normalized[0].isdigit():
            normalized = 'loc_' + normalized

        # 确保不为空
        if not normalized:
            normalized = 'unnamed'

        return normalized.lower()

    def solve(self, init_params: Dict[str, Any], goal_params: Dict[str, Any],
              distance_threshold: float = 100.0, solver_name: str = "fast-downward", debug: bool = False,
              timeout: float = 30.0) -> Dict[str, Any]:
        """
        求解PDDL问题
        
        Args:
            init_params: 初始状态参数
                - locations: List[str] - 地点列表
                - distance_matrix: Dict[Tuple[str, str], float] - 距离矩阵
                - drone_locations: Dict[str, str] - 无人机当前位置
                - package_locations: Dict[str, str] - 包裹当前位置
                - packages_on_drone: Dict[str, str] - 包裹是否在无人机上
            goal_params: 目标参数
                - package_destinations: Dict[str, str] - 包裹应该到达的地点
                - compute_locations: List[str] - 需要计算的地点
                - communicate_locations: List[str] - 需要通信的地点
                - inspect_locations: List[str] - 需要巡检的地点
                - drone_final_locations: Dict[str, str] - 无人机最终位置（可选）
            distance_threshold: 距离阈值，小于此值的地点视为connected
            solver_name: 求解器名称
            
        Returns:
            Dict包含求解结果
        """
        try:
            # 1. 构建连通图
            connected_pairs = self._build_connectivity_graph(
                init_params["locations"], 
                init_params["distance_matrix"], 
                distance_threshold
            )
            
            # 2. 创建临时problem文件
            problem_content = self._generate_problem_content(
                init_params, goal_params, connected_pairs
            )
            self.last_problem_content = problem_content
            # 3. 写入临时文件并解析（使用唯一文件名以支持多进程）
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', prefix='tccn_problem_', delete=False) as f:
                temp_problem_file = f.name
                f.write(problem_content)

            # 4. 解析problem
            problem = self.reader.parse_problem(str(self.domain_file), temp_problem_file)
            
            print(f"✓ Problem parsed successfully")
            print(f"  Objects: {len(problem.all_objects)}")
            print(f"  Actions: {len(problem.actions)}")
            print(f"  Goals: {len(problem.goals)}")
            
            # 5. 求解 (带超时)
            solver_params = {'heuristic': 'hadd'} if solver_name == 'tamer' else {}
            with OneshotPlanner(name=solver_name, params=solver_params, problem_kind=problem.kind) as planner:
                print(f"Using solver: {planner.name} (timeout={timeout}s)")
                result = planner.solve(problem, timeout=timeout)
                
                if result.plan is not None:
                    plan = result.plan
                    print(f"✓ Solution found with {len(plan.actions)} actions")
                    
                    # 解析解决方案
                    solution_steps = []
                    for i, action in enumerate(plan.actions, 1):
                        solution_steps.append({
                            "step": i,
                            "action": str(action),
                            "action_name": action.action.name,
                            "parameters": [str(param) for param in action.actual_parameters]
                        })
                    self.last_solution_content = solution_steps
                    return {
                        "success": True,
                        "solution": solution_steps,
                        "total_steps": len(plan.actions),
                        "solver_used": planner.name,
                        "problem_stats": {
                            "objects": len(problem.all_objects),
                            "actions": len(problem.actions),
                            "goals": len(problem.goals)
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": f"No solution found. Status: {result.status}",
                        "solver_used": planner.name
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # 清理临时文件（调试模式下保留）
            if not debug and 'temp_problem_file' in locals() and os.path.exists(temp_problem_file):
                os.remove(temp_problem_file)
            elif debug and 'temp_problem_file' in locals() and os.path.exists(temp_problem_file):
                print(f"🔍 Debug mode: Temporary problem file saved at: {temp_problem_file}")
    
    def _build_connectivity_graph(self, locations: List[str], 
                                 distance_matrix: Dict[Tuple[str, str], float],
                                 threshold: float) -> List[Tuple[str, str]]:
        """
        基于距离矩阵和阈值构建连通图
        
        使用最小生成树确保所有节点连通，同时添加距离小于阈值的边
        """
        # 创建图
        G = nx.Graph()
        G.add_nodes_from(locations)
        
        # 添加所有边（用于构建最小生成树）
        edges_with_weights = []
        for (loc1, loc2), distance in distance_matrix.items():
            if loc1 != loc2:
                edges_with_weights.append((loc1, loc2, distance))
        
        # 构建最小生成树确保连通性
        G.add_weighted_edges_from(edges_with_weights)
        mst = nx.minimum_spanning_tree(G)
        
        # 获取MST的边
        connected_pairs = list(mst.edges())
        
        # 添加距离小于阈值的额外边
        for (loc1, loc2), distance in distance_matrix.items():
            if loc1 != loc2 and distance < threshold:
                if (loc1, loc2) not in connected_pairs and (loc2, loc1) not in connected_pairs:
                    connected_pairs.append((loc1, loc2))
        
        # 确保双向连接
        bidirectional_pairs = []
        for loc1, loc2 in connected_pairs:
            bidirectional_pairs.extend([(loc1, loc2), (loc2, loc1)])
        
        print(f"✓ Built connectivity graph with {len(bidirectional_pairs)} connections")
        return bidirectional_pairs

    def _generate_problem_content(self, init_params: Dict[str, Any],
                                 goal_params: Dict[str, Any],
                                 connected_pairs: List[Tuple[str, str]]) -> str:
        """
        生成PDDL problem文件内容
        """
        # 提取参数
        locations = init_params["locations"]
        drone_locations = init_params["drone_locations"]
        package_locations = init_params["package_locations"]
        packages_on_drone = init_params.get("packages_on_drone", {})

        # 提取所有无人机和包裹，并规范化名称
        drones = [self._normalize_pddl_name(drone) for drone in drone_locations.keys()]
        packages = [self._normalize_pddl_name(package) for package in package_locations.keys()]

        # 规范化位置名称
        normalized_locations = [self._normalize_pddl_name(loc) for loc in locations]

        # 开始构建problem内容
        content = f"""(define (problem tccn-auto-problem)
    (:domain tccn-planning)

    (:objects"""

        # 添加无人机对象（如果有的话）
        if drones:
            content += f"""
        ;; 无人机
        {' '.join(drones)} - drone"""

        # 添加地点对象
        content += f"""
        ;; 地点
        {' '.join(locations)} - location"""

        # 添加包裹对象（如果有的话）
        if packages:
            content += f"""
        ;; 包裹
        {' '.join(packages)} - package"""

        content += """
    )

    (:init
"""

        # 添加无人机位置
        for drone, location in drone_locations.items():
            content += f"        (at {drone} {location})\n"

        # 收集携带package的drone
        drones_with_packages = set()

        # 添加包裹位置和状态
        for package, location in package_locations.items():
            # 规范化位置名称，处理None值
            normalized_location = self._normalize_pddl_name(str(location)) if location is not None else "unknown_location"

            if package in packages_on_drone:
                # 包裹在无人机上
                drone = packages_on_drone[package]
                drones_with_packages.add(drone)  # 记录携带package的drone
                content += f"        (on-drone {package} {drone})\n"
                # 如果包裹在无人机上，位置应该是无人机的位置
                drone_location = drone_locations.get(drone, "base")
                normalized_drone_location = self._normalize_pddl_name(str(drone_location))
                content += f"        (package-at {package} {normalized_drone_location})\n"
            else:
                # 包裹在地面，但要确保位置有效
                if location is not None and str(location).lower() != 'none':
                    content += f"        (package-at {package} {normalized_location})\n"
                else:
                    # 如果位置无效，假设在base
                    content += f"        (package-at {package} base)\n"

        # 添加drone-free状态（只有没有携带package的drone才是free的）
        for drone in drone_locations.keys():
            if drone not in drones_with_packages:
                content += f"        (drone-free {drone})\n"

        # 添加地点可用性
        for location in locations:
            content += f"        (available {location})\n"

        # 添加连通关系
        for loc1, loc2 in connected_pairs:
            content += f"        (connected {loc1} {loc2})\n"

        content += "    )\n\n    (:goal\n        (and\n"

        # 添加目标（避免重复）
        added_goals = set()

        # 包裹配送目标
        if "package_destinations" in goal_params:
            for package, destination in goal_params["package_destinations"].items():
                # 添加两个约束：
                # 1. 包裹必须在目标位置
                package_at_goal = f"(package-at {package} {destination})"
                if package_at_goal not in added_goals:
                    content += f"            {package_at_goal}\n"
                    added_goals.add(package_at_goal)

                # 2. 该包裹在该位置已完成配送（修改：绑定到具体的package和location）
                delivered_goal = f"(isDelivered {package} {destination})"
                if delivered_goal not in added_goals:
                    content += f"            {delivered_goal}\n"
                    added_goals.add(delivered_goal)

        # 计算任务目标
        if "compute_locations" in goal_params:
            for location in goal_params["compute_locations"]:
                goal = f"(isComputed {location})"
                if goal not in added_goals:
                    content += f"            {goal}\n"
                    added_goals.add(goal)

        # 通信任务目标
        if "communicate_locations" in goal_params:
            for location in goal_params["communicate_locations"]:
                goal = f"(isCommunicated {location})"
                if goal not in added_goals:
                    content += f"            {goal}\n"
                    added_goals.add(goal)

        # 巡检任务目标
        if "inspect_locations" in goal_params:
            for location in goal_params["inspect_locations"]:
                goal = f"(isInspected {location})"
                if goal not in added_goals:
                    content += f"            {goal}\n"
                    added_goals.add(goal)

        # 无人机最终位置目标（可选）
        if "drone_final_locations" in goal_params:
            for drone, location in goal_params["drone_final_locations"].items():
                content += f"            (at {drone} {location})\n"

        content += "        )\n    )\n)"
        
        return content
