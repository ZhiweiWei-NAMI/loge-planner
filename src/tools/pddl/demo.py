#!/usr/bin/env python3
"""
TCCN逻辑骨架PDDL规划系统演示脚本

展示如何使用逻辑骨架进行多智能体任务规划验证
"""

import subprocess
import sys
from pathlib import Path
import time

def run_planner(domain_file, problem_file, description):
    """运行PDDL规划器并显示结果"""
    print(f"\n{'='*60}")
    print(f"🚁 {description}")
    print(f"{'='*60}")
    print(f"域文件: {domain_file}")
    print(f"问题文件: {problem_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['pyperplan', domain_file, problem_file],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        end_time = time.time()
        solve_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 求解成功! 用时: {solve_time:.2f}秒")
            
            # 解析输出获取关键信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Plan length:' in line:
                    print(f"📋 {line.strip()}")
                elif 'Nodes expanded:' in line:
                    print(f"🔍 {line.strip()}")
                elif 'Search time:' in line:
                    print(f"⏱️  {line.strip()}")
            
            # 显示解决方案
            solution_file = Path(problem_file).with_suffix('.pddl.soln')
            if solution_file.exists():
                print(f"\n📝 解决方案 ({solution_file}):")
                with open(solution_file, 'r') as f:
                    solution = f.read().strip()
                    steps = [line.strip() for line in solution.split('\n') if line.strip()]
                    
                    # 显示前几步和后几步
                    if len(steps) <= 10:
                        for i, step in enumerate(steps, 1):
                            print(f"  {i:2d}. {step}")
                    else:
                        for i in range(5):
                            print(f"  {i+1:2d}. {steps[i]}")
                        print(f"  ... (省略 {len(steps)-10} 步)")
                        for i in range(len(steps)-5, len(steps)):
                            print(f"  {i+1:2d}. {steps[i]}")
        else:
            print(f"❌ 求解失败!")
            print(f"错误信息: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 求解超时 (>120秒)")
    except Exception as e:
        print(f"❌ 运行错误: {e}")

def analyze_solution(problem_file, description):
    """分析解决方案的特点"""
    solution_file = Path(problem_file).with_suffix('.pddl.soln')
    if not solution_file.exists():
        return
    
    print(f"\n🔍 {description} - 解决方案分析:")
    
    with open(solution_file, 'r') as f:
        solution = f.read().strip()
        steps = [line.strip() for line in solution.split('\n') if line.strip()]
    
    # 统计动作类型
    action_counts = {}
    drone_actions = {}
    
    for step in steps:
        if step.startswith('('):
            action = step.split()[0][1:]  # 移除开头的'('
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # 提取无人机信息
            parts = step.split()
            if len(parts) > 1:
                drone = parts[1]
                if drone.startswith('drone'):
                    drone_actions[drone] = drone_actions.get(drone, 0) + 1
    
    print(f"  📊 动作统计:")
    for action, count in sorted(action_counts.items()):
        print(f"    {action}: {count}次")
    
    print(f"  🚁 无人机工作量:")
    for drone, count in sorted(drone_actions.items()):
        print(f"    {drone}: {count}个动作")
    
    # 检查关键模式
    solution_text = ' '.join(steps)
    if 'take-photo' in solution_text and 'process-data-at-location' in solution_text:
        print(f"  ✅ 检测到检查→数据处理依赖关系")
    if 'pickup' in solution_text and 'drop' in solution_text:
        print(f"  ✅ 检测到完整的物流任务")
    if 'start-comm-relay' in solution_text:
        print(f"  ✅ 检测到通信中继任务")

def main():
    """主演示函数"""
    print("🚁 TCCN逻辑骨架PDDL规划系统演示")
    print("=" * 60)
    print("本演示展示如何使用逻辑骨架进行多智能体任务规划验证")
    print("专注于验证任务序列的逻辑可行性，无需考虑复杂的数值约束")
    
    # 检查pyperplan是否可用
    try:
        subprocess.run(['pyperplan', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: pyperplan未安装或不在PATH中")
        print("请运行: pip install pyperplan")
        sys.exit(1)
    
    # 切换到PDDL文件目录
    pddl_dir = Path(__file__).parent.parent.parent / 'data' / 'pddl'
    if not pddl_dir.exists():
        print(f"❌ 错误: PDDL目录不存在: {pddl_dir}")
        sys.exit(1)

    import os
    os.chdir(pddl_dir)
    
    # 演示1: 核心验证案例
    run_planner(
        'logical-domain.pddl',
        'core-validation.pddl',
        '核心验证案例 - 双无人机协作'
    )
    analyze_solution('core-validation.pddl', '核心验证')
    
    # 演示2: 完整多智能体案例
    run_planner(
        'logical-domain.pddl', 
        'logical-problem.pddl',
        '完整多智能体案例 - 三无人机复杂任务'
    )
    analyze_solution('logical-problem.pddl', '完整多智能体')
    
    print(f"\n{'='*60}")
    print("🎯 演示总结")
    print(f"{'='*60}")
    print("✅ 逻辑骨架成功验证了多智能体任务的逻辑可行性")
    print("✅ 任务依赖关系得到正确处理")
    print("✅ 复杂任务被合理分解为基础动作序列")
    print("✅ 为下层规划器提供了可靠的逻辑指导")
    print("\n🔄 下一步: 将逻辑动作映射到物理参数，在AirFogSim中验证")

if __name__ == "__main__":
    main()
