#!/usr/bin/env python3
"""
PDDL Problem Solver and Validator

This script validates PDDL domain and problem files, then attempts to solve them
using available planners.
"""

import argparse
import sys
from pathlib import Path
from pddl import parse_domain, parse_problem
from pddl.formatter import domain_to_string, problem_to_string

def validate_pddl_files(domain_file, problem_file):
    """Validate PDDL domain and problem files."""
    print(f"Validating domain file: {domain_file}")
    try:
        with open(domain_file, 'r') as f:
            domain_text = f.read()
        domain = parse_domain(domain_text)
        print("✓ Domain file is valid")
    except Exception as e:
        print(f"✗ Domain file validation failed: {e}")
        return False, None, None

    print(f"Validating problem file: {problem_file}")
    try:
        with open(problem_file, 'r') as f:
            problem_text = f.read()
        problem = parse_problem(problem_text)
        print("✓ Problem file is valid")
    except Exception as e:
        print(f"✗ Problem file validation failed: {e}")
        return False, domain, None

    return True, domain, problem

def analyze_problem(domain, problem):
    """Analyze the PDDL problem and provide insights."""
    print("\n=== Problem Analysis ===")
    print(f"Domain: {domain.name}")
    print(f"Problem: {problem.name}")
    print(f"Objects: {len(problem.objects)}")
    print(f"Initial facts: {len(problem.init)}")
    print(f"Goal conditions: {len(problem.goal.operands) if hasattr(problem.goal, 'operands') else 1}")
    print(f"Actions in domain: {len(domain.actions)}")
    
    print("\nObjects by type:")
    objects_by_type = {}
    for obj in problem.objects:
        obj_type = str(obj.type_tags[0]) if obj.type_tags else "object"
        if obj_type not in objects_by_type:
            objects_by_type[obj_type] = []
        objects_by_type[obj_type].append(str(obj))
    
    for obj_type, objs in objects_by_type.items():
        print(f"  {obj_type}: {', '.join(objs)}")
    
    print("\nActions available:")
    for action in domain.actions:
        print(f"  - {action.name}")

def solve_with_pyperplan(domain_file, problem_file):
    """Attempt to solve using pyperplan."""
    print("\n=== Attempting to solve with pyperplan ===")
    try:
        import pyperplan
        from pyperplan.planner import _parse, _ground, _search
        from pyperplan.heuristics import blind_heuristic
        
        # Parse domain and problem
        domain = _parse(str(domain_file))
        problem = _parse(str(problem_file))
        
        print("Grounding the problem...")
        task = _ground(domain, problem)
        
        print("Searching for solution...")
        solution = _search(task, blind_heuristic, 'astar')
        
        if solution:
            print(f"✓ Solution found with {len(solution)} steps!")
            print("\nSolution plan:")
            for i, action in enumerate(solution, 1):
                print(f"{i:2d}. {action}")
            return True, solution
        else:
            print("✗ No solution found")
            return False, None
            
    except ImportError:
        print("pyperplan not available")
        return False, None
    except Exception as e:
        print(f"Error during solving: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Validate and solve PDDL problems")
    parser.add_argument("--domain", type=Path, required=True, help="Path to domain file")
    parser.add_argument("--problem", type=Path, required=True, help="Path to problem file")
    parser.add_argument("--solve", action="store_true", help="Attempt to solve the problem")
    args = parser.parse_args()
    
    if not args.domain.exists():
        print(f"Domain file not found: {args.domain}")
        sys.exit(1)
    
    if not args.problem.exists():
        print(f"Problem file not found: {args.problem}")
        sys.exit(1)
    
    # Validate files
    valid, domain, problem = validate_pddl_files(args.domain, args.problem)
    
    if not valid:
        print("Validation failed. Please fix the PDDL files.")
        sys.exit(1)
    
    # Analyze problem
    analyze_problem(domain, problem)
    
    # Attempt to solve if requested
    if args.solve:
        success, solution = solve_with_pyperplan(args.domain, args.problem)
        if not success:
            print("\nTrying alternative approach...")
            print("Note: This problem involves durative actions which may require")
            print("specialized temporal planners like OPTIC, TFD, or POPF.")
    
    print("\n=== Summary ===")
    print("✓ PDDL files are syntactically valid")
    if args.solve:
        if success:
            print("✓ Solution found")
        else:
            print("✗ No solution found with basic planner")
            print("  Consider using temporal planners for durative actions")

if __name__ == "__main__":
    main()
