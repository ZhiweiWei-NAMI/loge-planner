#!/usr/bin/env python3
"""
Simple PDDL Problem Solver using pyperplan

This script attempts to solve PDDL problems using pyperplan.
"""

import argparse
import sys
from pathlib import Path

def solve_with_pyperplan(domain_file, problem_file):
    """Attempt to solve using pyperplan."""
    print(f"Domain file: {domain_file}")
    print(f"Problem file: {problem_file}")
    
    try:
        from pyperplan import planner
        from pyperplan.heuristics import blind_heuristic, add_heuristic
        
        print("Parsing domain and problem...")
        domain = planner._parse(str(domain_file))
        problem = planner._parse(str(problem_file))

        print("Domain parsed successfully")
        print(f"  Actions: {len(domain.actions)}")
        print(f"  Predicates: {len(domain.predicates)}")

        print("Problem parsed successfully")
        print(f"  Objects: {len(problem.objects)}")
        print(f"  Initial state facts: {len(problem.initial_state)}")

        print("Grounding the problem...")
        task = planner._ground(domain, problem)
        print(f"Grounded task has {len(task.operators)} operators")

        print("Searching for solution with blind heuristic...")
        solution = planner._search(task, blind_heuristic, 'astar')
        
        if solution:
            print(f"✓ Solution found with {len(solution)} steps!")
            print("\nSolution plan:")
            for i, action in enumerate(solution, 1):
                print(f"{i:2d}. {action}")
            return True, solution
        else:
            print("✗ No solution found with blind heuristic")
            
            print("Trying with add heuristic...")
            solution = planner._search(task, add_heuristic, 'astar')
            
            if solution:
                print(f"✓ Solution found with add heuristic with {len(solution)} steps!")
                print("\nSolution plan:")
                for i, action in enumerate(solution, 1):
                    print(f"{i:2d}. {action}")
                return True, solution
            else:
                print("✗ No solution found with add heuristic")
                return False, None
            
    except ImportError:
        print("pyperplan not available")
        return False, None
    except Exception as e:
        print(f"Error during solving: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Solve PDDL problems with pyperplan")
    parser.add_argument("--domain", type=Path, required=True, help="Path to domain file")
    parser.add_argument("--problem", type=Path, required=True, help="Path to problem file")
    args = parser.parse_args()
    
    if not args.domain.exists():
        print(f"Domain file not found: {args.domain}")
        sys.exit(1)
    
    if not args.problem.exists():
        print(f"Problem file not found: {args.problem}")
        sys.exit(1)
    
    success, solution = solve_with_pyperplan(args.domain, args.problem)
    
    if success:
        print("\n=== SUCCESS ===")
        print("A solution was found for the multi-agent problem!")
    else:
        print("\n=== NO SOLUTION ===")
        print("No solution found. This could be due to:")
        print("1. The problem is unsolvable")
        print("2. The search space is too large")
        print("3. Durative actions are not supported by this planner")
        print("4. There are syntax errors in the PDDL files")

if __name__ == "__main__":
    main()
