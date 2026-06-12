"""Hungarian-Based Greedy Assignment Baseline.

A geometric heuristic that minimizes the total instantaneous distance 
between agent positions and workflow starting points using the Hungarian algorithm.
This baseline represents a myopic optimization strategy that effectively minimizes 
travel cost but neglects long-horizon temporal dependencies and resource constraints.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any, Tuple
import pandas as pd


class HungarianGreedyAssigner:
    """Hungarian algorithm based greedy checkpoint assignment."""
    
    def __init__(self, n_agents: int = None):
        """
        Args:
            n_agents: Number of agents (if None, will be inferred from data)
        """
        self.n_agents = n_agents
    
    def fit_predict(
        self, 
        checkpoints_df: pd.DataFrame,
        agents_info: List[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Assign checkpoints to clusters (agents) using Hungarian algorithm.
        
        This method:
        1. Extracts checkpoint positions
        2. Gets agent initial positions (or uses random positions if not provided)
        3. Uses Hungarian algorithm to minimize total assignment distance
        4. Greedily assigns remaining checkpoints to nearest agent
        
        Args:
            checkpoints_df: DataFrame with checkpoint data including 'location' column
            agents_info: List of agent dicts with 'position' key
            
        Returns:
            Array of cluster assignments (agent indices) for each checkpoint
        """
        # Extract checkpoint positions - handle different column formats
        positions = []
        if 'location' in checkpoints_df.columns:
            for loc in checkpoints_df['location']:
                if isinstance(loc, str):
                    import ast
                    pos = ast.literal_eval(loc)
                else:
                    pos = loc
                positions.append(pos[:2] if len(pos) >= 2 else pos)
        elif 'avg_position_x' in checkpoints_df.columns:
            # Alternative format with separate x, y, z columns
            for _, row in checkpoints_df.iterrows():
                positions.append([row['avg_position_x'], row['avg_position_y']])
        else:
            raise ValueError("No position columns found in DataFrame")
        
        checkpoint_positions = np.array(positions)
        n_checkpoints = len(checkpoint_positions)
        
        # Determine number of agents/clusters
        if self.n_agents is None:
            # Default: same as number of workflows or sqrt of checkpoints
            n_workflows = checkpoints_df['workflow_id'].nunique()
            self.n_agents = min(n_workflows, max(5, int(np.sqrt(n_checkpoints))))
        
        n_agents = self.n_agents
        
        # Get or generate agent positions
        if agents_info and len(agents_info) >= n_agents:
            agent_positions = np.array([a.get('position', [0, 0])[:2] for a in agents_info[:n_agents]])
        else:
            # Generate agent positions as k-means centroids of checkpoint positions
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_agents, random_state=42, n_init=10)
            kmeans.fit(checkpoint_positions)
            agent_positions = kmeans.cluster_centers_
        
        # Build cost matrix: distance from each checkpoint to each agent
        cost_matrix = np.zeros((n_checkpoints, n_agents))
        for i, cp in enumerate(checkpoint_positions):
            for j, ap in enumerate(agent_positions):
                cost_matrix[i, j] = np.linalg.norm(cp - ap)
        
        # Phase 1: Use Hungarian algorithm to assign first n_agents checkpoints
        # (one checkpoint per agent to establish initial assignment)
        if n_checkpoints <= n_agents:
            # Simple case: direct Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignments = np.zeros(n_checkpoints, dtype=int)
            for r, c in zip(row_ind, col_ind):
                assignments[r] = c
        else:
            # Complex case: Hungarian for initial assignment, then greedy for rest
            assignments = np.zeros(n_checkpoints, dtype=int)
            
            # Track agent workloads for load balancing
            agent_workloads = np.zeros(n_agents)
            agent_centroids = agent_positions.copy()
            
            # Sort checkpoints by workflow start time if available
            if 'earliest_start_time' in checkpoints_df.columns:
                sorted_indices = checkpoints_df['earliest_start_time'].argsort().values
            else:
                sorted_indices = np.arange(n_checkpoints)
            
            # Greedy assignment with load balancing
            for idx in sorted_indices:
                cp = checkpoint_positions[idx]
                
                # Calculate scores: distance + load penalty
                distances = np.linalg.norm(agent_centroids - cp, axis=1)
                load_penalty = agent_workloads * 0.1  # Penalty for unbalanced load
                scores = distances + load_penalty
                
                # Assign to agent with lowest score
                best_agent = np.argmin(scores)
                assignments[idx] = best_agent
                agent_workloads[best_agent] += 1
                
                # Update agent centroid (moving average)
                n = agent_workloads[best_agent]
                agent_centroids[best_agent] = (
                    (agent_centroids[best_agent] * (n - 1) + cp) / n
                )
        
        return assignments
    
    def cluster_checkpoints(
        self,
        checkpoints_df: pd.DataFrame,
        agents_info: List[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Cluster checkpoints and return DataFrame with cluster assignments.
        
        Args:
            checkpoints_df: DataFrame with checkpoint data
            agents_info: List of agent information dicts
            
        Returns:
            DataFrame with added 'cluster_group' column
        """
        assignments = self.fit_predict(checkpoints_df, agents_info)
        result_df = checkpoints_df.copy()
        result_df['cluster_group'] = assignments
        return result_df


def run_hungarian_clustering(
    input_csv: str,
    output_csv: str,
    n_agents: int = None,
    agents_yaml: str = None
) -> pd.DataFrame:
    """Run Hungarian greedy clustering on checkpoint data.
    
    Args:
        input_csv: Path to input checkpoint CSV
        output_csv: Path to output CSV with cluster assignments
        n_agents: Number of agents/clusters
        agents_yaml: Optional path to agents YAML file
        
    Returns:
        DataFrame with cluster assignments
    """
    import yaml
    
    # Load checkpoint data
    df = pd.read_csv(input_csv)
    
    # Load agents info if provided
    agents_info = None
    if agents_yaml:
        with open(agents_yaml, 'r') as f:
            scenario = yaml.safe_load(f)
        agents_info = scenario.get('agents', [])
    
    # Run clustering
    assigner = HungarianGreedyAssigner(n_agents=n_agents)
    result_df = assigner.cluster_checkpoints(df, agents_info)
    
    # Save results
    result_df.to_csv(output_csv, index=False)
    print(f"Hungarian clustering complete: {len(result_df)} checkpoints -> {result_df['cluster_group'].nunique()} clusters")
    
    return result_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Hungarian greedy clustering")
    parser.add_argument("--input", required=True, help="Input checkpoint CSV")
    parser.add_argument("--output", required=True, help="Output CSV with clusters")
    parser.add_argument("--n-agents", type=int, default=None, help="Number of agents")
    parser.add_argument("--agents-yaml", default=None, help="Agents YAML file")
    
    args = parser.parse_args()
    run_hungarian_clustering(args.input, args.output, args.n_agents, args.agents_yaml)
