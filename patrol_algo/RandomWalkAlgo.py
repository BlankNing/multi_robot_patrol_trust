from .AbstractAlgo import Algo
from utils.astar_shortest_path import calculate_shortest_path
from utils.bfs_search_path_junction import get_full_path
import random
import copy


class RandomWalkAlgo(Algo):
    def __init__(self, algo_config):
        self.nodes_num = algo_config['nodes_num']
        self.robots_num = algo_config['robots_num']
        self.pgm_map_matrix = algo_config['pgm_map_matrix']
        self.node_pos_matrix = algo_config['node_pos_matrix']
        self.map_adj_matrix = algo_config['map_adj_matrix']
        # Initialize precomputed paths if available
        self.precomputed_paths = algo_config['precomputed_paths'].copy()

    def determine_goal(self, robot_id, current_node):
        # Randomly select a goal node
        goal_node = random.randint(0, self.nodes_num - 1)
        return goal_node

    def calculate_next_path(self, robot_id, current_node):
        goal_node = self.determine_goal(robot_id, current_node)
        while current_node == goal_node:
            goal_node = self.determine_goal(robot_id, current_node)
        # Check if precomputed paths are available
        if self.precomputed_paths is not None:
            # Use precomputed paths if available
            path = copy.deepcopy(self.precomputed_paths.get((current_node, goal_node), 'Not found'))
            while not path:
                goal_node = self.determine_goal(robot_id, current_node)
                path = copy.deepcopy(self.precomputed_paths.get((current_node, goal_node), 'Not found'))
                if path == False:
                    path = get_full_path(copy.deepcopy(self.precomputed_paths), copy.deepcopy(self.map_adj_matrix),
                                         current_node, goal_node)
            return path
        else:
            # Fallback to calculating the shortest path if precomputed paths are not available
            start = self.node_pos_matrix[current_node]
            goal = self.node_pos_matrix[goal_node]
            print(f'{start} {goal} not existed in precomputed paths, something is wrong')
            return calculate_shortest_path(self.pgm_map_matrix, start, goal)
