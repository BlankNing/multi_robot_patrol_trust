from .AbstractAlgo import Algo
from utils.astar_shortest_path import calculate_shortest_path
from utils.bfs_search_path_junction import get_full_path
import copy

class PartitionAlgo(Algo):
    def __init__(self, algo_config):
        self.algo_config = algo_config
        self.nodes_num = algo_config['nodes_num']
        self.robots_num = algo_config['robots_num']
        self.pgm_map_matrix = algo_config['pgm_map_matrix']
        self.node_pos_matrix = algo_config['node_pos_matrix']
        self.map_adj_matrix = algo_config['map_adj_matrix']
        self.precomputed_paths = copy.deepcopy(algo_config['precomputed_paths'])

    def determine_goal(self, robot_id, current_node):
        charge_num = self.nodes_num/self.robots_num
        goal_node = int((current_node + 1) % charge_num + robot_id * charge_num)
        return self.node_pos_matrix[goal_node]

    def determine_goal_node(self, robot_id, current_node):
        charge_num = self.nodes_num/self.robots_num
        goal_node = int((current_node + 1) % charge_num + robot_id * charge_num)
        return goal_node

    def calculate_next_path(self, robot_id, current_node):

        if self.precomputed_paths is not None:
            goal_node = self.determine_goal_node(robot_id, current_node)
            path = copy.deepcopy(self.precomputed_paths.get((current_node, goal_node), 'Not found'))
            assert path != []
            if path == False:
                path = get_full_path(copy.deepcopy(self.precomputed_paths), copy.deepcopy(self.map_adj_matrix), current_node, goal_node)
            return path

        else:
            start = self.node_pos_matrix[current_node]  # (,) pos
            goal = self.determine_goal(robot_id, current_node)
            print(f'{start} {goal} not existed in precomputed paths, something is wrong')
            return calculate_shortest_path(self.pgm_map_matrix, start, goal)

