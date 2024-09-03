from .AbstractAlgo import Algo
from utils.astar_shortest_path import calculate_shortest_path
from utils.bfs_search_path_junction import get_full_path
import numpy as np
import networkx as nx
import os
import copy


class CGGAlgo(Algo):
    def __init__(self, algo_config):
        self.nodes_num = algo_config['nodes_num']
        self.pgm_map_matrix = algo_config['pgm_map_matrix']
        self.node_pos_matrix = algo_config['node_pos_matrix']
        self.map_adj_matrix = algo_config['map_adj_matrix']
        self.map_name = algo_config['map_name']
        self.current_node = 0
        self.cgg_path = None
        self.cgg_index = 0
        # Initialize precomputed paths if available
        self.precomputed_paths = algo_config['precomputed_paths'].copy()
        self.initialize_cgg_path()

    def initialize_cgg_path(self):
        cgg_path_filename = f'./maps/{self.map_name}/{self.map_name}_cgg_path.npy'

        if os.path.isfile(cgg_path_filename):
            self.cgg_path = np.load(cgg_path_filename)
            self.shift_path_start_node()
        else:
            self.cgg_path = self.calculate_cgg_path()
            np.save(cgg_path_filename, self.cgg_path)
            self.shift_path_start_node()

    def calculate_cgg_path(self):
        graph = nx.from_numpy_array(self.map_adj_matrix, create_using=nx.MultiGraph)
        try:
            return nx.approximation.greedy_tsp(graph, source=self.current_node)
        except nx.NetworkXError:
            print('The graph does not contain a Hamiltonian path, using TSP approximation instead')
            return nx.approximation.traveling_salesman_problem(graph)

    def shift_path_start_node(self):
        start_node_location = np.where(self.cgg_path == self.current_node)[0][0]
        self.cgg_path = np.roll(self.cgg_path, -start_node_location)

    def determine_goal(self, robot_id, current_node):
        if self.cgg_index >= len(self.cgg_path) - 1:
            self.cgg_index = 0  # Loop back to the start of the path
        else:
            self.cgg_index += 1
        goal_node = self.cgg_path[self.cgg_index]
        return goal_node

    def calculate_next_path(self, robot_id, current_node):
        self.current_node = current_node
        goal_node = self.determine_goal(robot_id, current_node)

        # Check if precomputed paths are available
        if self.precomputed_paths is not None:
            path = copy.deepcopy(self.precomputed_paths.get((current_node, goal_node), 'Not found'))
            if path == False:
                path = get_full_path(copy.deepcopy(self.precomputed_paths), copy.deepcopy(self.map_adj_matrix),
                                     current_node, goal_node)
            # Use precomputed paths if available
            return path
        else:
            # Fallback to calculating the shortest path if precomputed paths are not available
            start = self.node_pos_matrix[current_node]
            goal = self.node_pos_matrix[goal_node]
            print(f'{start} {goal} not existed in precomputed paths, something is wrong')
            return calculate_shortest_path(self.pgm_map_matrix, start, goal)
