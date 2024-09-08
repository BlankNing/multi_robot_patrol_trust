from .AbstractAlgo import Algo
from utils.astar_shortest_path import calculate_shortest_path
from utils.bfs_search_path_junction import get_full_path
import numpy as np
import math
import random
import copy

class SEBSAlgo(Algo):
    def __init__(self, algo_config):
        """
        Initialize SEBSAlgo with given configuration.

        Parameters:
        - algo_config: Dictionary containing algorithm configuration including map matrices, node positions, etc.
        """
        self.nodes_num = algo_config['nodes_num']
        self.robots_num = algo_config['robots_num']
        self.pgm_map_matrix = algo_config['pgm_map_matrix']
        self.node_pos_matrix = algo_config['node_pos_matrix']
        self.map_adj_matrix = algo_config['map_adj_matrix']
        self.neighbour_matrix = algo_config['neighbour_matrix']
        # Initialize precomputed paths if available
        self.precomputed_paths = algo_config['precomputed_paths'].copy()


    def update_robot_count(self, new_robot_count):
        self.robots_num = new_robot_count
        self.intention_table = np.full(self.robots_num, self.nodes_num + 1)

    def update_intention_table(self, robot_id, new_intention):
        self.intention_table[robot_id] = new_intention

    def remove_robot(self, robot_id):
        self.intention_table = np.delete(self.intention_table, robot_id)
        self.robots_num -= 1

    def add_robot(self):
        self.robots_num += 1
        self.intention_table = np.append(self.intention_table, self.nodes_num + 1)

    def count_intention(self, node, intention_table):
        """
        Return the number of other agents that intend to visit the node in question.

        Parameters:
        - node: Node in question to check against.
        - intention_table: Table of intentions of other agents.

        Returns:
        - Int of number of other agents intending to visit that node.
        """
        sum_val = 0
        for intention in intention_table:
            if intention == node:
                sum_val += 1
        return sum_val

    def determine_goal(self, idleness_log, intention_table, current_node, gain1=0.2, gain2=20.0, edge_min=30.0):
        if len(self.neighbour_matrix[current_node]) > 1:
            log_result = math.log10(1.0 / gain1)
            posterior_probability = np.zeros(len(self.neighbour_matrix[current_node]))

            for i, neighbour_id in enumerate(self.neighbour_matrix[current_node]):
                neighbour_edge_weight = self.map_adj_matrix[current_node][neighbour_id]
                neighbour_edge_weight = min(neighbour_edge_weight, edge_min)
                gain = idleness_log[neighbour_id] / neighbour_edge_weight

                if gain < gain2:
                    exp_param = (log_result / gain2) * gain
                    posterior_probability[i] = gain1 * math.exp(exp_param)
                else:
                    posterior_probability[i] = 1.0

                # Count the number of other agents that are intending to visit this node
                count = self.count_intention(neighbour_id, intention_table)
                if count > 0:
                    num_agents = len(intention_table)
                    p_gain_state = math.pow(2, num_agents - count) / (math.pow(2, num_agents) - 1.0)
                    posterior_probability[i] *= p_gain_state

            # Choose the neighbour with the highest posterior probability
            max_posterior_prob_index = np.where(posterior_probability == np.max(posterior_probability))[0]
            number_of_max_probs = len(max_posterior_prob_index)

            if number_of_max_probs > 1:
                random_index = random.randint(0, number_of_max_probs - 1)
                next_vertex = self.neighbour_matrix[current_node][max_posterior_prob_index[random_index]]
            else:
                next_vertex = self.neighbour_matrix[current_node][max_posterior_prob_index[0]]
        else:
            next_vertex = self.neighbour_matrix[current_node][0]

        return next_vertex

    def calculate_next_path(self, robot_id, idleness_log, intention_table, current_node):
        goal_node = self.determine_goal(idleness_log, intention_table, current_node)

        # Check if precomputed paths are available
        if self.precomputed_paths is not None:
            # Use precomputed paths if available
            path = copy.deepcopy(self.precomputed_paths.get((current_node, goal_node), 'Not found'))
            if path == False:
                path = get_full_path(copy.deepcopy(self.precomputed_paths), copy.deepcopy(self.map_adj_matrix), current_node, goal_node)
        else:
            # Fallback to calculating the shortest path if precomputed paths are not available
            start = self.node_pos_matrix[current_node]
            goal = self.node_pos_matrix[goal_node]
            print(f'{start} {goal} not existed in precomputed paths, something is wrong')
            path = calculate_shortest_path(self.pgm_map_matrix, start, goal)
        try:
            assert path != []
        except:
            print(current_node, goal_node)
        return path, goal_node
