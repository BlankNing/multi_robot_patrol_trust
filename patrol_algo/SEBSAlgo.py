from .AbstractAlgo import Algo
from utils.astar_shortest_path import calculate_shortest_path
import numpy as np
import math
import random

class SEBSAlgo(Algo):
    def __init__(self, algo_config):
        """
        @param idleness_log: An array representing the idleness values of each node in the graph.
        @param agent: An object representing the agent.
        @param intention_table: A list of intentions of other agents to visit certain nodes.
        @param gain1: A float representing the gain factor.
        @param gain2: A float representing the gain threshold.
        @param edge_min: A float representing the minimum edge weight.

        @return: An integer representing the next vertex to travel to.
        """

        self.nodes_num = algo_config['nodes_num']
        self.robots_num = algo_config['robots_num']
        self.pgm_map_matrix = algo_config['pgm_map_matrix']
        self.node_pos_matrix = algo_config['node_pos_matrix']
        self.map_adj_matrix = algo_config['map_adj_matrix']
        self.neighbour_matrix = algo_config['neighbour_matrix']
        # self.idleness_log =
        # self.intention_table =
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
        Return the number of other agents that intend to visit the node in question
        @param node: Node in question to check against
        @param intention_table: Table of intentions of other agents
        @return: Int of number of other agents intending to visit that node
        """
        sum_val = 0
        for i in range(len(intention_table)):
            if intention_table[i] == node:
                sum_val += 1
        return sum_val
    def determine_goal(self, idleness_log, intention_table, current_node, gain1=0.2, gain2=20.0, edge_min=30.0):
        if len(self.neighbour_matrix[current_node]) > 1:
            log_result = math.log10(1.0/gain1)
            posterior_probability = np.zeros(len(self.neighbour_matrix[current_node]))

            for i in range(len(self.neighbour_matrix[current_node])):
                neighbour_id = self.neighbour_matrix[current_node][i]
                neighbour_edge_weight = self.map_adj_matrix[current_node][neighbour_id]
                neighbour_edge_weight = min(neighbour_edge_weight, edge_min)
                gain = idleness_log[neighbour_id] / neighbour_edge_weight

                if gain < gain2:
                    exp_param = (log_result / gain2) * gain

                    posterior_probability[i] = gain1 * math.exp(exp_param)
                else:
                    posterior_probability[i] = 1.0

                # Count the number of other agents that are intending to visit this node
                count = self.count_intention(self.neighbour_matrix[current_node][i], intention_table)
                # If count is larger than zero, another agent intends to visit it! Adjust state accordingly:
                if count > 0:
                    num_agents = len(intention_table)
                    p_gain_state = math.pow(2, num_agents - count) / (math.pow(2, num_agents) - 1.0)
                    posterior_probability[i] *= p_gain_state

                # Choose the one in the posterior probability with the largest value
                # return a numpy array, and if there are more than one include the index that each are found at
                # Return the 0th element of the tuple, as np.where returns a tuple
            max_posterior_prob_index = np.where(posterior_probability == np.max(posterior_probability))[0]
            number_of_max_probs = len(max_posterior_prob_index)
            # If there are more than 1 max_probs, choose randomly
            if number_of_max_probs > 1:
                # generate random int between 0 - number_of_max_probs
                random_index = random.randint(0, number_of_max_probs - 1)  # Not including total length
                # Choose randomly from the array of elements with the highest posterior_probability
                next_vertex = self.neighbour_matrix[current_node][max_posterior_prob_index[random_index]]
            else:
                # If there is only one choice of max_posterior_prob, use this value
                next_vertex = self.neighbour_matrix[current_node][max_posterior_prob_index[0]]

        else:
            # Only one neighbour, so travel to that
            next_vertex = self.neighbour_matrix[current_node][0]
        return next_vertex

    def calculate_next_path(self, robot_id, idleness_log, intention_table, current_node):
        goal_node = self.determine_goal(idleness_log, intention_table, current_node)
        start = self.node_pos_matrix[current_node]
        goal = self.node_pos_matrix[goal_node]
        return calculate_shortest_path(self.pgm_map_matrix, start, goal), goal_node
