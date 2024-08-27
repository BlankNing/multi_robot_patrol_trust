from utils.load_map import *
import itertools

map_name = 'museum'
patrol_algo = 'SEBS'
timesteps = 10000
robots_num = 8
trust_algo = 'FIRE'

def gen_neighbours(adjacency):
    """
    Returns a matrix representing the neighbors of each node in the PatrolMap.
    @return: A matrix where each row represents a node in the PatrolMap and the elements of the row represents the neighbors of that node.
    """
    num_nodes = adjacency.shape[0]
    neighbours_matrix = [[] for _ in range(num_nodes)]
    for i, j in itertools.product(range(num_nodes), range(num_nodes)):
        if adjacency[i][j] >= 1:
            neighbours_matrix[i].append(j)
    return neighbours_matrix

dynamic_trust_patrol_config = {
    'env_config':{
        'map_name': map_name,
        'node_pos_matrix':get_node_pos_matrix(map_name),
        'map_adj_matrix':get_map_adj_matrix(map_name),
        'pgm_map_matrix':get_pgm_map_matrix(map_name),
        'neighbour_matrix': gen_neighbours(get_map_adj_matrix(map_name))
    },
    'robot_config':{
        'robots_num': robots_num,
        'init_pos': get_default_init_pos(get_node_pos_matrix(map_name),robots_num),
        'true_positive_trustworthy': 1,
        'false_positive_trustworthy': 0,
        'true_positive_abnormal': 1,
        'false_positive_abnormal': 0.7,
        'uncooperativeness': 0.2,
        'required_tasks_list': [i for i in range(4)],
        'robots_capable_tasks':{i : [i % 4] for i in range(robots_num)},
        'extra_reward': 4000,
        'env_penalty': -4000,
        'service_select_strategy': 'trust', # random, good, bad, ignore0_num, trust
        'provider_select_strategy': 'trust', # random, determined, trust
        'trust_algo': trust_algo,
        'patrol_algo': patrol_algo,
        'guide_algo': 'Random',
        'provider_select_randomness': 'boltzmann', # determined, boltzmann
        'service_strategy_based_on_trust': {'threshold':0}, #{threshold: 0.3}, {function:which function}
        'communication_range': 200,
        'guide_robot_id': [8],
    },
    'algo_config':{
        'patrol_algo_name':patrol_algo,
    },
    'guide_algo_config': {
        'patrol_algo_name': 'Random'
    },
    'trust_config':{
        'trust_dynamic': {2000: {0:1, 4:0}, 5000: {0:0, 4:1}}, # {timestep_1: {robot_id: trustworthy 1 /untrustworthy 0 },}
        'cooperativeness_dynamic': {4000: {4:1}}, # {timestep_1: {robot_id: cooperative 1 /uncooperative 0 },}
        'untrust_list': [0],
        'uncooperative_list': [4],
        'trust_algo': trust_algo,
        'trust_mode': 'IT+WR',
        'malicious_reporter_list': [],
        'malicious_amplitude': -0.2,
    },
    'total_steps':timesteps,
    'result_dir_path': './results/dynamic/',
    'seed':600, #600,1000,3407,300,5000
}
