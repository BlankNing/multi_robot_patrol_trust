from utils.load_map import *

map_name = 'museum'
patrol_algo = 'partition'
timesteps = 10000
robots_num = 8
trust_algo = 'FIRE' # FIRE, TRAVOS, SUBJECTIVE, YUSINGH, FUZZY, ML


static_trust_patrol_config = {
    'env_config':{
        'map_name': map_name,
        'node_pos_matrix':get_node_pos_matrix(map_name),
        'map_adj_matrix':get_map_adj_matrix(map_name),
        'pgm_map_matrix':get_pgm_map_matrix(map_name),
    },
    'robot_config':{
        'robots_num': robots_num,
        'init_pos': get_default_init_pos(get_node_pos_matrix(map_name),robots_num),
        'true_positive_trustworthy': 1,
        'false_positive_trustworthy': 0,
        'true_positive_abnormal': 1,
        'false_positive_abnormal': 0.9,
        'uncooperativeness': 0.2,
        'required_tasks_list': [i for i in range(4)],
        'robots_capable_tasks':{i : [i % 4] for i in range(robots_num)},
        'extra_reward': 3000,
        'env_penalty': -1000,
        'service_select_strategy': 'trust', # random, good, bad, ignore0_num, trust
        'provider_select_strategy': 'trust', # random, determined, trust
        'trust_algo': trust_algo,
        'provider_select_randomness': 'boltzmann', # determined, boltzmann
        'service_strategy_based_on_trust': {'threshold': 0.90}, #{threshold: 0.3}, {function:which function}
        'communication_range': 100000,
    },
    'algo_config':{
        'patrol_algo_name':patrol_algo,
    },
    'trust_config':{
        'untrust_list': [0],
        'uncooperative_list': [],
        'trust_algo': trust_algo,
        'trust_mode': 'IT+WR',
    },
    'total_steps':timesteps,
    'result_dir_path': './results/static/',
    'seed':600, #600,1000,3407,300,5000
}
