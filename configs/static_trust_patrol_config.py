from utils.load_map import *

map_name = 'museum'
patrol_algo = 'partition'
timesteps = 3000
robots_num = 8

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
        'false_positive_abnormal': 0.7,
        'required_tasks_list': [i for i in range(4)],
        'robots_capable_tasks':{i : [i % 4] for i in range(8)},
        'extra_reward': 5000,
        'env_penalty': -5000,
        'service_select_strategy': 'random', # random, good, bad, ignore0_num, trust
        'provider_select_strategy': 'random', # random, determined, trust
    },
    'algo_config':{
        'patrol_algo_name':patrol_algo,
    },
    'trust_config':{
        'untrust_list': [0],
        'trust_algo':'FIRE',
    },
    'total_steps':timesteps,

}
