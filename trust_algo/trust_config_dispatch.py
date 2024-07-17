def get_trust_algo_config(config_file):
    map_name = config_file['env_config']['map_name']
    node_pos_matrix = config_file['env_config']['node_pos_matrix']
    map_adj_matrix = config_file['env_config']['map_adj_matrix']
    pgm_map_matrix = config_file['env_config']['pgm_map_matrix']
    patrol_algo = config_file['algo_config']['patrol_algo_name']
    init_pos = config_file['robot_config']['init_pos']
    robots_num = config_file['robot_config']['robots_num']
    trust_algo = config_file['trust_config']['trust_algo']

    if trust_algo == 'beta':
        trust_algo_config = {
            None,
        }
        return trust_algo_config

    elif patrol_algo == 'FIRE':
        pass

    elif patrol_algo == 'YUSIGH':
        pass

    else:
        return None