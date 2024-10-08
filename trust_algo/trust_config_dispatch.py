def get_trust_algo_config(config_file):
    map_name = config_file['env_config']['map_name']
    node_pos_matrix = config_file['env_config']['node_pos_matrix']
    map_adj_matrix = config_file['env_config']['map_adj_matrix']
    pgm_map_matrix = config_file['env_config']['pgm_map_matrix']
    patrol_algo = config_file['algo_config']['patrol_algo_name']
    init_pos = config_file['robot_config']['init_pos']
    robots_num = config_file['robot_config']['robots_num']
    trust_algo = config_file['trust_config']['trust_algo']
    trust_mode = config_file['trust_config']['trust_mode']


    if trust_algo == 'BETA':
        trust_algo_config = {
            None,
        }
        return trust_algo_config

    elif trust_algo == 'FIRE':
        return {}

    elif trust_algo == 'YUSINGH':
        return {}

    elif trust_algo == 'FUZZY':
        return {}

    elif trust_algo == 'TRAVOS':
        trust_algo_config = {
            'trust_mode': trust_mode,
        }
        return trust_algo_config

    elif trust_algo == 'SUBJECTIVE':
        trust_algo_config = {
            'robot_num': robots_num
        }
        return trust_algo_config

    elif trust_algo == 'ML':
        trust_algo_config = {
            'trust_model_path_provider': './trust_algo/ML_based_training/models/provider/svm_model_rbf.pkl',
            'trust_model_path_reporter': './trust_algo/ML_based_training/models/reporter/svm_model_rbf.pkl',
            'scaler_path_provider': './trust_algo/ML_based_training/models/provider/scaler.pkl',
            'scaler_path_reporter': './trust_algo/ML_based_training/models/reporter/scaler.pkl',
        }
        return trust_algo_config
    else:
        return None