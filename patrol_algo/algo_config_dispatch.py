def get_algo_config(config_file, algo = None):
    map_name = config_file['env_config']['map_name']
    node_pos_matrix = config_file['env_config']['node_pos_matrix']
    map_adj_matrix = config_file['env_config']['map_adj_matrix']
    pgm_map_matrix = config_file['env_config']['pgm_map_matrix']
    patrol_algo = config_file['algo_config']['patrol_algo_name']
    precomputed_paths = config_file['env_config']['precomputed_paths']
    try:
        guide_patrol_algo = config_file['guide_algo_config']['patrol_algo_name']
        sweep_patrol_algo = config_file['sweep_algo_config']['patrol_algo_name']
    except:
        pass
    init_pos = config_file['robot_config']['init_pos']
    robots_num = config_file['robot_config']['robots_num']

    if patrol_algo == 'partition' and algo == None:
        partition_algo_config  = {
            'robots_num': robots_num,
            'nodes_num': len(node_pos_matrix),
            'pgm_map_matrix': pgm_map_matrix,
            'node_pos_matrix': node_pos_matrix,
            'map_adj_matrix': map_adj_matrix,
            'precomputed_paths': precomputed_paths
        }
        return partition_algo_config

    elif patrol_algo == 'Random' and algo == None:
        algo_config  = {
            'robots_num': robots_num,
            'nodes_num': len(node_pos_matrix),
            'pgm_map_matrix': pgm_map_matrix,
            'node_pos_matrix': node_pos_matrix,
            'map_adj_matrix': map_adj_matrix,
            'precomputed_paths': precomputed_paths
        }
        return algo_config

    elif patrol_algo == 'SEBS' and algo == None:
        algo_config = {
            'robots_num': robots_num,
            'nodes_num': len(node_pos_matrix),
            'pgm_map_matrix': pgm_map_matrix,
            'node_pos_matrix': node_pos_matrix,
            'map_adj_matrix': map_adj_matrix,
            'neighbour_matrix': config_file['env_config']['neighbour_matrix'],
            'precomputed_paths': precomputed_paths
        }
        return algo_config

    elif guide_patrol_algo == 'Random' and algo == 'Random':
        algo_config  = {
            'robots_num': robots_num,
            'nodes_num': len(node_pos_matrix),
            'pgm_map_matrix': pgm_map_matrix,
            'node_pos_matrix': node_pos_matrix,
            'map_adj_matrix': map_adj_matrix,
            'precomputed_paths': precomputed_paths
        }
        return algo_config

    elif sweep_patrol_algo == 'CGG' and algo =='CGG':
        algo_config = {
            'robots_num': robots_num,
            'nodes_num': len(node_pos_matrix),
            'pgm_map_matrix': pgm_map_matrix,
            'node_pos_matrix': node_pos_matrix,
            'map_adj_matrix': map_adj_matrix,
            'map_name': map_name,
            'precomputed_paths': precomputed_paths,
        }
        return algo_config