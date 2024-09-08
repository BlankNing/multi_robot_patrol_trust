from multiprocessing import Pool
import logging
from datetime import datetime
import os
import random
import pandas as pd
# todo: copy the setting at the end of the page to config
# define single experiment code
def run_experiment(env_seed):
    from configs.dynamic_trust_patrol_config import dynamic_trust_patrol_config as config
    config['seed'] = env_seed
    config['robot_config']['run_communication_comparison'] = False
    reward = {}
    experiments = ['good', 'bad', 'random',  'FIRE', 'TRAVOS', 'SUBJECTIVE', 'YUSINGH', 'FUZZY']
    # experiments = ['bad', 'TRAVOS']

    for exper in experiments:
        if exper == 'good':
            config['robot_config']['service_select_strategy'] = 'good'
            config['robot_config']['provider_select_strategy'] = 'random'
        elif exper == 'bad':
            config['robot_config']['service_select_strategy'] = 'bad'
            config['robot_config']['provider_select_strategy'] = 'random'
        elif exper == 'random':
            config['robot_config']['service_select_strategy'] = 'random'
            config['robot_config']['provider_select_strategy'] = 'random'
        elif exper == 'FIRE':
            config['robot_config']['service_select_strategy'] = 'trust'
            config['robot_config']['provider_select_strategy'] = 'trust'
            config['robot_config']['trust_algo'] = 'FIRE'
            config['trust_config']['trust_algo'] = 'FIRE'
            config['robot_config']['service_strategy_based_on_trust'] = {'threshold': 0.90}
        elif exper == 'TRAVOS':
            config['robot_config']['service_select_strategy'] = 'trust'
            config['robot_config']['provider_select_strategy'] = 'trust'
            config['robot_config']['trust_algo'] = 'TRAVOS'
            config['trust_config']['trust_algo'] = 'TRAVOS'
            config['robot_config']['service_strategy_based_on_trust'] = {'threshold': 0.49}
        elif exper == 'YUSINGH':
            config['robot_config']['service_select_strategy'] = 'trust'
            config['robot_config']['provider_select_strategy'] = 'trust'
            config['robot_config']['trust_algo'] = 'YUSINGH'
            config['trust_config']['trust_algo'] = 'YUSINGH'
            config['robot_config']['service_strategy_based_on_trust'] = {'threshold': 0.9}
        elif exper == 'SUBJECTIVE':
            config['robot_config']['service_select_strategy'] = 'trust'
            config['robot_config']['provider_select_strategy'] = 'trust'
            config['robot_config']['trust_algo'] = 'SUBJECTIVE'
            config['trust_config']['trust_algo'] = 'SUBJECTIVE'
            config['robot_config']['service_strategy_based_on_trust'] = {'threshold': -0.01}
        elif exper == 'FUZZY':
            config['robot_config']['service_select_strategy'] = 'trust'
            config['robot_config']['provider_select_strategy'] = 'trust'
            config['robot_config']['trust_algo'] = 'FUZZY'
            config['trust_config']['trust_algo'] = 'FUZZY'
            config['robot_config']['service_strategy_based_on_trust'] = {'threshold': 0.648}

        trust_method = config['robot_config']['service_select_strategy']
        trust_algorithm = config['trust_config']['trust_algo']

        # training log
        result_dir_path = config['result_dir_path']
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result_dir = os.path.join(result_dir_path, current_time)
        os.makedirs(result_dir, exist_ok=True)
        
        # init env
        from envs.Dynamic_Trust.DynamicEnv import DynamicEnv as Env
        env = Env(config)
    
        # start simulating
        for t in range(config['total_steps']):
            if t % 2000 == 0:
                print(f"Seed {env_seed} - {t}/{config['total_steps']} {exper}")
            env.step(verbose=False)
    
        # save history to csv
        data = pd.DataFrame(env.monitor.histories)
        patrol_method = 'SEBS'
        title = f'{trust_algorithm}_{trust_method}_histories_{patrol_method}.csv'
        if trust_method in ['good', 'random', 'bad']:
            title = f'{trust_method}_histories_{patrol_method}.csv'
        data.to_csv(os.path.join(result_dir, title))

        re = env.monitor.average_reward_per_round()
        reward[exper] = re
        print(f'{env_seed}_{exper}: {re}')
    
    return reward

if __name__ == '__main__':
    # multiprocessing pool
    pool = Pool(processes=os.cpu_count())  # all available CPUs

    # execute mission
    rewards = pool.map(run_experiment, range(30))  # experiment number: 20

    # close multiprocessing pool
    pool.close()
    pool.join()

    # save all experiment data
    data = pd.DataFrame(rewards)
    data.to_csv('dynamic_model_comparison.csv')

'''
from utils.load_map import *
import itertools

map_name = 'cumberland'
patrol_algo = 'SEBS'
timesteps = 20000
robots_num = 12
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
        'neighbour_matrix': gen_neighbours(get_map_adj_matrix(map_name)),
        'precomputed_paths': get_predefined_path(map_name),
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
        'extra_reward': 2000,
        'env_penalty': -1000,
        'service_select_strategy': 'trust', # random, good, bad, ignore0_num, trust
        'provider_select_strategy': 'trust', # random, determined, trust
        'trust_algo': trust_algo,
        'patrol_algo': patrol_algo,
        'guide_algo': 'Random',
        'sweep_algo': 'CGG',
        'provider_select_randomness': 'boltzmann', # determined, boltzmann
        'run_communication_comparison': False,
        'service_strategy_based_on_trust': {'threshold':0.8}, #{threshold: 0.3}, {function:which function}
        'communication_range':100,
        'guide_robot_id': [4],
        'sweep_robot_id': [7],
    },
    'algo_config':{
        'patrol_algo_name':patrol_algo,
    },
    'guide_algo_config': {
        'patrol_algo_name': 'Random'
    },
    'sweep_algo_config': {
        'patrol_algo_name': 'CGG'
    },
    'trust_config':{
        'trust_dynamic': {2000: {0:1, 5:0}, 5000: {0:0}, 8000:{5:1}, 13000: {0:1}, 15000:{5:0}}, # {timestep_1: {robot_id: trustworthy 1 /untrustworthy 0 },}
        'cooperativeness_dynamic': {}, # {timestep_1: {robot_id: cooperative 1 /uncooperative 0 },}
        'untrust_list': [0],
        'uncooperative_list': [],
        'trust_algo': trust_algo,
        'trust_mode': 'IT+WR',
        'malicious_reporter_list': [],
        'malicious_target_list': [],
        'malicious_amplitude': -0.2,
    },
    'total_steps':timesteps,
    'result_dir_path': './results/dynamic/',
    'seed':600, #600,1000,3407,300,5000
}


'''