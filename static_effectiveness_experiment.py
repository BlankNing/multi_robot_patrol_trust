from multiprocessing import Pool
import logging
from datetime import datetime
import os
import random
import pandas as pd

# define single experiment code
def run_experiment(env_seed):
    from configs.static_trust_patrol_config import static_trust_patrol_config as config
    config['seed'] = env_seed
    reward = {}
    experiments = ['good', 'bad', 'random',  'FIRE', 'TRAVOS', 'SUBJECTIVE', 'YUSINGH', 'FUZZY', 'ML']
    experiments = ['ML']

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
        elif exper == 'ML':
            config['robot_config']['service_select_strategy'] = 'trust'
            config['robot_config']['provider_select_strategy'] = 'trust'
            config['robot_config']['trust_algo'] = 'ML'
            config['trust_config']['trust_algo'] = 'ML'
            config['robot_config']['service_strategy_based_on_trust'] = {'threshold': 0.5}

        trust_method = config['robot_config']['service_select_strategy']
        trust_algorithm = config['trust_config']['trust_algo']

        # training log
        result_dir_path = config['result_dir_path']
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result_dir = os.path.join(result_dir_path, current_time)
        os.makedirs(result_dir, exist_ok=True)
        
        # init env
        from envs.Static_Trust.StaticEnv import StaticEnv as Env
        env = Env(config)
    
        # start simulating
        for t in range(config['total_steps']):
            if t % 2000 == 0:
                print(f"Seed {env_seed} - {t}/{config['total_steps']} {exper}")
            env.step(verbose=False)
    
        # save history to csv
        data = pd.DataFrame(env.monitor.histories)
        data.to_csv(os.path.join(result_dir, f'{trust_algorithm}_{trust_method}_histories_{env_seed}.csv'))
        
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
    data.to_csv('static_effectiveness_experiment.csv')

'''
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
    'seed':1, #600,1000,3407,300,5000
}

'''