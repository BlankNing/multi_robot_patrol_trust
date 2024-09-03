'''

challenge the capability(report anomaly, speed), predictability(choose to cooperate or not, stop probability), integrity(cooperate situation)

Robots patrol in museum(map) environment,
central monitor maintains the performance record todo: failure and success
Each in charge of 4 nodes,
stop with a probability of 0.05,
report true anomaly with probability of 0.95, false anomaly 0.05
notice closest robot, todo
choose to cooperate with probability 0.7 todo
'''

experiment = 'dynamic'

if experiment == 'static':
    from envs.Static_Trust.StaticEnv import StaticEnv as Env
    from configs.static_trust_patrol_config import static_trust_patrol_config as config
else:
    from envs.Dynamic_Trust.DynamicEnv import DynamicEnv as Env
    from configs.dynamic_trust_patrol_config import dynamic_trust_patrol_config as config

import logging
from datetime import datetime
import os
import random
import copy

display_config = copy.deepcopy(config)
display_config['env_config'] = {}
print(display_config)

trust_method = config['robot_config']['service_select_strategy']
trust_algorithm = config['trust_config']['trust_algo']

# set the logging system, create necessary file folders
result_dir_path = config['result_dir_path']
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
result_dir = os.path.join(result_dir_path, current_time)
os.makedirs(result_dir, exist_ok=True)
log_file_path = os.path.join(result_dir, 'experiment.log')
# init logging system
logging.basicConfig(filename = log_file_path, level=logging.INFO,
                    format = '%(message)s')

# guarantee reproduciblility
try:
    random.seed(config['seed'])
except:
    pass

# prepare environment, robot, algo, trust model
env = Env(config)

# start simulation
for t in range(config['total_steps']):
    if t % 1000 ==0:
        print(f"{t}/{config['total_steps']}")
    env.step(verbose=False)

# plot idleness, confusion matrix, reward
# env.monitor.plot_idleness(0)
# env.monitor.plot_trust_value(0)
# env.monitor.create_patrol_screenshot(config, 200)
# env.monitor.create_patrol_gif(config)
# env.monitor.reward_with_untrustworthy_plot(0)
# env.monitor.trust_with_untrustworthy_plot(0)

print(env.monitor.histories)

# save interaction history as csv
import pandas as pd
data = pd.DataFrame(env.monitor.histories)
data.to_csv(os.path.join(result_dir, f'{trust_algorithm}_{trust_method}_histories.csv'))

# env.monitor.plot_idleness_in_range([i for i in range(6)])
# env.monitor.combined_reward_trust_with_all_robot_plot(0, config['robot_config']['service_select_strategy'])
# env.monitor.create_patrol_gif_new(config, 'SEBS_museum_recharge_service.gif')

re = env.monitor.average_reward_per_round()