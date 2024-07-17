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

from envs.Static_Trust.StaticEnv import StaticEnv as Env
from configs.static_trust_patrol_config import static_trust_patrol_config as config
import logging
from datetime import datetime
import os
import random

print(config)

# set the logging system
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
result_dir = os.path.join('results',current_time)
os.makedirs(result_dir, exist_ok=True)
log_file_path = os.path.join(result_dir, 'experiment.log')

logging.basicConfig(filename = log_file_path, level=logging.INFO,
                    format = '%(message)s')

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

no_zero_rewards = [i for i in env.monitor.rewards if i != 0]
print(f'average reward: {sum(no_zero_rewards) / len(no_zero_rewards)}')