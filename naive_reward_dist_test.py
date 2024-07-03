'''
Test in an environment without trust and reputation model
Robot 0 spread false alarm sometimes with a probability

Each time a robot release an alarm, it chooses parterner randomly and send request
The provider robot received the request and choose to provide certain services
1. always provide good service
2. always provide bad service
3. provide services randomly

We want to see what the reward distribution is like.
The ideal reward should be: good > complete randomly > bad

Then we investigate the circumstance that robots has discrimination against Robot 0,
who constantly send false alarms.

If the request is from Robot 0, other robots will have higher probability to provide bad service,
cause they knew this robot 0 is not trustworthy and don't what to provide good service which will damage its reward.

The rewards should make up for the distance penalty, but shoud not be too large to let the all good service circumstance
still receive good reward.
'''

from envs.Static_Trust.StaticEnv import StaticEnv as Env
from configs.static_trust_patrol_config import static_trust_patrol_config as config
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import multiprocessing as mp
import os

strategies = ['good', 'bad', 'random', 'ignore0_0.9', 'ignore0_0.6', 'ignore0_0.3']

config['total_steps'] = 3000
cycle_num = 23

def run_simulation(strategy):
    print(f'Testing strategy {strategy}')
    config['robot_config']['select_strategy'] = strategy
    env = Env(config)
    raw_data = []
    data = []
    anomaly_cycle_last_time = 0

    for index in range(cycle_num):
        print(f'Test Cycle: {index + 1}/{cycle_num} for strategy {strategy}')
        for t in range(config['total_steps']):
            if t % 1000 == 0:
                print(f"{t}/{config['total_steps']} for strategy {strategy}")
            env.step(verbose=False)

        no_zero_rewards = [i for i in env.monitor.rewards if i != 0]
        data.append(sum(no_zero_rewards) / len(no_zero_rewards))
        raw_data.extend(no_zero_rewards)
        print('anomaly detection cycle number: {}'.format(len(no_zero_rewards) - anomaly_cycle_last_time))
        anomaly_cycle_last_time = len(no_zero_rewards)
        print('average reward for a single anomaly detection cycle:', sum(no_zero_rewards) / len(no_zero_rewards))

    # No differences between using raw_data or data
    data = raw_data
    # plot part
    pic_name = f"strategy_{strategy}_hist_dist_chart_{cycle_num}"
    fig, ax1 = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax1.hist(data, bins=10, edgecolor='black', alpha=0.7)
    ax1.set_ylabel('Frequency')
    ax2 = ax1.twinx()
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 100)
    y_kde = kde(x_range)
    ax2.plot(x_range, y_kde, 'r-', linewidth=2)
    ax2.set_ylabel('Density')
    plt.title(f'Histogram and Distribution Curve (Strategy {strategy})')
    ax1.set_xlabel('Value')
    plt.savefig(f"./tmp_results/naive_reward_test/{cycle_num}/{pic_name}.png", dpi=300)
    plt.show()
    raw_data_np = np.array(raw_data)
    np.save(f"./tmp_results/naive_reward_test/{cycle_num}/{pic_name}.npy", raw_data_np)
    return strategy, data


if __name__ == '__main__':
    result_dir = f"./tmp_results/naive_reward_test/{cycle_num}"
    os.makedirs(result_dir, exist_ok=True)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_simulation, strategies)

    for strategy, data in results:
        print(f"Strategy {strategy} completed with data: {data}")
