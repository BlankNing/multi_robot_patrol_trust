import logging

from basic_patrol_class.Env import BasicEnv
from .StaticTrustRobot import StaticRobot
from .StaticTrustMonitor import StaticMonitor
from trust_algo.trust_config_dispatch import get_trust_algo_config
from trust_algo.TrustFactory import TrustFactory
from collections import deque
import random

class StaticEnv(BasicEnv):
    '''
    0. basic patrol
    report cycle
    1. reporter spot anomaly: Robot step
    2. ask provider for help: Robot step, change state, Env broadcast
    3. provide service based on trust model: Robot step
    4. record impression to the monitor Robot return, Monitor record
    5. receive reward: Env step
    6. update trust: Robot
    next report cycle
    '''
    def __init__(self, config_file):
        super().__init__(config_file)

        self.untrust_list = config_file['trust_config']['untrust_list']
        self.trust_algo = config_file['trust_config']['trust_algo']
        self.trust_algo_config = get_trust_algo_config(config_file)
        config_file['robot_config']['pgm_map_matrix'] = self.pgm_map_matrix
        self.robot_config = config_file['robot_config']
        # only 1 robot report anomaly at one time
        self.has_anomaly = False
        # init anomaly
        self.anomaly = -1
        # Init trust_engine
        self.trust_engine = TrustFactory().create_algo(self.trust_algo, self.trust_algo_config)
        # Static Env Monitor
        self.monitor = StaticMonitor()
        # Static Robot Init
        self.robots = [StaticRobot(i, self.algo_engine, self.node_pos_matrix, self.init_pos[i], self.untrust_list,
                                    self.monitor, self.trust_engine, self.robot_config) for i in range(self.robots_num)]
        self.cycle_history = deque(maxlen=3)
        # set initial anomaly position and preceived by the monitor
        self.update_anomaly_random_report()
        # load log system
        self.logger = logging.getLogger(__name__)


    def update_anomaly_random_report(self):
        '''
        update a new anomaly in the environment
        :return: None
        '''
        self.anomaly = random.randint(0, self.nodes_num)
        self.monitor.update_anomaly_pos(self.anomaly)

    def step(self, verbose=False):
        self.timestep += 1
        self.logger.info(f"Timestep {self.timestep}\nCurrent Anomaly {self.anomaly}")

        # robot move, update trust when calling for help
        robot_pos_records = []
        env_interaction_impressions= []
        for robot in self.robots:
            robot_pos_record, env_interaction_impression = robot.step(verbose=verbose, timestep=self.timestep)
            robot_pos_records.append(robot_pos_record)
            env_interaction_impressions.append(env_interaction_impression)

            # check if we are in an anomaly detection cycle
            current_states = [r.state for r in self.robots]
            self.cycle_history.append(min([i == 'Patrolling' for i in current_states]))
            if self.timestep == 1:
                if self.cycle_history[-1] == 0:
                    self.monitor.set_in_cycle_flag()
            else:
                # if a cycle just end, now all the robots are patroling
                if self.cycle_history[-1] > self.cycle_history[-2]:
                    self.monitor.cancel_in_cycle_flag()
                    self.update_anomaly_random_report()
                # if a cycle just begin, now some robots are reporting & providing service
                elif self.cycle_history[-1] < self.cycle_history[-2]:
                    self.monitor.set_in_cycle_flag()

        self.monitor.collect_robot_pos(robot_pos_records)
        self.monitor.collect_robot_impression(env_interaction_impressions)

        # calculate total reward
        provider_reward_record = {}
        reporter_reward_record = {}
        total_reward = 0
        max_distance = 0
        reporter_reward_total = 0
        reporter_id = -1
        for i in env_interaction_impressions:
            if i != {}:
                is_true_anomaly = i['is_true_anomaly'] # all the same
                service_quality = i['service_quality'] # not the same
                reporter_reward_matrix = {(0, 0): 0, (0, 1): 0, (1, 0): self.robot_config['env_penalty'], (1, 1): self.robot_config['extra_reward']}
                reporter_reward = reporter_reward_matrix[(is_true_anomaly, service_quality)]
                reporter_reward_total += reporter_reward
                provider_reward = i['reward']
                total_reward += provider_reward + reporter_reward
                provider_reward_record[i['service_robot']] = provider_reward
                reporter_id = i['request_robot']
                if service_quality == 1:
                    max_distance = i['distance'] if i['distance'] > max_distance else max_distance
        # if at this timestep, some robot come to help, the reporter have to wait until all the robots have came
        # thus total_reward - max_distance for all the robots coming to help
        total_reward -= max_distance
        reporter_reward_total -= max_distance
        reporter_reward_record[reporter_id] = reporter_reward_total
        self.monitor.collect_reward(total_reward)
        self.logger.info(f"Reward: Reporter {reporter_reward_record}; Provider {provider_reward_record},")

        # node record
        node_idleness_records = []
        for node in self.nodes:
            node_idleness_record = node.step(robot_pos_records)
            node_idleness_records.append(node_idleness_record)
        self.monitor.collect_node_idleness(node_idleness_records)