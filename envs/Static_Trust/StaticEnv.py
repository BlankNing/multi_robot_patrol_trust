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
        self.uncooperative_list = config_file['trust_config']['uncooperative_list']
        self.trust_algo = config_file['trust_config']['trust_algo']
        self.trust_algo_config = get_trust_algo_config(config_file)
        config_file['robot_config']['pgm_map_matrix'] = self.pgm_map_matrix
        self.max_distance = sum(self.pgm_map_matrix.shape)
        self.robot_config = config_file['robot_config']
        # only 1 robot report anomaly at one time
        self.has_anomaly = False
        # init anomaly
        self.anomaly = -1
        # Static Env Monitor
        self.monitor = StaticMonitor(self.robot_config['robots_num'])
        # Init trust_engine (instantiate trust engine and its configure file)
        self.trust_algo_config['history_monitor'] = self.monitor
        self.trust_engine = TrustFactory().create_algo(self.trust_algo, self.trust_algo_config)
        # Static Robot Init
        self.robots = [StaticRobot(i, self.algo_engine, self.node_pos_matrix, self.init_pos[i], self.untrust_list, self.uncooperative_list,
                                    self.monitor, self.trust_engine, self.robot_config) for i in range(self.robots_num)]
        # collect init position
        self.monitor.collect_robot_pos(config_file['robot_config']['init_pos'])
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
        interaction_histories = []
        self.timestep += 1
        self.logger.info(f"Timestep {self.timestep}\nCurrent Anomaly {self.anomaly}")

        # robot move, update trust when calling for help
        interaction_flag = False
        robot_pos_records = []
        env_interaction_impressions= []
        robot_current_states = []
        for robot in self.robots:
            robot_current_states.append(robot.state)
            robot_pos_record, env_interaction_impression = robot.step(verbose=verbose, timestep=self.timestep)
            robot_pos_records.append(robot_pos_record)
            env_interaction_impressions.append(env_interaction_impression)
            interaction_flag = not all(element == {} for element in env_interaction_impressions)
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

        # calculate total reward if interaction happens
        if interaction_flag:
            max_distance = 0
            max_distance_index = -1
            cnt = 0
            for index, i in enumerate(env_interaction_impressions):
                # i example: Request record: {'request_robot': 0, 'service_robot': 1, 'time': 1, 'task': 1, 'request_position': (160, 140),
                # 'is_true_anomaly': 0, 'service_quality': 1, 'distance': 122, 'reward': -244}, Trust record: {0: 1.0}
                if i != {}:
                    # reward system
                    reporter_reward_matrix = {(0, 0): 0, (0, 1): 0, (1, 0): self.robot_config['env_penalty'], (1, 1): self.robot_config['extra_reward']}

                    # rating system
                    if i['is_true_anomaly'] == 1 and i['service_quality'] == 1:
                        rating_to_provider = 1 - i['distance']/self.max_distance
                    elif i['is_true_anomaly'] == 1 and i['service_quality'] == 0:
                        rating_to_provider = -1
                    elif i['is_true_anomaly'] == 0 and i['service_quality'] == 1:
                        rating_to_provider = 1 - i['distance']/self.max_distance
                    else:
                        rating_to_provider = 0.1

                    interaction_history = {
                        'is_true_anomaly': i['is_true_anomaly'],
                        'reporter_id': i['request_robot'],
                        'provider_id': i['service_robot'],
                        'task_id': i['task'],
                        'is_same_type': i['is_same_type'],
                        'report_time': i['time'],
                        'provide_time': i['service_time'],
                        'report_position': i['request_position'],
                        'provide_position': i['service_position'],
                        'trust_towards_reporter': i['trust_value_towards_reporter'],
                        'trust_towards_provider': i['trust_value_to_provider'],
                        'provider_action': i['service_quality'],
                        'provider_reward': i['reward'],
                        'reporter_reward': reporter_reward_matrix[(i['is_true_anomaly'], i['service_quality'])],
                        'rating_to_reporter': 1 - i['distance']/self.max_distance if i['is_true_anomaly'] == 1  else -1, # simple rating system, can regularise it
                        'rating_to_provider': rating_to_provider, # complex rating system, related with max
                        'distance_penalty': i['distance'],
                    }
                    interaction_histories.append(interaction_history)

                    # collect data for TRAVOS


                    # monitor collect history [service quality/is true anomaly, reward]
                    if i['service_quality'] == 1:
                        max_distance = i['distance'] if i['distance'] > max_distance else max_distance
                        max_distance_index = cnt
                    cnt += 1

            # if at this timestep, some robot come to help, the reporter have to wait until all the robots have came
            # thus total_reward - max_distance for all the robots coming to help
            interaction_histories[max_distance_index]['reporter_reward'] -= max_distance
            self.monitor.collect_histories(interaction_histories)


        # node record
        node_idleness_records = []
        for node in self.nodes:
            node_idleness_record = node.step(robot_pos_records, robot_current_states)
            node_idleness_records.append(node_idleness_record)
        self.monitor.collect_node_idleness(node_idleness_records)