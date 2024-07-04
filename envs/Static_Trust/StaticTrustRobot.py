from basic_patrol_class.Robot import Robot
import logging
import numpy as np
import random


class StaticRobot(Robot):
    def __init__(self, id, algo_engine, node_pos_matrix, init_pos, untrust_list, monitor, trust_engine, config_file):
        super().__init__(id, algo_engine, node_pos_matrix, init_pos)

        # {robot id: [capable task list]}
        self.robots_capable_tasks = config_file['robots_capable_tasks']
        self.robot_capable_task = config_file['robots_capable_tasks'][self.id]
        self.required_tasks_list = config_file['required_tasks_list']
        self.env_penalty = config_file['env_penalty']
        self.extra_reward = config_file['extra_reward']
        self.service_select_strategy = config_file['service_select_strategy']
        self.provider_select_strategy = config_file['provider_select_strategy']
        self.service_strategy_based_on_trust = config_file['service_strategy_based_on_trust']
        self.monitor = monitor
        self.trust_engine = trust_engine
        self.service_time = 0
        self.task_to_robot = self.generate_task_to_robot()
        # load logging system
        self.logger = logging.getLogger(__name__)

        if self.id not in untrust_list:
            self.true_positive = config_file['true_positive_trustworthy']
            self.false_positve = config_file['false_positive_trustworthy']
        else:
            self.true_positive = config_file['true_positive_abnormal']
            self.false_positve = config_file['false_positive_abnormal']

        self.true_anomaly_pos = self.monitor.get_anomaly_pos()
        self.pgm_map_matrix = config_file['pgm_map_matrix']

    # todo: define two strategies (1) threshold based (2) continuous map function based
    def threshold_based_service_strategy(self, trust_value, threshold) -> int:
        if trust_value < threshold:
            return 0
        else:
            return 1

    def function_based_service_strategy(self, trust_value) -> int:
        return 1

    # todo: choose_service_provider based on trust engine
    def choose_service_provider(self, required_tasks):
        '''
        :param required_tasks: list eg: [0,2,3]; task_to_robot {task:[all robots that are capable of this task]}
        robot capable task: {robot id: [capable task list]}
        task_to_robots eg: {1:[1,5],2:[2,6]}
        Can get {1:[1,5],2:[2,6], 3:[3,7],4:[4,0]} initially
        now we received a required_tasks list [1,2]
        need to return {1:[1,5],2:[2,6]} -> {1:1,2:2}
        :return: task_ro_robot_assignment {task: robot_id}
        '''
        # 1. delete the other unrequired task
        task_to_robots = {key: self.task_to_robot[key] for key in required_tasks if key in self.task_to_robot}
        # 2. delete request robot id from the robot list
        for task, robots in task_to_robots.items():
            if self.id in robots:
                robots.remove(self.id) # A robot cannot call it self to help
        if self.provider_select_strategy == 'trust':
            pass
        elif self.provider_select_strategy == 'random':
            return {task: random.choice(task_to_robots[task]) for task in required_tasks}
        elif self.provider_select_strategy == 'determined':
            return {1: 1, 2: 2, 3: 3, 0: 4}

    # todo: choose_service_quality based on trust engine/random/determined
    def choose_service_quality(self, request_robot_id):
        if self.service_select_strategy == 'trust':
            history = self.monitor.get_history_as_provider(self.id, request_robot_id)
            trust_value = self.trust_engine.calculate_trust_value(history)

            # decide what to do: (1) reach threshold then dead/ (2) map function between the trust value and the strategy
            if 'threshold' in self.service_strategy_based_on_trust:
                return self.threshold_based_service_strategy(trust_value, threshold=float(self.service_strategy_based_on_trust['threshold']))
            elif self.service_strategy_based_on_trust == 'function':
                return self.function_based_service_strategy(trust_value)

        elif self.service_select_strategy =='good':
            return 1
        elif self.service_select_strategy == 'bad':
            return 0
        elif self.service_select_strategy == 'random':
            return random.randint(0, 1)
        elif 'ignore0' in self.service_select_strategy:
            probability = float(self.service_select_strategy.split('_')[1]) # probability that other robot will ignore Robot 0
            if request_robot_id == 0 and random.random() < probability:
                return 0
            else:
                return 1

    def check_node(self):
        return np.where((self.node_pos_matrix == self.current_pos).all(axis=1))[0]

    def generate_task_to_robot(self):
        task_to_robot = {}
        for robot, tasks in self.robots_capable_tasks.items():
            for task in tasks:
                task_to_robot[task] = []
        for robot, tasks in self.robots_capable_tasks.items():
            for task in tasks:
                task_to_robot[task].append(robot)
        return task_to_robot

    def step(self, verbose=False, **kwargs):
        timestep = kwargs.get('timestep')
        impression = {}

        # If is in service state:
        if self.service_time != 0:
            self.service_time -= 1

        # If reach an interest point, could find anomaly
        if self.path_list == []:
            # check current anomaly point position
            self.true_anomaly_pos = self.monitor.get_anomaly_pos()
            # check which node robot is on
            self.last_node = int(self.check_node())
            # calculate the next place to go
            self.path_list = self.algo_engine.calculate_next_path(self.id, self.last_node)
            # check if it's still in anomaly cycle
            anomaly_detect_cycle_flag = self.monitor.get_in_cycle_flag()

            # report anomaly with probability when arriving at a node
            if self.last_node == self.true_anomaly_pos:
                # if detected and no progressing anomaly detection cycle
                if random.random() < self.true_positive and not anomaly_detect_cycle_flag:
                    # set the state to reporting
                    self.state = 'True Requesting'
                    required_tasks = random.sample(self.required_tasks_list,
                                                   random.randint(1, len(self.required_tasks_list)))
                    # todo: choose service provider based on trust
                    name_list = self.choose_service_provider(required_tasks)
                    self.monitor.inform_request(self.id, name_list, self.current_pos, 1, timestep)
                    # todo: determine service_time based on astar distance
                    self.service_time = 1  # speed up the procedure
                    self.logger.info(
                        f"Robot {self.id}, Current Position: {self.current_pos}, Current State: {self.state}, Last Node: {self.last_node},"
                        f" Required tasks: {required_tasks}, Required robot namelist: {name_list}, True/False anomaly: True ")
            else:
                if random.random() < self.false_positve and not anomaly_detect_cycle_flag:
                    # set the state to reporting
                    self.state = 'False Requesting'
                    required_tasks = random.sample(self.required_tasks_list,
                                                   random.randint(1, len(self.required_tasks_list)))
                    # todo: choose service provider based on trust
                    name_list = self.choose_service_provider(required_tasks)
                    self.monitor.inform_request(self.id, name_list, self.current_pos, 0, timestep)
                    self.service_time = 1  # speed up the procedure
                    self.logger.info(
                        f"Robot {self.id}, Current Position: {self.current_pos}, Current State: {self.state}, Last Node: {self.last_node},"
                        f" Required tasks: {required_tasks}, Required robot namelist: {name_list}, True/False anomaly: False,")

        # if didn't find anomaly or providing services, robot move
        if self.service_time == 0:
            self.state = 'Patrolling'
            # move 1 step
            self.current_pos = self.path_list[0]
            self.path_list.pop(0)
            self.logger.info(
                f"Robot {self.id}, Current Position: {self.current_pos}, Current State: {self.state}, Last Node: {self.last_node},")

        # If some one is requesting for help at this timestep, switch to provider mode
        if self.monitor.check_request(self.id, timestep) != None:
            self.state = 'Providing'
            current_request = self.monitor.check_request(self.id, timestep)
            impression = current_request
            request_robot_id = current_request['request_robot']
            # todo: choose_service_quality based on trust
            service_quality = self.choose_service_quality(request_robot_id)
            impression['service_quality'] = service_quality

            is_true_anomaly = impression['is_true_anomaly']

            if service_quality == 1:
                # Only when providing good service will service provider receive negative reward proportional to distance
                # distance = len(
                #     calculate_shortest_path(self.pgm_map_matrix, self.current_pos, current_request['request_position']))
                distance = abs(self.current_pos[0] - current_request['request_position'][0]) + abs(
                    self.current_pos[1] - current_request['request_position'][1])
                # self.service_time = distance
                self.service_time = 0  # trick for speeding up the simulation
                impression['distance'] = distance
                # record reward
                if is_true_anomaly == 1:
                    impression['reward'] = self.extra_reward - 2 * distance
                else:  # false alarm, negative reward
                    impression['reward'] = - 2 * distance

            elif service_quality == 0:
                # if bad service, calculate the reward of this interaction
                if is_true_anomaly == 1:
                    impression['reward'] = self.env_penalty
                else:
                    impression['reward'] = 0

            self.logger.info(
                f"Robot {self.id}, Current Position: {self.current_pos}, Current State: {self.state}, Last Node: {self.last_node},"
                f" Request record: {impression}, ")

        # For visualisation
        if verbose == True:
            print(f"Robot_{self.id} {self.state} at {self.current_pos}")

        return self.current_pos, impression
