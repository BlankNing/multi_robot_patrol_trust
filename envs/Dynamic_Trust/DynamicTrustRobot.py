from basic_patrol_class.Robot import Robot
import logging
import numpy as np
import random

class DynamicRobot(Robot):
    def __init__(self, id, algo_engine, node_pos_matrix, init_pos, untrust_list, uncooperative_list, trust_dynamic, cooperativeness_dynamic, monitor, trust_engine, guide_engine, sweep_engine, config_file):
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
        self.communication_range = config_file['communication_range']
        self.provider_select_randomness = config_file['provider_select_randomness']
        self.trust_algo = config_file['trust_algo']
        self.patrol_algo = config_file['patrol_algo']
        self.robot_num = config_file['robots_num']
        self.guide_robot_id = config_file['guide_robot_id']
        self.sweep_robot_id = config_file['sweep_robot_id']
        self.run_communication_comparison = config_file['run_communication_comparison']
        self.is_guide_robot = self.id in self.guide_robot_id
        self.is_sweep_robot = self.id in self.sweep_robot_id
        if self.is_guide_robot:
            self.algo_engine = guide_engine
            self.patrol_algo = 'Random'
        if self.is_sweep_robot:
            self.algo_engine = sweep_engine
            self.patrol_algo = 'CGG'

        self.monitor = monitor
        # set robot communication range
        self.limited_communication_range = config_file['communication_range']
        self.monitor.robot_communication_range = self.limited_communication_range
        self.unlimited_communication_range = 999999999
        # set trust engine
        self.trust_engine = trust_engine
        self.service_time = 0
        self.last_node = int(self.check_node())
        self.trust_dynamic_timestep = trust_dynamic
        self.cooperativeness_dynamic_timestep = cooperativeness_dynamic
        # battery simulation and recharge time:
        self.battery_time = random.randint(2000, 9000)
        self.recharge_time = random.randint(300, 500)
        if self.patrol_algo == 'SEBS':
            self.goal_node = self.algo_engine.determine_goal(np.zeros(len(self.monitor.get_latest_idleness())),
                                                             np.full(self.robot_num, config_file['dimension'] + 1),
                                                             self.last_node)
        self.task_to_robot = self.generate_task_to_robot()
        # load logging system
        self.logger = logging.getLogger(__name__)
        # inform monitor robot's communication range
        self.monitor.update_robot_communication_range(self.communication_range)
        # set up untrustworthy robot
        self.trustworthy_robot_setting = (config_file['true_positive_trustworthy'], config_file['false_positive_trustworthy'])
        self.untrustworthy_robot_setting = (config_file['true_positive_abnormal'], config_file['false_positive_abnormal'])

        if self.id not in untrust_list:
            self.true_positive = config_file['true_positive_trustworthy']
            self.false_positve = config_file['false_positive_trustworthy']
        else:
            self.true_positive = config_file['true_positive_abnormal']
            self.false_positve = config_file['false_positive_abnormal']

        # set up uncooperative robot
        self.uncooperative_robot_setting = config_file['uncooperativeness']

        if self.id in uncooperative_list:
            self.uncooperativeness = config_file['uncooperativeness']
        else:
            self.uncooperativeness = 0

        self.true_anomaly_pos = self.monitor.get_anomaly_pos()
        self.pgm_map_matrix = config_file['pgm_map_matrix']

    def threshold_based_service_strategy(self, trust_value, threshold) -> int:
        if trust_value < threshold:
            return 0
        else:
            return 1

    def function_based_service_strategy(self, trust_value) -> int:
        return 1

    def choose_service_provider_FIRE(self, timestep, task_to_robots):
        final_task_list = {} # {1:1, 2:5, 3:6}
        trust_value_records = {} # {1 :{1:{'trust_value':xxx, 'reliability':xxxx}, 4:{'trust_value':xxx, 'reliability':xxxx}}, 2: {}}
        for task, robots in task_to_robots.items():
            if len(robots) == 1:
                final_task_list[task] = robots[0]
            else:
                trust_value_record = {}
                trust_value_record_key = {}
                has_no_trust = []

                # calculate the trust values of different robots working on the same task, store in trust_value
                for i, provider_robot_id in enumerate(robots):
                    self.monitor.robot_communication_range = self.limited_communication_range
                    trust_record = self.trust_engine.calculate_trust_value_reporter(self.id, provider_robot_id, task,
                                                                                    timestep, self.robots_capable_tasks)
                    if self.run_communication_comparison:
                        self.monitor.robot_communication_range = self.unlimited_communication_range
                        unlimited_trust_record = self.trust_engine.calculate_trust_value_reporter(self.id, provider_robot_id, task,
                                                                                        timestep, self.robots_capable_tasks)
                        self.monitor.collect_trust_values(timestep, self.id, provider_robot_id, 'reporter_to_provider', trust_record, unlimited_trust_record)
                    trust_value = trust_record['trust_value']
                    # if there's no history, set trust value to np.nan
                    trust_value_record[provider_robot_id] = trust_record
                    if np.isnan(trust_value):  # use isnan()
                        has_no_trust.append(provider_robot_id)
                    else:
                        trust_value_record_key[provider_robot_id] = trust_value

                # after getting the trust value towards different robots for the same task
                # choose provider based on determined or boltzmann methods
                if self.provider_select_randomness == 'determined':
                    try:
                        max_value = max(trust_value_record.values())
                        max_keys = [key for key, value in trust_value_record.items() if value == max_value]
                        most_trustworthy_robot_id = random.choice(max_keys)
                    except:
                        most_trustworthy_robot_id = random.choice(has_no_trust)
                    final_task_list[task] = most_trustworthy_robot_id

                elif self.provider_select_randomness == 'boltzmann':
                    try:
                        max_value = max(trust_value_record_key.values())
                        max_keys = [key for key, value in trust_value_record_key.items() if value == max_value]
                        # simple boltzmann, 70% 30%
                        if has_no_trust != []:
                            chosen_list = random.choices([max_keys, has_no_trust], weights=[0.85, 0.15])[0]
                        else:
                            chosen_list = max_keys
                        most_trustworthy_robot_id = random.choice(chosen_list)
                        final_task_list[task] = most_trustworthy_robot_id
                    except:
                        try:
                            most_trustworthy_robot_id = random.choice(has_no_trust)
                            final_task_list[task] = most_trustworthy_robot_id
                        except: # all has been down, ignore this task
                            pass
                trust_value_records[task] = trust_value_record

        return final_task_list, trust_value_records

    def choose_service_provider_TRAVOS(self, timestep, task_to_robots):
        final_task_list = {}  # {1:1, 2:5, 3:6}
        trust_value_records = {}  # {1 :{1:{'trust_value_direct':xxx, 'direct_confidence':xxxx, 'trust_value_combined':xxx, 'reputation_confidence':xxxx}, 4:{'trust_value':xxx, 'reliability':xxxx}}, 2: {}}
        # {'trust_value': trust_value, 'trust_type': 'direct', 'gamma': gamma, 'delta': delta, 'confidence': confidence, 'alpha': alpha, 'beta':beta}
        for task, robots in task_to_robots.items():
            if len(robots) == 1:
                final_task_list[task] = robots[0]
            elif len(robots) == 0: # ignore this task
                pass
            else:
                trust_value_record = {}
                wait_reputation_robots = []
                combined_trustworthy_robots = []

                # calculate the trust values of different robots working on the same task, store in trust_value
                for i, provider_robot_id in enumerate(robots):
                    trust_record = self.trust_engine.calculate_direct_trust_value_reporter(self.id, provider_robot_id, task,
                                                                                    timestep, self.robots_capable_tasks)
                    if trust_record['direct_confidence'] < 0.95:
                        wait_reputation_robots.append(provider_robot_id)

                    trust_value_record[provider_robot_id] = trust_record

                if wait_reputation_robots == []:
                    most_direct_trustworthy_robot_id = random.choice(robots)
                    final_task_list[task] = most_direct_trustworthy_robot_id
                else:
                    # calculate combined trust to decide which robot to interact
                    for i, provider_robot_id in enumerate(wait_reputation_robots):
                        self.monitor.robot_communication_range = self.limited_communication_range
                        trust_record = self.trust_engine.calculate_trust_value_reporter(self.id, provider_robot_id, task,
                                                                                    timestep, self.robots_capable_tasks)
                        if self.run_communication_comparison:
                            self.monitor.robot_communication_range = self.unlimited_communication_range
                            unlimited_trust_record = self.trust_engine.calculate_trust_value_reporter(self.id,
                                                                                                      provider_robot_id,
                                                                                                      task,
                                                                                                      timestep,
                                                                                                      self.robots_capable_tasks)
                            self.monitor.collect_trust_values(timestep, self.id, provider_robot_id,
                                                              'reporter_to_provider', trust_record,
                                                              unlimited_trust_record)

                        trust_value_record[provider_robot_id] = trust_record

                    max_trust = max(item['trust_value'] for item in trust_value_record.values())
                    max_items = [key for key, item in trust_value_record.items() if item['trust_value'] == max_trust]
                    final_task_list[task] = random.choice(max_items)

                trust_value_records[task] = trust_value_record

        return final_task_list, trust_value_records

    def choose_service_provider_YUSINGH(self, timestep, task_to_robots):
        final_task_list = {} # {1:1, 2:5, 3:6}
        trust_value_records = {} # {1 :{1:{'trust_value':xxx, 'reliability':xxxx}, 4:{'trust_value':xxx, 'reliability':xxxx}}, 2: {}}
        for task, robots in task_to_robots.items():
            if len(robots) == 1:
                final_task_list[task] = robots[0]
            else:
                trust_value_record = {}
                trust_value_record_key = {}
                has_no_trust = []

                # calculate the trust values of different robots working on the same task, store in trust_value
                for i, provider_robot_id in enumerate(robots):
                    self.monitor.robot_communication_range = self.limited_communication_range
                    trust_record = self.trust_engine.calculate_trust_value_reporter(self.id, provider_robot_id, task,
                                                                                    timestep, self.robots_capable_tasks)
                    if self.run_communication_comparison:
                        self.monitor.robot_communication_range = self.unlimited_communication_range
                        unlimited_trust_record = self.trust_engine.calculate_trust_value_reporter(self.id,
                                                                                                  provider_robot_id,
                                                                                                  task,
                                                                                                  timestep,
                                                                                                  self.robots_capable_tasks)
                        self.monitor.collect_trust_values(timestep, self.id, provider_robot_id,
                                                          'reporter_to_provider', trust_record,
                                                          unlimited_trust_record)
                    trust_value = trust_record['trust_value']
                    # if there's no history, set trust value to np.nan
                    trust_value_record[provider_robot_id] = trust_record
                    if np.isnan(trust_value):  # use isnan()
                        has_no_trust.append(provider_robot_id)
                    else:
                        trust_value_record_key[provider_robot_id] = trust_value

                # after getting the trust value towards different robots for the same task
                # choose provider based on determined or boltzmann methods
                if self.provider_select_randomness == 'determined':
                    try:
                        max_value = max(trust_value_record.values())
                        max_keys = [key for key, value in trust_value_record.items() if value == max_value]
                        most_trustworthy_robot_id = random.choice(max_keys)
                    except:
                        most_trustworthy_robot_id = random.choice(has_no_trust)
                    final_task_list[task] = most_trustworthy_robot_id

                elif self.provider_select_randomness == 'boltzmann':
                    try:
                        max_value = max(trust_value_record_key.values())
                        max_keys = [key for key, value in trust_value_record_key.items() if value == max_value]
                        # simple boltzmann, 70% 30%
                        if has_no_trust != []:
                            chosen_list = random.choices([max_keys, has_no_trust], weights=[0.85, 0.15])[0]
                        else:
                            chosen_list = max_keys
                        most_trustworthy_robot_id = random.choice(chosen_list)
                        final_task_list[task] = most_trustworthy_robot_id
                    except:
                        try:
                            most_trustworthy_robot_id = random.choice(has_no_trust)
                            final_task_list[task] = most_trustworthy_robot_id
                        except:
                            pass

                trust_value_records[task] = trust_value_record

        return final_task_list, trust_value_records


    def choose_service_provider_SUBJECTIVE(self, timestep, task_to_robots):
        final_task_list = {} # {1:1, 2:5, 3:6}
        trust_value_records = {} # {1 :{1:{'trust_value':xxx, 'reliability':xxxx}, 4:{'trust_value':xxx, 'reliability':xxxx}}, 2: {}}
        for task, robots in task_to_robots.items():
            if len(robots) == 1:
                final_task_list[task] = robots[0]
            else:
                trust_value_record = {}
                trust_value_record_key = {}
                has_no_trust = []
                trust_robot = []
                uncertain_robot = []
                untrust_robot = []

                # calculate the trust values of different robots working on the same task, store in trust_value
                for i, provider_robot_id in enumerate(robots):
                    self.monitor.robot_communication_range = self.limited_communication_range
                    trust_record = self.trust_engine.calculate_trust_value_reporter(self.id, provider_robot_id, task,
                                                                                    timestep, self.robots_capable_tasks)
                    if self.run_communication_comparison:
                        self.monitor.robot_communication_range = self.unlimited_communication_range
                        unlimited_trust_record = self.trust_engine.calculate_trust_value_reporter(self.id,
                                                                                                  provider_robot_id,
                                                                                                  task,
                                                                                                  timestep,
                                                                                                  self.robots_capable_tasks)
                        self.monitor.collect_trust_values(timestep, self.id, provider_robot_id,
                                                          'reporter_to_provider', trust_record,
                                                          unlimited_trust_record)
                    trust_value = trust_record['trust_value']
                    # if there's no history, set trust value to np.nan
                    trust_value_record[provider_robot_id] = trust_record
                    if np.isnan(trust_value):  # use isnan()
                        has_no_trust.append(provider_robot_id)
                    elif trust_value == 1:
                        trust_robot.append(provider_robot_id)
                    elif trust_value == 0:
                        uncertain_robot.append(provider_robot_id)
                    elif trust_value == -1:
                        untrust_robot.append(provider_robot_id)

                # after getting the trust value towards different robots for the same task
                # choose provider based on determined or boltzmann methods
                if self.provider_select_randomness == 'determined':
                    if trust_robot != []:
                        most_trustworthy_robot_id = random.choice(trust_robot)
                    else:
                        try:
                            most_trustworthy_robot_id = random.choice(uncertain_robot)
                        except:
                            most_trustworthy_robot_id = random.choice(untrust_robot)
                    final_task_list[task] = most_trustworthy_robot_id

                elif self.provider_select_randomness == 'boltzmann':
                    if trust_robot != [] and uncertain_robot != []:
                        chosen_list = random.choices([trust_robot, uncertain_robot], weights=[0.85, 0.15])[0]
                        most_trustworthy_robot_id = random.choice(chosen_list)
                    else:
                        try:
                            most_trustworthy_robot_id = random.choice(uncertain_robot)
                            final_task_list[task] = most_trustworthy_robot_id
                        except:
                            try:
                                most_trustworthy_robot_id = random.choice(untrust_robot)
                                final_task_list[task] = most_trustworthy_robot_id
                            except:
                                pass

                trust_value_records[task] = trust_value_record

        return final_task_list, trust_value_records

    def choose_service_provider(self, required_tasks, timestep):
        '''
        :param required_tasks: list eg: [0,2,3]; task_to_robot {task:[all robots that are capable of this task]}
        robot capable task: {robot id: [capable task list]}
        task_to_robots eg: {1:[1,5],2:[2,6]}
        Can get {1:[1,5],2:[2,6], 3:[3,7],4:[4,0]} initially
        now we received a required_tasks list [1,2]
        need to return {1:[1,5],2:[2,6]} -> {1:1,2:2}
        :return: task_ro_robot_assignment {task: robot_id}
                    trust_value_records_example = {
                        1: {  # Task ID 1
                            5: 0.85,  # Trust value for robot 5
                            7: 0.78   # Trust value for robot 7
                        },
                        2: {  # Task ID 2
                            2: 0.92,  # Trust value for robot 2
                            6: 0.89   # Trust value for robot 6
                        }
                    }
        '''
        # todo: update task_to_robot to prevent requesting recharging robot
        # 1. delete the other unrequired task
        task_to_robots = {key: self.task_to_robot[key] for key in required_tasks if key in self.task_to_robot}
        # 2. delete request robot id from the robot list
        for task, robots in task_to_robots.items():
            if self.id in robots:
                robots.remove(self.id) # A robot cannot call itself to help
            # 3. delete recharging robot
            for robot in robots:
                if robot in self.monitor.get_recharging_robots():
                    robots.remove(robot)

        # select provider based on trust model
        if self.provider_select_strategy == 'trust':
            if self.trust_algo =='FIRE':
                return self.choose_service_provider_FIRE(timestep, task_to_robots)
            elif self.trust_algo =='TRAVOS':
                return self.choose_service_provider_TRAVOS(timestep, task_to_robots)
            elif self.trust_algo =='YUSINGH':
                return self.choose_service_provider_YUSINGH(timestep, task_to_robots)
            elif self.trust_algo == 'FUZZY':
                return self.choose_service_provider_FIRE(timestep, task_to_robots)
            elif self.trust_algo =='SUBJECTIVE':
                return self.choose_service_provider_SUBJECTIVE(timestep, task_to_robots)
            elif self.trust_algo =='ML':
                return self.choose_service_provider_FIRE(timestep, task_to_robots)



        # select provider randomly across all the available robots
        elif self.provider_select_strategy == 'random':
            return {task: random.choice(task_to_robots[task])
                    for task in required_tasks
                    if task_to_robots[task]}, 'random'

        # select provider with determined strategy
        elif self.provider_select_strategy == 'determined':
            return {1: 1, 2: 2, 3: 3, 0: 4}, 'determined'

    def choose_service_quality(self, request_robot_id, task_info, timestep):
        if self.service_select_strategy == 'trust':

            if self.trust_algo == 'TRAVOS':
                self.monitor.robot_communication_range = self.limited_communication_range
                trust_record = self.trust_engine.calculate_direct_trust_value_provider(request_robot_id, self.id, task_info, timestep, self.robots_capable_tasks)
                if self.run_communication_comparison:
                    self.monitor.robot_communication_range = self.unlimited_communication_range
                    unlimited_trust_record = self.trust_engine.calculate_direct_trust_value_provider(request_robot_id, self.id, task_info, timestep, self.robots_capable_tasks)
                    self.monitor.collect_trust_values(timestep, request_robot_id,  self.id,
                                                      'provider_to_reporter', trust_record,
                                                      unlimited_trust_record)
                if trust_record['direct_confidence'] < 0.95:
                    self.monitor.robot_communication_range = self.limited_communication_range
                    trust_record = self.trust_engine.calculate_trust_value_provider(request_robot_id, self.id,
                                                                                           task_info, timestep,
                                                                                           self.robots_capable_tasks)
                    if self.run_communication_comparison:
                        self.monitor.robot_communication_range = self.unlimited_communication_range
                        unlimited_trust_record = self.trust_engine.calculate_trust_value_provider(request_robot_id, self.id,
                                                                                           task_info, timestep,
                                                                                           self.robots_capable_tasks)
                        self.monitor.collect_trust_values(timestep, request_robot_id, self.id,
                                                          'provider_to_reporter', trust_record,
                                                          unlimited_trust_record)


            else:
                self.monitor.robot_communication_range = self.limited_communication_range
                trust_record = self.trust_engine.calculate_trust_value_provider(request_robot_id, self.id, task_info, timestep, self.robots_capable_tasks)
                if self.run_communication_comparison:
                    self.monitor.robot_communication_range = self.unlimited_communication_range
                    unlimited_trust_record = self.trust_engine.calculate_trust_value_provider(request_robot_id, self.id, task_info, timestep, self.robots_capable_tasks)
                    self.monitor.collect_trust_values(timestep, request_robot_id, self.id,
                                                      'provider_to_reporter', trust_record,
                                                      unlimited_trust_record)

            # if self.trust_algo == 'FIRE':
            #     self.monitor.robot_communication_range = self.limited_communication_range
            #     trust_record = self.trust_engine.calculate_trust_value_provider(request_robot_id, self.id, task_info,
            #                                                                     timestep, self.robots_capable_tasks)
            # elif self.trust_algo == 'YUSINGH':
            #     trust_record = self.trust_engine.calculate_trust_value_provider(request_robot_id, self.id, task_info, timestep, self.robots_capable_tasks)
            # elif self.trust_algo =='FUZZY':
            #     trust_record = self.trust_engine.calculate_trust_value_provider(request_robot_id, self.id, task_info, timestep, self.robots_capable_tasks)
            # elif self.trust_algo =='SUBJECTIVE':
            #     trust_record = self.trust_engine.calculate_trust_value_provider(request_robot_id, self.id, task_info, timestep, self.robots_capable_tasks)
            # elif self.trust_algo == 'ML':
            #     trust_record = self.trust_engine.calculate_trust_value_provider(request_robot_id, self.id, task_info, timestep, self.robots_capable_tasks)

            # decide what to do based on the trust value: (1) reach threshold then dead
            # (2) map function between the trust value and the strategy
            trust_value = trust_record['trust_value']
            if 'threshold' in self.service_strategy_based_on_trust:
                return self.threshold_based_service_strategy(trust_value, threshold=float(self.service_strategy_based_on_trust['threshold'])),trust_record
            elif self.service_strategy_based_on_trust == 'function':
                return self.function_based_service_strategy(trust_value), trust_record

        elif self.service_select_strategy =='good':
            return 1, 'good'
        elif self.service_select_strategy == 'bad':
            return 0, 'bad'
        elif self.service_select_strategy == 'random':
            return random.randint(0, 1), 'random'
        elif 'ignore0' in self.service_select_strategy:
            # probability that other robot will ignore Robot 0
            probability = float(self.service_select_strategy.split('_')[1])
            if request_robot_id == 0 and random.random() < probability:
                return 0, self.service_select_strategy
            else:
                return 1, self.service_select_strategy

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

    def update_battery_time(self):
        self.battery_time = random.randint(2000,8000)

    def update_recharging_time(self):
        self.recharge_time = random.randint(300, 600)

    def check_trustworthy(self):
        if (self.true_positive, self.false_positve) == self.trustworthy_robot_setting:
            return 1
        else:
            return 0

    def check_coopertaiveness(self):
        if self.uncooperativeness == 0:
            return 1
        else:
            return 0

    def step(self, verbose=False, **kwargs):
        timestep = kwargs.get('timestep')
        intention_table = kwargs.get('intention_table')
        idleness_log = kwargs.get('idleness_log')
        impression = {}

        if self.battery_time > 0:
            self.battery_time -= 1

        # robot battery reaches 0, or below 0 because if it recahes 0 during reporting/serving, we let it finish its work
        if self.battery_time <= 0 and self.state == 'Patrolling':
            # if exist other robots that are capable of this task
            if self.monitor.check_if_rechargable(self.task_to_robot, self.id):
                self.state = 'Recharging' # the stragedy of a space
                self.update_recharging_time()
                self.service_time = self.recharge_time
                self.monitor.set_recharging_robot(self.id)
            else:
                # no robot is capable of carrying out this task, unable to leave and recharge
                self.update_battery_time()

        # If is in service state, update service time
        if self.service_time > 0:
            self.service_time -= 1

        # check trust dynamic
        if timestep in self.trust_dynamic_timestep.keys() and self.id in self.trust_dynamic_timestep[timestep].keys():
            if self.trust_dynamic_timestep[timestep][self.id]: # become trustworthy
                self.true_positive = self.trustworthy_robot_setting[0]
                self.false_positve = self.trustworthy_robot_setting[1]
            else: # become untrustworthy
                self.true_positive = self.untrustworthy_robot_setting[0]
                self.false_positve = self.untrustworthy_robot_setting[1]

        # check cooperativeness dynamic
        if timestep in self.cooperativeness_dynamic_timestep.keys() and self.id in self.cooperativeness_dynamic_timestep[timestep].keys():
            if self.cooperativeness_dynamic_timestep[timestep][self.id]: # become cooperative
                self.uncooperativeness = 0
            else: # become uncooperative
                self.uncooperativeness = self.uncooperative_robot_setting

        # parameters all updated, start patrol/request/provide
        # If reach an interest point, and it's not in a detection cycle, then could find anomaly
        # and it's not a service robot, service robot only provide help
        if self.path_list == [] and self.is_guide_robot:
            # check which node robot is on
            self.last_node = int(self.check_node())
            self.path_list = self.algo_engine.calculate_next_path(self.id, self.last_node)

        if self.path_list == [] and self.is_sweep_robot:
            # check which node robot is on
            self.last_node = int(self.check_node())
            self.path_list = self.algo_engine.calculate_next_path(self.id, self.last_node)

        if self.path_list == [] and not self.is_guide_robot and not self.is_sweep_robot:
            # check current anomaly point position
            self.true_anomaly_pos = self.monitor.get_anomaly_pos()
            # check which node robot is on
            self.last_node = int(self.check_node())

            if self.patrol_algo == 'partition':
                # calculate the next place to go
                self.path_list = self.algo_engine.calculate_next_path(self.id, self.last_node)

            elif self.patrol_algo == 'SEBS':
                # get the latest idleness log
                self.path_list, self.goal_node = self.algo_engine.calculate_next_path(self.id, idleness_log, intention_table, self.last_node)

            # start anomaly detection and cooperation
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
                    # choose service provider based on trust
                    name_list, trust_record = self.choose_service_provider(required_tasks, timestep)
                    self.monitor.inform_request(self.id, name_list, self.current_pos, 1, timestep, trust_record, self.check_trustworthy())
                    # determine service_time based on astar distance
                    self.service_time = 2  # waiting for report
                    self.logger.info(
                        f"Reporter_id {self.id}, Reporter Position: {self.current_pos}, Last Node: {self.last_node}, Service time: {self.service_time}"
                        f" Required tasks: {required_tasks}, Required robot namelist: {name_list}, True/False anomaly: True, Trust record: {trust_record}")
            else:
                # may generate false alarm
                if random.random() < self.false_positve and not anomaly_detect_cycle_flag:
                    # set the state to reporting
                    self.state = 'False Requesting'
                    required_tasks = random.sample(self.required_tasks_list,
                                                   random.randint(1, len(self.required_tasks_list)))
                    # choose service provider based on trust
                    name_list, trust_record = self.choose_service_provider(required_tasks, timestep)
                    self.monitor.inform_request(self.id, name_list, self.current_pos, 0, timestep, trust_record, self.check_trustworthy())
                    self.service_time = 2 # wait 2 timestep for the monitor to gather providers' decision, and then update waiting time
                    self.logger.info(
                        f"Reporter_id {self.id}, Reporter Position: {self.current_pos}, Last Node: {self.last_node}, Service time: {self.service_time}"
                        f" Required tasks: {required_tasks}, Required robot namelist: {name_list}, True/False anomaly: False, Trust record: {trust_record}")

        # see how long have to wait until the service robot come to provide
        if 'Requesting' in self.state and self.service_time == 0:
            self.service_time = self.monitor.check_waiting_time_reporter(self.id, timestep)

        # this is behind anomaly reporting section because we want robot to move away after one anomaly detection cycle
        if self.service_time <= 1 and self.battery_time <= 0 and self.state == 'Recharging':
            # update battery time, cant put it in the next if because service_time=0 can result from requesting/serving
            self.update_battery_time()
            # update task_to_robot, this robot is ready to serve again
            self.monitor.release_recharging_robot(self.id)

        # if didn't find anomaly or providing services, robot move
        if self.service_time <= 0:
            self.service_time = 0
            self.state = 'Patrolling'
            try:
                # move back to the path_list, move 1 step
                self.current_pos = self.path_list[0]
                self.path_list.pop(0)
            except IndexError:
                pass
                # todo: see what's wrong and why work?
                # print(self.id, timestep, self.algo_engine, self.path_list)

        # If someone is requesting for help at this timestep, switch to provider mode
        if self.monitor.check_request(self.id, timestep) != None:
            self.state = 'Providing'
            current_request = self.monitor.check_request(self.id, timestep)
            impression = current_request
            request_robot_id = current_request['request_robot']
            # for uncooperative robot,
            if random.random() < self.uncooperativeness:
                service_quality = 0
                trust_record = 'uncooperative'
            else:
                # if not uncooperative, choose_service_quality based on trust
                service_quality, trust_record = self.choose_service_quality(request_robot_id, current_request['task'], timestep)
            impression['service_quality'] = service_quality
            impression['trust_value_towards_reporter'] = trust_record
            impression['service_position'] = self.current_pos
            impression['service_time'] = timestep
            impression['is_same_type'] = 1 if self.robots_capable_tasks[request_robot_id] == self.robots_capable_tasks[self.id] else 0
            impression['provider_cooperativeness'] = self.check_coopertaiveness()
            is_true_anomaly = impression['is_true_anomaly']

            if service_quality == 1:
                # Only when providing good service will service provider receive negative reward proportional to distance
                # distance = len(
                #     calculate_shortest_path(self.pgm_map_matrix, self.current_pos, current_request['request_position']))
                distance = abs(self.current_pos[0] - current_request['request_position'][0]) + abs(
                    self.current_pos[1] - current_request['request_position'][1])
                # self.service_time = distance
                self.service_time = distance  # in dynamic environment, you have to show the service time.
                impression['distance'] = distance
                # record reward
                if is_true_anomaly == 1:
                    impression['reward'] = self.extra_reward - 2 * distance
                else:  # false alarm, negative reward
                    impression['reward'] = - 2 * distance

            elif service_quality == 0:
                # if bad service, calculate the reward of this interaction
                impression['distance'] = 0
                if is_true_anomaly == 1:
                    impression['reward'] = self.env_penalty
                else:
                    impression['reward'] = 0

            self.logger.info(
                f"Provider_id {self.id}, Provider Position: {self.current_pos}, Last Node: {self.last_node},"
                f" Request record: {impression}, Service time: {self.service_time}")

        self.logger.info(
            f"Robot {self.id}, Current Position: {self.current_pos}, Current State: {self.state}, Last Node: {self.last_node}, Service time: {self.service_time}, Battery time: {self.battery_time}")

        # For visualisation
        if verbose == True:
            print(f"Robot_{self.id} {self.state} at {self.current_pos}")

        return self.current_pos, impression, self.state
