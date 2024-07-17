from basic_patrol_class.Monitor import Monitor

class StaticMonitor(Monitor):
    def __init__(self, robot_num):
        super().__init__()
        self.anomaly = -1
        self.current_request = {}
        self.rewards = []
        self.in_cycle_flag = 0
        self.true_report_num = 0
        self.false_report_num = 0
        self.robot_num = robot_num

        # history data for trust model calculation: {1:{2:[[come or not/true or false, reward, timestep],[]],3:[histories]...}}
        self.reporter_histories = self.generate_history_dict()
        self.provider_histories = self.generate_history_dict()
        self.informative_impressions = []

    def generate_history_dict(self):
        '''
        {1:{2:[histories],3:[],4:[]},
         2:{1:[],3:[],4:[]},
         3:{1:[],2:[],4:[]},
         4:{1:[],2:[],3:[]},
        }
        :return:
        '''
        robot_dict = {}
        for i in range(0, self.robot_num):
            robot_dict[i] = {j: [] for j in range(0, self.robot_num) if j != i}
        return robot_dict

    def update_anomaly_pos(self,new_pos):
        self.anomaly = new_pos

    def get_anomaly_pos(self):
        return self.anomaly

    def collect_reward(self, single_reward):
        self.rewards.append(single_reward)

    def collect_infomative_impressions(self, impression):
        self.informative_impressions.append(impression)

    def check_request(self, service_robot_id, time_step):
        try:
            if self.current_request[service_robot_id]['time'] == time_step - 1:
                return self.current_request[service_robot_id]
        except:
            return None

    def inform_request(self, request_robot_id, name_list, request_pos, is_true_anomaly, timestep, trust_value_towards_provider):
        '''
        :param request_robot_id:
        :param name_list: {1:0, 2:2, 3:1}
        :return:
        '''
        if is_true_anomaly == 1:
            self.true_report_num += 1
        else:
            self.false_report_num += 1

        for task_id, robot_id in name_list.items():
            self.current_request[robot_id] = {'request_robot':request_robot_id, 'service_robot': robot_id, 'time': timestep,
                                          'task':task_id, 'request_position': request_pos, 'is_true_anomaly': is_true_anomaly,
                                              'trust_value_towards_provider': trust_value_towards_provider,}


    def collect_reporter_history(self, reporter_history):
        reporter_id = reporter_history[0]
        provider_id = reporter_history[1]
        add_history = reporter_history[2]
        self.reporter_histories[reporter_id][provider_id].append(add_history)

    def collect_provider_history(self, provider_history):
        reporter_id = provider_history[0]
        provider_id = provider_history[1]
        add_history = provider_history[2]
        self.provider_histories[provider_id][reporter_id].append(add_history)

    def set_in_cycle_flag(self):
        self.in_cycle_flag = 1

    def cancel_in_cycle_flag(self):
        self.in_cycle_flag = 0

    def get_in_cycle_flag(self):
        return self.in_cycle_flag

    def get_history_as_provider(self, provider_id, reporter_id):
        return self.provider_histories[provider_id][reporter_id]

    def get_history_as_reporter(self, reporter_id, provider_id):
        return self.reporter_histories[reporter_id][provider_id]

    def get_history_as_reporter_witness(self, reporter_id, provider_id, communication_range):
        witness_history = {}
        try:
            current_robot_pos = self.robot_pos[-1]
        except: # spot anomaly in the first round, no pos information recorded yet
            return []

        reporter_pos = current_robot_pos[reporter_id]
        for i, pos in enumerate(current_robot_pos):
            distance = ((pos[0]-reporter_pos[0])**2 + (pos[1]-reporter_pos[1])**2) ** 0.5
            if distance <= communication_range and i != reporter_id:
                witness_history[i] = self.provider_histories[i][provider_id]
        return witness_history
    def get_history_as_provider_witness(self, provider_id, reporter_id, communication_range):
        witness_history = {}
        try:
            current_robot_pos = self.robot_pos[-1]
        except:  # spot anomaly in the first round, no pos information recorded yet
            return []

        provider_pos = current_robot_pos[reporter_id]
        for i, pos in enumerate(current_robot_pos):
            distance = ((pos[0] - provider_pos[0]) ** 2 + (pos[1] - provider_pos[1]) ** 2) ** 0.5
            if distance <= communication_range and i != provider_id:
                witness_history[i] = self.provider_histories[i][provider_id]
        return witness_history


