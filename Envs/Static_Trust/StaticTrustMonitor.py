from patrol_class.Monitor import Monitor

class StaticMonitor(Monitor):
    def __init__(self):
        super().__init__()
        self.anomaly = -1
        self.current_request = {}
        self.impressions = []
        self.rewards = []
        self.in_cycle_flag = 0
        self.true_report_num = 0
        self.false_report_num = 0

    def update_anomaly_pos(self,new_pos):
        self.anomaly = new_pos

    def get_anomaly_pos(self):
        return self.anomaly

    def collect_reward(self, single_reward):
        self.rewards.append(single_reward)

    def check_request(self, service_robot_id, time_step):
        try:
            if self.current_request[service_robot_id]['time'] == time_step - 1:
                return self.current_request[service_robot_id]
        except:
            return None

    def inform_request(self, request_robot_id, name_list, request_pos, is_true_anomaly, timestep):
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
                                          'task':task_id, 'request_position': request_pos, 'is_true_anomaly': is_true_anomaly}


    def collect_robot_impression(self, env_interaction_impression):
        self.impressions.append(env_interaction_impression)

    def set_in_cycle_flag(self):
        self.in_cycle_flag = 1

    def cancle_in_cycle_flag(self):
        self.in_cycle_flag = 0

    def get_in_cycle_flag(self):
        return self.in_cycle_flag



