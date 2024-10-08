from basic_patrol_class.Monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
import random
class DynamicMonitor(Monitor):
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
        self.histories = []
        self.communication_comparison_experiment = []
        self.informative_impressions = []
        self.recharging_robots = []
        self.robot_communication_range = -1
    def update_robot_communication_range(self, range):
        self.robot_communication_range = range

    def calculate_distance(self, robot_id1, robot_id2):
        robot_1_pos = self.robot_pos[-1][robot_id1]
        robot_2_pos = self.robot_pos[-1][robot_id2]
        return ((robot_1_pos[0]-robot_2_pos[0])**2 + (robot_1_pos[1]-robot_2_pos[1])**2) ** 0.5

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

    def get_latest_idleness(self):
        return self.node_idleness[-1]

    def update_anomaly_pos(self,new_pos):
        self.anomaly = new_pos

    def get_anomaly_pos(self):
        return self.anomaly

    def collect_reward(self, single_reward):
        self.rewards.append(single_reward)

    def collect_infomative_impressions(self, impression):
        self.informative_impressions.append(impression)

    def collect_histories(self, histories):
        self.histories.extend(histories)
        for h in histories:
            # history: [timestep, rating, task]
            self.collect_reporter_history([h['reporter_id'], h['provider_id'],[h['provide_time'], h['rating_to_provider'], h['task_id'], h['is_same_type']]])
            self.collect_provider_history([h['reporter_id'], h['provider_id'],[h['provide_time'], h['rating_to_reporter'], h['task_id'], h['is_same_type']]])

    def check_request(self, service_robot_id, time_step):
        try:
            if self.current_request[service_robot_id]['time'] == time_step - 1:
                return self.current_request[service_robot_id]
        except:
            return None

    def check_waiting_time_reporter(self, reporter_id, timestep):
        try:
            distance = max([history['distance_penalty'] for history in self.histories if history['reporter_id'] == reporter_id and history['report_time'] == timestep - 2])
        except:
            distance = 0
        # because reporter only need to wait until all robots come to help, and then it can run away.
        return distance/2

    def inform_request(self, request_robot_id, name_list, request_pos, is_true_anomaly, timestep, trust_value_towards_provider, reporter_trustworthiness):
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
            try:
                trust_value_to_provider = trust_value_towards_provider[task_id][robot_id]
            except:
                trust_value_to_provider = 'only'
            self.current_request[robot_id] = {'request_robot':request_robot_id, 'service_robot': robot_id, 'time': timestep,
                                          'task':task_id, 'request_position': request_pos, 'is_true_anomaly': is_true_anomaly,
                                              'trust_value_towards_provider': trust_value_towards_provider,
                                              'trust_value_to_provider': trust_value_to_provider,
                                              'reporter_trustworthiness': reporter_trustworthiness}

    def collect_trust_values(self, timestep, reporter_id, provider_id, mode, limited_trust_record, unlimited_trust_record):
        record = {
            'timestep': timestep,
            'reporter_id': reporter_id,
            'provider_id': provider_id,
            'trust_direction': mode,
            'limited_trust_record': limited_trust_record,
            'unlimited_trust_record': unlimited_trust_record
        }
        self.communication_comparison_experiment.append(record)


    def get_recharging_robots(self):
        return self.recharging_robots

    def set_recharging_robot(self, robot_id):
        self.recharging_robots.append(robot_id)

    def check_if_rechargable(self, task_to_robot, robot_id):
        is_rechargable = True
        # only keep those tasks related with the robot himself:
        task_to_robots = {task: robots for task, robots in task_to_robot.items() if robot_id in robots}
        for task, robots in task_to_robots.items():
            for robot in self.recharging_robots:
                try:
                    robots.remove(robot)
                except:
                    pass
            if len(robots) == 1:
                is_rechargable = False
                return is_rechargable

        return is_rechargable
    def release_recharging_robot(self, robot_id):
        self.recharging_robots.remove(robot_id)

    def average_reward_per_round(self):
        reward = []
        for h in self.histories:
            reward.append(h['reporter_reward'] + h['provider_reward'])
        average_reward = sum(reward)/len(reward)
        print(average_reward)
        return average_reward

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

    def get_history_as_provider(self, reporter_id, provider_id):
        return self.provider_histories[provider_id][reporter_id]

    def get_history_as_reporter(self, reporter_id, provider_id):
        return self.reporter_histories[reporter_id][provider_id]

    def get_history_as_reporter_witness_FIRE(self, reporter_id, provider_id, local_history_length = 10, referral_length=2):

        witness_history = {}
        max_jumps = referral_length
        try:
            current_robot_pos = self.robot_pos[-1]
        except:  # Spot anomaly in the first round, no pos information recorded yet
            return []

        robot_within_communication = [(reporter_id, 0)]  # Tuple with robot ID and jump level
        visited_robot = {reporter_id}

        # Use a while loop to avoid modifying the list while iterating over it
        while robot_within_communication:
            robot, current_jump = robot_within_communication.pop(
                0)  # Process the first robot in the list with its jump level
            center_robot_pos = current_robot_pos[robot]

            if current_jump >= max_jumps:
                continue  # Stop processing if the maximum number of jumps is reached

            for i, pos in enumerate(current_robot_pos):
                distance = ((pos[0] - center_robot_pos[0]) ** 2 + (pos[1] - center_robot_pos[1]) ** 2) ** 0.5

                if distance <= self.robot_communication_range and i not in visited_robot and i != provider_id:
                    witness_history[i] = self.reporter_histories[i][provider_id][-local_history_length:]
                    # Add to the list to process if not already visited
                    robot_within_communication.append((i, current_jump + 1))
                    visited_robot.add(i)

        return sum(list(witness_history.values()),[])

    def get_history_as_provider_witness_FIRE(self, reporter_id, provider_id, local_history_length=10, referral_length=2):
        witness_history = {}
        max_jumps = referral_length
        try:
            current_robot_pos = self.robot_pos[-1]
        except:  # Spot anomaly in the first round, no pos information recorded yet
            return []

        # Initialize the list with the provider and set the initial jump count to 0
        robot_within_communication = [(provider_id, 0)]  # Tuple with robot ID and jump level
        visited_robot = {provider_id}

        # Use a while loop to avoid modifying the list while iterating over it
        while robot_within_communication:
            robot, current_jump = robot_within_communication.pop(0)  # Process the first robot in the list with its jump level
            center_robot_pos = current_robot_pos[robot]

            if current_jump >= max_jumps:
                continue  # Stop processing if the maximum number of jumps is reached

            for i, pos in enumerate(current_robot_pos):
                distance = ((pos[0] - center_robot_pos[0]) ** 2 + (pos[1] - center_robot_pos[1]) ** 2) ** 0.5

                if distance <= self.robot_communication_range and i not in visited_robot and i != provider_id and i != reporter_id:
                    witness_history[i] = self.provider_histories[i][reporter_id][-local_history_length:]
                    # Add to the list to process if not already visited
                    robot_within_communication.append((i, current_jump + 1))
                    visited_robot.add(i)

        return sum(list(witness_history.values()), [])

    def get_history_as_reporter_certified_witness_FIRE(self, reporter_id, provider_id):
        witness_history = {}
        try:
            current_robot_pos = self.robot_pos[-1]
        except: # spot anomaly in the first round, no pos information recorded yet
            return []

        for i, pos in enumerate(current_robot_pos):
            if i != provider_id and i != reporter_id:
                witness_history[i] = self.reporter_histories[i][provider_id]

        return sum(list(witness_history.values()),[])

    def get_history_as_provider_certified_witness_FIRE(self, reporter_id, provider_id):
        witness_history = {}
        try:
            current_robot_pos = self.robot_pos[-1]
        except: # spot anomaly in the first round, no pos information recorded yet
            return []

        for i, pos in enumerate(current_robot_pos):
            if i != provider_id and i != reporter_id:
                witness_history[i] = self.provider_histories[i][reporter_id]

        return sum(list(witness_history.values()),[])

    def get_history_as_reporter_witness_TRAVOS(self, reporter_id, provider_id, local_history_length=10,
                                               referral_length=2):
        witness_history = {}
        max_jumps = referral_length  # Set the maximum number of jumps allowed

        try:
            current_robot_pos = self.robot_pos[-1]
        except:  # Spot anomaly in the first round, no pos information recorded yet
            return {}

        # Initialize the list with the reporter and set the initial jump count to 0
        robot_within_communication = [(reporter_id, 0)]  # Tuple with robot ID and jump level
        visited_robot = {reporter_id}

        # Use a while loop to avoid modifying the list while iterating over it
        while robot_within_communication:
            robot, current_jump = robot_within_communication.pop(
                0)  # Process the first robot in the list with its jump level
            center_robot_pos = current_robot_pos[robot]

            if current_jump >= max_jumps:
                continue  # Stop processing if the maximum number of jumps is reached

            for i, pos in enumerate(current_robot_pos):
                distance = ((pos[0] - center_robot_pos[0]) ** 2 + (pos[1] - center_robot_pos[1]) ** 2) ** 0.5

                if distance <= self.robot_communication_range and i not in visited_robot and i != reporter_id and i != provider_id:
                    witness_history[i] = self.reporter_histories[i][provider_id][-local_history_length:]
                    # Add to the list to process if not already visited
                    robot_within_communication.append((i, current_jump + 1))
                    visited_robot.add(i)

        return witness_history

    def get_history_as_provider_witness_TRAVOS(self, reporter_id, provider_id, local_history_length=10,
                                               referral_length=2):
        witness_history = {}
        max_jumps = referral_length  # Set the maximum number of jumps allowed

        try:
            current_robot_pos = self.robot_pos[-1]
        except:  # Spot anomaly in the first round, no pos information recorded yet
            return {}

        # Initialize the list with the provider and set the initial jump count to 0
        robot_within_communication = [(provider_id, 0)]  # Tuple with robot ID and jump level
        visited_robot = {provider_id}

        # Use a while loop to avoid modifying the list while iterating over it
        while robot_within_communication:
            robot, current_jump = robot_within_communication.pop(
                0)  # Process the first robot in the list with its jump level
            center_robot_pos = current_robot_pos[robot]

            if current_jump >= max_jumps:
                continue  # Stop processing if the maximum number of jumps is reached

            for i, pos in enumerate(current_robot_pos):
                distance = ((pos[0] - center_robot_pos[0]) ** 2 + (pos[1] - center_robot_pos[1]) ** 2) ** 0.5

                if distance <= self.robot_communication_range and i not in visited_robot and i != provider_id and i != reporter_id:
                    witness_history[i] = self.provider_histories[i][reporter_id][-local_history_length:]
                    # Add to the list to process if not already visited
                    robot_within_communication.append((i, current_jump + 1))
                    visited_robot.add(i)

        return witness_history

    def get_history_as_reporter_witness_SUBJECTIVE(self, reporter_id, provider_id, last_interaction_timestep,
                                                   local_history_length=10, referral_length=2):
        witness_history = {}
        max_jumps = referral_length  # Set the maximum number of jumps allowed

        try:
            current_robot_pos = self.robot_pos[-1]
        except:  # Spot anomaly in the first round, no pos information recorded yet
            return {}

        # Initialize the list with the reporter and set the initial jump count to 0
        robot_within_communication = [(reporter_id, 0)]  # Tuple with robot ID and jump level
        visited_robot = {reporter_id}

        # Use a while loop to avoid modifying the list while iterating over it
        while robot_within_communication:
            robot, current_jump = robot_within_communication.pop(
                0)  # Process the first robot in the list with its jump level
            center_robot_pos = current_robot_pos[robot]

            if current_jump >= max_jumps:
                continue  # Stop processing if the maximum number of jumps is reached

            for i, pos in enumerate(current_robot_pos):
                distance = ((pos[0] - center_robot_pos[0]) ** 2 + (pos[1] - center_robot_pos[1]) ** 2) ** 0.5

                if distance <= self.robot_communication_range and i not in visited_robot and i != reporter_id and i != provider_id:
                    # Filter the history based on the last interaction timestep
                    old_history = [history for history in self.reporter_histories[i][provider_id] if
                                   history[1] < last_interaction_timestep]
                    witness_history[i] = old_history[-local_history_length:]

                    # Add to the list to process if not already visited
                    robot_within_communication.append((i, current_jump + 1))
                    visited_robot.add(i)

        return witness_history

    def get_history_as_provider_witness_SUBJECTIVE(self, reporter_id, provider_id, last_interaction_timestep,
                                                   local_history_length=10, referral_length=2,):
        witness_history = {}
        max_jumps = referral_length  # Set the maximum number of jumps allowed

        try:
            current_robot_pos = self.robot_pos[-1]
        except:  # Spot anomaly in the first round, no pos information recorded yet
            return {}

        # Initialize the list with the provider and set the initial jump count to 0
        robot_within_communication = [(provider_id, 0)]  # Tuple with robot ID and jump level
        visited_robot = {provider_id}

        # Use a while loop to avoid modifying the list while iterating over it
        while robot_within_communication:
            robot, current_jump = robot_within_communication.pop(
                0)  # Process the first robot in the list with its jump level
            center_robot_pos = current_robot_pos[robot]

            if current_jump >= max_jumps:
                continue  # Stop processing if the maximum number of jumps is reached

            for i, pos in enumerate(current_robot_pos):
                distance = ((pos[0] - center_robot_pos[0]) ** 2 + (pos[1] - center_robot_pos[1]) ** 2) ** 0.5

                if distance <= self.robot_communication_range and i not in visited_robot and i != provider_id and i != reporter_id:
                    # Filter the history based on the last interaction timestep
                    old_history = [history for history in self.provider_histories[i][reporter_id] if
                                   history[1] < last_interaction_timestep]
                    witness_history[i] = old_history[-local_history_length:]

                    # Add to the list to process if not already visited
                    robot_within_communication.append((i, current_jump + 1))
                    visited_robot.add(i)

        return witness_history


    # history plot

    # history example:
    # {'is_true_anomaly': 0, 'reporter_id': 0, 'provider_id': 3, 'task_id': 3, 'report_time': 1, 'provide_time': 2,
    #  'report_position': (160, 140), 'provide_position': (430, 262), 'trust_towards_reporter': nan,
    #  'trust_towards_provider': nan, 'provider_action': 1, 'provider_reward': -784, 'reporter_reward': -392,
    #  'rating_to_reporter': -1, 'rating_to_provider': -1, 'distance_penalty': 392}
    def reward_with_untrustworthy_plot(self, untrust_robot_id):
        reward = []
        timestep = []
        for h in self.histories:
            if h['reporter_id'] == untrust_robot_id:
                reward.append(h['reporter_reward'])
                timestep.append(h['report_time'])

        plt.plot(timestep, reward)
        plt.show()

    def trust_with_untrustworthy_plot(self, untrust_robot_id):
        is_true_anomaly = []
        timestep = []

        trust_towards_reporter = []
        provider_id = []

        for h in self.histories:
            if h['reporter_id'] == untrust_robot_id:
                is_true_anomaly.append(h['is_true_anomaly'])
                timestep.append(h['report_time'])
                if h['trust_towards_reporter'] == 'good':
                    trust_towards_reporter.append(1.0)
                else:
                    trust_towards_reporter.append(h['trust_towards_reporter'])

                provider_id.append(h['provider_id'])

        # Convert to numpy arrays for easier handling
        timestep = np.array(timestep)
        is_true_anomaly = np.array(is_true_anomaly)
        trust_towards_reporter = np.array(trust_towards_reporter)
        provider_id = np.array(provider_id)

        # Create a color map for different providers
        unique_providers = np.unique(provider_id)
        colors = plt.cm.get_cmap('tab10', len(unique_providers))

        # Plot the anomalies
        plt.figure(figsize=(12, 6))

        # True anomalies (green)
        plt.scatter(timestep[is_true_anomaly == 1], [0] * sum(is_true_anomaly == 1),
                    color='green', label='True Anomaly', marker='x')

        # False anomalies (red)
        plt.scatter(timestep[is_true_anomaly == 0], [0] * sum(is_true_anomaly == 0),
                    color='red', label='False Anomaly', marker='x')

        # Plot the trust values with different colors for each provider
        for i, provider in enumerate(unique_providers):
            provider_mask = provider_id == provider
            plt.scatter(timestep[provider_mask], trust_towards_reporter[provider_mask],
                        color=colors(i), label=f'Provider {provider}')

        # Set labels and title
        plt.xlabel('Timestep')
        plt.ylabel('Trust Value / Anomaly Indicator')
        plt.title(f'Trust Towards Untrustworthy Robot {untrust_robot_id}')

        # Draw a horizontal line at y=0 for clarity
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        # Add legend
        plt.legend()

        # Show plot
        plt.show()

    def combined_reward_trust_with_untrustworthy_robot_plot(self, untrust_robot_id, strategy_name):
        # Data for the reward plot
        reward = []
        timestep_reward = []

        # Data for the trust plot
        is_true_anomaly = []
        timestep_trust = []
        trust_towards_reporter = []
        provider_id = []

        # Extracting data for trust, reward and anomalies
        for h in self.histories:
            if h['reporter_id'] == untrust_robot_id:
                is_true_anomaly.append(h['is_true_anomaly'])
                timestep_trust.append(h['report_time'])
                reward.append(h['provider_reward'])
                if h['trust_towards_reporter'] == 'good':
                    trust_towards_reporter.append(1.0)
                else:
                    trust_towards_reporter.append(h['trust_towards_reporter'])
                provider_id.append(h['provider_id'])

        # Convert to numpy arrays for easier handling
        timestep_trust = np.array(timestep_trust)
        timestep_reward = timestep_trust
        is_true_anomaly = np.array(is_true_anomaly)
        trust_towards_reporter = np.array(trust_towards_reporter)
        provider_id = np.array(provider_id)

        # Create a color map for different providers
        unique_providers = np.unique(provider_id)
        colors = plt.cm.get_cmap('tab10', len(unique_providers))

        # Create the main figure and axis for the trust values
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot trust values with different colors for each provider on ax1
        for i, provider in enumerate(unique_providers):
            provider_mask = provider_id == provider
            ax1.scatter(timestep_trust[provider_mask], trust_towards_reporter[provider_mask],
                        color=colors(i), label=f'Provider {provider}')

        # Plot the anomalies on ax1
        ax1.scatter(timestep_trust[is_true_anomaly == 1], [0] * sum(is_true_anomaly == 1),
                    color='green', label='True Anomaly', marker='x')
        ax1.scatter(timestep_trust[is_true_anomaly == 0], [0] * sum(is_true_anomaly == 0),
                    color='red', label='False Anomaly', marker='x')

        # Set labels and title for ax1
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Trust Value / Anomaly Indicator')
        ax1.set_title(f'Trust and Reward Towards Untrustworthy Robot {untrust_robot_id} with {strategy_name}, Reward per Round:{round(sum(reward)/len(reward),2)}')

        # Create a second y-axis for the reward plot
        ax2 = ax1.twinx()

        # Plot the reward data on ax2
        ax2.plot(timestep_reward, reward, color='blue', label='Reward', linestyle='-', marker='o')

        # Set the label for ax2
        ax2.set_ylabel('Reward')

        # Calculate and set the y-limits for both axes
        trust_min, trust_max = ax1.get_ylim()
        reward_min, reward_max = ax2.get_ylim()

        # Determine the offset needed to align the zeros
        trust_range = trust_max - trust_min
        reward_range = reward_max - reward_min

        # Ensure the zero points align
        if trust_min < 0 and reward_min < 0:
            ax1.set_ylim(trust_min, trust_max)
            ax2.set_ylim(reward_min, reward_max)
        elif trust_min >= 0 and reward_min >= 0:
            ax1.set_ylim(0, trust_max)
            ax2.set_ylim(0, reward_max)
        else:
            ax1.set_ylim(trust_min, trust_max)
            ax2.set_ylim(trust_min * (reward_range / trust_range), reward_max)

        # Draw a horizontal line at y=0 for both axes for clarity
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        # Add legend for ax1 and ax2
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Show the combined plot
        plt.show()

    def combined_reward_trust_with_all_robot_plot(self, untrust_robot_id, strategy_name):
        # Data for the reward plot
        reward = []
        timestep_reward = []

        # Data for the trust plot
        is_true_anomaly = []
        timestep_trust = []
        trust_towards_reporter = []
        provider_id = []

        # Data for distinguishing between untrustworthy and trustworthy robots
        is_untrustworthy = []

        # Extracting data for trust, reward, and anomalies
        for h in self.histories:
            is_untrustworthy.append(h['reporter_id'] == untrust_robot_id)
            is_true_anomaly.append(h['is_true_anomaly'])
            timestep_trust.append(h['report_time'])
            reward.append(h['provider_reward'] + h['reporter_reward'])
            trust_value_towards_reporter = h['trust_towards_reporter']['trust_value']
            if trust_value_towards_reporter == 'good' or np.isnan(trust_value_towards_reporter):
                trust_towards_reporter.append(1.0)
            elif trust_value_towards_reporter == 'bad':
                trust_towards_reporter.append(-1.0)
            else:
                trust_towards_reporter.append(trust_value_towards_reporter)
            provider_id.append(h['provider_id'])

        # Convert to numpy arrays for easier handling
        timestep_trust = np.array(timestep_trust)
        is_true_anomaly = np.array(is_true_anomaly)
        trust_towards_reporter = np.array(trust_towards_reporter)
        provider_id = np.array(provider_id)
        is_untrustworthy = np.array(is_untrustworthy)

        # Create a color map for different providers
        unique_providers = np.unique(provider_id)
        colors = plt.cm.get_cmap('tab10', len(unique_providers))

        # Plot the anomalies
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # True anomalies reported by the untrustworthy robot (green)
        ax1.scatter(timestep_trust[(is_true_anomaly == 1) & is_untrustworthy],
                    [0] * sum((is_true_anomaly == 1) & is_untrustworthy),
                    color='green', label='True Anomaly (Untrustworthy)', marker='x')

        # True anomalies reported by trustworthy robots (blue)
        ax1.scatter(timestep_trust[(is_true_anomaly == 1) & ~is_untrustworthy],
                    [0] * sum((is_true_anomaly == 1) & ~is_untrustworthy),
                    color='purple', label='True Anomaly (Trustworthy)', marker='x')

        # False anomalies reported by the untrustworthy robot (red)
        ax1.scatter(timestep_trust[(is_true_anomaly == 0) & is_untrustworthy],
                    [0] * sum((is_true_anomaly == 0) & is_untrustworthy),
                    color='red', label='False Anomaly (Untrustworthy)', marker='x')

        # False anomalies reported by trustworthy robots (orange)
        ax1.scatter(timestep_trust[(is_true_anomaly == 0) & ~is_untrustworthy],
                    [0] * sum((is_true_anomaly == 0) & ~is_untrustworthy),
                    color='orange', label='False Anomaly (Trustworthy)', marker='x')

        # Plot the trust values with different colors for each provider
        for i, provider in enumerate(unique_providers):
            provider_mask = provider_id == provider
            ax1.scatter(timestep_trust[provider_mask], trust_towards_reporter[provider_mask],
                        color=colors(i), label=f'Provider {provider} Trust')

        # Set labels and title
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Trust Value / Anomaly Indicator')
        ax1.set_title(f'Trust and Reward towards Robot {untrust_robot_id} ({strategy_name})') # Reward per round:{round(sum(reward)/len(reward),2)}

        # Draw a horizontal line at y=0 for clarity
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        # Add a secondary y-axis for reward
        ax2 = ax1.twinx()
        ax2.plot(timestep_trust, reward, color='blue', label='Reward', linestyle='-', marker='o')
        ax2.set_ylabel('Reward')

        # Align y=0 for both axes
        ax2.set_ylim(bottom=ax1.get_ylim()[0])

        # Add legends
        fig.legend(loc='upper right')

        # Show plot
        plt.show()


