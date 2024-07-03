from patrol_class.Monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np

class TrustMonitor(Monitor):
    def __init__(self):
        super().__init__()
        self.trust_value = []

    def get_observable_history(self, robot_id):
        # if distributed 1.within radius 2.return position info & accuracy info
        return self.robot_pos

    def collect_trust_values(self, nodes_trust_record):
        self.trust_value.append(nodes_trust_record)

    def plot_trust_value(self,robot_num):
        data_array = np.array(self.trust_value)
        column_data = data_array[:, robot_num]
        plt.plot(range(len(column_data)), column_data)
        plt.title(f"Trust Value of Robot {robot_num} Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Trust Value")
        plt.grid(True)
        plt.show()