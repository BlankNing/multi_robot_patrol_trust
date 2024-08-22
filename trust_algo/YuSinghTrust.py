from .AbstractTrust import Trust
import numpy as np

class YuSinghTrust(Trust):
    def __init__(self, config):
        # relevant items
        self.history_monitor = config['history_monitor']
        # useful info



    def calculate_trust_value_reporter(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks): # return (0,1)
        pass



    def calculate_trust_value_provider(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks): # return (0,1)
        pass
