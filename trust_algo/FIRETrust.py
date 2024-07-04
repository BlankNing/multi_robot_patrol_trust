from .AbstractTrust import Trust
import numpy as np

class FIRETrust(Trust):
    def __init__(self):
        self.alpha = 0
        self.beta = 0

    def calculate_trust_value(self, observed_pos_history) -> float: # return (0,1)
        if observed_pos_history == []:
            return 1.0
        else:
            observed_pos_history_np = np.array(observed_pos_history)
            return sum(observed_pos_history_np[:,0])/len(observed_pos_history_np[:,0])