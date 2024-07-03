from .AbstractTrust import Trust
import numpy as np

class BetaTrust(Trust):
    def __init__(self, trust_algo_config):
        self.alpha = 0
        self.beta = 0

    def calculate_trust_value(self, observed_pos_history):
        if len(observed_pos_history) == 1 or observed_pos_history == []:
            return [1 for _ in range(8)]
        else:
            paths_array = np.array(observed_pos_history)
            changes = paths_array[:-1] != paths_array[1:]
            changes_int = changes.astype(int)
            changes_matrix = np.any(changes_int, axis=2).astype(int)
            sum_changes = np.sum(changes_matrix, axis=0)
            num_timesteps = changes_matrix.shape[0]
            beta_trust = sum_changes/num_timesteps
            return beta_trust