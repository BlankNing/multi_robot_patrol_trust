from .AbstractTrust import Trust
import numpy as np

class TRAVOSTrust(Trust):
    def __init__(self, trust_algo_config):
        self.alpha = 0
        self.beta = 0
        self.trust_mode = trust_algo_config['trust_mode']

    def calculate_trust_value(self, observed_pos_history) -> float: # return (0,1)
        direct_history = observed_pos_history['direct']
        witness_history = observed_pos_history['witness']
        aggregate_trust = {}

        if 'IT' in self.trust_mode:
            try:
                direct_history_np = np.array(direct_history)
                direct_trust = (sum(direct_history_np[:,0])+1)/(len(direct_history_np[:,0])+1)
                aggregate_trust['direct_trust'] = direct_trust
            except:
                aggregate_trust['direct_trust'] = 1.0


        community_trust = {}
        if 'WR' in self.trust_mode:
            for witness_id, witness_his in witness_history.items():
                try:
                    witness_history_np = np.array(witness_his)
                    witness_trust = (sum(witness_history_np[:, 0]) + 1) / (len(witness_history_np[:, 0]) + 1)
                    community_trust[witness_id] = witness_trust
                except:
                    community_trust[witness_id] = 1.0

        aggregate_trust['community_trust'] = community_trust
        # todo: how to aggregate community_trust and individual trust?


        return 0