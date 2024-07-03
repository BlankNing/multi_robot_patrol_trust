from trust_algo.BetaTrust import BetaTrust

class TrustFactory():
    @staticmethod
    def create_algo(trust_algo, trust_algo_config):
        if trust_algo == 'beta':
            return BetaTrust(trust_algo_config)
        else:
            # for test
            return None
