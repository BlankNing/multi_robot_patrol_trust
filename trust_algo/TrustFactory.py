from trust_algo.BetaTrust import BetaTrust
from trust_algo.FIRETrust import FIRETrust

class TrustFactory():
    @staticmethod
    def create_algo(trust_algo, trust_algo_config):
        if trust_algo == 'BETA':
            return BetaTrust(trust_algo_config)
        elif trust_algo == 'FIRE':
            return FIRETrust()
        elif trust_algo == 'YUSIGNH':
            return FIRETrust()
        else:
            # for test
            return None
