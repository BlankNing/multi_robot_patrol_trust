from trust_algo.BetaTrust import BetaTrust
from trust_algo.FIRETrust import FIRETrust
from trust_algo.TRAVOSTrust import TRAVOSTrust

class TrustFactory():
    @staticmethod
    def create_algo(trust_algo, trust_algo_config):
        if trust_algo == 'BETA':
            return BetaTrust(trust_algo_config)
        elif trust_algo == 'FIRE':
            return FIRETrust(trust_algo_config)
        elif trust_algo == 'YUSIGNH':
            return FIRETrust()
        elif trust_algo == 'TRAVOS':
            return TRAVOSTrust(trust_algo_config)
        else:
            # for test
            return None
