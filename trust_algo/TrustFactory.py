from trust_algo.BetaTrust import BetaTrust
from trust_algo.FIRETrust import FIRETrust
from trust_algo.TRAVOSTrust import TRAVOSTrust
from trust_algo.YuSinghTrust import YuSinghTrust
from trust_algo.FuzzyTrust import FuzzyTrust
from trust_algo.SubjectiveLogicTrust import SubjectiveLogicTrust
from trust_algo.MLTrust import MLTrust

class TrustFactory():
    @staticmethod
    def create_algo(trust_algo, trust_algo_config):
        if trust_algo == 'BETA':
            return BetaTrust(trust_algo_config)
        elif trust_algo == 'FIRE':
            return FIRETrust(trust_algo_config)
        elif trust_algo == 'YUSINGH':
            return YuSinghTrust(trust_algo_config)
        elif trust_algo == 'TRAVOS':
            return TRAVOSTrust(trust_algo_config)
        elif trust_algo == 'FUZZY':
            return FuzzyTrust(trust_algo_config)
        elif trust_algo == 'ML':
            return MLTrust(trust_algo_config)
        elif trust_algo == 'SUBJECTIVE':
            return SubjectiveLogicTrust(trust_algo_config)
        else:
            # for test
            return None
