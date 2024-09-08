from patrol_algo.PartitionAlgo import PartitionAlgo
from patrol_algo.SEBSAlgo import SEBSAlgo
from patrol_algo.RandomWalkAlgo import RandomWalkAlgo
from patrol_algo.CGGAlgo import CGGAlgo

class AlgoFactory():
    @staticmethod
    def create_algo(patrol_algo, algo_config):
        if patrol_algo == 'partition':
            return PartitionAlgo(algo_config)
        elif patrol_algo == 'SEBS':
            return SEBSAlgo(algo_config)
        elif patrol_algo == 'Random':
            return RandomWalkAlgo(algo_config)
        elif patrol_algo == 'CGG':
            return CGGAlgo(algo_config)
