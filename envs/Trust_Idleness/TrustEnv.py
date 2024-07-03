from basic_patrol_class.Env import BasicEnv

from .TrustMonitor import TrustMonitor
from trust_algo.TrustFactory import TrustFactory
from trust_algo.trust_config_dispatch import get_trust_algo_config


class TrustEnv(BasicEnv):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.is_distributed = None
        if config_file['coordination'] == 'distributed':
            from .TrustRobotDistributed import TrustRobotDistributed as TrustRobot
            self.is_distributed = True
        else:
            from .TrustRobotCentralised import TrustRobotCentralised as TrustRobot
            self.is_distributed = False

        # Load extra variables for trust scenario
        self.untrust_list = config_file['trust_config']['untrust_list']
        self.malfunc_prob = config_file['trust_config']['malfunc_prob']
        self.trust_algo = config_file['trust_config']['trust_algo']
        self.trust_algo_config = get_trust_algo_config(config_file)


        # Init TrustRobot and TrustMonitor, don't need to modify Algorithm and Node
        self.monitor = TrustMonitor()
        self.trust_engine = TrustFactory().create_algo(self.trust_algo, self.trust_algo_config)
        if self.is_distributed:
            self.robots = [TrustRobot(i, self.algo_engine, self.node_pos_matrix, self.init_pos[i], self.untrust_list, self.malfunc_prob, self.monitor, self.trust_engine) for i in range(self.robots_num)]
        else:
            self.robots = [TrustRobot(i, self.algo_engine, self.node_pos_matrix, self.init_pos[i], self.untrust_list, self.malfunc_prob) for i in range(self.robots_num)]

    def step(self,verbose=False):
        # robot move
        robot_pos_records = []
        robot_trust_records = []
        for robot in self.robots:
            robot_pos_record, robot_trust_record = robot.step(verbose=verbose)
            robot_pos_records.append(robot_pos_record)
            robot_trust_records.append(robot_trust_record)
        self.monitor.collect_robot_pos(robot_pos_records)

        if self.is_distributed:
            self.monitor.collect_trust_values(robot_trust_records)
        else:
            self.monitor.collect_trust_values(self.trust_engine.calculate_trust_value(self.monitor.robot_pos))

        # node record
        node_idleness_records = []
        for node in self.nodes:
            node_idleness_record = node.step(robot_pos_records)
            node_idleness_records.append(node_idleness_record)
        self.monitor.collect_node_idleness(node_idleness_records)
