from basic_patrol_class.Robot import Robot

import random

class TrustRobotDistributed(Robot):
    def __init__(self, id, algo_engine, node_pos_matrix, init_pos, untrust_list, malfunc_prob, monitor, trust_engine):
        super().__init__(id, algo_engine, node_pos_matrix, init_pos)
        self.untrust_list = untrust_list
        self.is_trust = 1 if self.id not in self.untrust_list else 0
        self.malfunc_prob = malfunc_prob
        self.monitor = monitor
        self.trust_engine = trust_engine
        self.current_trust = []

    def calculate_trust(self):
        '''
        Distributed robots can calculate trust by it self.
        :return:
        '''
        return self.trust_engine.calculate_trust_value(self.monitor.get_observable_history(self.id))

    def step(self,verbose = False):
        if self.path_list == []:
            self.last_node = int(self.check_node())
            self.path_list = self.algo_engine.calculate_next_path(self.id, self.last_node)

        if self.is_trust:
            # move 1 step
            self.state = 'Patrolling'
            self.current_pos = self.path_list[0]
            self.path_list.pop(0)
        else:
            if random.random() > self.malfunc_prob:
                self.state = 'Patrolling'
                self.current_pos = self.path_list[0]
                self.path_list.pop(0)
            else:
                self.state = 'Resting'

        if verbose == True:
            print(f"Robot_{self.id} {self.state} at {self.current_pos} {self.last_node}")

        # calculate trust of other robots
        self.current_trust = self.calculate_trust()

        return self.current_pos, self.current_trust