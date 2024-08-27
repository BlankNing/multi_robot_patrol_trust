class Node():
    def __init__(self, id, pos_info):
        self.id = id
        self.pos_info = tuple(pos_info)
        self.idleness = 0

    def step(self, robot_current_pos, robot_current_states):
        if self.pos_info in robot_current_pos and robot_current_states[robot_current_pos.index(self.pos_info)] == 'Patrolling':
            self.reset()
        else:
            self.idleness += 1
        return self.idleness

    def reset(self):
        self.idleness = 0
