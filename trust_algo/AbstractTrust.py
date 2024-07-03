import abc

class Trust(abc.ABC):
    @abc.abstractmethod
    def calculate_trust_value(self, observed_pos_history):
        '''
        calculate the trust value of the rest of the group, always 1 towards itself
        :param target_robot_list:
        :param observed_history:
        :return:
        '''
        pass