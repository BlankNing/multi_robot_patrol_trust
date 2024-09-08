import abc

class Trust(abc.ABC):
    @abc.abstractmethod
    def calculate_trust_value_reporter(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks):
        '''
        As reporter, calculate the trust value towards a single provider
        :param invovled robot entities
        :return:
        '''
        pass

    @abc.abstractmethod
    def calculate_trust_value_provider(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks):
        '''
        As provider, calculate the trust value towards the reporter
        :param invovled robot entities
        :return:
        '''
        pass