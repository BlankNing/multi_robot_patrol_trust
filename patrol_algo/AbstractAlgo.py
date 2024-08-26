import abc
from typing import Any

class Algo(abc.ABC):
    @abc.abstractmethod
    def calculate_next_path(self, *args: Any, **kwargs: Any):
        '''
        given robot number and current node position, determine which node to go next. return planned path

        cannot handle change path at middle of the journey.

        :param robot_id:
        :param current_node:
        :return:
        '''
        pass
