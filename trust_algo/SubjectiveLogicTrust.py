from .AbstractTrust import Trust
import itertools
import numpy as np

class SubjectiveLogicTrust(Trust):
    def __init__(self, config):
        # relevant items
        self.history_monitor = config['history_monitor']
        # useful info
        self.H = 20
        self.lower_threshold = 0.2
        self.upper_threshold = 0.8
        self.accept_threshold = 0.1
        self.nBF = 2


    def direct_trust_reliability_calculation(self, filtered_histories):
        total = len(filtered_histories)

        m_T = sum(1 for h in filtered_histories if h >= self.upper_threshold) /total
        m_not_T = sum(1 for h in filtered_histories if h <= self.lower_threshold) /total
        m_uncertain = 1 - m_T - m_not_T

        return {
            frozenset(['T']): m_T,
            frozenset(['not_T']): m_not_T,
            frozenset(['T','not_T']): m_uncertain
        }

    def dempster_rule_of_combination(self, m1, m2):
        """
        Apply Dempster's rule of combination on two belief functions m1 and m2.
        m1 and m2 are dictionaries where the keys are frozensets representing the focal elements
        and the values are the corresponding mass functions.
        """
        combined = {}
        for A, B in itertools.product(m1.keys(), m2.keys()):
            intersection = A & B
            if intersection:
                combined[intersection] = combined.get(intersection, 0) + m1[A] * m2[B]

        normalization_factor = 1 - sum(m1[A] * m2[B] for A, B in itertools.product(m1.keys(), m2.keys()) if not (A & B))

        for key in combined:
            combined[key] /= normalization_factor

        return combined

    def combined_trust_reliability_calculation(self, filtered_histories_dict):
        beliefs = []
        all_empty_flag = 1
        for key, history in filtered_histories_dict.items():
            if len(history) != 0:
                all_empty_flag = 0
                belief = self.direct_trust_reliability_calculation(history)
                beliefs.append(belief)
        # no reputation
        if all_empty_flag:
            return 'none', {
                    frozenset(['T']): 0,
                    frozenset(['not_T']): 0,
                    frozenset(['T', 'not_T']): 1
                }
        else: # has reputaion
            aggregated_belief = beliefs[0]
            try:
                for b in beliefs[1:]:
                    aggregated_belief = self.dempster_rule_of_combination(aggregated_belief, b)
            except:
                pass
            return beliefs, aggregated_belief

    def trust_decision(self, total_belief):
        belief_T = total_belief.get(frozenset(['T']), 0)
        belief_not_T = total_belief.get(frozenset(['not_T']), 0)
        belief_uncertain = total_belief.get(frozenset(['T', 'not_T']), 0)
        if belief_T - belief_not_T >= self.accept_threshold or belief_uncertain == 1:
            return 1
        else:
            return 0


    def calculate_trust_value_reporter(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks): # return (0,1)
        histories = self.history_monitor.get_history_as_reporter(reporter_id, provider_id)
        filtered_histories = [history[1] for history in histories if history[2] == task_info][-self.H:]

        # can use direct history/trust
        if len(filtered_histories) != 0:
            belief = self.direct_trust_reliability_calculation(filtered_histories)
            trust_value = self.trust_decision(belief)
            return {
                'trust_value': trust_value,
                'belief': belief
            }
        else:
            # get reputation history
            histories_dict = self.history_monitor.get_history_as_reporter_witness_TRAVOS(reporter_id, provider_id,
                                                                                         local_history_length=self.H,
                                                                                         referral_length=self.nBF)
            filtered_histories_dict = {}
            for key, value in histories_dict.items():
                filtered_histories_dict[key] = [history[1] for history in value if history[2] == task_info]
            beliefs, total_belief = self.combined_trust_reliability_calculation(filtered_histories_dict)
            trust_value = self.trust_decision(total_belief)
            return {
                'trust_value': trust_value,
                'beliefs': beliefs,
                'total_belief': total_belief
            }

    def calculate_trust_value_provider(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks): # return (0,1)
        histories = self.history_monitor.get_history_as_provider(reporter_id, provider_id)
        filtered_histories = [history[1] for history in histories if history[2] == task_info][-self.H:]

        # can use direct history/trust
        if len(filtered_histories) != 0:
            belief = self.direct_trust_reliability_calculation(filtered_histories)
            trust_value = self.trust_decision(belief)
            return {
                'trust_value': trust_value,
                'belief': belief
            }
        else:
            # get reputation history
            histories_dict = self.history_monitor.get_history_as_provider_witness_TRAVOS(reporter_id, provider_id,
                                                                                         local_history_length=self.H,
                                                                                         referral_length=self.nBF)
            # no reputation as well, completely uncertain, but trust
            if histories_dict == {}:
                return {
                    'trust_value': 1,
                    'belief': {
                        frozenset(['T']): 0,
                        frozenset(['not_T']): 0,
                        frozenset(['T', 'not_T']): 1
                    }
                }
            else:  # has reputation, calculate aggregated trust
                filtered_histories_dict = {}
                for key, value in histories_dict.items():
                    filtered_histories_dict[key] = [history[1] for history in value if history[2] == task_info]
                beliefs, total_belief = self.combined_trust_reliability_calculation(filtered_histories_dict)
                trust_value = self.trust_decision(total_belief)
                return {
                    'trust_value': trust_value,
                    'belief': total_belief
                }

