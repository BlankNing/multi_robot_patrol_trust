import math

from .AbstractTrust import Trust
import numpy as np
import scipy.integrate as integrate

class TRAVOSTrust(Trust):
    # todo: implement reliability calculation
    def __init__(self, trust_algo_config):
        # relevant items
        self.history_monitor = trust_algo_config['history_monitor']
        self.use_reputation = False

        # algorithm parameters
        self.H = 99999999
        self.nBF = 2
        self.confidence_rate = 0.95
        self.epsilon = 0.2


    def reputation_reliability_factor_collect(self):
        pass


    def direct_trust_reliability_calculation(self, filtered_histories):
        '''
        Calculate the trust value based on the history of the provider
        :param filtered_histories: ratings of each interaction [0,1,1,0,1,-1,0,1,-1]
        :return:
        '''
        self.use_reputation = False
        alpha = len([x for x in filtered_histories if x > 0]) + 1
        beta = len([x for x in filtered_histories if x <= 0]) + 1

        trust_value = alpha / (alpha + beta)

        delta = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        gamma = 1 - math.sqrt(12 * delta / ((alpha + beta) ** 2 * (alpha + beta + 1)))

        def integrand_B(B, alpha, beta):
            return (B ** (alpha - 1)) * ((1 - B) ** (beta - 1))

        def integrand_U(U, alpha, beta):
            return (U ** (alpha - 1)) * ((1 - U) ** (beta - 1))

        # Perform the integration for the numerator and denominator
        numerator, _ = integrate.quad(integrand_B, trust_value - self.epsilon, trust_value + self.epsilon, args=(alpha, beta))
        denominator, _ = integrate.quad(integrand_U, 0, 1, args=(alpha, beta))

        # Calculate the final gamma value
        confidence = numerator / denominator

        return {'trust_value': trust_value, 'trust_type': 'direct', 'gamma': gamma, 'delta': delta, 'direct_confidence': confidence, 'alpha': alpha, 'beta':beta}



    def combined_trust_reliability_calculation(self, filtered_histories_dict, direct_trust):
        '''
        :param filtered_histories: ratings of each interaction 1:[0,1,1,0,1,-1,0,1,-1] 3:[]
        :return:
        '''
        self.use_reputation = True
        # calculate reputation
        N = 0
        M = 0
        alpha_beta_dict = {} #{1:{'alpha':10,'beta':5}}
        for key, filtered_histories in filtered_histories_dict.items():
            alpha_beta_dict[key] = {}
            alpha_beta_dict[key]['alpha'] = len([x for x in filtered_histories if x > 0]) + 1
            alpha_beta_dict[key]['beta'] = len([x for x in filtered_histories if x <= 0]) + 1
            N += alpha_beta_dict[key]['alpha']
            M += alpha_beta_dict[key]['beta']
        N += 1
        M += 1
        reputaion_trust = N / (N + M)

        modified_trust = {
            'trust_value': reputaion_trust,
            'trust_type': 'reputation',
            'direct_trust_value': direct_trust['trust_value'],
            'direct_trust_confidence': direct_trust['direct_confidence'],
            'N': N,
            'M': M
        }

        return modified_trust

    def calculate_direct_trust_value_reporter(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks) -> dict:
        '''
        As a reporter, get the individual interaction history with a single provider
        :return:
        '''
        histories = self.history_monitor.get_history_as_reporter(reporter_id, provider_id)
        filtered_histories = [history[1] for history in histories if history[2] == task_info][-self.H:]

        return self.direct_trust_reliability_calculation(filtered_histories)

    def calculate_trust_value_reporter(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks):
        '''
        As a reporter, get the individual interaction history with a single provider
        :return:
        '''
        histories_dict = self.history_monitor.get_history_as_reporter_witness_TRAVOS(reporter_id, provider_id,
                                                                                local_history_length=self.H,
                                                                                referral_length=self.nBF)

        filtered_histories_dict = {}
        for key, value in histories_dict.items():
            filtered_histories_dict[key] = [history[1] for history in value if history[2] == task_info]

        direct_trust = self.calculate_direct_trust_value_reporter(reporter_id, provider_id, task_info, timestep, robot_capable_tasks)

        return self.combined_trust_reliability_calculation(filtered_histories_dict, direct_trust)


    def calculate_direct_trust_value_provider(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks) -> dict:
        '''
        As a provider, get the individual interaction history with a single reporter
        :return:
        '''
        histories = self.history_monitor.get_history_as_provider(reporter_id, provider_id)
        filtered_histories = [history[1] for history in histories if history[2] == task_info][-self.H:]

        return self.direct_trust_reliability_calculation(filtered_histories)

    def calculate_trust_value_provider(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks):
        '''
        As a provider, get the individual interaction history with a single reporter
        :return:
        '''
        histories_dict = self.history_monitor.get_history_as_provider_witness_TRAVOS(reporter_id, provider_id,
                                                                                local_history_length=self.H,
                                                                                referral_length=self.nBF)

        filtered_histories_dict = {}
        for key, value in histories_dict.items():
            filtered_histories_dict[key] = [history[1] for history in value if history[2] == task_info]

        direct_trust = self.calculate_direct_trust_value_provider(reporter_id, provider_id, task_info, timestep, robot_capable_tasks)

        return self.combined_trust_reliability_calculation(filtered_histories_dict, direct_trust)