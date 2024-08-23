import math

from .AbstractTrust import Trust

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyTrust(Trust):
    def __init__(self, trust_algo_config):
        # relevant items
        self.history_monitor = trust_algo_config['history_monitor']

        # Define the universe of discourse (range of inputs)
        trust_range = np.arange(-1, 1.1, 0.1)  # From -1 to 1 in steps of 0.1

        # Define fuzzy variables
        interaction_trust = ctrl.Antecedent(trust_range, 'interaction_trust')
        witness_reputation = ctrl.Antecedent(trust_range, 'witness_reputation')
        certified_reputation = ctrl.Antecedent(trust_range, 'certified_reputation')
        overall_trust = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'overall_trust')

        # Define fuzzy membership functions for each input
        interaction_trust['L'] = fuzz.trimf(interaction_trust.universe, [-1, -1, 0])
        interaction_trust['M'] = fuzz.trimf(interaction_trust.universe, [-1, 0, 1])
        interaction_trust['H'] = fuzz.trimf(interaction_trust.universe, [0, 1, 1])

        witness_reputation['L'] = fuzz.trimf(witness_reputation.universe, [-1, -1, 0])
        witness_reputation['M'] = fuzz.trimf(witness_reputation.universe, [-1, 0, 1])
        witness_reputation['H'] = fuzz.trimf(witness_reputation.universe, [0, 1, 1])

        certified_reputation['L'] = fuzz.trimf(certified_reputation.universe, [-1, -1, 0])
        certified_reputation['M'] = fuzz.trimf(certified_reputation.universe, [-1, 0, 1])
        certified_reputation['H'] = fuzz.trimf(certified_reputation.universe, [0, 1, 1])

        # Define fuzzy membership functions for the output
        overall_trust['VL'] = fuzz.trimf(overall_trust.universe, [0, 0, 0.25])
        overall_trust['L'] = fuzz.trimf(overall_trust.universe, [0, 0.25, 0.5])
        overall_trust['M'] = fuzz.trimf(overall_trust.universe, [0.25, 0.5, 0.75])
        overall_trust['H'] = fuzz.trimf(overall_trust.universe, [0.5, 0.75, 1])
        overall_trust['VH'] = fuzz.trimf(overall_trust.universe, [0.75, 1, 1])

        # Define fuzzy rules based on the table provided
        rule1 = ctrl.Rule(interaction_trust['L'] & witness_reputation['L'] & certified_reputation['L'],
                          overall_trust['VL'])
        rule2 = ctrl.Rule(interaction_trust['M'] & witness_reputation['L'] & certified_reputation['L'],
                          overall_trust['L'])
        rule3 = ctrl.Rule(interaction_trust['L'] & witness_reputation['M'] & certified_reputation['L'],
                          overall_trust['L'])
        rule4 = ctrl.Rule(interaction_trust['L'] & witness_reputation['L'] & certified_reputation['M'],
                          overall_trust['L'])
        rule5 = ctrl.Rule(interaction_trust['M'] & witness_reputation['M'] & certified_reputation['L'],
                          overall_trust['L'])
        rule6 = ctrl.Rule(interaction_trust['L'] & witness_reputation['M'] & certified_reputation['M'],
                          overall_trust['L'])
        rule7 = ctrl.Rule(interaction_trust['M'] & witness_reputation['L'] & certified_reputation['M'],
                          overall_trust['L'])
        rule8 = ctrl.Rule(interaction_trust['H'] & witness_reputation['L'] & certified_reputation['L'],
                          overall_trust['L'])
        rule9 = ctrl.Rule(interaction_trust['L'] & witness_reputation['H'] & certified_reputation['L'],
                          overall_trust['L'])
        rule10 = ctrl.Rule(interaction_trust['L'] & witness_reputation['L'] & certified_reputation['H'],
                           overall_trust['L'])
        rule11 = ctrl.Rule(interaction_trust['L'] & witness_reputation['M'] & certified_reputation['H'],
                           overall_trust['M'])
        rule12 = ctrl.Rule(interaction_trust['L'] & witness_reputation['H'] & certified_reputation['M'],
                           overall_trust['M'])
        rule13 = ctrl.Rule(interaction_trust['M'] & witness_reputation['L'] & certified_reputation['H'],
                           overall_trust['M'])
        rule14 = ctrl.Rule(interaction_trust['M'] & witness_reputation['H'] & certified_reputation['L'],
                           overall_trust['M'])
        rule15 = ctrl.Rule(interaction_trust['H'] & witness_reputation['L'] & certified_reputation['M'],
                           overall_trust['M'])
        rule16 = ctrl.Rule(interaction_trust['H'] & witness_reputation['M'] & certified_reputation['L'],
                           overall_trust['M'])
        rule17 = ctrl.Rule(interaction_trust['M'] & witness_reputation['M'] & certified_reputation['M'],
                           overall_trust['M'])
        rule18 = ctrl.Rule(interaction_trust['H'] & witness_reputation['H'] & certified_reputation['L'],
                           overall_trust['H'])
        rule19 = ctrl.Rule(interaction_trust['L'] & witness_reputation['H'] & certified_reputation['H'],
                           overall_trust['H'])
        rule20 = ctrl.Rule(interaction_trust['H'] & witness_reputation['L'] & certified_reputation['H'],
                           overall_trust['H'])
        rule21 = ctrl.Rule(interaction_trust['H'] & witness_reputation['M'] & certified_reputation['M'],
                           overall_trust['H'])
        rule22 = ctrl.Rule(interaction_trust['M'] & witness_reputation['M'] & certified_reputation['H'],
                           overall_trust['H'])
        rule23 = ctrl.Rule(interaction_trust['M'] & witness_reputation['H'] & certified_reputation['M'],
                           overall_trust['H'])
        rule24 = ctrl.Rule(interaction_trust['M'] & witness_reputation['H'] & certified_reputation['H'],
                           overall_trust['H'])
        rule25 = ctrl.Rule(interaction_trust['H'] & witness_reputation['H'] & certified_reputation['M'],
                           overall_trust['H'])
        rule26 = ctrl.Rule(interaction_trust['H'] & witness_reputation['M'] & certified_reputation['H'],
                           overall_trust['H'])
        rule27 = ctrl.Rule(interaction_trust['H'] & witness_reputation['H'] & certified_reputation['H'],
                           overall_trust['VH'])

        # Create the control system and simulation
        trust_control_system = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
            rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19,
            rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27
        ])

        self.trust_simulation = ctrl.ControlSystemSimulation(trust_control_system)
        # hyperparameters:
        self.H = 20
        self.nBF = 2


        # Example: Set input values and compute the overall trust
    def calculate_trust(self, it_value, wr_value, cr_value):
        self.trust_simulation.input['interaction_trust'] = it_value
        self.trust_simulation.input['witness_reputation'] = wr_value
        self.trust_simulation.input['certified_reputation'] = cr_value
        self.trust_simulation.compute()
        return self.trust_simulation.output['overall_trust']

    def calculate_trust_value(self, filtered_histories):
        alpha = len([x for x in filtered_histories if x > 0]) + 1
        beta = len([x for x in filtered_histories if x <= 0]) + 1
        trust_value = alpha / (alpha + beta)
        return round(trust_value, 1)


    def calculate_trust_value_reporter(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks):
        '''
        As a reporter, get the individual interaction history with a single provider
        :return:
        '''
        histories = self.history_monitor.get_history_as_reporter(reporter_id, provider_id)
        direct_filtered_histories = [history[1] for history in histories if history[2] == task_info][-self.H:]
        IT = self.calculate_trust_value(direct_filtered_histories)

        histories = self.history_monitor.get_history_as_reporter_witness_FIRE(reporter_id, provider_id, local_history_length = self.H, referral_length=self.nBF)
        witness_filtered_histories = [history[1] for history in histories if history[2] == task_info]
        WR = self.calculate_trust_value(witness_filtered_histories)
        CR = WR

        trust_value = self.calculate_trust(IT, WR, CR)

        return {'trust_value': trust_value, 'IT': IT, 'WR': WR, 'CR': CR}


    def calculate_trust_value_provider(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks):
        '''
        As a provider, get the individual interaction history with a single reporter
        :return:
        '''
        histories = self.history_monitor.get_history_as_provider(reporter_id, provider_id)
        direct_filtered_histories = [history[1] for history in histories if history[2] == task_info][-self.H:]
        IT = self.calculate_trust_value(direct_filtered_histories)

        histories = self.history_monitor.get_history_as_provider_witness_FIRE(reporter_id, provider_id,
                                                                              local_history_length=self.H,
                                                                              referral_length=self.nBF)
        witness_filtered_histories = [history[1] for history in histories if history[2] == task_info]
        WR = self.calculate_trust_value(witness_filtered_histories)
        CR = WR

        trust_value = self.calculate_trust(IT, WR, CR)

        return {'trust_value': trust_value, 'IT': IT, 'WR': WR, 'CR': CR}

