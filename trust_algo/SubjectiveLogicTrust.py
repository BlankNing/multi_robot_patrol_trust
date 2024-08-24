from .AbstractTrust import Trust
import math


# Define the Opinion class to hold belief, disbelief, and uncertainty
class Opinion:
    def __init__(self, belief, disbelief, uncertainty):
        self.belief = belief
        self.disbelief = disbelief
        self.uncertainty = uncertainty

    def __repr__(self):
        return f"Opinion(b={self.belief:.3f}, d={self.disbelief:.3f}, u={self.uncertainty:.3f})"


# Evidence-to-Opinion operator
def evidence_to_opinion(p, n, k):
    """
    Convert evidence (positive, negative, and time interval) into an opinion.
    :param p: positive evidence
    :param n: negative evidence
    :param k: time interval or unobserved evidence
    :return: Opinion object
    """
    total = p + n + k
    if total == 0:
        return Opinion(0, 0, 1)

    b = p / total
    d = n / total
    u = k / total

    return Opinion(b, d, u)


# Consensus operator (⊕)
def consensus_opinion(op1, op2):
    """
    Combine two opinions using the consensus operator.
    :param op1: Opinion from source 1
    :param op2: Opinion from source 2
    :return: Combined Opinion object
    """
    if op1.uncertainty + op2.uncertainty - op1.uncertainty * op2.uncertainty !=0:

        combined = op1.uncertainty + op2.uncertainty - op1.uncertainty * op2.uncertainty
        b_combined = (op1.belief * op2.uncertainty + op2.belief * op1.uncertainty) / combined
        d_combined = (op1.disbelief * op2.uncertainty + op2.disbelief * op1.uncertainty) / combined
        u_combined = (op1.uncertainty * op2.uncertainty)/combined

        return Opinion(b_combined, d_combined, u_combined)

    else:
        gamma = op2.uncertainty/op1.uncertainty
        combined = gamma + 1
        b_combined = (gamma * op1.belief + op2.belief) / combined
        d_combined = (gamma * op1.disbelief + op2.disbelief) / combined
        u_combined = 0
        return Opinion(b_combined, d_combined, u_combined)

# Discounting operator (⊗)
def discount_opinion(op1, op2):
    """
    Apply discounting operator to combine an opinion with a recommended opinion.
    :param op1: Opinion of the recommender
    :param op2: Opinion provided by the recommender about another node
    :return: Discounted Opinion object
    """
    b = op1.belief * op2.belief
    d = op1.belief * op2.disbelief
    u = op1.disbelief + op2.uncertainty + op1.uncertainty * op2.belief

    return Opinion(b, d, u)


# Fading operator (∅)
def fading_opinion(op, m, decay_factor=0.1):
    """
    Apply fading operator to increase uncertainty over time.
    :param op: Opinion object
    :param m: Time intervals since last interaction
    :param decay_factor: Rate of decay per time interval
    :return: Faded Opinion object
    """
    fading = math.exp(-decay_factor * m)
    new_b = fading * op.belief
    new_d = fading * op.disbelief
    new_u = op.uncertainty + abs(new_d + new_b - (op.belief + op.disbelief)* (1 - fading))

    return Opinion(new_b, new_d, new_u)


# Comparison operator (≥)
def compare_opinion(op, threshold=0.6):
    """
    Compare an opinion against a trust threshold to classify it.
    :param op: Opinion object
    :param threshold: Trust threshold for belief
    :return: Trust classification: 'Trustworthy', 'Untrustworthy', 'Uncertain'
    """
    if op.belief >= threshold and op.disbelief < threshold:
        return 1
    elif op.disbelief >= threshold and op.uncertainty < threshold:
        return -1
    else:
        return 0

class SubjectiveLogicTrust(Trust):
    def __init__(self, config):
        # relevant items
        self.history_monitor = config['history_monitor']
        self.robot_num = config['robot_num']
        # useful info
        self.H = 20
        self.nBF = 1
        self.reporter_id = -1
        self.provider_id = -1
        self.task_info = -1
        self.last_timestep_interaction_with_reporter = self.generate_history_dict()
        self.last_timestep_interaction_with_provider = self.generate_history_dict()

    def generate_history_dict(self):
        '''
        {1:{2:-1,3:-1,4:-1},
         2:{1:-1,3:-1,4:-1},
         3:{1:[],2:[],4:[]},
         4:{1:[],2:[],3:[]},
        }
        :return:
        '''
        robot_dict = {}
        for i in range(0, self.robot_num):
            robot_dict[i] = {j: math.inf for j in range(0, self.robot_num) if j != i}
        return robot_dict

    def calculate_direct_trust(self, filtered_histories):

        p = len([x for x in filtered_histories if x > 0.2])
        n = len([x for x in filtered_histories if x <= 0])
        k = len([x for x in filtered_histories if 0 <= x <= 0.2])

        return evidence_to_opinion(p, n, k)

    def calculate_old_reputation(self, filtered_histories_dict, last_interaction_timestep_with_provider):
        init_opinion = Opinion(0,0,1)

        for recommender, histories in filtered_histories_dict.items():
            refer_opinion = self.calculate_direct_trust(histories)

            histories = self.history_monitor.get_history_as_reporter(self.reporter_id, recommender)
            filtered_histories = [history[1] for history in histories if history[2] == self.task_info and history
                                  and history[1] < last_interaction_timestep_with_provider][-self.H:]
            opinion_to_refer = self.calculate_direct_trust(filtered_histories)

            opinion = discount_opinion(opinion_to_refer, refer_opinion)
            init_opinion = consensus_opinion(init_opinion, opinion)

        return init_opinion

    def calcluate_new_reputation(self, filtered_histories_dict, old_reputation):
        init_opinion = old_reputation

        for recommender, histories in filtered_histories_dict.items():
            refer_opinion = self.calculate_direct_trust(histories)

            histories = self.history_monitor.get_history_as_reporter(self.reporter_id, recommender)
            filtered_histories = [history[1] for history in histories if history[2] == self.task_info and history][-self.H:]
            opinion_to_refer = self.calculate_direct_trust(filtered_histories)

            opinion = discount_opinion(opinion_to_refer, refer_opinion)
            init_opinion = consensus_opinion(init_opinion, opinion)

        return init_opinion


    def calculate_trust_value_reporter(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks): # return (0,1)
        self.reporter_id = reporter_id
        self.provider_id = provider_id
        self.task_info = task_info

        last_interaction_timestep_with_provider = self.last_timestep_interaction_with_provider[reporter_id][provider_id]

        # get all history
        histories = self.history_monitor.get_history_as_reporter(reporter_id, provider_id)
        filtered_histories = [history[1] for history in histories if history[2] == task_info][-self.H:]

        # calculate direct opinion
        direct_opinion = self.calculate_direct_trust(filtered_histories)

        # calculate old reputaion
        init_op = Opinion(0,0,1)
        # only get the history before last interaction
        histories_dict = self.history_monitor.get_history_as_reporter_witness_SUBJECTIVE(reporter_id, provider_id,
                                                                                     last_interaction_timestep_with_provider,
                                                                                     local_history_length=self.H,
                                                                                     referral_length=self.nBF)
        filtered_histories_dict = {}
        for key, value in histories_dict.items():
            filtered_histories_dict[key] = [history[1] for history in value if history[2] == task_info]

        old_reputation_opinion = self.calculate_old_reputation(filtered_histories_dict, last_interaction_timestep_with_provider)

        # fade old reputation
        if last_interaction_timestep_with_provider != math.inf:
            delta_timestep = timestep - last_interaction_timestep_with_provider
        else:
            delta_timestep = 0
        revised_reputation_opinion = fading_opinion(old_reputation_opinion, delta_timestep)

        # calculate new reputation
        histories_dict = self.history_monitor.get_history_as_reporter_witness_TRAVOS(reporter_id, provider_id,
                                                                                     local_history_length=self.H,
                                                                                     referral_length=self.nBF)
        filtered_histories_dict = {}
        for key, value in histories_dict.items():
            filtered_histories_dict[key] = [history[1] for history in value if history[2] == task_info]

        new_reputation_opinion = self.calcluate_new_reputation(filtered_histories_dict, revised_reputation_opinion)

        # judge trust or not
        trust_value = compare_opinion(new_reputation_opinion)

        # update last interaction
        self.last_timestep_interaction_with_provider[reporter_id][provider_id] = timestep

        return {'trust_value': trust_value, 'direct_opinion': direct_opinion, 'old_reputation': old_reputation_opinion, 'new_reputation': new_reputation_opinion}



    def calculate_trust_value_provider(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks): # return (0,1)
        self.reporter_id = reporter_id
        self.provider_id = provider_id
        self.task_info = task_info

        last_timestep_interaction_with_reporter = self.last_timestep_interaction_with_reporter[reporter_id][provider_id]

        # get all history
        histories = self.history_monitor.get_history_as_provider(reporter_id, provider_id)
        filtered_histories = [history[1] for history in histories if history[2] == task_info][-self.H:]

        # calculate direct opinion
        direct_opinion = self.calculate_direct_trust(filtered_histories)

        # calculate old reputaion
        init_op = Opinion(0, 0, 1)
        # only get the history before last interaction
        histories_dict = self.history_monitor.get_history_as_provider_witness_SUBJECTIVE(reporter_id, provider_id,
                                                                                         last_timestep_interaction_with_reporter,
                                                                                         local_history_length=self.H,
                                                                                         referral_length=self.nBF)
        filtered_histories_dict = {}
        for key, value in histories_dict.items():
            filtered_histories_dict[key] = [history[1] for history in value if history[2] == task_info]

        old_reputation_opinion = self.calculate_old_reputation(filtered_histories_dict,
                                                               last_timestep_interaction_with_reporter)

        # fade old reputation
        if last_timestep_interaction_with_reporter != math.inf:
            delta_timestep = timestep - last_timestep_interaction_with_reporter
        else:
            delta_timestep = 0
        revised_reputation_opinion = fading_opinion(old_reputation_opinion, delta_timestep)

        # calculate new reputation
        histories_dict = self.history_monitor.get_history_as_provider_witness_TRAVOS(reporter_id, provider_id,
                                                                                     local_history_length=self.H,
                                                                                     referral_length=self.nBF)
        filtered_histories_dict = {}
        for key, value in histories_dict.items():
            filtered_histories_dict[key] = [history[1] for history in value if history[2] == task_info]

        new_reputation_opinion = self.calcluate_new_reputation(filtered_histories_dict, revised_reputation_opinion)

        # judge trust or not
        trust_value = compare_opinion(new_reputation_opinion)

        # update last interaction
        self.last_timestep_interaction_with_reporter[reporter_id][provider_id] = timestep

        return {'trust_value': trust_value, 'direct_opinion': direct_opinion, 'old_reputation': old_reputation_opinion,
                'new_reputation': new_reputation_opinion}



# Example usage of the operators
if __name__ == "__main__":
    # Example evidence to opinion conversion
    opinion_a = evidence_to_opinion(3, 1, 1)
    opinion_b = evidence_to_opinion(2, 2, 1)
    print(f"Opinion A: {opinion_a}")
    print(f"Opinion B: {opinion_b}")

    # Combine opinions using consensus operator
    combined_opinion = consensus_opinion(opinion_a, opinion_b)
    print(f"Combined Opinion: {combined_opinion}")

    # Discount an opinion
    discounted_opinion = discount_opinion(opinion_a, opinion_b)
    print(f"Discounted Opinion: {discounted_opinion}")

    # Apply fading to an opinion
    faded_opinion = fading_opinion(opinion_a, 5)
    print(f"Faded Opinion: {faded_opinion}")

    # Compare an opinion to a trust threshold
    trust_status = compare_opinion(opinion_a)
    print(f"Trust Status of Opinion A: {trust_status}")