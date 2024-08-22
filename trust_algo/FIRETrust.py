from .AbstractTrust import Trust
import numpy as np

class FIRETrust(Trust):
    def __init__(self, config):
        # relevant items
        self.history_monitor = config['history_monitor']
        # useful info
        self.reporter_id = -1
        self.provider_id = -1
        self.task_info = -1
        self.current_time = -1
        self.robot_capable_tasks = {}
        # hyperparameters
        self.nBF = 2
        self.RL = 5
        self._lambda = 100
        self.H = 10

        self.rho_IT = -np.log(0.5)
        self.rho_RT = -np.log(0.5)
        self.rho_WR = -np.log(0.5)
        self.rho_CR = -np.log(0.5)

        self.WI = 2.0
        self.WR = 2.0
        self.WW = 1.0
        self.WC = 0.5

    def trust_reliability_calculation(self, filtered_histories, rho):
        if filtered_histories.shape[0] == 0:
            return 0,0
        else:
            # calculate w weight
            timesteps = filtered_histories[:, 0]
            delta_timestep = timesteps - self.current_time
            w = np.exp(delta_timestep/self._lambda)

            # calculate average sum of rating and weight w
            weighted_ratings = w * filtered_histories[:, 1]
            TR = np.sum(weighted_ratings)/np.sum(w)

            # reliability calculation
            rho_RK = 1 - np.exp(-rho * np.sum(w))
            rho_DK = 1 - 0.5 * np.sum(w * np.abs(filtered_histories[:, 1] - TR))/np.sum(w)
            rho_K = rho_DK * rho_RK

            return TR, rho_K

    def get_IT_reporter(self, reporter_id, provider_id, task_info):
        '''
        As a reporter, get the individual interaction history with a single provider
        :return:
        '''
        histories = self.history_monitor.get_history_as_reporter(reporter_id, provider_id)
        filtered_histories = np.array([history for history in histories if history[2] == task_info][-self.H:])
        return self.trust_reliability_calculation(filtered_histories, self.rho_IT)

    def get_WR_reporter(self, reporter_id, provider_id, task_info):
        '''
        As a reporter, get the witness reputation history towards a single provider
        :return:
        '''
        histories = self.history_monitor.get_history_as_reporter_witness_FIRE(reporter_id, provider_id,
                                                                local_history_length = self.H, referral_length=self.nBF)
        filtered_histories = np.array([history for history in histories if history[2] == task_info])

        return self.trust_reliability_calculation(filtered_histories, self.rho_WR)


    def get_CR_reporter(self, reporter_id, provider_id, task_info):
        '''
        As a reporter, get the certified reputation history towards a single reporter
        :return:
        '''
        filtered_histories = self.history_monitor.get_history_as_reporter_certified_witness_FIRE(reporter_id, provider_id)
        sorted_histories = sorted(filtered_histories, key=lambda x: x[1], reverse=False)[-10:]
        return self.trust_reliability_calculation(np.array(sorted_histories), self.rho_CR)

    def get_RB_reporter(self, reporter_id, provider_id, task_info):
        reporter_task = self.robot_capable_tasks[reporter_id]
        provider_task = self.robot_capable_tasks[provider_id]

        # todo: hyperparameter
        if reporter_task == provider_task:
            TR_RT = 1
            rho_RT = 0.8
        else:
            TR_RT = 0.5
            rho_RT = 0.3

        return TR_RT, rho_RT


    def calculate_trust_value_reporter(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks): # return (0,1)
        self.reporter_id = reporter_id
        self.provider_id = provider_id
        self.current_time = timestep
        self.task_info  = task_info
        self.robot_capable_tasks = robot_capable_tasks

        TR_IT, rho_IT = self.get_IT_reporter(reporter_id, provider_id, task_info)
        TR_RT, rho_RT = self.get_RB_reporter(reporter_id, provider_id, task_info)
        TR_WR, rho_WR = self.get_WR_reporter(reporter_id, provider_id, task_info)
        TR_CR, rho_CR = self.get_CR_reporter(reporter_id, provider_id, task_info)

        Tk = np.array([TR_IT, TR_RT, TR_WR, TR_CR])
        rho_K = np.array([rho_IT, rho_RT, rho_WR, rho_CR])
        Wk = np.array([self.WI, rho_WR, self.WW, self.WC])
        wk = rho_K * Wk

        T = np.sum(Tk * wk)/np.sum(wk)
        rho_T = np.sum(wk)/np.sum(Wk)

        return {'trust_value': T, 'reliability_of_trust': rho_T}



    def get_IT_provider(self, reporter_id, provider_id, task_info):
        '''
        As a provider, get the individual interaction history with a single reporter
        :return:
        '''
        histories = self.history_monitor.get_history_as_provider(reporter_id, provider_id)
        filtered_histories = np.array([history for history in histories if history[2] == task_info][-self.H:])

        return self.trust_reliability_calculation(filtered_histories, self.rho_IT)

    def get_WR_provider(self, reporter_id, provider_id, task_info):
        '''
        As a provider, get the witness reputation history towards a single reporter
        :return:
        '''
        histories = self.history_monitor.get_history_as_provider_witness_FIRE(reporter_id, provider_id,
                                                                local_history_length = self.H, referral_length=self.nBF)
        filtered_histories = np.array([history for history in histories if history[2] == task_info])

        return self.trust_reliability_calculation(filtered_histories, self.rho_WR)

    def get_CR_provider(self, reporter_id, provider_id, task_info):
        '''
        As a provider, get the certified reputation history towards a single reporter
        :return:
        '''
        filtered_histories = self.history_monitor.get_history_as_provider_certified_witness_FIRE(reporter_id, provider_id)
        sorted_histories = sorted(filtered_histories, key=lambda x: x[1], reverse=False)[-10:]
        return self.trust_reliability_calculation(np.array(sorted_histories), self.rho_CR)

    def get_RB_provider(self, reporter_id, provider_id, task_info):
        reporter_task = self.robot_capable_tasks[reporter_id]
        provider_task = self.robot_capable_tasks[provider_id]
        # todo: hyperparameter
        if reporter_task == provider_task:
            TR_RT = 1
            rho_RT = 0.8
        else:
            TR_RT = 0.5
            rho_RT = 0.3

        return TR_RT, rho_RT
    def calculate_trust_value_provider(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks): # return (0,1)
        self.reporter_id = reporter_id
        self.provider_id = provider_id
        self.current_time = timestep
        self.task_info  = task_info
        self.robot_capable_tasks = robot_capable_tasks

        TR_IT, rho_IT = self.get_IT_provider(reporter_id, provider_id, task_info)
        TR_RT, rho_RT = self.get_RB_provider(reporter_id, provider_id, task_info)
        TR_WR, rho_WR = self.get_WR_provider(reporter_id, provider_id, task_info)
        TR_CR, rho_CR = self.get_CR_provider(reporter_id, provider_id, task_info)

        Tk = np.array([TR_IT, TR_RT, TR_WR, TR_CR])
        rho_K = np.array([rho_IT, rho_RT, rho_WR, rho_CR])
        Wk = np.array([self.WI, rho_WR, self.WW, self.WC])
        wk = rho_K * Wk

        T = np.sum(Tk * wk)/np.sum(wk)
        rho_T = np.sum(wk)/np.sum(Wk)

        return {'trust_value': T, 'reliability_of_trust': rho_T}
