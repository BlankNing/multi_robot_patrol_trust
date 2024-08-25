from .AbstractTrust import Trust
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MLTrust(Trust):
    def __init__(self, config):
        # relevant items
        self.history_monitor = config['history_monitor']
        # useful info
        self.reporter_model = joblib.load(config['trust_model_path_reporter'])
        self.provider_model = joblib.load(config['trust_model_path_provider'])
        self.reporter_scaler = joblib.load(config['scaler_path_provider'])
        self.provider_scaler = joblib.load(config['scaler_path_provider'])
        # hyperparameters
        self.nBF = 2
        self.H = 10

    def trust_label_to_value(self, label):
        if label == 1:
            return 0.0
        elif label == 0:
            return 1.0

    def obtain_features(self, history, current_timestep, distance):
        avg_rating_300 = avg_rating_1000 = avg_rating_3000 = 0
        interaction_count_300 = interaction_count_1000 = interaction_count_3000 = 0
        same_type = None
        # Filter relevant interactions within different time windows
        relevant_300 = [entry for entry in history if current_timestep - entry[0] <= 300]
        relevant_1000 = [entry for entry in history if current_timestep - entry[0] <= 1000]
        relevant_3000 = [entry for entry in history if current_timestep - entry[0] <= 3000]
        # Calculate average ratings for each window
        if relevant_300:
            avg_rating_300 = np.mean([entry[1] for entry in relevant_300])
            interaction_count_300 = len(relevant_300)
        if relevant_1000:
            avg_rating_1000 = np.mean([entry[1] for entry in relevant_1000])
            interaction_count_1000 = len(relevant_1000)
        if relevant_3000:
            avg_rating_3000 = np.mean([entry[1] for entry in relevant_3000])
            interaction_count_3000 = len(relevant_3000)
        if history != []:
            last_interaction = history[-1]
            same_type = last_interaction[3]

        # Assemble features into a dictionary
        features = {
            'avg_rating_300': avg_rating_300,
            'avg_rating_1000': avg_rating_1000,
            'avg_rating_3000': avg_rating_3000,
            'interaction_count_300': interaction_count_300,
            'interaction_count_1000': interaction_count_1000,
            'interaction_count_3000': interaction_count_3000,
            'distance': distance,
            'same_type': same_type
        }
        return features

    def calculate_trust_value_reporter(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks): # return (0,1)
        histories = self.history_monitor.get_history_as_reporter(reporter_id, provider_id)
        filtered_histories = np.array([history for history in histories if history[2] == task_info][-self.H:])
        # get features
        distance = self.history_monitor.calculate_distance(reporter_id, provider_id)
        features = self.obtain_features(filtered_histories, timestep, distance)

        features_df = pd.DataFrame([features])
        features_df.fillna(0, inplace=True)
        scaled_features = self.reporter_scaler.fit_transform(features_df)

        # model predict trust level
        predict_label = self.reporter_model.predict(scaled_features)
        trust_value = self.trust_label_to_value(predict_label)

        return {'trust_value': trust_value, 'features': features}


    def calculate_trust_value_provider(self, reporter_id, provider_id, task_info, timestep, robot_capable_tasks): # return (0,1)

        histories = self.history_monitor.get_history_as_provider(reporter_id, provider_id)
        filtered_histories = np.array([history for history in histories if history[2] == task_info][-self.H:])
        # get features
        distance = self.history_monitor.calculate_distance(reporter_id, provider_id)
        features = self.obtain_features(filtered_histories, timestep, distance)

        features_df = pd.DataFrame([features])
        features_df.fillna(0, inplace=True)
        scaled_features = self.provider_scaler.fit_transform(features_df)

        # model predict trust level
        predict_label = self.provider_model.predict(scaled_features)
        trust_value = self.trust_label_to_value(predict_label)

        return {'trust_value': trust_value, 'features': features}
