import pandas as pd
import numpy as np

env_number = 300
# Load the CSV file into a DataFrame
df = pd.read_csv(f'./histories_{env_number}.csv')


# Function to calculate average rating, interaction count, and distance
def calculate_features(df, reporter_id, provider_id, current_time, window_size):
    # Filter interactions within the time window
    relevant_interactions = df[(df['reporter_id'] == reporter_id) &
                               (df['provider_id'] == provider_id) &
                               (df['report_time'] >= current_time - window_size) &
                               (df['report_time'] < current_time)]

    if len(relevant_interactions) == 0:
        return None, None

    # Average rating
    avg_rating = relevant_interactions['rating_to_reporter'].mean()

    # Count of interactions
    interaction_count = len(relevant_interactions)

    return avg_rating, interaction_count


# Create a new DataFrame to store the features
features = []

# Iterate through each interaction in the original DataFrame
for index, row in df.iterrows():
    reporter_id = row['reporter_id']
    provider_id = row['provider_id']
    current_time = row['report_time']

    # Calculate features for different time windows
    avg_rating_300, interaction_count_300 = calculate_features(df, reporter_id, provider_id, current_time, 300)
    avg_rating_1000, interaction_count_1000 = calculate_features(df, reporter_id, provider_id, current_time, 1000)
    avg_rating_3000, interaction_count_3000 = calculate_features(df, reporter_id, provider_id, current_time, 3000)

    # Distance between two robots
    distance = row['distance_penalty']

    # Same type
    same_type = row['is_same_type']

    # Append features to the list
    features.append([
        reporter_id,
        provider_id,
        current_time,
        avg_rating_300, avg_rating_1000, avg_rating_3000,
        interaction_count_300, interaction_count_1000, interaction_count_3000,
        distance,
        same_type
    ])

# Create a DataFrame from the features list
features_df = pd.DataFrame(features, columns=[
    'reporter_id', 'provider_id', 'current_time',
    'avg_rating_300', 'avg_rating_1000', 'avg_rating_3000',
    'interaction_count_300', 'interaction_count_1000', 'interaction_count_3000',
    'distance', 'same_type'
])

# Save the features DataFrame to a new CSV file
features_df.to_csv(f'./derived_features_as_provider_{env_number}.csv', index=False)

print(f"Features derived and saved to 'derived_features_as_provider_{env_number}.csv'.")