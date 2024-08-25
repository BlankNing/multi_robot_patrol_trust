import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
# Load the train and test datasets
train_df = pd.read_csv('dataset/reporter/labeled_train.csv')
test_df = pd.read_csv('dataset/reporter/labeled_test.csv')
# Features used for training
features = [
    'avg_rating_300', 'avg_rating_1000', 'avg_rating_3000',
    'interaction_count_300', 'interaction_count_1000', 'interaction_count_3000',
    'distance', 'same_type'
]
# Extracting features and labels
X_train = train_df[features]
y_train = train_df['cluster']
X_test = test_df[features]
y_test = test_df['cluster']
# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train the SVM model with an RBF kernel
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
# Make predictions on the test data
y_pred = svm_model.predict(X_test_scaled)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

model_filename = 'models/reporter/svm_model_rbf.pkl'
joblib.dump(svm_model, model_filename)

scaler_filename = 'models/reporter/scaler.pkl'
joblib.dump(scaler, scaler_filename)

print(f'Model saved to {model_filename}')