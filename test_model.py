import pandas as pd
import joblib

# Paths to model and test data
model_path = 'trained_model_all_features.pkl'  # Updated model file path
test_data_path = 'test_metadata_last_100.csv'  # Replace with your test metadata file path

# Load the model
model = joblib.load(model_path)

# Load test data
test_data = pd.read_csv(test_data_path)

# Select all features
features = ['std_magnitude', 'max_magnitude', 'total_power', 'low_freq_power', 'high_freq_power']
X_test = test_data[features]

# Generate predictions
predictions = model.predict(X_test)

# Add predictions to the test data
test_data['PredictedWear'] = predictions

# Save results to a new CSV
output_file = 'predicted_test_results_all_features.csv'
test_data.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
