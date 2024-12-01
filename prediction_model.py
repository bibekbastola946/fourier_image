from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import joblib

def train_model_with_all_features(csv_path='enhanced_fourier_analysis.csv'):
    # Load the data
    df = pd.read_csv(csv_path)

    # Use all available features
    features = ['std_magnitude', 'max_magnitude', 'total_power', 'low_freq_power', 'high_freq_power']
    target = 'PercentWear'

    # Drop rows with missing target or features
    df = df.dropna(subset=[target] + features)

    # Define X (features) and y (target)
    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluation Metrics
    metrics = {
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Train MSE': mean_squared_error(y_train, y_train_pred),
        'Test MSE': mean_squared_error(y_test, y_test_pred),
        'Train R²': r2_score(y_train, y_train_pred),
        'Test R²': r2_score(y_test, y_test_pred),
    }

    # Print Results
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

    # Save the model
    joblib.dump(model, 'trained_model_all_features.pkl')
    print("Model saved as 'trained_model_all_features.pkl'.")

    return model, metrics

# Train the model with all features
train_model_with_all_features()
