"""
Train a housing price prediction model and export it with YAML configuration
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import yaml
from datetime import datetime
import json

# Load the data
print("Loading data...")
train_data = pd.read_csv('train_100k.csv')

# Display basic info
print(f"Training data shape: {train_data.shape}")
print(f"Columns: {list(train_data.columns)}")

# Identify target column (likely 'SalePrice' or similar)
target_cols = [col for col in train_data.columns if 'price' in col.lower() or 'sale' in col.lower()]
if target_cols:
    target_column = target_cols[0]
else:
    # Assume last column is target
    target_column = train_data.columns[-1]

print(f"Target column: {target_column}")

# Separate features and target
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

print(f"\nCategorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Simple preprocessing: One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Handle missing values
X_encoded = X_encoded.fillna(X_encoded.median())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Accuracy: {r2*100:.2f}%")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Save the model and scaler
print("\nSaving model and scaler...")
joblib.dump(model, 'housing_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature names
with open('feature_names.json', 'w') as f:
    json.dump(list(X_train.columns), f)

# Create YAML configuration
model_config = {
    'model_info': {
        'name': 'Housing Price Prediction Model',
        'type': 'RandomForestRegressor',
        'version': '1.0',
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Random Forest model for predicting housing prices'
    },
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'data_info': {
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'total_features': len(X_train.columns),
        'categorical_features': len(categorical_cols),
        'numerical_features': len(numerical_cols),
        'target_column': target_column
    },
    'performance_metrics': {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'accuracy_percentage': float(r2 * 100)
    },
    'feature_importance': {
        row['feature']: float(row['importance']) 
        for _, row in feature_importance.head(20).iterrows()
    },
    'preprocessing': {
        'scaling': 'StandardScaler',
        'categorical_encoding': 'One-Hot Encoding',
        'missing_value_strategy': 'Median Imputation'
    },
    'files': {
        'model_file': 'housing_price_model.pkl',
        'scaler_file': 'scaler.pkl',
        'feature_names_file': 'feature_names.json'
    }
}

# Save YAML configuration
print("Saving model configuration to YAML...")
with open('model_config.yaml', 'w') as f:
    yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)

print("\n" + "="*50)
print("✓ Model saved to: housing_price_model.pkl")
print("✓ Scaler saved to: scaler.pkl")
print("✓ Feature names saved to: feature_names.json")
print("✓ Configuration saved to: model_config.yaml")
print("="*50)
print("\nYou can now run the Streamlit app with: streamlit run app.py")
