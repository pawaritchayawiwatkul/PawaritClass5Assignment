import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ── 1. Load saved model, label encoder, and scaler ──────────────────────
model = keras.models.load_model('models/activity_model_final.h5')
label_encoder = joblib.load('models/label_encoder.pkl')
scaler = joblib.load('models/scaler.pkl')
print("Loaded model, label_encoder, and scaler from models/")
model.summary()

# ── 2. Load and preprocess test data ────────────────────────────────────
print("\nLoading test data ...")
df = pd.read_csv('data/test_data.csv')

# Fill missing values
df = df.fillna(df.median(numeric_only=True))
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Same feature groups as training
demographic_features = ['Age', 'Gender', 'MaritalStatus', 'MonthlyIncome', 'Designation']
travel_features = ['NumberOfTrips', 'Passport', 'OwnCar']
interaction_features = [
    'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
    'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched',
    'PreferredPropertyStar', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting'
]
all_features = demographic_features + travel_features + interaction_features

X = df[all_features]
y = df['ProdTaken']

y_test = label_encoder.transform(y)

categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Align columns to match what the scaler expects
train_columns = scaler.feature_names_in_
X_encoded = X_encoded.reindex(columns=train_columns, fill_value=0)

X_test_scaled = scaler.transform(X_encoded)

# ── 3. Evaluate ─────────────────────────────────────────────────────────
print("\nEvaluating the model on the test set ...")
test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC:      {test_auc:.4f}")

# ── 4. Predictions & reports ────────────────────────────────────────────
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Taken', 'Taken']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))