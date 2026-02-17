import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

# Load the data (use full dataset for better training)
print("Loading data...")
df = pd.read_csv('data/data.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nTarget distribution:\n{df['ProdTaken'].value_counts()}")

# Split into train and test DataFrames (80-20) before any preprocessing
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)
print(f"\nSplit saved: data/train_data.csv ({len(train_df)}), data/test_data.csv ({len(test_df)})")

# Handle missing values (fit on train, apply same to test)
print(f"\nMissing values:\n{train_df.isnull().sum()[train_df.isnull().sum() > 0]}")
train_medians = train_df.median(numeric_only=True)
train_df = train_df.fillna(train_medians)
test_df = test_df.fillna(train_medians)
for col in train_df.select_dtypes(include=['object']).columns:
    mode_val = train_df[col].mode()[0]
    train_df[col] = train_df[col].fillna(mode_val)
    test_df[col] = test_df[col].fillna(mode_val)


# 2.1 Extract the features
# ============================================================

# Demographic attributes (Age, Gender, MaritalStatus, Income, etc.)
demographic_features = ['Age', 'Gender', 'MaritalStatus', 'MonthlyIncome', 'Designation']

# Travel behavior (Trips, Passport, Car ownership, etc.)
travel_features = ['NumberOfTrips', 'Passport', 'OwnCar']

# Interaction data (Product pitched, Satisfaction, Followups, etc.)
interaction_features = [
    'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
    'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched',
    'PreferredPropertyStar', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting'
]

# Target: ProdTaken (0 = No, 1 = Yes)
target = 'ProdTaken'

# Combine all feature groups
all_features = demographic_features + travel_features + interaction_features

print("\n--- Feature Extraction ---")
print(f"Demographic attributes ({len(demographic_features)}): {demographic_features}")
print(f"Travel behavior ({len(travel_features)}): {travel_features}")
print(f"Interaction data ({len(interaction_features)}): {interaction_features}")
print(f"Target: {target}")
print(f"Total features: {len(all_features)}")

# Validate that all expected features exist in the dataset
missing_features = [f for f in all_features if f not in df.columns]
if missing_features:
    print(f"\nWARNING: Missing features in dataset: {missing_features}")
else:
    print("\nAll expected features found in dataset.")


# Plot class distribution (training set)
plt.figure(figsize=(8, 5))
train_df['ProdTaken'].value_counts().plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Class Distribution of ProdTaken')
plt.xlabel('ProdTaken')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=100)

# Separate features and target from train/test DataFrames
X_train_raw = train_df[all_features]
y_train_raw = train_df[target]
X_test_raw = test_df[all_features]
y_test_raw = test_df[target]

# Encode target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

# Identify categorical vs numerical features
categorical_columns = X_train_raw.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X_train_raw.select_dtypes(include=['number']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_columns)}): {categorical_columns}")
print(f"Numerical columns ({len(numerical_columns)}): {numerical_columns}")

# One-hot encode categorical features (align columns between train and test)
X_train_encoded = pd.get_dummies(X_train_raw, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_raw, columns=categorical_columns, drop_first=True)

# Align columns: ensure test has same columns as train
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

print(f"Features after encoding: {X_train_encoded.shape[1]}")
print(f"\nTraining set size: {X_train_encoded.shape[0]}")
print(f"Testing set size: {X_test_encoded.shape[0]}")


# Balance the training set by randomly removing samples with y=0
class_0_indices = np.where(y_train == 0)[0]
class_1_indices = np.where(y_train == 1)[0]

num_class_1 = len(class_1_indices)
indices_to_keep = np.random.choice(class_0_indices, num_class_1, replace=False)
balanced_indices = np.sort(np.concatenate([indices_to_keep, class_1_indices]))

X_train_encoded = X_train_encoded.iloc[balanced_indices]
y_train = y_train[balanced_indices]

print(f"Training set balanced: {np.sum(y_train == 0)} samples with label 0, {np.sum(y_train == 1)} samples with label 1")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Save label encoder and scaler for inference
joblib.dump(label_encoder, 'models/label_encoder.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Saved models/label_encoder.pkl and models/scaler.pkl")

print(f"\nTraining set size: {X_train_scaled.shape[0]}")
print(f"Testing set size: {X_test_scaled.shape[0]}")

# Build an improved neural network for better accuracy
l2 = keras.regularizers.l2(2e-4)
print("\nBuilding neural network model...")
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", kernel_regularizer=l2, input_shape=(X_train_scaled.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.35),

    keras.layers.Dense(32, activation="relu", kernel_regularizer=l2),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),

    keras.layers.Dense(16, activation="relu", kernel_regularizer=l2),
    keras.layers.Dropout(0.15),

    keras.layers.Dense(1, activation="sigmoid"),
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

# Display model summary
model.summary()


def compile_model_with_custom_loss(model, loss_function, optimizer='adam', metrics=None):
    """
    Compile the model with a custom loss function.

    Parameters:
    -----------
    model : keras.Model
        The model to compile
    loss_function : str or callable
        Either a string for built-in loss functions (e.g., 'binary_crossentropy')
        or a custom loss function that takes (y_true, y_pred) as arguments
    optimizer : str or keras.optimizers.Optimizer
        Optimizer to use (default: 'adam')
    metrics : list
        List of metrics to track (default: ['accuracy', AUC])

    Returns:
    --------
    model : keras.Model
        The compiled model

    Example:
    --------
    # Using built-in loss
    compile_model_with_custom_loss(model, 'binary_crossentropy')

    # Using custom loss
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_exp = tf.exp(-bce)
        focal_loss = alpha * tf.pow(1 - bce_exp, gamma) * bce
        return tf.reduce_mean(focal_loss)

    compile_model_with_custom_loss(model, focal_loss)
    """
    if metrics is None:
        metrics = ['accuracy', keras.metrics.AUC(name='auc')]

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )

    return model


def weighted_binary_crossentropy(y_true, y_pred, pos_weight=2.0):
    """
    Weighted binary crossentropy loss for imbalanced datasets.
    Applies higher weight to positive class errors.
    """
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weights = y_true * (pos_weight - 1.0) + 1.0
    weighted_bce = bce * weights
    return tf.reduce_mean(weighted_bce)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance.
    Focuses on hard-to-classify examples.

    Parameters:
    -----------
    alpha : float
        Weighting factor (default: 0.25)
    gamma : float
        Focusing parameter (default: 2.0)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # Calculate focal loss
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)

    focal_loss_value = weight * cross_entropy
    return tf.reduce_mean(focal_loss_value)


def combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.01):
    """
    Combined loss: alpha * binary_crossentropy + beta * L1_regularization on model weights

    This applies L1 regularization to the model's trainable weights instead of predictions.
    This is more commonly used for feature selection and model sparsity.

    Parameters:
    -----------
    model : keras.Model
        The model whose weights will be regularized
    alpha : float
        Weight for binary cross-entropy term (default: 1.0)
    beta : float
        Weight for L1 regularization on weights (default: 0.01)

    Returns:
    --------
    loss_function : callable
        A loss function that takes (y_true, y_pred) as arguments

    Example:
    --------
    custom_loss = combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.001)
    compile_model_with_custom_loss(model, custom_loss)
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # Binary cross-entropy component
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_loss = tf.reduce_mean(bce)

        # L1 regularization on model weights
        l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])

        # Combined loss
        total_loss = alpha * bce_loss + beta * l1_reg

        return total_loss

    return loss


# Compile with standard binary crossentropy (best for balanced dataset)
compile_model_with_custom_loss(model, 'binary_crossentropy')

# Create output directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    verbose=2,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
)

# Save the trained model
model.save('models/activity_model_final.h5')
print("Saved model to models/activity_model_final.h5")

# ── 3. Evaluate ─────────────────────────────────────────────────────────
print("\nEvaluating the model on the training set ...")
test_loss, test_accuracy, test_auc = model.evaluate(X_train_scaled, y_train, verbose=0)
print(f"Train Loss:     {test_loss:.4f}")
print(f"Train Accuracy: {test_accuracy:.4f}")
print(f"Train AUC:      {test_auc:.4f}")

# ── 4. Predictions & reports ────────────────────────────────────────────
y_pred_proba = model.predict(X_train_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_train, y_pred, target_names=['Not Taken', 'Taken']))

print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred))

# Plot training history
plt.figure(figsize=(14, 5))

# Plot accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot AUC
plt.subplot(1, 3, 3)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100)
plt.show()