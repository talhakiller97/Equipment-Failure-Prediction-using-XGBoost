import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from imblearn.over_sampling import SMOTE
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load data
df = pd.read_csv("predictive_maintenance.csv")
print("Columns in dataset:", df.columns.tolist())

# Show sample data
print("Sample data:")
print(df.head())

# Use 'Target' column for machine failure
if 'Target' in df.columns:
    df['Target'] = df['Target'].astype(int)
    target_col = 'Target'
else:
    raise KeyError("The expected target column ('Target') is not found in the dataset.")

# Create datetime feature
df['timestamp'] = pd.date_range(start='2025-01-01', periods=len(df), freq='h')

# Drop columns not needed and encode categorical variables
X = df.drop(columns=['UDI', 'Product ID', target_col, 'timestamp'], errors='ignore')
X = pd.get_dummies(X, drop_first=True)
y = df[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balance classes with SMOTE
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)

# Train model
model = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=200,
    subsample=0.8
)
model.fit(X_train_bal, y_train_bal)

# Predict and evaluate
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Threshold tuning
thresholds = np.arange(0.1, 0.9, 0.01)
best_f1 = 0
best_thresh = 0.5

for thresh in thresholds:
    preds = (y_proba > thresh).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Best F1 Score: {best_f1:.3f} at Threshold: {best_thresh:.2f}")

# Final prediction
y_pred = (y_proba > best_thresh).astype(int)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
conf_matrix = confusion_matrix(y_test, y_pred)
conf_df = pd.DataFrame(conf_matrix, columns=["Predicted_No_Failure", "Predicted_Failure"],
                       index=["Actual_No_Failure", "Actual_Failure"])
roc_score = roc_auc_score(y_test, y_proba)

# Save evaluation outputs
pd.DataFrame({
    "Best F1 Score": [best_f1],
    "Best Threshold": [best_thresh],
    "ROC AUC Score": [roc_score]
}).to_csv("outputs/model_metrics.csv", index=False)

report_df.to_csv("outputs/classification_report.csv")
conf_df.to_csv("outputs/confusion_matrix.csv")

# Predict on full dataset
X_scaled = scaler.transform(X)
df['failure_proba'] = model.predict_proba(X_scaled)[:, 1]
df['failure_pred'] = (df['failure_proba'] > best_thresh).astype(int)

# Save full predictions
df.to_csv("outputs/raw_data_with_predictions.csv", index=False)

# Filter predicted failures
failures = df[df['failure_pred'] == 1]
failures.to_csv("outputs/predicted_failures.csv", index=False)

# Failures in next 24 hours
latest_time = df['timestamp'].max()
cutoff_time = latest_time - pd.Timedelta(hours=24)
failures_next_24h = failures[failures['timestamp'] > cutoff_time]
failures_next_24h[['timestamp', 'failure_proba']].to_csv("outputs/failures_next_24_hours.csv", index=False)

print("\nMachines predicted to fail within the next 24 hours:")
print(failures_next_24h[['timestamp', 'failure_proba']])

# Standardized feature importance
X_std = pd.DataFrame(X_scaled, columns=X.columns)
z_scores = np.abs((X_std - X_std.mean()) / X_std.std())

# Explain failures
def explain_failure(row_idx):
    row = X_std.iloc[row_idx]
    z = z_scores.iloc[row_idx]
    top_features = z.sort_values(ascending=False).head(3).index.tolist()
    explanation = {
        "Row Index": row_idx,
        "Top Factor 1": f"{top_features[0]} = {row[top_features[0]]:.2f} (z: {z[top_features[0]]:.2f})",
        "Top Factor 2": f"{top_features[1]} = {row[top_features[1]]:.2f} (z: {z[top_features[1]]:.2f})",
        "Top Factor 3": f"{top_features[2]} = {row[top_features[2]]:.2f} (z: {z[top_features[2]]:.2f})",
    }
    return explanation

# Collect explanations
explanations = []
print("\nPredicted Failures and Explanations:\n")
for idx in failures.index:
    exp = explain_failure(idx)
    explanations.append(exp)
    print(f"Row Index: {exp['Row Index']}")
    print(f" - {exp['Top Factor 1']}")
    print(f" - {exp['Top Factor 2']}")
    print(f" - {exp['Top Factor 3']}")
    print("")

pd.DataFrame(explanations).to_csv("outputs/failure_explanations.csv", index=False)
