import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import joblib
import os

# Ensure directories exist
if not os.path.exists('model'):
    os.makedirs('model')
if not os.path.exists('static/plots'):
    os.makedirs('static/plots')

# Set aesthetic style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# 1. Load Data
print("Loading dataset...")
try:
    df = pd.read_csv('ds_cvd_w1.csv')
except FileNotFoundError:
    print("Error: ds_cvd_w1.csv not found!")
    exit()

if 'id' in df.columns:
    df = df.drop(columns=['id'])

if df['age'].mean() > 150:
    print("Pre-processing: Converting age to years...")
    df['age'] = (df['age'] / 365.25).round().astype(int)

X = df.drop(columns=['cardio'])
y = df['cardio']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 4. Save Model
print("Saving model...")
joblib.dump(model, 'model/cardio_model.pkl')

# --- VISUALIZATION GENERATION ---

print("Generating Visualizations...")

# A. Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 14})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('static/plots/confusion_matrix.png', dpi=300)
plt.close()

# B. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[features[i] for i in indices], hue=[features[i] for i in indices], palette='viridis', legend=False)
plt.title('Feature Importance', fontsize=16, fontweight='bold')
plt.xlabel('Relative Importance', fontsize=12)
plt.tight_layout()
plt.savefig('static/plots/feature_importance.png', dpi=300)
plt.close()

# C. ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('static/plots/roc_curve.png', dpi=300)
plt.close()

# D. Prediction Probability Distribution
plt.figure(figsize=(8, 6))
sns.histplot(y_prob, bins=20, kde=True, color='purple')
plt.title('Prediction Probability Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Probability of Heart Disease')
plt.tight_layout()
plt.savefig('static/plots/prob_dist.png', dpi=300)
plt.close()

# E. Learning Curve (Sub-sample for speed)
print("Generating Learning Curve (this might take a few seconds)...")
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.title('Learning Curve', fontsize=16, fontweight='bold')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('static/plots/learning_curve.png', dpi=300)
plt.close()

# F. Age Distribution (Data Insight)
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', hue='cardio', element="step", stat="density", common_norm=False, palette='coolwarm')
plt.title('Age Distribution by Health Status', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('static/plots/age_dist.png', dpi=300)
plt.close()

print("All plots generated successfully in static/plots/!")
