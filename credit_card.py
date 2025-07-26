import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv('creditcard.csv')
print("Original Class Distribution:")
print(df['Class'].value_counts())

# Step 2: Split before SMOTE
X = df.drop('Class', axis=1)
y = df['Class']

# Step 3: Split original data into training and testing (keep test untouched)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print("\nAfter SMOTE on Training Set:")
print(pd.Series(y_train_sm).value_counts())

# Step 6: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_sm, y_train_sm)

# Step 7: Predict on original, untouched test data
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluation
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Visualize Confusion Matrix (waits until you manually close the window)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix on Real Data (492 frauds)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()  # <-- stays open until manually closed

# Step 10: Get predicted fraud transactions
print("\nStep 10: Extracting predicted fraud transactions...")
X_test_original = X_test.copy()
X_test_original['Actual_Class'] = y_test
X_test_original['Predicted_Class'] = y_pred

# Filter transactions where the model predicted fraud (Predicted_Class == 1)
predicted_frauds = X_test_original[X_test_original['Predicted_Class'] == 1]

# Display first few predicted fraud transactions
print("\nPredicted Fraud Transactions:")
print(predicted_frauds.head(10))  # Show more if needed

# Count how many were actually fraud
actual_frauds_detected = predicted_frauds[predicted_frauds['Actual_Class'] == 1]
print(f"\nDetected actual frauds: {len(actual_frauds_detected)} out of 492 total frauds in test data")
