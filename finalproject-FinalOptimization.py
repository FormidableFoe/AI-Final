import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
data = pd.read_csv('emg_all_features_labeled.csv', header=None)

# Remove duplicate rows
data = data.drop_duplicates()

# Remove rows with any missing values
data = data.dropna()

# Separate features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# Perform untuned runs
models_untuned = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Untuned accuracies
untuned_accuracies = {}
for name, model in models_untuned.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    untuned_accuracies[name] = accuracy_score(y_test, predictions)

# Perform tuned runs with best parameters
models_tuned = {
    "Decision Tree": DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=5, random_state=42),
    "Random Forest": RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200, random_state=42),
    "SVM": SVC(C=10, gamma='scale', kernel='rbf', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(max_depth=12, learning_rate=0.1, n_estimators=200, random_state=42)
}

# Tuned accuracies
tuned_accuracies = {}
for name, model in models_tuned.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    tuned_accuracies[name] = accuracy_score(y_test, predictions)

# Prepare data for plotting
untuned = list(untuned_accuracies.values())
tuned = list(tuned_accuracies.values())
model_names = list(models_untuned.keys())

# Plot comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(model_names))  # Label locations
width = 0.35  # Width of bars

plt.bar(x - width/2, untuned, width, label='Untuned', color='lightcoral', edgecolor='black')
plt.bar(x + width/2, tuned, width, label='Tuned', color='lightblue', edgecolor='black')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Untuned and Tuned Model Accuracies')
plt.xticks(x, model_names)
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y')

for i, v in enumerate(untuned):
    plt.text(i - 0.35, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
for i, v in enumerate(tuned):
    plt.text(i + 0.35, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
