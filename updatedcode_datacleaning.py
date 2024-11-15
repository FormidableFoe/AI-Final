# -*- coding: utf-8 -*-
"""FinalProject.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1emsr3BPOr0mcDKE_--Gjmk3CATO_JN06

1. Upload the file by running the first code block

2. Then run the second block to run the decision tree algorithm

**The rest of this is the description of the data**

Features are along x-axis(columns 1 to 80)
Samples are along the y-axis(rows)
The last column(81) consists labels such that:

1 == index_finger

2 == middle_finger

3 == ring_finger

4 == little_finger

5 == thumb

6 == rest

7 == victory_gesture

There are 80 columns because there were 8 electrodes and 10 features were extracted for each electrode.

Features are in the order {standard_deviation; root_mean_square; minimum; maximum; zero_crossings; average_amplitude_change; amplitude_first_burst; mean_absolute_value; wave_form_length; willison_amplitude}

First 8 columns are standard_deviation, the next 8 columns are root_mean_square and so on according to the order described above...

Note: You may want to normalize some features because their ranges are dramatically different.
"""

# Run this to upload the CSV file
from google.colab import files
uploaded = files.upload()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('emg_all_features_labeled.csv', header=None)

# Remove duplicate rows
data = data.drop_duplicates()

# Handle missing values (remove rows with any missing values)
data = data.dropna()

# Separate features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize
plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, max_depth=3)
plt.show()
