# Kaden DuBois
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # For splitting only

# Load dataset
df = pd.read_csv("secondary_data.csv", sep = ';')
df = df.sample(n = 2500, random_state = 69)

# Encode categorical variables
target_col = 'class' # Target column

# Label encode the target
label_mapping = {label: idx for idx, label in enumerate(df[target_col].unique())}
y = df[target_col].map(label_mapping).values

# One-hot encode features
X = pd.get_dummies(df.drop(columns = [target_col])).values.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69)

# Distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# kNN prediction
def knn_predict(X_train, y_train, x_test, k = 5):
    distances = [euclidean_distance(x_test, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

k_values = [3, 5, 11, 25, 51, 101]
accuracies = []

for k in k_values:
    correct = 0
    for i in range(len(X_test)):
        pred = knn_predict(X_train, y_train, X_test[i], k = k)
        if pred == y_test[i]:
            correct += 1

    # Print Outcome
    accuracy = correct / len(X_test)
    accuracies.append(accuracy * 100)  # Convert to percentage
    print(f"Accuracy for k = {k}: {accuracy * 100:.2f}%")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title("kNN Accuracy vs. Number of Neighbors (k)")
plt.xlabel("k (Number of Neighbors)", fontsize = 16)
plt.ylabel("Accuracy (%)", fontsize = 16)
plt.grid(True)
plt.tight_layout()
plt.show()