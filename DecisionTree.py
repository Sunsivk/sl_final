# Kaden DuBois
import pandas as pd
import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Entropy function to calculate the entropy of a set of labels
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

# Information Gain function to calculate the information gain of a feature
def information_gain(data, feature, target):
    # Drop rows where the feature is NaN
    data = data[data[feature].notna()]
    
    if data.empty:
        return 0  # No information gain if we can't evaluate the feature

    # Calculate the entropy of the target variable
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    
    # Calculate the weighted entropy for the feature
    weighted_entropy = 0
    for value in values:
        subset = data[data[feature] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset[target])
    
    return total_entropy - weighted_entropy


class DecisionNode:
    def __init__(self, feature = None, children  = None, is_leaf = False, prediction = None):
        self.feature = feature
        self.children = children or {}
        self.is_leaf = is_leaf
        self.prediction = prediction

def build_tree(data, features, target):
    labels = data[target]
    
    # Base cases
    if len(set(labels)) == 1:
        return DecisionNode(is_leaf=True, prediction=labels.iloc[0])
    
    if not features:
        most_common = labels.mode()[0]
        return DecisionNode(is_leaf = True, prediction = most_common)
    
    # Choose best feature
    gains = {feature: information_gain(data, feature, target) for feature in features}
    best_feature = max(gains, key = gains.get)

    node = DecisionNode(feature = best_feature)
    feature_values = data[best_feature].unique()

    # Handle NaN values by creating a separate branch
    for value in feature_values:
        subset = data[data[best_feature] == value]
        if subset.empty:
            most_common = labels.mode()[0]
            node.children[value] = DecisionNode(is_leaf = True, prediction = most_common)
        else:
            remaining_features = [f for f in features if f != best_feature]
            node.children[value] = build_tree(subset, remaining_features, target)
    
    return node

# Predict function to traverse the tree and make predictions
def predict(instance, node):
    if node.is_leaf:
        return node.prediction
    
    feature_value = instance[node.feature]
    child = node.children.get(feature_value)
    
    if child:
        return predict(instance, child)
    else:
        return None  # Handle unknown value

# Evaluate function to calculate accuracy
def evaluate(data, tree, target):
    predictions = data.apply(lambda row: predict(row, tree), axis = 1)
    accuracy = (predictions == data[target]).mean()
    return accuracy

# Load and preprocess dataset
df = pd.read_csv("secondary_data.csv", sep = ';')

# Count the number of NaNs per row
nan_counts = df.isnull().sum(axis = 1)

# Remove rows where the count of NaNs is greater than or equal to 5
df = df[nan_counts < 4]

df = df.sample(frac = 1, random_state = 42).reset_index(drop = True)  # Shuffle the dataset

features = list(df.columns)
features.remove('class')  # Remove target column

# Train-test splits
train_df = df.sample(frac = 0.8, random_state = 42)
test_df = df.drop(train_df.index)

# Use the same 300 points across all evaluations
test_sample = test_df.sample(n = 300, random_state = 42)  # adjust if test set < 1000

# Define training set sizes (e.g., from 100 to full training set)
train_sizes = np.linspace(100, len(train_df), num = 10, dtype = int)

accuracies = []

print("Training Size\tAccuracy")

for size in train_sizes:
    subset = train_df.sample(n = size, random_state = 42)
    tree = build_tree(subset, features, target = 'class')
    acc = evaluate(test_sample, tree, target = 'class')
    accuracies.append(acc * 100)
    print(f"{size}\t\t{acc:.2%}")

# Plot accuracy vs. training sample size
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, accuracies, marker = 's', linestyle = '-', color = 'blue')
plt.title("Decision Tree Accuracy vs. Training Set Size", fontsize = 16)
plt.xlabel("Number of Training Samples", fontsize = 16)
plt.ylabel("Accuracy (%)", fontsize = 16)
plt.grid(True)
plt.tight_layout()
plt.show()