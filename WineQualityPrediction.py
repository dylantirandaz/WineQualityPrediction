import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import time
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
wine = pd.read_csv('winequality.csv')

# Perform EDA and visualizations
print(wine.head())
print(wine.describe())
print(wine.info())

plt.figure(figsize=(10, 6))
sns.countplot(wine['quality'])
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 10))
sns.pairplot(wine)
plt.title('Pairplot of Wine Quality Dataset')
plt.show()

plt.figure(figsize=(12, 10))
corr = wine.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Feature engineering and preprocessing
X = wine.drop('quality', axis=1)
y = wine['quality']

# Convert 'quality' to binary classification
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
y = wine['quality']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate different models
models = [
    LogisticRegression(random_state=42),
    SVC(kernel='linear', random_state=42),
    SVC(kernel='rbf', random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42)
]

results = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append([model.__class__.__name__, accuracy, precision, recall, f1])

# Model optimization and selection
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

best_model = models[results_df['Accuracy'].idxmax()]
print("Best Model:", best_model.__class__.__name__)

# Make predictions and interpret the results
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
