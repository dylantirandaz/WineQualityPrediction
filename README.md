# Wine Quality Classification

This project involves building and evaluating machine learning models to classify wine quality based on various physicochemical properties. The dataset used is the "Wine Quality" dataset, which contains information about different types of wine and their quality ratings.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering and Preprocessing](#feature-engineering-and-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Comparison](#model-comparison)
- [Best Model Selection](#best-model-selection)
- [Conclusion](#conclusion)

## Installation

To run this project, you need to have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
## Dataset

The dataset used in this project is in the "Wine Quality" dataset. It can be downloaded from UCI Machine Learning Repository(https://archive.ics.uci.edu/dataset/186/wine+quality)

Load the dataset:

```bash
wine = pd.read_csv('winequality.csv')
```

## Exploratory Data Analysis (EDA)

Perform basic EDA to understand the dataset:

```bash
print(wine.head())
print(wine.describe())
print(wine.info())
```
Visualize the distribution of wine quality:

```bash
plt.figure(figsize=(10, 6))
sns.countplot(wine['quality'])
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()
```
Pairplot of the dataset:

```bash
plt.figure(figsize=(12, 10))
sns.pairplot(wine)
plt.title('Pairplot of Wine Quality Dataset')
plt.show()
```
Correlation matrix heatmap:

```bash
plt.figure(figsize=(12, 10))
corr = wine.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
```

## Feature Engineering and Preprocessing

Prepare the data for modeling:
1. Drop the target variable `quality` from the features.
2. Convert `quality` to a binary classification: `bad` (2-6.5) and `good` (6.5-8).
3. Standardize the features using `StandardScaler`.

```bash
X = wine.drop('quality', axis=1)
y = wine['quality']

bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
y = wine['quality']

scaler = StandardScaler()
X = scaler.fit_transform(X)
```
Split the data into training and testing sets:
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Model Training and Evaluation

Train and evaluate different models:
1. Logistic Regression
2. Support Vector Machine (SVM) with linear kernel
3. SVM with RBF kernel
4. Random Forest Classifier

```bash
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
```

## Model Comparison

Compare the performance of the models:

```bash
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()
```

## Best Model Selection

Select the best model based on accuracy:

```bash
best_model = models[results_df['Accuracy'].idxmax()]
print("Best Model:", best_model.__class__.__name__)

y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
```

## Conclusion 
This project demonstrates how to build and evaluate different machine learning models to classify wine quality. The best performing model can be selected based on various performance metrics such as accuracy, precision, recall, and F1 score.




