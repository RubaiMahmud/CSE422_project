# -*- coding: utf-8 -*-
"""MushroomDatasetAnalysis_422Project

Original file is located at
    https://colab.research.google.com/drive/1T9ai9Ho1oCuNtV97b-9r7YtjOxC2Ho8f
"""

import pandas as pd
import numpy as np
from google.colab import drive
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

drive.mount('/content/drive')

"""# **Load Dataset**"""

file_path = "/content/drive/MyDrive/dataset/mushroom_dataset.csv"

dataset = pd.read_csv(file_path, sep=';')

print("\nFirst 10 rows of the dataset:")
dataset.head(10)

"""# **Summarize Data**"""

dataset

"""# **Dataset** **Shape**"""

print ('Shape of the dataset is {}. This dataset contains {} row and {} columns.'.format(dataset.shape,dataset.shape[0],dataset.shape[1]))

"""# **Features and Datatypes**"""

dataset.info()

num_data_points = dataset.shape[0]
num_features = dataset.shape[1] - 1

"""# **Number** **of** **Features**"""

print(f"Number of features: {num_features}")

"""# **Number of Datapoints**"""

print(f"Number of data points: {num_data_points}")

"""# **Feature** **Types** (Data Splitting)  """

##Selecting numerical features
numerical_data = dataset.select_dtypes(include='number')

#append the features of numerical_data to list
numerical_features=numerical_data.columns.tolist()

print(f'There are {len(numerical_features)} numerical features:', '\n')
print(numerical_features)

#Selecting categoricalfeatures
categorical_data = dataset.select_dtypes(include= 'object')

#append the features of categorical_data to list
categorical_features=categorical_data.columns.tolist()

print(f'There are {len(categorical_features)} categorical features:', '\n')
print(categorical_features)

"""# **Descriptive Analysis**

**Summary of Numerical Features**
"""

# Transposed stats for numerical features
numerical_data.describe().T

"""**Summary of Categorical Features**"""

# Transposed stats for categorical features
categorical_data.describe().T

"""**Varience of each numerical feature**"""

numerical_data.var()

numerical_data.skew()

"""# **Histograms and Box Plot**"""

dataset.hist(figsize=(12,12),bins=20)
plt.show()

numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns

# Set up the figure
plt.figure(figsize=(20, 30))

# Plot boxplots for each numerical feature including the target variable 'OUTCOME'
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols), 1, i)
    sns.boxplot(x=dataset[col], color='skyblue')
    plt.title(f'Boxplot of {col}', fontsize=12)
    plt.tight_layout()

plt.show()

"""# **Number Unique values in each feature**

# **Numerical**
"""

numerical_data.nunique()

"""# **Categorical**"""

categorical_data.nunique()

"""# **Missing Values**"""

dataset.isnull().sum()

"""# **Barplot of unique value counts in every categorical features**"""

for col in categorical_features:
    plt.title(f'Distribution of {col}')
    categorical_data[col].value_counts().sort_index().plot(kind='bar', rot=0, xlabel=col,ylabel='count')
    plt.show()

"""# **Correlation Analysis**

**Correlation matrix of whole dataset**
"""

# Calculate the correlation matrix
correlation_matrix = numerical_data.corr()
correlation_matrix

"""# **Correlation Heatmap plot of whole dataset**"""

# Calculate the correlation matrix for the encoded dataset
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.3)
plt.show()

"""# **Correlation plot between numerical features and target**"""

fig, ax = plt.subplots(3,1, figsize=(10, 10))

# Combine numerical data and the target variable for correlation calculation
data_for_correlation = pd.concat([numerical_data, dataset['class']], axis=1)

# Encode the target variable 'class' for correlation calculation
data_for_correlation['class'] = LabelEncoder().fit_transform(data_for_correlation['class'])

## Correlation coefficient using different methods
corr1 = data_for_correlation.corr('pearson')[['class']].sort_values(by='class', ascending=False)
corr2 = data_for_correlation.corr('spearman')[['class']].sort_values(by='class', ascending=False)
corr3 = data_for_correlation.corr('kendall')[['class']].sort_values(by='class', ascending=False)

#setting titles for each plot
ax[0].set_title('Pearson method')
ax[1].set_title('spearman method')
ax[2].set_title('Kendall method')

## Generating heatmaps of each methods
sns.heatmap(corr1, ax=ax[0], annot=True)
sns.heatmap(corr2, ax=ax[1], annot=True)
sns.heatmap(corr3, ax=ax[2], annot=True)

plt.tight_layout()
plt.show()

"""# **Check Imbalance in data**"""

#group instances based on the classes in class variable
class_counts=dataset.groupby("class").size()

columns=['outcome','count','percentage']
outcome=class_counts.index.tolist()
count=class_counts.values.tolist()
percentage=list()

#Calculate the percentage of each value of the class variable from total
total_data_points = dataset.shape[0]
for val_count in count:
    percent=(val_count/total_data_points)*100
    percentage.append(percent)

# Convert the calulated values into a dataframe
imbalance_df=pd.DataFrame(list(zip(outcome,count,percentage)),columns=columns)
imbalance_df

"""# **Bar chart of dataset imbalance**"""

class_counts = dataset['class'].value_counts()

print("Class distribution:")
print(class_counts)

# Plot the bar chart
plt.figure(figsize=(8, 5))
class_counts.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title("Distribution of Classes in Dataset", fontsize=14)
plt.xlabel("Class", fontsize=12)
plt.ylabel("Number of Instances", fontsize=12)
plt.xticks(rotation=0)
plt.show()

"""# **Density plots of numerical features**"""

numerical_data.plot(kind='density',figsize=(14,14),subplots=True,layout=(6,2),title="Density plot of Numerical features",sharex=False)
plt.show()

"""# **Pre Processing**

**Drop Columns with Null values**
"""

dataset = dataset.drop(['stem-root', 'veil-type', 'veil-color', 'spore-print-color', 'stem-surface'], axis = 1)

dataset.shape

dataset.isnull().sum()

"""# **Imputing Null Columns**"""

# Select categorical columns
cat_cols = ['cap-surface', 'gill-attachment', 'gill-spacing', 'ring-type']

# Apply most frequent imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
dataset[cat_cols] = cat_imputer.fit_transform(dataset[cat_cols])
dataset.isnull().sum()

"""# **One-Hot Encoding**"""

dataset_encoded = dataset.copy()

# Apply One-Hot Encoding on all categorical columns
dataset_encoded = pd.get_dummies(dataset_encoded, drop_first=False)

# Show first 5 rows
dataset_encoded.head()

"""# **Train-Test Split + Scaling**"""

# Assuming 'class_p' is the target variable column after one-hot encoding
X = dataset_encoded.drop(['class_e', 'class_p'], axis=1)
y = dataset_encoded['class_p'] # Select one of the one-hot encoded class columns as target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""# **Model Training**

## **Logistic Regression**
"""

# Instantiate the Logistic Regression model
logistic_model = LogisticRegression(random_state=42)

# Train the model using the scaled training data
logistic_model.fit(X_train_scaled, y_train)
y_pred_log = logistic_model.predict(X_test_scaled)

print("Logistic Regression model trained successfully!")

"""## **K-Nearest Neighbors (KNN)**"""

# Instantiate the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model using the scaled training data
knn_model.fit(X_train_scaled, y_train)
y_pred = knn_model.predict(X_test_scaled)
y_pred_prob = knn_model.predict_proba(X_test_scaled)[:, 1]
print("KNN model trained successfully!")

"""## **Neural Network Model**"""

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the neural network model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=64, validation_split=0.2)

"""# **Model Evaluation**

## **Prediction Accuracy**
"""

# Predict on the test set for each model
y_pred_log = logistic_model.predict(X_test_scaled)
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_nn = (model.predict(X_test_scaled) > 0.5).astype("int32") # Get predictions for NN

# Calculate accuracy for each model
accuracy_log = accuracy_score(y_test, y_pred_log)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

# Create a DataFrame to store accuracy results
accuracy_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Neural Network'],
    'Accuracy': [accuracy_log, accuracy_knn, accuracy_nn]
})

# Sort by accuracy
accuracy_df = accuracy_df.sort_values(by='Accuracy', ascending=False)

print("Prediction Accuracy of each model:")
display(accuracy_df)

# Plotting the accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=accuracy_df, palette='viridis')
plt.title('Model Prediction Accuracy Comparison', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1) # Set y-axis limit from 0 to 1 for accuracy
plt.show()

"""## **Precision, Recall, and F1-Score Comparison**"""

# Calculate Precision, Recall, and F1-Score for each model
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)

# Create a DataFrame to store the results
metrics_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Neural Network'],
    'Precision': [precision_log, precision_knn, precision_nn],
    'Recall': [recall_log, recall_knn, recall_nn],
    'F1-Score': [f1_log, f1_knn, f1_nn]
})

print("Precision, Recall, and F1-Score of each model:")
display(metrics_df)

# Plotting the comparison
metrics_df_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 7))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df_melted, palette='viridis')
plt.title('Model Performance Metrics Comparison (Precision, Recall, F1-Score)', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)
plt.show()

"""## **Confusion Matrices**"""

# Calculate Confusion Matrix for each model
cm_log = confusion_matrix(y_test, y_pred_log)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_nn = confusion_matrix(y_test, y_pred_nn)

# Plotting Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('KNN Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', ax=axes[2])
axes[2].set_title('Neural Network Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.show()

"""## **AUC Score and ROC Curve**"""

# Get prediction probabilities for each model
y_pred_prob_log = logistic_model.predict_proba(X_test_scaled)[:, 1]
y_pred_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]
y_pred_prob_nn = model.predict(X_test_scaled).ravel() # Get probabilities for NN and flatten

# Calculate ROC curve and AUC for each model
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_prob_log)
roc_auc_log = auc(fpr_log, tpr_log)

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_prob_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)

# Plotting ROC Curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_log, tpr_log, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
plt.plot(fpr_knn, tpr_knn, color='green', lw=2, label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_nn, tpr_nn, color='red', lw=2, label=f'Neural Network (AUC = {roc_auc_nn:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Print AUC scores
print("\nAUC Scores:")
print(f"Logistic Regression: {roc_auc_log:.2f}")
print(f"KNN: {roc_auc_knn:.2f}")
print(f"Neural Network: {roc_auc_nn:.2f}")

"""# **K-Means (Unsupervised Learning)**

## Elbow Method to Determine Optimal Clusters for K-Means
"""

# Calculate WCSS for a range of cluster numbers
wcss = []
range_n_clusters = range(1, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X) # Use the feature data (X)
    wcss.append(kmeans.inertia_) # inertia_ is the WCSS

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range_n_clusters)
plt.grid(True)
plt.show()

"""# **K-Means Clustering**"""

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # n_init is set to 10 explicitly
clusters = kmeans.fit_predict(X) # Use the feature data (X)

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', marker='o', s=50)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
            pca.transform(kmeans.cluster_centers_)[:, 1],
            c= "red", marker='X', s=200, label = "centroid")
plt.title('K-Means Clustering Results (PCA Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()
