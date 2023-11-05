#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdatalabel.xlsx")
df


# In[2]:


import matplotlib.pyplot as plt


feature1 = df['embed_0']
feature2 = df['embed_1']

plt.scatter(feature1, feature2)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Scatter Plot of Feature1 vs Feature2')
plt.show()

# In[26]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the data

# Create a DataFrame


# Choose Feature1 as the independent variable and Feature2 as the dependent variable
X = df['embed_0'].values.reshape(-1, 1)
y = df['embed_1']

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions using the model
y_pred = model.predict(X)

# Calculate the mean squared error
mse = mean_squared_error(y, y_pred)

# Print the mean squared error
print("Mean Squared Error:", mse)

# Plot the data and the regression line
plt.scatter(X, y )
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel('embed_0')
plt.ylabel('embed_1')
plt.show()

# In[47]:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data into a DataFrame


# Select two classes for binary classification (ClassA and ClassB)
binary_data = df[df['Label'].isin([0, 1])]

# Define features and target variable
X = binary_data[['embed_0', 'embed_1']]
y = (binary_data['Label'] == 0).astype(int)  # 1 for ClassA, 0 for ClassB

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy}")


# In[62]:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# Choose two classes for binary classification (ClassA and ClassB)
binary_df = df[(df['Label'] == 0) | (df['Label'] == 1)]

# Define features and target variable
X = binary_df[['embed_0', 'embed_1']]
y = (binary_df['Label'] == 0).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a logistic regression classifier
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_regression.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# In[75]:

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create a DataFrame from the given data

# Split the data into training and testing sets
X = df[['embed_0', 'embed_1']]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
tree_regressor = DecisionTreeRegressor()
tree_regressor.fit(X_train, y_train)

# Predict values using the Decision Tree Regressor
tree_predictions = tree_regressor.predict(X_test)

# Train a k-NN Regressor
knn_regressor = KNeighborsRegressor(n_neighbors=3)  # You can adjust the number of neighbors (k) as needed.
knn_regressor.fit(X_train, y_train)

# Predict values using the k-NN Regressor
knn_predictions = knn_regressor.predict(X_test)

# Evaluate the models
tree_mse = mean_squared_error(y_test, tree_predictions)
knn_mse = mean_squared_error(y_test, knn_predictions)

print("Decision Tree MSE:", tree_mse)
print("k-NN Regressor MSE:", knn_mse)

