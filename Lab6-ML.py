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


# In[41]:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define the data


# Label encoding for 'Class' column
df['Label'] = df['Label'].astype('category')
df['Label'] = df['Label'].cat.codes

# Split the data into training and test sets
X = df[['embed_0', 'embed_1']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model and fit it to the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the classes on the test data and evaluate the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


