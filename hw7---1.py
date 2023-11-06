#!/usr/bin/env python
# coding: utf-8

# # Homework: Decision Tree
# ### Due Sun, November 5, 11:59 PM
# 
# This homework is based on the materials covered in week 6 and 7 about decision tree algorithms.
# 
# You will work with a dataset containing a bank's customer information. You will act as a Data scientist for the bank to build a model that will help the bank's marketing department to identify the potential customers who have a higher probability of purchasing a loan product.
# 
# The dataset contains the following attributes:
# - ID: Customer ID
# - Age: Customer’s age in completed years
# - Experience: #years of professional experience
# - Income: Annual income of the customer (in thousand dollars)
# - ZIP Code: Home Address ZIP code.
# - Family: the Family size of the customer
# - CCAvg: Average spending on credit cards per month (in thousand dollars)
# - Education: Education Level. 1: Undergrad; 2: Graduate;3: Advanced/Professional
# - Mortgage: Value of house mortgage if any. (in thousand dollars)
# - Personal_Loan: Did this customer accept the personal loan offered in the last campaign?
# - Securities_Account: Does the customer have securities account with the bank?
# - CD_Account: Does the customer have a certificate of deposit (CD) account with the bank?
# - Online: Do customers use internet banking facilities?
# - CreditCard: Does the customer use a credit card issued by any other Bank (excluding All life Bank)?
# 
# Please use Python as the coding language. You are free to use existing Python libraries. Please make sure your codes can run successfully, no points will be given for a question if the code fails to run.

# ### Preparation: load dataset
# Download the dataset 'bank_customer.csv' and store it in the same folder as this Jupyter notebook. Use the following codes to load the dataset.

# In[64]:


import numpy as np
import pandas as pd
df = pd.read_csv('bank_customer.csv')
df.info()


# To simplify the following analysis, we drop 'ID', 'ZIP Code', 'Family', 'Mortgage' columns from the dataframe.

# In[3]:


df = pd.read_csv('bank_customer.csv')
df = df.drop(['ID','ZIP Code','Family','Mortgage'],axis=1)
df.info()


# ### Q1. Understand the dataset (1 point)
# From the output from `df.info()`, we observe that this dataset contains records on 5000 customers and there is no missing data. The dataset contains a mix of numerical and categorical attributes, and all categorical data are represented with numbers. Note: you do not need to consider the 4 columns that have been dropped.
# 
# 1. Identify all categorical attributes, and use the proper command to report the number of unique values in each categorical column. (0.5 point)
# 2. Identify all numerical attributes, and use the proper command to report the range and quartile of each numerical column. (0.5 point)

# *Space reserved for writing explanation for Q1.1*
# 

# In[46]:


categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    unique_values = df[column].nunique()
    print(f"Column '{column}' has {unique_values} unique values.")


# *Space reserved for writing explanation for Q1.2*
# 

# In[47]:


# Enter your code for Q1.2 here
numerical_columns = df.select_dtypes(include=['int', 'float']).columns
for column in numerical_columns:
    column_range = df[column].max() - df[column].min()
    quartiles = df[column].quantile([0.25, 0.5, 0.75])
    print(f"Column '{column}':")
    print(f"Range: {column_range}")
    print(f"25th Percentile (Q1): {quartiles[0.25]}")
    print(f"Median (Q2): {quartiles[0.5]}")
    print(f"75th Percentile (Q3): {quartiles[0.75]}")



# ### Q2. Create separate arrays to store features and target label (2 point)
# In this dataset, the target label is indicated in the column 'Personal Loan': a '1' value means the customer accepted a loan in the previous campaign, and a '0' values means the customer did not accept. We use all other columns as features to predict the classification in 'Personal Loan'.
# 
# You need to create the following arrays.
# - X: stores all predictor variables
# - y: stores all target label
# 
# For all numerical attributes, you need to encode them as categorical attributes based on the quartile. For example, for the 'Age' attribute, you need to encode it as a new attribute 'Age_cat':
# - Age_cat = 0, if Age <= Q1, where Q1 is the first quartile (25 percent value among all ages)
# - Age_cat = 1, if Age > Q1 and Age <= Q2, where Q2 is the second quartile (50 percent value among all ages)
# - Age_cat = 2, if Age > Q2 and Age <= Q3, where Q3 is the third quartile (75 percent value among all ages)
# - Age_cat = 3, if Age > Q3
# 
# 1. Create y (0.5 point)
# 2. Create X and properly encode all numerical values (1.5 point) Note: after encoding, remember to drop the original numerical attributes. The shape of the generated X should be (5000,9).

# *Space reserved for writing explanation for Q2*
# 

# In[48]:


# Enter your code for Q2.1 here
y = df['Personal Loan']


# In[49]:


# Enter your code for Q2.2 here
numerical_columns = df.select_dtypes(include=['int', 'float']).columns
categorical_columns = df.select_dtypes(include=['object']).columns
X = df[categorical_columns]  # Start with the categorical columns


# Concatenate the one-hot encoded categorical columns with the target variable 'Personal Loan'
X = pd.concat([X, df[numerical_columns]], axis=1)


# ### Q3. Train and test a decision tree with 80/20 split (2 point)
# You will now train a decision tree on the dataset.
# 
# 1. Use the proper commands to split (X,y) into training set (80% of all data), and testing set (20% of all data). Use random_state=0 when creating the splits. (0.5 point)
# 2. Use the proper commands to fit a decision tree model on the training set with 'Gini impurity' as splitting criterion, and 'random_state=123' in `scikit-learn`. (1 point)
# 3. Use the proper commands to report the training and testing accuracy of the decision tree. (0.5 point)

# *Space reserved for writing explanation for Q3*
# 

# In[50]:


# Enter your code for Q3.1 here
from sklearn.model_selection import train_test_split

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[51]:


# Enter your code for Q3.2 here
from sklearn.tree import DecisionTreeClassifier

# Create and train a decision tree classifier
decision_tree = DecisionTreeClassifier(criterion='gini', random_state=123)
decision_tree.fit(X_train, y_train)


# In[52]:


# Enter your code for Q3.3 here
# Calculate training accuracy
training_accuracy = decision_tree.score(X_train, y_train)

# Calculate testing accuracy
testing_accuracy = decision_tree.score(X_test, y_test)

print(f"Training accuracy: {training_accuracy:.2f}")
print(f"Testing accuracy: {testing_accuracy:.2f}")


# ### Q4. Train and test a smaller decision tree (1 point)
# In the previous question, we did not impose any restriction on the size of the decision tree. For interpretability, it is often desirable to have a smaller decision tree. You will next train a smaller decision tree, and observe its performance. The same training and testing sets will be used.
# 
# 1. Use the proper commands to fit a decision tree with a maximum depth of 3 on the training set with 'Gini impurity' as splitting criterion, 'random_state=123'. (0.5 point)
# 2. Use the proper commands to report the training and testing accuracy of the new decision tree. (0.5 point)

# *Space reserved for writing explanation for Q4*
# 

# In[53]:


# Enter your code for Q4.1 here
from sklearn.tree import DecisionTreeClassifier

# Create and train a smaller decision tree with a maximum depth of 3
small_decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=123)
small_decision_tree.fit(X_train, y_train)


# In[54]:


# Enter your code for Q4.2 here
# Calculate training accuracy for the smaller decision tree
small_training_accuracy = small_decision_tree.score(X_train, y_train)

# Calculate testing accuracy for the smaller decision tree
small_testing_accuracy = small_decision_tree.score(X_test, y_test)

print(f"Training accuracy (small tree): {small_training_accuracy:.2f}")
print(f"Testing accuracy (small tree): {small_testing_accuracy:.2f}")


# ### Q5. Use cross validation to tune a smaller decision tree (2 point)
# As we expect, restricting the size of the decision tree leads to a drop in accuracy. You will next use 5-fold cross validation on the training dataset to find the criterion and max_depth setup, which leads to the best validation accuracy, from the following range:
# - criterion is either Gini impurity or Information gain
# - max_depth is 2, 3 or 4
# 
# 1. Use GridSearchCV function to perform the cross validation and report the best parameter setup. (1 point)
# 2. Train a new decision tree with the found best parameters on the same training data, then report training and testing accuracy of the new decision tree. (1 point)

# *Space reserved for writing explanation for Q5*
# 

# In[55]:


# Enter your code for Q5.1 here
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define the parameter grid to search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4]
}

# Create a decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=123)

# Create the GridSearchCV object
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Report the best parameter setup
best_params = grid_search.best_params_
print("Best parameter setup:", best_params)


# In[56]:


# Enter your code for Q5.2 here
# Train a new decision tree with the best parameters on the training data
best_decision_tree = DecisionTreeClassifier(criterion=best_params['criterion'], max_depth=best_params['max_depth'], random_state=123)
best_decision_tree.fit(X_train, y_train)

# Calculate training accuracy for the best decision tree
best_training_accuracy = best_decision_tree.score(X_train, y_train)

# Calculate testing accuracy for the best decision tree
best_testing_accuracy = best_decision_tree.score(X_test, y_test)

print(f"Training accuracy (best decision tree): {best_training_accuracy:.2f}")
print(f"Testing accuracy (best decision tree): {best_testing_accuracy:.2f}")


# ### Q6. Visualize a decision tree, and make observations about decision rules (2 point)
# You will now visualize the decision tree created in Q5. You can use either the `plot_tree` function or the `export_graphviz` function. For both functions, to increase readability, use the following code to set feature and class label names in the generated figure:
# - `feature_names=X.columns, class_names=['Not Accept','Accept']`
# 
# 1. Visualize the decision tree trained in Q5 (1 point)
# 2. Observe one decision rule from the generated tree (you can choose any rule based on your own tree), and report the support and confidence of the rule. (1 point)

# *Space reserved for writing explanation for Q6.1*
# 

# In[65]:


# Enter your code for Q6.1
from sklearn.tree import export_graphviz
import graphviz

# Set the feature and class label names
feature_names = X.columns
class_names = ['Not Accept', 'Accept']

# Visualize the decision tree
dot_data = export_graphviz(
    best_decision_tree,
    out_file=None,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    special_characters=True
)


# *Space reserved for writing explanation for Q6.2*
# 
# 

# In[66]:


graph = graphviz.Source(dot_data)

graph.render("decision_tree")

graph.view("decision_tree")


# In[ ]:




