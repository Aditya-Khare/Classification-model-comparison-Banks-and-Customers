#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The data set I used contains details of a bank's customers.

# The target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) 
# or s/he continues to be a customer.

# In order to predict target variables, I will use sklearn classification algorithms:

#Logistic Regression Classification
# K-Nearest Neighbour (KNN) Classification
# Support Vector Machine (SVM) Classification
# Naive Bayes Classification
# Decision Tree Classification
# Random Forest Classification
# After implementing each model, I used score and confusion matrix methods to compare models


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
import os


# In[3]:


data = pd.read_csv('Churn_Modelling.csv')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


# we don't need some of the columns for our analysis, like customer ID, surname, etc.
data.drop(['RowNumber', 'CustomerId', 'Surname', 'Geography'], axis=1, inplace=True)

# we will change customers' gender from Male/Female to 1/0
data.Gender = [1 if each == 'Male' else 0 for each in data.Gender]

data.sample(5)


# In[7]:


# Question arises : How many of the customers left the bank?
plt.figure(figsize=[5,5])
sns.set(style='darkgrid')
ax = sns.countplot(x='Exited', data=data, palette='Set3')
data.loc[:,'Exited'].value_counts()


# In[8]:


# 2. SEPARETING FEATURES AND TARGET
# Now separete target feature (y) from other features (x_data).

y = data.Exited.values
x_data = data.drop(['Exited'], axis=1)

x_data.describe()


# In[9]:


# I can see the x_data features has a large scale of numbers. So I need to normalize all the features between 0 and 1.


# In[10]:


# 3. NORMALIZATION PROCESS

x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))
x.head()


# In[11]:


# 4. SPLITTING DATA FOR TRAINING AND TESTING
# I am going to split my data set into as train (x_train, y_train) and test (x_test, y_test) datas.
# Then I am going to teach my machine learning algorithms by using trainig data set.
# Later I will use my trained model to predict my test data (y_pred).
# Finally I will compare my predictions (y_pred) with my test data (y_test).

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=7)

print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)


# In[12]:


# 5. LOGISTIC REGRESSION CLASSIFICATION:


# In[13]:


from sklearn.linear_model import LogisticRegression

# Defining the model
lr = LogisticRegression()

# Training the model:
lr.fit(x_train, y_train)

# Predicting target values by using x_test and our model:
y_pred0 = lr.predict(x_test)


# In[14]:


# Confusion matrix for visulalization of our prediction accuracy:
from sklearn.metrics import confusion_matrix

# Creating the confusion matrix:
lr_cm = confusion_matrix(y_test, y_pred0)

#Visualization:
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(lr_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax, cmap='BrBG')
plt.title('Logistic Regression Classification Confusion Matrix')
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.show()


# In[15]:


score_lr = lr.score(x_test, y_test)
print(score_lr)


# In[16]:


# 6. KNN CLASSIFICATION ALGORITHM:


# In[17]:


from sklearn.neighbors import KNeighborsClassifier

# Defining the model with a k number of 13:
knn = KNeighborsClassifier(n_neighbors=13)

# Training the model:
knn.fit(x_train, y_train)

# Predicting target values by using x_test and our model:
y_pred1 = knn.predict(x_test)


# In[18]:


# Confusion matrix for visualization our prediction accuracy:
from sklearn.metrics import confusion_matrix

# Creating the confusion matrix:
knn_cm = confusion_matrix(y_test, y_pred1)

# Visualization:
f, ax = plt.subplots(figsize=(4,4))
sns.heatmap(knn_cm, annot=True, linewidth=1, linecolor='cyan', fmt='.0f', ax=ax, cmap='BrBG')
plt.title('KNN Classification Confusion Matrix')
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.show()


# In[19]:


score_knn = knn.score(x_test, y_test)
print(score_knn)


# In[20]:


# 7. SUPPORT VECTOR MACHINE (SVM) ALGORITHM:
# I will use the same x_train and y_train data sets to teach a SVM model.


# In[21]:


from sklearn.svm import SVC

# Defining SVM model
svm = SVC(random_state=2)

# Training model:
svm.fit(x_train, y_train)

# Predicting target values by using x_test and our model:
y_pred2 = svm.predict(x_test)


# In[22]:


# Confusion matrix for visualization our prediction accuracy:
from sklearn.metrics import confusion_matrix

# Creating the confusion matrix:
svm_cm = confusion_matrix(y_test, y_pred2)

# Visualization:
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(svm_cm, annot=True, linewidth=1.0, linecolor='cyan', fmt='.0f', ax=ax, cmap='BrBG')
plt.title('SVM Classification Confusion Matrix')
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.show()


# In[23]:


score_svm = svm.score(x_test, y_test)
print(score_svm)


# In[24]:


# 8. NAIVE BAYES ALGORITHM:


# In[25]:


from sklearn.naive_bayes import GaussianNB

# Defining model:
nb = GaussianNB()

# Training the model:
nb.fit(x_train, y_train)

# Predicting:
y_pred3 = nb.predict(x_test)


# In[26]:


# Confusion matrix for visualization our prediction accuracy:
from sklearn.metrics import confusion_matrix

# Creating the confusion matrix:
nb_cm = confusion_matrix(y_test, y_pred3)

# Visualization:
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(nb_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax, cmap='BrBG')
plt.title('Naive Bayes Classification Confusion Matrix')
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.show()


# In[27]:


score_nb = nb.score(x_test, y_test)
print(score_nb)


# In[28]:


# 9. DECISION TREE ALGORITHM:


# In[29]:


from sklearn.tree import DecisionTreeClassifier

# Defining the model:
dt = DecisionTreeClassifier()

# Training:
dt.fit(x_train, y_train)

# Predicting:
y_pred4 = dt.predict(x_test)


# In[30]:


# Confusion matrix for visualization our prediction accuracy:
from sklearn.metrics import confusion_matrix

# Creating the confusion matrix:
dt_cm = confusion_matrix(y_test, y_pred4)

# Visualization:
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(dt_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax, cmap='BrBG')
plt.title('Decision Tree Classification Confusion Matrix')
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.show()


# In[31]:


score_dt = dt.score(x_test, y_test)
print(score_dt)


# In[32]:


# 10. RANDOM FOREST CLASSIFICATION ALGORITHM:


# In[33]:


from sklearn.ensemble import RandomForestClassifier

# Defining:
rf = RandomForestClassifier(n_estimators=100, random_state=3)

# Training:
rf.fit(x_train, y_train)

# Predicting:
y_pred5 = rf.predict(x_test)


# In[34]:


# Confusion matrix for visualization our prediction accuracy:
from sklearn.metrics import confusion_matrix

# Creating the confusion matrix:
rf_cm = confusion_matrix(y_test, y_pred5)

# Visualization:
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(rf_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax, cmap='BrBG')
plt.title('Random Forest Classification Confusion Matrix')
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.show()


# In[35]:


score_rf = rf.score(x_test, y_test)
print(score_rf)


# In[36]:


# 11. COMPARISON OF ALGORITHMS:
# Now it's time to compare our models.


# In[37]:


# First of all compare the accuracies:

data_scores = pd.Series([score_lr, score_knn, score_svm, score_nb, score_dt, score_rf], 
                        index=['logistic_regression_score', 'knn_score', 'svm_score', 'naive_bayes_score',
                               'decision_tree_score', 'random_forest_score']) 
data_scores


# In[38]:


# From the accuracy comparison I can see random forest classification gave the best result.

# Now I want to see y_test and my models' y_pred values manually:

d = {'y_test': y_test, 'log_reg_pred': y_pred0,'knn_prediction': y_pred1, 
     'svm_prediction': y_pred2, 'naive_bayes_prediction': y_pred3, 
     'decision_tree_prediction': y_pred4, 'random_forest_prediction': y_pred5}
data01 = pd.DataFrame(data=d)
data01.T


# In[39]:


# Finally I want to show my models' confusion matrixes side by side:

fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3, 3, 1) # row, column, position
ax1.set_title('Logistic Regression Classification')

ax2 = fig.add_subplot(3, 3, 2) # row, column, position
ax2.set_title('KNN Classification')

ax3 = fig.add_subplot(3, 3, 3)
ax3.set_title('SVM Classification')

ax4 = fig.add_subplot(3, 3, 4)
ax4.set_title('Naive Bayes Classification')

ax5 = fig.add_subplot(3, 3, 5)
ax5.set_title('Decision Tree Classification')

ax6 = fig.add_subplot(3, 3, 6)
ax6.set_title('Random Forest Classification')

sns.heatmap(data=lr_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax1, cmap='BrBG')
sns.heatmap(data=knn_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax2, cmap='BrBG')   
sns.heatmap(data=svm_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax3, cmap='BrBG')
sns.heatmap(data=nb_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax4, cmap='BrBG')
sns.heatmap(data=dt_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax5, cmap='BrBG')
sns.heatmap(data=rf_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax6, cmap='BrBG')
plt.show()


# In[ ]:




