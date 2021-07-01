#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install numpy pandas sklearn


# In[3]:


import numpy as np


# In[4]:


import pandas as pd


# In[5]:


import itertools


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[9]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[15]:


#Read the data
df=pd.read_csv('C:\\Users\\A.SEKAR\\Downloads\\news\\news.csv')


# In[16]:


#Get shape and head
df.shape
df.head()


# In[17]:


#DataFlair - Get the labels
labels=df.label
labels.head()


# In[18]:


#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[19]:


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# In[20]:


#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[21]:


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# In[22]:


#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[23]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:




