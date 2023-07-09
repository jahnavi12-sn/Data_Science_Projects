#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


print(stopwords.words('english'))


# In[4]:


news_dataset = pd.read_csv('train.csv')


# In[5]:


news_dataset.shape


# In[6]:


news_dataset.head()


# In[7]:


news_dataset.isnull().sum()


# In[8]:


news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author']+' '+ news_dataset['title']
print(news_dataset['content'])


# In[9]:


port_stem = PorterStemmer()
def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split() #for converting in list
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content


# In[10]:


news_dataset['content']=news_dataset['content'].apply(stemming)


# In[11]:


print(news_dataset['content'])


# In[12]:


X=news_dataset['content'].values
Y=news_dataset['label'].values
print(X)
print(Y)


# In[13]:


vectorizer=TfidfVectorizer() #Tf- repeated numbers given a specific number, idf- (inverse frequency)if repeated too many times reduces its importance value
vectorizer.fit(X) #to create vocab from train data
X=vectorizer.transform(X) #copy transformed data to train dataset


# In[14]:


print(X)


# In[15]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[16]:


model=LogisticRegression()
model.fit(X_train,Y_train)


# In[17]:


X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy score of training data: ',training_data_accuracy)


# In[18]:


X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy score of testing data: ',testing_data_accuracy)


# In[19]:


prediction=model.predict(X_test[0])
print(prediction)
if (prediction==0):
    print('The news is real')
else:
    print('The news is fake')


# In[20]:


print(Y_test[0])

