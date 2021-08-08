#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader as pdr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB 
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


# In[2]:


start_date = "2010-11-01"
end_date =  "2020-11-01"

company_name = "AMZN" #Amazon

#Getting Stock market data from start_date to end_date of "Amazon"
df = pdr.get_data_yahoo(company_name, start_date,end_date )


# In[3]:


df.describe()


# In[4]:


df.head()


# In[5]:


plt.figure(figsize=(20,7))
plt.plot(df["Close"],label="Amazon Stock")
plt.title("Closing Price vs Date")
plt.xlabel("Date")
plt.ylabel("Close Price in USD")
plt.legend()
plt.savefig("Closing Price vs Date.png",dpi=300)
plt.show()


# In[6]:


# Calculating the difference in closing prices
df["Diff"] = df.Close.diff()

# calculating moving average of closing price over 2 days to smoothen the curve
df["SMA_2"] = df.Close.rolling(2).mean()

# calculating index
df["Force_Index"] = df["Close"] * df["Volume"]

# assigning lable y = 1 if stock price has increased and 0 otherwise
df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else -1).shift(-1)

#removing redundant columns and cleaning data
df = df.dropna()


# In[7]:


# converting dataframe to numpy array
data = df.to_numpy()
# normalizing data
data = normalize(data)


# In[8]:


#saperating normalized features
X = data[:,:-1]

# assinig lable values 
y = df["y"].values

#splitting in test and training set
X_train, X_test, y_train, y_test = train_test_split(
   X,
   y,
   test_size=0.064,
   shuffle=False,
)


# In[9]:


#creating classifier
clf = BernoulliNB()

#training 
clf.fit(X_train,y_train)

#predicting based on X_test
y_pred = clf.predict(X_test)

#Calculating accuracy
print("Accuracy by using Bernoulli Naive Bayes classifier : " , accuracy_score(y_test, y_pred)*100 ,"%")


# In[10]:


#creating classifier
clf = GaussianNB()

#training 
clf.fit(X_train,y_train)

#predicting based on X_test
y_pred = clf.predict(X_test)

#Calculating accuracy
print("Accuracy by using Gaussian Naive Bayes classifier : " , accuracy_score(y_test, y_pred)*100 ,"%")


# In[11]:


#creating classifier
clf = CategoricalNB()

#training 
clf.fit(X_train,y_train)

#predicting based on X_test
y_pred = clf.predict(X_test)

#Calculating accuracy
print("Accuracy by using Gaussian Naive Bayes classifier : " , accuracy_score(y_test, y_pred)*100 ,"%")


# In[ ]:





# In[ ]:




