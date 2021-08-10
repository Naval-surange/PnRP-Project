#!/usr/bin/env python
# coding: utf-8

# ## Importing improtant libraries

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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ## Pulling Data of stock

# In[2]:


start_date = "2010-11-01"
end_date =  "2020-11-01"

company_name = "AMZN" #Amazon

#Getting Stock market data from start_date to end_date of "Amazon"
df = pdr.get_data_yahoo(company_name, start_date,end_date )


# ## analysing stock data

# In[3]:


df.describe()


# In[4]:


df.head()


# ## Visualising Trends of the stocks 

# In[5]:


df['Date'] = df.index

fig1 = plt.figure(figsize=(20,7))
ax1 = fig1.add_subplot(111)
ax1.plot(df["Date"],df["Open"],label="Open Price")
ax1.plot(df["Date"],df["Close"],label="Close Price")
ax1.legend(loc=4)


axins1 = zoomed_inset_axes(ax1, zoom = 6, loc=2)
axins1.plot(df["Date"],df["Open"])
axins1.plot(df["Date"],df["Close"])
x1, x2, y1, y2 = datetime.strptime("2018-01-01", "%Y-%m-%d"), datetime.strptime("2018-07-01", "%Y-%m-%d"),1400,1800
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)

axins1.set_xticks([])
axins1.set_yticks([])

mark_inset(ax1, axins1, loc1=1, loc2=4, fc="none", ec="0.5")


ax1.set_xlabel("Date",fontsize=30)
ax1.set_ylabel("Price in USD",fontsize=30)


plt.savefig("fig1.png",dpi=500)
plt.show()


# In[6]:


fig1 = plt.figure(figsize=(20,7))
ax1 = fig1.add_subplot(111)
ax1.plot(df["Date"],df["High"],label="Highest Price")
ax1.plot(df["Date"],df["Low"],label="Lowest Price")
ax1.legend(loc=4)


axins1 = zoomed_inset_axes(ax1, zoom = 6, loc=2)
axins1.plot(df["Date"],df["High"])
axins1.plot(df["Date"],df["Low"])
x1, x2, y1, y2 = datetime.strptime("2018-01-01", "%Y-%m-%d"), datetime.strptime("2018-07-01", "%Y-%m-%d"),1300,1800
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)

axins1.set_xticks([])
axins1.set_yticks([])

mark_inset(ax1, axins1, loc1=1, loc2=4, fc="none", ec="0.5")


ax1.set_xlabel("Date",fontsize=30)
ax1.set_ylabel("Price in USD",fontsize=30)

plt.savefig("fig2.png",dpi=500)
plt.show()


# In[7]:


plt.figure(figsize=(20,7))
plt.plot(df["Volume"],label="Volume of Amazon Stock",linewidth=1)
yhat = savgol_filter(df["Volume"], 25, 3)
plt.plot(df["Date"],yhat,linewidth=4,label="Smoothened Curve")
plt.xlabel("Date",fontsize=20)
plt.ylabel("Volume of stock in flow",fontsize=20)
plt.legend()
plt.savefig("fig3.png",dpi=500)
plt.show()


# ## Prepossessing of Data

# In[8]:


# Calculating the difference in closing prices
df["Diff"] = df.Close.diff()

# calculating moving average of closing price over 2 days to smoothen the curve
df["SMA_2"] = df.Close.rolling(2).mean()

# calculating index
df["Force_Index"] = df["Close"] * df["Volume"]

# assigning lable y = 1 if stock price has increased and 0 if decreased
df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else -1).shift(-1)

#removing redundant columns and cleaning data
df = df.drop("Date",axis=1)
df = df.dropna()


# ## Normalizing Data

# In[9]:


# converting dataframe to numpy array
data = df.to_numpy()
# normalizing data
data = normalize(data)


# ## Training Different models and finding there accuracies

# In[10]:


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


# In[11]:


#creating classifier
clf = BernoulliNB()

#training 
clf.fit(X_train,y_train)

#predicting based on X_test
y_pred = clf.predict(X_test)

#Calculating accuracy
print("Accuracy by using Bernoulli Naive Bayes classifier : " , accuracy_score(y_test, y_pred)*100 ,"%")


# In[12]:


#creating classifier
clf = GaussianNB()

#training 
clf.fit(X_train,y_train)

#predicting based on X_test
y_pred = clf.predict(X_test)

#Calculating accuracy
print("Accuracy by using Gaussian Naive Bayes classifier : " , accuracy_score(y_test, y_pred)*100 ,"%")


# In[13]:


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




