#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy as np


# In[2]:


dataset = pandas.read_csv("iris.csv")
dataset = dataset.sample(frac=1)


# In[3]:


#Getting the input Data.
X = dataset[["sepal_length","sepal_width","petal_length","petal_width"]]
print(X.head())
X = np.array(X)
print(f"\nshape of X is {X.shape}")


# In[4]:


#Getting the Classes. 
Y = dataset["species"]
print(Y.head())
Y = np.array(Y).reshape((150,1))
print(f"\nShape of Y is {Y.shape}")


# In[5]:


#Pre-Processing the output Classes.
Y[Y=="Iris-setosa"] = 0
Y[Y=="Iris-versicolor"] = 1
Y[Y=="Iris-virginica"] = 2


# In[6]:


print(f"Shape of Y after Pre-Processing is {Y.shape}\n")
print(np.squeeze(Y))


# In[7]:


from sklearn import preprocessing 


# In[8]:


#One Hot Encoding Y.
one_hot = preprocessing.OneHotEncoder()
O_e =  one_hot.fit(Y)
Y_e = O_e.transform(Y)
Y = Y_e.toarray()
print(f"Shape of Y is {Y.shape}")


# In[9]:


import tensorflow as tf

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout


# In[10]:


model = Sequential()
model.add(Dense(10,input_shape=(4,),activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3))


# In[11]:


predictions = model(X).numpy()


# In[12]:


#Initial Predictions 
Y_beta = tf.nn.softmax(predictions).numpy()
print(Y_beta[:5])
print(np.sum(Y_beta[:5]))
Y_beta = np.argmax(Y_beta,axis = 1)
print(Y_beta[:5])


# In[13]:


lossSoft = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


# In[14]:


print(lossSoft(Y[:5],predictions[:5]).numpy())


# In[15]:


X_train =  X[:-50]
Y_train = Y[:-50]
X_test = X[100:-1]
Y_test = Y[100:-1]
model.compile(optimizer = "adam",loss=lossSoft,metrics=["accuracy"])


# In[21]:


model.fit(X,Y,epochs=10)


# In[22]:


probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])


# In[23]:


print(np.argmax(probability_model(X[:100]).numpy(),axis=1))
print(Y[:100])


# In[ ]:




