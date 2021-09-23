#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()


# In[3]:


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# In[4]:


plt.figure(figsize=(15,2))
plt.imshow(x_train[0])


# In[5]:


print('Number of unique classes: ', len(np.unique(y_train)))
print('Classes: ', np.unique(y_train))


# In[ ]:





# In[6]:


x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)
x_train_flattened.shape


# In[7]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
X_train = minmax.fit_transform(x_train_flattened)
X_test = minmax.transform(x_test_flattened)


# In[8]:


X_train.shape


# In[9]:


Ann_model=keras.Sequential([
    keras.layers.Dense(360,input_dim = X_train.shape[1],activation="relu"),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(170,activation="relu"),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(100,activation="relu"),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(50,activation="relu"),
    keras.layers.Dense(10,activation="sigmoid")
])
Ann_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
Ann_model.fit(X_train,y_train,epochs=12)


# In[10]:


Ann_model.evaluate(X_test,y_test)


# In[11]:


X_train=X_train.reshape(60000,28,28,1)
X_test=X_test.reshape(10000,28,28,1)


# In[12]:


Cnn_model=keras.Sequential([
    #CNN
    keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    #ANN
    keras.layers.Flatten(),
    keras.layers.Dense(100,activation='relu'),
    
    keras.layers.Dense(10,activation='softmax')
])
Cnn_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
Cnn_model.fit(X_train,y_train,epochs=10)


# In[13]:


Cnn_model.evaluate(X_test,y_test)


# In[ ]:




