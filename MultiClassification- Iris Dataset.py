#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt


# In[2]:


#import Iris dataset 
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target


# In[3]:


#turn categorical targets to one-hot encode 
df=pd.get_dummies(df,columns=['target'])


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


X=df.iloc[:,:4]
y=df.iloc[:,4:]
scale=StandardScaler()
X=scale.fit_transform(X)
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3, random_state=10)


# In[7]:


#One hidden layers seams sufficient for the current problem 
#findind the best number of nodes in the hidden layer by the means of a sugested equation 
def number_of_nodes(samples, inp,out, alpha):
    return (samples)/alpha*(inp+out)
#alpha can be a variable between 2 an 10. it is suggested that 2 will avoid over fitting 
#here the number of training samples is 100, number of inputs=4 and number of outputs=3
n=number_of_nodes(100,4,3,2)

    


# In[8]:


#define DL model
model=Sequential()
#the current model include three layers: input, one hidden layer, and output 
#number of nodes in the input layer is number of features plus one to consider the constant value 
#number of nodes in the output layer is equal to number of labels 
model.add(Dense(5, activation='relu', input_shape=(4,)))
model.add(Dense(n,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['accuracy','binary_accuracy','categorical_accuracy'])


# In[9]:


model.summary()


# In[10]:


model.output_shape


# In[11]:


model.get_config()


# In[12]:


model.get_weights()


# In[24]:


D=[]
for i in [2,4,5,10,20,25,50]:
    for j in [10,100,500,1000]:
        A=model.fit(X_train, y_train,validation_split=0.2,epochs=j, batch_size=i, verbose=1 )
        D.append(A)
        


# In[59]:


fig, axs = plt.subplots(7, 4,figsize=(20,10),sharey=True)
axs[0, 0].plot(D[0].history['val_categorical_accuracy'])
axs[0, 1].plot(D[1].history['val_categorical_accuracy'])
axs[0, 2].plot(D[2].history['val_categorical_accuracy'])
axs[0, 3].plot(D[3].history['val_categorical_accuracy'])
axs[1, 0].plot(D[4].history['val_categorical_accuracy'])
axs[1, 1].plot(D[5].history['val_categorical_accuracy'])
axs[1, 2].plot(D[6].history['val_categorical_accuracy'])
axs[1, 3].plot(D[7].history['val_categorical_accuracy'])
axs[2, 0].plot(D[8].history['val_categorical_accuracy'])
axs[2, 1].plot(D[9].history['val_categorical_accuracy'])
axs[2, 2].plot(D[10].history['val_categorical_accuracy'])
axs[2, 3].plot(D[11].history['val_categorical_accuracy'])
axs[3, 0].plot(D[12].history['val_categorical_accuracy'])
axs[3, 1].plot(D[13].history['val_categorical_accuracy'])
axs[3, 2].plot(D[14].history['val_categorical_accuracy'])
axs[3, 3].plot(D[15].history['val_categorical_accuracy'])
axs[4, 0].plot(D[16].history['val_categorical_accuracy'])
axs[4, 1].plot(D[17].history['val_categorical_accuracy'])
axs[4, 2].plot(D[18].history['val_categorical_accuracy'])
axs[4, 3].plot(D[19].history['val_categorical_accuracy'])
axs[5, 0].plot(D[20].history['val_categorical_accuracy'])
axs[5, 1].plot(D[21].history['val_categorical_accuracy'])
axs[5, 2].plot(D[22].history['val_categorical_accuracy'])
axs[5, 3].plot(D[23].history['val_categorical_accuracy'])
axs[6, 0].plot(D[24].history['val_categorical_accuracy'])
axs[6, 1].plot(D[25].history['val_categorical_accuracy'])
axs[6, 2].plot(D[26].history['val_categorical_accuracy'])
axs[6, 3].plot(D[27].history['val_categorical_accuracy'])


# In[58]:


fig2, axs = plt.subplots(7, 4,figsize=(20,10),sharey=True)
axs[0, 0].plot(D[0].history['val_loss'])
axs[0, 1].plot(D[1].history['val_loss'])
axs[0, 2].plot(D[2].history['val_loss'])
axs[0, 3].plot(D[3].history['val_loss'])
axs[1, 0].plot(D[4].history['val_loss'])
axs[1, 1].plot(D[5].history['val_loss'])
axs[1, 2].plot(D[6].history['val_loss'])
axs[1, 3].plot(D[7].history['val_loss'])
axs[2, 0].plot(D[8].history['val_loss'])
axs[2, 1].plot(D[9].history['val_loss'])
axs[2, 2].plot(D[10].history['val_loss'])
axs[2, 3].plot(D[11].history['val_loss'])
axs[3, 0].plot(D[12].history['val_loss'])
axs[3, 1].plot(D[13].history['val_loss'])
axs[3, 2].plot(D[14].history['val_loss'])
axs[3, 3].plot(D[15].history['val_loss'])
axs[4, 0].plot(D[16].history['val_loss'])
axs[4, 1].plot(D[17].history['val_loss'])
axs[4, 2].plot(D[18].history['val_loss'])
axs[4, 3].plot(D[19].history['val_loss'])
axs[5, 0].plot(D[20].history['val_loss'])
axs[5, 1].plot(D[21].history['val_loss'])
axs[5, 2].plot(D[22].history['val_loss'])
axs[5, 3].plot(D[23].history['val_loss'])
axs[6, 0].plot(D[24].history['val_loss'])
axs[6, 1].plot(D[25].history['val_loss'])
axs[6, 2].plot(D[26].history['val_loss'])
axs[6, 3].plot(D[27].history['val_loss'])


# In[60]:


#I chose the best parameters based on the minimum validation loss which means batch size=2 and epoch=10 
model.fit(X_train, y_train,validation_split=0.2,epochs=10, batch_size=2, verbose=1 )
plt.plot(model.evaluate(X_test, y_test,return_dict=True,verbose=1)


# In[ ]:




