#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


# Importing dataset
df = pd.read_csv("apt.csv")


# In[ ]:


df.head()


# # Random Forest Classification
# 

# In[ ]:


#features selection for the random forest classification 
df1 = df[['src_ip','src_port','dst_ip','dst_port','proto','type']]


# In[ ]:


df1.head()


# In[ ]:


df1 = df1.rename({'proto': 'Protocol'}, axis=1)


# In[ ]:


df1.head()


# In[ ]:


df1 = df1.rename({'type': 'Attack_type'}, axis=1)


# In[ ]:


#Show total attacks 
df1['Attack_type'].value_counts()


# In[ ]:


df1.head()


# In[ ]:


#radical approach for cleaning the data
df1 = df1.dropna() #if values are NA the drop them
df1.isnull().sum()


# In[ ]:


#counts total protocols used for attack
df1['Protocol'].value_counts()


# In[ ]:


#show all the variables of attack in array
df1["Attack_type"].unique()


# In[ ]:


#converts data into numbers for classification
from sklearn.preprocessing import LabelEncoder


# In[ ]:


le_Attack = LabelEncoder()
df1['Attack_type'] = le_Attack.fit_transform(df1['Attack_type'])
df1["Attack_type"].unique()


# In[ ]:


df1.head()


# In[ ]:


df1['Protocol'].unique()


# In[ ]:


le_protocol = LabelEncoder()
df1['Protocol'] = le_protocol.fit_transform(df1['Protocol'])
df1["Protocol"].unique()


# In[ ]:


df1.head()


# In[ ]:


le_srcip = LabelEncoder()
df1['src_ip'] = le_srcip.fit_transform(df1['src_ip'])
df1["src_ip"].unique()


# In[ ]:


le_dstip = LabelEncoder()
df1['dst_ip'] = le_dstip .fit_transform(df1['dst_ip'])
df1["dst_ip"].unique()


# In[ ]:


#Specify X and Y 
X = df1.drop("Attack_type", axis=1)
y = df1["Attack_type"]


# In[ ]:


X


# In[ ]:


y


# In[ ]:


labels = df1.pop('Attack_type')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model = RandomForestClassifier()


# In[ ]:


# fit the model on the whole dataset
model.fit(X, y.values)


# In[ ]:


y_pred = model.predict(X)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y.values, y_pred)
cm


# In[ ]:


get_ipython().system('pip install seaborn')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(8,8))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:


#find error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


error


# In[ ]:


import time

start = time.time()
y_pred = model.predict(X)
end = time.time()
print("Time Taken: ",end-start)


# In[ ]:


get_ipython().system('pip3 install -U scikit-learn scipy matplotlib')


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(f"Accuracy: {accuracy_score(y.values, y_pred)}%")


# In[ ]:


import numpy as np


# In[ ]:


#Input the details of system for APT attack prediction
row = np.array([['192.168.1.30',24,'192.168.1.190',24,'icmp']])
row


# In[ ]:


final = row[:,2]
final


# In[ ]:


final2 = row[:,1]
final2


# In[ ]:


row[:,0] = le_srcip.transform(row[:,0])
row[:,2] = le_dstip.transform(row[:,2])
row[:,4] = le_protocol.transform(row[:,4])
row = row.astype(float)
row


# In[ ]:


y_pred = model.predict(row)
y_pred


# In[ ]:





# # Time Series Forecasting

# In[ ]:


df.head()


# In[ ]:


#df2 = df.head(78)
#df2 = df.iloc[:278]


# In[ ]:


#Feature Selection For Forecasting
df2=df.reset_index()['label']


# In[ ]:


#Drop null values from the data
df2 = df2.dropna()


# In[ ]:


df2= df2.head(1258)


# In[ ]:


df2


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
plt.plot(df2)


# In[ ]:


import numpy as np


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df2=scaler.fit_transform(np.array(df2).reshape(-1,1))


# In[ ]:


print(df2)


# In[ ]:


#splitting dataset into train and test split
training_size=int(len(df2)*0.65)
test_size=len(df2)-training_size
train_data,test_data=df2[0:training_size,:],df2[training_size:len(df2),:1]


# In[ ]:


training_size,test_size


# In[ ]:


train_data


# In[ ]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[ ]:


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[ ]:


print(X_train.shape), print(y_train.shape)


# In[ ]:


print(X_test.shape), print(ytest.shape)


# In[ ]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[ ]:


import tensorflow as tf
tf.__version__


# In[ ]:


# Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[ ]:


# Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[ ]:


#Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[ ]:


# Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[ ]:


# Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[ ]:


# Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df2)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df2)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df2)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df2))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[ ]:


len(test_data)



# In[ ]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[ ]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input


# In[ ]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[ ]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


len(df1)


# # Network Graph Analysis

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 10)
import matplotlib as mpl
import numpy as np


# In[ ]:


df.head()


# In[ ]:


#Feature selection for Network Analysis
df4 = df.melt(
    ['src_ip','dst_ip','proto','type'],
    var_name = 'Time',value_name='Attack_type')


# In[ ]:


df4= df4.head(1999)


# In[ ]:


df4.head()


# In[ ]:


G = nx.from_pandas_edgelist(df4, 
                            source='src_ip',
                            target='dst_ip',
                            edge_attr='type',
                            create_using=nx.DiGraph())


# In[ ]:


print(nx.info(G))


# In[ ]:


#Graph showing which source ip can target the destination ip 
nx.draw_networkx(G)


# In[ ]:


G.out_degree(weight='dst_ip')


# However in degree is the one that determines the victory

# In[ ]:


h = plt.hist(dict(G.in_degree(weight='src_ip')).values())


# # Ensembling for final prediction

# In[ ]:


#Forecasting of the next 10 days
plt.plot(day_new,scaler.inverse_transform(df2[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[ ]:


df3=df2.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[ ]:


df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)


# In[ ]:


if(y_pred == 5):
    print("You are safe...No attack predict")
elif(y_pred == 2):
    print("Warning!! It's a Dos attack on IP", final[0], "from port", final2[0], "within 10 days.")
elif(y_pred == 1):
    print("Warning!! It's a Ddos attack on IP", final[0], "from port", final2[0], "within 10 days.")
elif(y_pred == 0):
    print("Warning!! It's a Backdoor attack on IP", final[0],"from port", final2[0], "within 10 days.")
elif(y_pred == 7):
    print("Warning!! It's a Ransomware attack on IP", final[0], "from port", final2[0], "within 10 days.")
elif(y_pred == 8):
    print("Warning!! It's a Scanning attack on IP", final[0], "from port", final2[0], "within 10 days.")
elif(y_pred == 6):
    print("Warning!! It's a Password attack on IP", final[0], "from port", final2[0], "within 10 days.")
elif(y_pred == 3):
    print("Warning!! It's a Injection attack on IP", final[0], "from port", final2[0], "within 10 days.")
elif(y_pred == 9):
    print("Warning!! It's a Xss attack on IP", final[0], "from port", final2[0], "within 10 days.")
else:
    print("Warning!! It's a Mitm attack on IP", final[0], "from port", final2[0], "within 10 days.")
    


# In[ ]:


print("Accuracy of a model by finding error which is", error)


# In[ ]:




