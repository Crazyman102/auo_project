# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 17:24:37 2021

@author: Wu Jia Zhan
"""
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from numpy import random
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
# function 
def series_to_supervised(data,n_in,n_out):
    nvar=1 if type(data) is list else data.shape[1]
    cols,names=list(),list()
    df=DataFrame(data)
    # input sequence (t-n,...,t-1)
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names+=[('var%d(t-%d)' % (j+1,i)) for j in range(nvar)]
    # output sequence (t,...,t+n)
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i==0:
            names+=[('var%d(t)'%(j+1)) for j in range(nvar)]
        else:
            names+=[('var%d(t+%d)'% (j+1,i)) for j in range(nvar)]
    supervise_data=concat(cols,axis=1)
    supervise_data.columns=names
    
    
    supervise_data.dropna(inplace=True)
    return supervise_data
        
def del_data(data,start,end,feature_count):
    delcol=[]
    for i in range(start,end):
        if i % feature_count !=0:
            delcol.append(i)
    
    data=data.drop(data.columns[delcol],axis=1)
    return data
        

def write_to_pkl(df,filename):
    data_pkl=MinMaxScaler(feature_range=(0,1)).fit(df)
    with open(filename,'wb') as outfile:
        pkl.dump(data_pkl, outfile)
    data=data_pkl.transform(df)
    return data

def make_feedforward_model(inputs,outputs):
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(inputs,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(7,activation='relu'))
    model.add(layers.Dense(outputs,activation='linear'))
    model.compile(loss='mean_absolute_error',optimizer=keras.optimizers.Adam(learning_rate=0.003),metrics=['mae'])
    return model


def make_cnn_model(inputs,outputs):
    return
    

# read dataset data
filepath='dataset/TrainSevenDay.csv'
data=read_csv(filepath)

# del column('DateTime')
del data['DateTime']

# data type DateFrame=> array
timeseries=data.values
# 將資料轉換成監督是學習
reframe=series_to_supervised(timeseries, 14, 7)
# 將資料欄位去除預測的7天(5向feature)
start_col=timeseries.shape[1]*14
end_col=reframe.shape[1]
new_reframe=del_data(reframe,start_col,end_col,timeseries.shape[1])

# 將資料做shuffle
random.shuffle(new_reframe.values)
train_X=new_reframe.iloc[:,:84]
train_y=new_reframe.iloc[:,-7:]

# 將 feature label 做標準化
train_X=write_to_pkl(train_X, 'model/feature_nomethod.pkl')
train_y=write_to_pkl(train_y, 'model/label_nomethod.pkl')

# model
model=make_feedforward_model(train_X.shape[1],7)
model.summary()
path=("model/sevenday_nomothod.h5").format()
checkpoint=ModelCheckpoint(path,monitor='mean_absolute_error',verbose=2,save_best_only=True,mode='min')
callbacklist=[checkpoint]
history=model.fit(train_X,train_y,epochs=500,batch_size=24,validation_split=0.05,callbacks=callbacklist)
model.save(path)
a=history.history

loss=a['loss']
val_loss=a['val_loss']
mae=a['mae']
val_mae=a['val_mae']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'r',label='training loss')
plt.plot(epochs,val_loss,'b',label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs,mae,'r',label='mae')
plt.plot(epochs,val_mae,'b',label='val_mae')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.legend()
plt.show()
