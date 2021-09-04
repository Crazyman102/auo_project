# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 20:24:10 2021

@author: Wu Jia Zhan
"""
# =============================================================================
# 1.讀取資料
# 2.將資料做supervised
# 3.將output不要的特徵移除
# 4.取出所有的output欄位
# 5.將y 的values複製到5000比
# 6.將原始資料做supervised 增加到5000比
# =============================================================================

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from numpy import random
from numpy import array 
from numpy import vstack
import numpy as np
def series_to_supervised(data,n_in,n_out,dropnan=True):
    n_vars=1 if type(data) is list else data.shape[1]
    df=DataFrame(data)
    cols,names=list(),list()
    # input sequences
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names+=[('var%d(t-%d)'%(j+1,i)) for j in range(n_vars)]
    # output sequence
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i==0:
            names+=[('var%d(t)'%(j+1)) for j in range(n_vars)]
        else:
            names+=[('var%d(t+%d)'% (j+1,i)) for j in range(n_vars)]
    # concat data
    agg=concat(cols,axis=1)
    agg.columns=names
    if dropnan:
        agg.dropna(inplace=True)    
    return agg

def del_data(data,strat,end,feature):
    del_col=[]
    for i in range(start,end+1):
        if i % feature !=0:
            del_col.append(i)
    data=data.drop(data.columns[del_col],axis=1)
    return data

def new_data(data):
    names = locals()
  
    names['top1%s'%i],names['top2%s'%i],names['temp%s'%i],names['hum%s'%i],names['L6B%s'%i],names['CELL%s'%i]=new_random(data)
    names['totoal_data%s'%i]=vstack((names['top1%s'%i],names['top2%s'%i],names['temp%s'%i],names['hum%s'%i],names['L6B%s'%i],names['CELL%s'%i]))
    
    names['totoal_data%s'%i] =  names['totoal_data%s'%i].T
   
    names['totoal_data%s'%i] = names['totoal_data%s'%i].astype('float32')
    timeseries=names['totoal_data%s'%i] #將所有資料統整(原本、生成)
    return timeseries  

def new_random(input_data):
    top1,top2,temp,hum,L6B,CELL=[],[],[],[],[],[]    
    row,col=input_data.shape    
    for i in range(row):
        for j in range(col):
            value=input_data[i][j]
            if j==0:
                new_value=value+(random.uniform(0,373))
                top1.append(new_value)
            elif j==1:
                new_value=input_data[i][j]+(random.uniform(-373,0))
                top2.append(new_value)
            elif j==2:
                temp.append(input_data[i][j])
            elif j==3:
                hum.append(input_data[i][j])
            elif j==4:
                L6B.append(input_data[i][j])
            elif j==5:
                CELL.append(input_data[i][j])                
    return top1,top2,temp,hum,L6B,CELL



def data_augment(timeseries):
    # shape : (1270, 6)
    timeseries[:, 0:1] += np.random.uniform(0, 373, size=(timeseries.shape[0], 1))
    timeseries[:, 1:2] += np.random.uniform(-373, 0, size=(timeseries.shape[0], 1))
    return timeseries.astype('float32')

data=read_csv('dataset/TrainSevenDay.csv')
del data['DateTime']
dataseries=data.values

reframe=series_to_supervised(dataseries,14,7)
# 將資料欄位去除預測的7天(5個feature)
start=data.shape[1]*14 #原始資料*14
end=reframe.shape[1] #監督是學習後總col
reframe=del_data(reframe,start,end,data.shape[1])# 將不是top1的刪掉

# step 4 取出資料生成的結果
y_answer=reframe.iloc[:,84:] 
# step 5
y_new_answer=[]
for i in range(5):    
    y_new_answer.extend(y_answer.values)


#step 6
new_data=[]
for i in range(4):    
    new_timeseries=data_augment(dataseries)
    reframe_2=series_to_supervised(new_timeseries, 14, 7)
    reframe_2=reframe_2.iloc[:,:84]
    new_data.extend(reframe_2.values)
new_data.extend(reframe.iloc[:,:84].values)
    
new_data = np.array(new_data)
y_new_answer = np.array(y_new_answer)

# step 7 data to shuffle
def shuffle_sample(inputs, target):
    ind = np.arange(0, inputs.shape[0])
    np.random.shuffle(ind)
    new_inputs = inputs[ind, :]
    new_target = target[ind, :]
    return new_inputs, new_target

new_inputs, new_target = shuffle_sample(new_data, y_new_answer)



from sklearn.preprocessing import MinMaxScaler

train_X=MinMaxScaler(feature_range=(0,1)).fit_transform(new_inputs)
train_y=MinMaxScaler(feature_range=(0,1)).fit_transform(new_target)


from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

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
# model
model=make_feedforward_model(train_X.shape[1],7)
model.summary()
path=("model/sevenday_addnoise.h5").format()
checkpoint=ModelCheckpoint(path,monitor='mean_absolute_error',verbose=2,save_best_only=True,mode='min')
callbacklist=[checkpoint]
history=model.fit(train_X,train_y,epochs=2000,batch_size=24,validation_split=0.05,callbacks=callbacklist)
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