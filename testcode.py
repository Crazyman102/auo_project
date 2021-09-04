# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 20:39:56 2021

@author: Wu Jia Zhan
"""

from numpy.core.arrayprint import set_printoptions
from pandas import read_csv
import pickle as pkl
from keras import models
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from numpy import mean


month=[1,7,11]
date=[1,7,14,21]
test_path=[]
actual_path=[]
for i in month:
    for j in date:
        test_name='test-max-%d-%d.csv'%(i,j)
        actual_name='actual-max-%d-%d.csv'%(i,j)
        test_path.append(test_name)
        actual_path.append(actual_name)
    

all_MAE=[]
for i in range(0, len(actual_path)):    
    # read data
    path='dataset/'+test_path[i]
    data=read_csv(path)
    del data['DateTime']
    timeseries=data.values
    
    # change data shape
    windowsize=timeseries.shape[0]*timeseries.shape[1]
    timeseries=timeseries.reshape(1,windowsize)
    
    # scalar feature 
    with open('model/feature_nomethod.pkl','rb') as infile:
        feature_pkl=pkl.load(infile)
    values=feature_pkl.transform(timeseries)
    
    # sclar label
    with open('model/label_nomethod.pkl','rb') as infile:
        label_pkl=pkl.load(infile)
    
    
    # load model and put the data into model
    # model=models.load_model('model/sevenday_nomothod.h5')
    model=models.load_model('model/sevenday_addnoise.h5')
    pred=model.predict(values)
    pred=pred.astype(np.float32)
    pred=label_pkl.inverse_transform(pred)
    pred=pred.reshape(7,)
    
    # load actual data
    path2='dataset/'+actual_path[i]
    actual_data=read_csv(path2)
    del actual_data['DateTime']
    actual_values=np.array(actual_data.values).reshape(7,)
    
    
    # calculate mean square error
    MAE=mean_absolute_error(actual_values,pred)
    all_MAE.append(MAE)
    
    # make result picture
    days=range(1,8)
    plt.plot(days,pred,'red',label='predict_value')
    plt.plot(days,actual_values,'green',label='actual_value')
    plt.xlabel('days')
    plt.ylabel('value')
    plt.title(actual_path[i])
    plt.legend()
    plt.show()
    print(actual_path[i]+'MAE:',MAE)