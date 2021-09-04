# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:03:56 2021

@author: Wu Jia Zhan
"""
# https://blog.csdn.net/weixin_44935235/article/details/104036259
from pandas import read_csv

# read data
data=read_csv('dataset/TrainSevenDay.csv')
dataseries=data['Top1'].values

import numpy as np
def ave_value(data):
    return np.sum(data)/len(data)

def sigma(data,avg):
    sigma_squ=np.sum(np.power((data-avg),2))/len(data)
    return np.power(sigma_squ,0.5)

def prob(data,avg,sig):
    print(data)
    sqrt_2pi=np.power(2*np.pi,0.5)
    coef=1/(sqrt_2pi*sig)
    powercoef=-1/(2*np.power(sig,2))
    mypow=powercoef*(np.power((data-avg),2))
    return coef*(np.exp(mypow))

# 樣本均值
ave2=ave_value(dataseries)
ave=np.mean(dataseries)
# 樣本數標準差
sig=dataseries.std()
sig2=sigma(dataseries, ave)
# 高斯分布概率
x=np.arange(min(dataseries),max(dataseries),10)
pdf2=prob(x,ave,sig)


from matplotlib import pyplot as plt
plt.plot(x,pdf2, linewidth=3)
plt.grid()
plt.xlabel("HLB")
plt.ylabel("prob density")
plt.title("Gaussian distrbution")
plt.tick_params(labelsize=10) 
plt.show()


