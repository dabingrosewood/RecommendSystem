# -*- coding: utf-8 -*-
"""
@author: Yitan Lou,Yixiao Tang
"""
import csv
import numpy as np
import time


#csv文件倒入
def loadCSVfile(str):
    s=str
    tmp = np.loadtxt(s, dtype=np.str, delimiter=",")
#简单数据集
    data = tmp[1:,0:].astype(np.str)#加载数据部分
    label = tmp[0:1].astype(np.str)#加载类别标签部分


#分解数据集
    timeUnix = data[0:,3].astype(np.int)  #原格式时间
    timeNormal=[time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timeUnix[i])) for i in range(timeUnix.size)] #标准格式时间

    rating = data[0:,2].astype(np.float)
    userId = data[0:,1].astype(np.str)
    movieID = data[0:,0].astype(np.str)

#    return timeNormal[100],timeUnix[100] 时间测试

    return  data

#dat文件倒入
def loadDATfile(str):
    s=str;
    ratings = np.genfromtxt(s, usecols=(0, 1, 2, 3), delimiter='::', dtype='int')
    nfolds = 5

    return ratings



if __name__=="__main__":

    print(loadDATfile("data/ml-1m/ratings.dat"))
    #print(loadCSVfile("data/ml-1m/ratings.dat"))

