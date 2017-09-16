import csv
import numpy as np
import time

def loadCSVfile2():
    tmp = np.loadtxt("data/ratings.csv", dtype=np.str, delimiter=",")
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

    return  label,data[100]


print(loadCSVfile2())
