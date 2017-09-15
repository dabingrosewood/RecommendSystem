import numpy as np
import csv
with open("E:\AiDM\Assignment1\ml-latest-small\ml-latest-small\\ratings.csv", "r", encoding="utf-8") as csvfile:
    # 读取csv文件，返回的是迭代类型
    read = csv.reader(csvfile)
    data=[]
    for i in read:
        data.append(i)