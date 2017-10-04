# -*- coding: utf-8 -*-
"""
@author: Yitan Lou,Yixiao Tang
"""



import numpy as np

import importer
import time


ratings = importer.loadDATfile("data/ml-1m/ratings.dat") #import the data
nfolds = 5
# allocate memory for results:
err_train = np.zeros(nfolds)
err_test = np.zeros(nfolds)
MAE_train = np.zeros(nfolds)
MAE_test = np.zeros(nfolds)
# to make sure you are able to repeat results, set the random seed to something:
np.random.seed(29)
max_user = np.max(ratings[:,0])
max_item = np.max(ratings[:,1])
eachnum_user = np.zeros(max_user+1)  # the number of rate of each user or item
eachnum_item = np.zeros(max_item+1)
avg_user = np.zeros(max_user+1)
avg_item = np.zeros(max_item+1)
seqs = [x % nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)
t=np.zeros(nfolds)


for fold in range(nfolds):
    start = time.clock()
    train_sel = np.array([x != fold for x in seqs])
    test_sel = np.array([x == fold for x in seqs])
    train = ratings[train_sel]
    test = ratings[test_sel]
    gmr = np.mean(train[:, 2])
    for x in range(len(train)): #caculate average of users and average of items
        eachnum_user[train[x][0]] = eachnum_user[train[x][0]]+1
        eachnum_item[train[x][1]] = eachnum_item[train[x][1]]+1
        avg_user[train[x][0]] = avg_user[train[x][0]]+train[x][2]
        avg_item[train[x][1]] = avg_item[train[x][1]]+train[x][2]
    # find the item or the user which has no sample in the train set
    tmp = np.array([x == 0 for x in eachnum_user])
    avg_user[tmp] = gmr
    eachnum_user[tmp] = 1
    avg_user = avg_user / eachnum_user
    tmp = np.array([x == 0 for x in eachnum_item])
    avg_item[tmp] = gmr
    eachnum_item[tmp] = 1
    avg_item = avg_item / eachnum_item
    X1 = np.zeros(len(train))
    X2 = np.zeros(len(train))
    for x in range(len(train)):
        X1[x] = avg_user[train[x][0]]
        X2[x] = avg_item[train[x][1]]
    Y = train[:,2]
    X = np.vstack([X1, X2, np.ones(len(train))]).T
    S = np.linalg.lstsq(X, Y)
    print(S[0])
    pred_train = np.zeros(len(train))
    for x in range(len(train)):
        pred_train[x] = X1[x] * S[0][0] + X2[x] * S[0][1] + S[0][2]
    err_train[fold] = np.sqrt(np.mean((train[:, 2] - pred_train) ** 2))
    MAE_train[fold] = np.mean(np.abs(train[:, 2] - pred_train))
    # test phase
    X1 = np.zeros(len(test))
    X2 = np.zeros(len(test))
    for x in range(len(test)):
        X1[x] = avg_user[test[x][0]]
        X2[x] = avg_item[test[x][1]]
    Y = test[:, 2]
    X = np.vstack([X1, X2, np.ones(len(test))]).T
    S = np.linalg.lstsq(X, Y)
    pred_test = np.zeros(len(test))
    for x in range(len(test)):
        pred_test[x] = X1[x] * S[0][0] + X2[x] * S[0][1] + S[0][2]
    err_test[fold] = np.sqrt(np.mean((test[:, 2] - pred_test) ** 2))
    MAE_test[fold] = np.mean(np.abs(test[:, 2] - pred_test))
    end = time.clock()
    t[fold] = (end - start) / 60



print('RMSE of train:')
print(err_train)
print('RMSE of test:')
print(err_test)
print('MAE of train:')
print(MAE_train)
print('MAE of test:')
print(MAE_test)
print("time is:")
print(t)
