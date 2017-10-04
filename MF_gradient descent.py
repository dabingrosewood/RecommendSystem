# -*- coding: utf-8 -*-
"""
@author: Yitan Lou,Yixiao Tang
"""

import numpy as np
import random
import importer
import time
import matplotlib.pyplot as plt

ratings = importer.loadDATfile("data/ml-1m/ratings.dat") #import the data

nfolds = 5
K = 10 #num_factors = 10
reg = 0.05 # regularization
learn_rate = 0.005
iteration = 75

# allocate memory for results:
err_train = np.zeros(nfolds)
err_test = np.zeros(nfolds)
MAE_train = np.zeros(nfolds)
MAE_test = np.zeros(nfolds)
t=np.zeros(nfolds)

# to make sure you are able to repeat results, set the random seed to something:
np.random.seed(29)
max_user = np.max(ratings[:,0])
max_item = np.max(ratings[:,1])
features_user = np.zeros([max_user+1,K])  # the features of users
features_item = np.zeros([max_item+1,K])  # the features of items
seqs = [x % nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)
last_RMSE_tain = 10  # the RMSE of the train of the last time
RMSE_tain = 0
MAE_tain = 0

for fold in range(nfolds):

    #initiation process
    start = time.clock()
    RMSE_record,MAE_record=[],[]
    train_sel = np.array([x != fold for x in seqs])
    test_sel = np.array([x == fold for x in seqs])
    train = ratings[train_sel]
    test = ratings[test_sel]
    for x in range(max_user+1):
        for y in range(K):
            features_user[x,y] = random.random()
    for x in range(max_item+1):
        for y in range(K):
            features_item[x,y] = random.random()
    pred_train = np.zeros(len(train))
    pred_test = np.zeros(len(test))


    #test session out of the iteration and the main iteration for gradient descent
    for x in range(len(train)):
        pred_train[x] = np.dot(features_user[train[x][0], :], features_item[train[x][1], :])
    RMSE_tain = np.sqrt(np.mean((train[:, 2] - pred_train) ** 2))
    MAE_tain = np.mean(np.abs(train[:, 2] - pred_train))

    for s in range(iteration):
        print(RMSE_tain)
        last_RMSE_tain = RMSE_tain
        last_MAE_tain = MAE_tain
        for x in range(len(train)):
            pred = np.dot(features_user[train[x][0], :], features_item[train[x][1], :]).T
            e = train[x, 2] - pred
            features_user[train[x][0],:] = features_user[train[x][0],:] + learn_rate * (2 * e * features_item[train[x][1],:] - reg * features_user[train[x][0],:])
            features_item[train[x][1],:] = features_item[train[x][1],:] + learn_rate * (2 * e * features_user[train[x][0],:] - reg * features_item[train[x][1],:])
        for x in range(len(train)):
            pred_train[x] = np.dot(features_user[train[x][0], :], features_item[train[x][1], :])
        RMSE_tain = np.sqrt(np.mean((train[:, 2] - pred_train) ** 2))
        MAE_tain =  np.mean(np.abs(train[:, 2] - pred_train))
        RMSE_record.append(last_RMSE_tain)
        MAE_record.append(last_MAE_tain)



    #test
    for x in range(len(test)):
        pred_test[x] = np.dot(features_user[test[x][0], :], features_item[test[x][1], :])

    #CACULATE THE INDICATES
    err_train[fold] = RMSE_tain
    err_test[fold] = np.sqrt(np.mean((test[:, 2] - pred_test) ** 2))
    MAE_train[fold] = np.mean(np.abs(train[:, 2] - pred_train))
    MAE_test[fold] = np.mean(np.abs(test[:, 2] - pred_test))



    end = time.clock()
    t[fold] = (end - start) / 60

    plt.figure(fold + 1)
    hor = np.arange(iteration)
    plt.plot(hor + 1, RMSE_record, label="RMSE of train")
    plt.plot(hor + 1, MAE_record, label="MAE of train")
    plt.title("RMSE & MAE")
    plt.xlabel("Iterations")
    plt.show()


    #OUTPUT
    print("time is:")
    print(t)
    print("RMSE_train")
    print(err_train)
    print("RMSE_test")
    print(err_test)
    print("MAE_train")
    print(MAE_train)
    print("MAE_test")
    print(MAE_test)
