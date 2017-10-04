"""
Created on Fri Sep 18 21:09:10 2015

@author: Wojtek Kowalczyk

This script demonstrates how implement the "global average rating" recommender 
and validate its accuracy with help of 5-fold cross-validation.



@modified by Yitan Lou and Yixiao Tang
"""

import numpy as np
import importer
import time

# load data
# ratings=read_data("ratings.dat")
ratings = importer.loadDATfile("data/ml-1m/ratings.dat")
ratings = np.array(ratings)

"""
Alternatively, instead of reading data file line by line you could use the Numpy
genfromtxt() function. For example:

ratings = np.genfromtxt("ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')

will create an array with 3 columns.

Additionally, you may now save the rating matrix into a binary file 
and later reload it very quickly: study the np.save and np.load functions.
"""

# split data into 5 train and test folds
nfolds = 5

# allocate memory for results:
err_train = np.zeros(nfolds)
err_test = np.zeros(nfolds)

# to make sure you are able to repeat results, set the random seed to something:
np.random.seed(29)

seqs = [x % nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)
start = time.clock()
# for each fold:
for fold in range(nfolds):
    train_sel = np.array([x != fold for x in seqs])
    test_sel = np.array([x == fold for x in seqs])
    train = ratings[train_sel]
    test = ratings[test_sel]

    # calculate model parameters: mean rating over the training set:
    gmr = np.mean(train[:, 2])

    # apply the model to the train set:
    err_train[fold] = np.sqrt(np.mean((train[:, 2] - gmr) ** 2))

    # apply the model to the test set:
    err_test[fold] = np.sqrt(np.mean((test[:, 2] - gmr) ** 2))

    # print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))


end = time.clock()
t = (end - start) / 60

# print the final conclusion:
print("\n")
print("Mean error on TRAIN: " + str(np.mean(err_train)))
print("Mean error on  TEST: " + str(np.mean(err_test)))
print("run time",t)

# Just in case you need linear regression: help(np.linalg.lstsq) will tell you 
# how to do it!
