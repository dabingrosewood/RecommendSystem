import numpy as np
import csv
with open("E:\AiDM\Assignment1\ml-latest-small\ml-latest-small\\ratings.csv", "r", encoding="utf-8") as csvfile:
    # 读取csv文件，返回的是迭代类型
    read = csv.reader(csvfile)
    data=[]
    for i in read:
        data.append(i)
    data.pop(0)
    ratings=np.array(data)

    # split data into 5 train and test folds
    nfolds = 5

    # allocate memory for results:
    err_train = np.zeros(nfolds)
    err_test = np.zeros(nfolds)

    # to make sure you are able to repeat results, set the random seed to something:
    np.random.seed(17)  # 方法改变随机数生成器的种子

    seqs = [x % nfolds for x in range(len(ratings))]
    np.random.shuffle(seqs)  # 方法将序列的所有元素随机排序,seq保存每一个样本的归属编号（1-nflod）

    # for each fold:
    for fold in range(1):
        train_sel = np.array([x != fold for x in seqs])
        test_sel = np.array([x == fold for x in seqs])
        train = ratings[train_sel]
        test = ratings[test_sel]
        test = test.astype(np.float)
        train = train.astype(np.float)
       # calculate model parameters: mean rating over the training set:
        gmr = np.mean(train[:,2])  # global mean rating

        # apply the model to the train set:
        err_train[fold] = np.sqrt(np.mean((train[:, 2] - gmr) ** 2))

        # apply the model to the test set:
        err_test[fold] = np.sqrt(np.mean((test[:, 2] - gmr) ** 2))

        # print errors:
        print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))
        # print the final conclusion:
        print("\n")
        print("Mean error on TRAIN: " + str(np.mean(err_train)))
        print("Mean error on  TEST: " + str(np.mean(err_test)))

