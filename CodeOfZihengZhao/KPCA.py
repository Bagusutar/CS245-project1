# coding=utf-8
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score
from utli import *

#实验说明
#Step1：用PCA（linear kernel）对x数据降维，从2048降维到8、16、64、128、256
#Step2：分别对8、16、64、128、256维特征用10折交叉验证找到最好的C：0.01, 0.1, 1, 10

def KPCA(x, required_d):
    # 对输入x，用PCA方法降维到required_d维，并将降维后的数据保存为np文件，方便下次调用
    kpca = PCA(n_components=required_d)
    x_kpca = kpca.fit_transform(x)
    np.save('PCA/np_x_PCA_'+str(required_d), x_kpca)
    return x_kpca

#--------------------------------------Step1: Run at the first time-----------------------------------------

"""
x = np.load("np_x.npy")
for i in ['8', '16', '32', '64', '128', '256']:
    KPCA(x, int(i))
"""

#-----------------------------------------------Step2-------------------------------------------------------

dimens = ['8', '16', '32', '64', '128', '256']
C = [0.01, 0.1, 1 , 10]
prefix = "PCA/np_x_PCA_"
y = np.load("np_y.npy")
for i in range(len(dimens)):
    overall_scores = [0 for x in range(4)]
    kpca_x = np.load(prefix+dimens[i]+".npy")  # 加载feature
    train_x, train_y, test_x, test_y = partition_no_valid(kpca_x, y, 0.6)
    t = time.time()

    for j in range(4):
        svc = svm.SVC(C=C[j], kernel='linear', decision_function_shape='ovr')  # 初始化分类器
        scores = cross_val_score(svc, train_x, train_y, cv=10)  # K-fold交叉验证
        overall_scores[j] = np.sum(scores) / 10.0

    bestC = overall_scores.index(max(overall_scores))
    print('10-fold average scores：', overall_scores)
    print("The Best C for PCA "+dimens[i]+" = ", C[bestC])
    svc = svm.SVC(C=C[bestC], kernel='linear', decision_function_shape='ovr')
    svc.fit(train_x, train_y)
    print("Train Accuracy：", svc.score(train_x, train_y))  # 测试
    print("Test Accuracy：", svc.score(test_x, test_y))
    print('\n')


dimens = ['8', '16', '32', '64', '128']
C = [0.01, 0.1, 1 , 10]
prefix = "LLE/np_x_LLE_"
y = np.load("np_y.npy")
for i in range(len(dimens)):
    overall_scores = [0 for x in range(4)]
    lle_x = np.load(prefix+dimens[i]+".npy")  # 加载feature
    train_x, train_y, test_x, test_y = partition_no_valid(lle_x, y, 0.6)
    t = time.time()

    for j in range(4):
        svc = svm.SVC(C=C[j], kernel='linear', decision_function_shape='ovr')  # 初始化分类器
        scores = cross_val_score(svc, train_x, train_y, cv=10)  # K-fold交叉验证
        overall_scores[j] = np.sum(scores) / 10.0

    bestC = overall_scores.index(max(overall_scores))
    print('10-fold average scores：', overall_scores)
    print("The Best C for LLE "+dimens[i]+" = ", C[bestC])
    svc = svm.SVC(C=C[bestC], kernel='linear', decision_function_shape='ovr')
    svc.fit(train_x, train_y)
    print("Train Accuracy：", svc.score(train_x, train_y))  # 测试
    print("Test Accuracy：", svc.score(test_x, test_y))
    print('\n')


