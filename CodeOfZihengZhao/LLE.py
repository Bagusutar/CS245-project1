# coding=utf-8
import numpy as np
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
import time
from sklearn import svm
from sklearn.model_selection import cross_val_score
from utli import *

def wrap_lle(x, required_d, neighbors):
    # 对输入x，用LLE方法降维到required_d维，并将降维后的数据保存为np文件，方便下次调用
    lle = LocallyLinearEmbedding(n_components=required_d, n_neighbors=neighbors)
    lle.fit(x)
    x_lle = lle.embedding_
    np.save('LLE/np_x_LLE_'+str(required_d)+str(neighbors), x_lle)
    return x_lle

#--------------------------------------Run at the first time-----------------------------------------
"""
x = np.load("PCA/np_x_PCA_64.npy")
t = time.time()
wrap_lle(x, 8, 5)
print('LLE deduce 8 5 time:', time.time()-t)

x = np.load("PCA/np_x_PCA_64.npy")
t = time.time()
wrap_lle(x, 8, 10)
print('LLE deduce 8 10 time:', time.time()-t)

x = np.load("PCA/np_x_PCA_64.npy")
t = time.time()
wrap_lle(x, 8, 15)
print('LLE deduce 8 15 time:', time.time()-t)

x = np.load("PCA/np_x_PCA_64.npy")
t = time.time()
wrap_lle(x, 16, 15)
print('LLE deduce 16 15 time:', time.time()-t)

x = np.load("PCA/np_x_PCA_64.npy")
t = time.time()
wrap_lle(x, 32, 15)
print('LLE deduce 32 15 time:', time.time()-t)
"""
#-----------------------------------------------main-------------------------------------------------
nbs = ['5', '10', '15']
C = [0.01, 0.1, 1 , 10]
prefix = "LLE/np_x_LLE_8"
y = np.load("np_y.npy")
for i in range(len(nbs)):
    overall_scores = [0 for x in range(4)]
    lle_x = np.load(prefix+nbs[i]+".npy")  # 加载feature
    train_x, train_y, test_x, test_y = partition_no_valid(lle_x, y, 0.6)
    t = time.time()

    for j in range(4):
        svc = svm.SVC(C=C[j], kernel='linear', decision_function_shape='ovr')  # 初始化分类器
        scores = cross_val_score(svc, train_x, train_y, cv=10)  # K-fold交叉验证
        overall_scores[j] = np.sum(scores) / 10.0

    bestC = overall_scores.index(max(overall_scores))
    print('10-fold average scores：', overall_scores)
    print("The Best C for LLE 8 dimension "+nbs[i]+" neighours = ", C[bestC])
    svc = svm.SVC(C=C[bestC], kernel='linear', decision_function_shape='ovr')
    svc.fit(train_x, train_y)
    print("Train Accuracy：", svc.score(train_x, train_y))  # 测试
    print("Test Accuracy：", svc.score(test_x, test_y))
    print('\n')


dimens = ['8', '16']
C = [0.01, 0.1, 1 , 10]
prefix = "LLE/np_x_LLE_"
subfix = "15"
y = np.load("np_y.npy")
for i in range(len(dimens)):
    overall_scores = [0 for x in range(4)]
    lle_x = np.load(prefix+dimens[i]+subfix+".npy")  # 加载feature
    train_x, train_y, test_x, test_y = partition_no_valid(lle_x, y, 0.6)
    t = time.time()

    for j in range(4):
        svc = svm.SVC(C=C[j], kernel='linear', decision_function_shape='ovr')  # 初始化分类器
        scores = cross_val_score(svc, train_x, train_y, cv=10)  # K-fold交叉验证
        overall_scores[j] = np.sum(scores) / 10.0

    bestC = overall_scores.index(max(overall_scores))
    print('10-fold average scores：', overall_scores)
    print("The Best C for LLE "+dimens[i]+" dimensions 15 neighbors = ", C[bestC])
    svc = svm.SVC(C=C[bestC], kernel='linear', decision_function_shape='ovr')
    svc.fit(train_x, train_y)
    print("Train Accuracy：", svc.score(train_x, train_y))  # 测试
    print("Test Accuracy：", svc.score(test_x, test_y))
    print('\n')
