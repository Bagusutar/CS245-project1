from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np

method = "LLE"  # RFE PCA LLE
dim = 8  # 8 16 32 64 128 256 512 1024 2048

features = np.load("{}/feature_{}.npy".format(method, dim))
labels = np.load("data/label.npy".format(method))

if dim == 2048:
    features = np.load("data/feature.npy".format(method, dim))

# 把要调整的参数以及其候选值 列出来；
param_grid = {"C": [10000, 50000, 100000]}

# 维数较低时用SVC效果较好，维数高时用LinearSVC速度较快
if dim < 32:
    model = SVC(kernel='linear', decision_function_shape='ovr')
else:
    model = LinearSVC()

grid_search = GridSearchCV(model, param_grid, cv=5, verbose=10, n_jobs=5)  # 实例化一个GridSearchCV类
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=10, stratify=labels)
grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。

print("Dim:{}".format(dim))
print("Parameters:{}".format(param_grid))
print("Best parameters:{}".format(grid_search.best_params_))
print("Best score on train set:{:.4f}".format(grid_search.best_score_))
print("Test set score:{:.4f}".format(grid_search.score(X_test, y_test)))
