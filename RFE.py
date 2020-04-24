from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import numpy as np


def KRFE(feature, label):
    feature_normalized = preprocessing.normalize(feature, norm='l1')
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(LogisticRegression(), n_features_to_select=8, step=8, verbose=1)
    rfe.fit(feature_normalized, label)

    indexes = [i for i in range(feature.shape[1])]
    rank = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), indexes))

    rank_list = []
    for i in rank:
        rank_list.append(i[1])
    print(rank)
    print(rank_list)
    return rank_list


feature = np.load("RFE/feature.npy")
label = np.load("RFE/label.npy")

rank_l = KRFE(feature, label)

for dim in [8, 16, 32, 64, 128, 256, 512, 1024]:
    feature_ = np.zeros((feature.shape[0], dim))
    rank_l_ = sorted(rank_l[:dim])

    j = 0
    for i in rank_l_:
        feature_[:, j] = feature[:, i]
        j += 1

    np.save("RFE_/feature_{}.npy".format(dim), feature_)
