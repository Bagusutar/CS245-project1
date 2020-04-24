from sklearn.decomposition import PCA
import numpy as np


def KPCA(feature, dim):
    pca = PCA(n_components=dim)
    feature_ = pca.fit_transform(feature)
    np.save('PCA/feature_' + str(dim), feature_)


feature = np.load("data/feature.npy")
for dim in [8, 16, 32, 64, 128, 256, 512, 1024]:
    KPCA(feature, dim)
