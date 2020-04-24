from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
import time


def KLLE(feature, dim):
    print("dim:", dim)
    t = time.time()
    lle = LocallyLinearEmbedding(n_components=dim, n_jobs=4, neighbors_algorithm='ball_tree')
    feature_ = lle.fit_transform(feature)
    np.save('LLE/feature_' + str(dim), feature_)
    print("time:", time.time() - t)


feature = np.load("LLE/feature.npy")
for dim in [1024, 512, 256, 128, 64, 32, 16, 8]:
    KLLE(feature, dim)
