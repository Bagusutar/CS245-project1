import os
import numpy as np

path = "G:\\Animals_with_Attributes2\\Features\\ResNet101"
dirs = os.listdir(path)
features_path = os.path.join(path, dirs[0])
filenames_path = os.path.join(path, dirs[1])
labels_path = os.path.join(path, dirs[2])

features = open(features_path, 'r')
feature_set = []
for line in features.readlines():
    data = line.rstrip().split(' ')
    data = [float(i) for i in data]
    feature_set.append(data)

labels = open(labels_path, 'r')
label_set = []
for line in labels.readlines():
    data = int(line.rstrip())
    label_set.append(data)

feature_array = np.array(feature_set)
label_array = np.array(label_set)

print(feature_array.shape)
print(label_array.shape)
np.save("data/feature.npy", feature_array)
np.save("data/label.npy", label_array)
