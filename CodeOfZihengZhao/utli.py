# coding=utf-8
import numpy as np

def numpy_xy():
    # 将txt格式的feature与label转为array格式并储存
    filename = 'data/AwA2-features.txt'
    x = []
    file_to_read = open(filename, 'r')
    count = 0
    while True:
        lines = file_to_read.readline()
        if not lines:
          break
        a_image = lines.split(" ")
        a_image[-1] = a_image[-1][:-1]
        if count == 0:
            print(len(a_image))
            print(a_image)
            count += 1
        x.append(a_image)
    x = np.array(x)
    np.save("np_x", x)

    filename = 'data/AwA2-labels.txt'
    y = []
    file_to_read = open(filename, 'r')
    count = 0
    while True:
        lines = file_to_read.readline()
        if not lines:
          break
        a_label = lines[:-1]
        y.append(a_label)
    y = np.array(y)
    np.save("np_y", y)

def check_labels():
    # 统计各个类下的样本数
    filename = 'data/AwA2-labels.txt'
    labels = [0 for i in range(51)]
    file_to_read = open(filename, 'r')
    while True:
        lines = file_to_read.readline()
        if not lines:
            break
        labels[int(lines[:-1])] += 1
    assert len(labels) == 51
    labels = np.array(labels)
    np.save("np_labels", labels)

def partition_no_valid(x, y, train_prop):
    # 根据训练样本比例均匀划分每个类目
    labels = np.load("np_labels.npy")
    train_labels = labels*train_prop  # train_labes[i]即第i类图片有多少划为train data
    x = x.tolist()
    y = y.tolist()
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    cursor = 0
    for i in range(1, 51):
        for j in range(int(train_labels[i])):   # 带小数的train sample数就统一退一位，如2.7到2
            train_x.append(x[cursor])
            train_y.append(y[cursor])
            cursor += 1
        for j in range(int(train_labels[i]), labels[i]):
            test_x.append(x[cursor])
            test_y.append(y[cursor])
            cursor += 1
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return train_x, train_y, test_x, test_y