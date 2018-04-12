# -*- coding:UTF-8 -*-
import numpy as np

from KNN.knnfirst.knnTest import create_dateset


def test_create_dataset(sample_to_be_tested, dataset, labels, k):
    dataset_size = dataset.shape[0]
    # 计算要被分类的输入向量x,y与给定个点的差
    differcdeEDence_set = np.tile(sample_to_be_tested, (dataset_size, 1)) - dataset
    x_y_square = difference_set ** 2
    distance_unfinish = x_y_square.sum(axis=1)
    distance = distance_unfinish ** 0.5
    # 将元素从xiao到da排列，返回其对应的索引
    indexes_sorted = np.argsort(distance)
    print indexes_sorted
    label_count = {}
    for i in range(k):
        label_selected = labels[indexes_sorted[i]]
        label_count[label_selected] = label_count.get(label_selected, 0) + 1

    category = sorted(label_count, key=lambda item: label_count.values(), reverse=True)
    return category[0]


if __name__ == '__main__':
    data, labels = create_dateset()
    print test_create_dataset([10, 2], data, labels, 4)
