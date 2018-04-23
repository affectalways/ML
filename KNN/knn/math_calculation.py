# -*- coding:UTF-8 -*-
import numpy as np
import KDTreeNode


# 获得平均数
def get_mean_value(data):
    '''
    参数data为np.ndarray类型
    :param data:
    :return:
    '''
    # 计算每一列的平均数
    return data.mean(axis=0)


# test ok
# 获取方差，以进行哪一维度划分
def get_variance(data):
    '''
    参数data为np.ndarray类型
    :param data:
    :return:
    '''
    return np.var(data, axis=0)
    # data_variance_numpy = data.var(axis=0)
    # index = np.where(data_variance_numpy == np.max(data_variance_numpy))
    # return index[0][0]
    # 计算每一列的方差
    # if len(data) == 0:
    #     return 0
    # 按列求均值
    # data_mean = np.mean(data, axis=0)
    # 均值差
    # mean_difference = data - data_mean
    # 方差
    # data_var = np.sum(mean_difference ** 2, axis=0) / data.shape[0]
    # return data_var


# 计算欧式距离
def get_distance(unclassified_element, node):
    # ord是指进行几次方，若ord=2， 则距离为(（x1-x2）^2 + (y1-y2)^2)^1/2
    # distance = np.linalg.norm(unclassified_element - node.value, axis=0, ord=len(unclassified_element))
    distance = 0.0
    for i in range(len(unclassified_element)):
        distance += (unclassified_element[i] - node.value[i]) ** 2
    # distance = np.linalg.norm(unclassified_element - node.value, axis=0, ord=2)
    # return distance
    return np.math.sqrt(distance)


if __name__ == "__main__":
    # get_variance(np.array([[1, 2, 3], [1, 2, 3]]))
    node = KDTreeNode.Node(value=np.array([1, 2, 1, 1], dtype=np.float64), label=1, split_dimension=1,
                           left_child_node=None,
                           right_child_node=None)
    tmp = np.array([1, 1, 1, 4])
    print get_distance(tmp, node)
    print get_distance(unclassified_element=np.array([0, 0, 0, 4], dtype=np.float64), node=node)
