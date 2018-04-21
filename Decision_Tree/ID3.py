# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 12:57
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : ID3.py
# @Software: PyCharm
import fractions
import math
import sys

import numpy as np

import vector_data


# test ok
# 判断类的个数
def determine_number_category(data):
    # 转置的原因是 可以获取任何维度的数据的 类
    categories = np.unique(data.T[-1])
    # print categories
    # 若为空，返回False
    return True if categories.shape[0] > 1 else False


# 获取类在数据集中de个数
def get_category_count(data):
    categories_from_data_T = data.T[-1]
    # 获取类别
    categories = np.unique(categories_from_data_T)
    # 存储每个类别出现的次数
    categories_dict = {}
    for category in categories:
        categories_dict[category] = categories_from_data_T[categories_from_data_T == category].size
    return categories_dict


# test ok
# 判断特征是否为空
def determine_feature_is_not_empty(data):
    features = data.T[:-1]
    # print features
    # 若为空，返回False
    return True if features.size > 0 else False


# 获取每个特征取值空间, 和有多少特征
def get_feature(data):
    feature_data = data.T[:-1]
    features = []
    for row in feature_data:
        features.append(np.unique(row))
    features = np.array(features)
    return len(features), features


# 获取经验熵
# 该方法需要在判断特征是否为空之后执行
def get_empirical_entropy(data):
    # 样本数
    sample_size = len(data)
    # 每个类别出现的次数,用字典封装
    category_dict = get_category_count(data)
    # 经验熵
    empirical_entropy = 0

    for category_number in category_dict.values():
        probability = fractions.Fraction(category_number, sample_size)
        log2 = math.log(probability, 2) if probability != 0 else 0
        empirical_entropy -= probability * log2
    return empirical_entropy


# 经验条件概率
def get_empirical_conditional_entropy(data, origin_sample_size):
    # 经验条件概率
    empirical_conditional_entropy = 0
    # 每个类别出现的次数,用字典封装
    # category_dict = get_category_count(data)
    probability = fractions.Fraction(len(data), origin_sample_size)
    # 经验熵
    empirical_entropy = get_empirical_entropy(data)

    # 经验条件熵 -= 特征取指定值的个数 / 数据集样本总数 * 特征取值定值的经验熵
    empirical_conditional_entropy -= probability * empirical_entropy
    return empirical_conditional_entropy


# 信息增益
def get_information_gain_dict(data):
    '''
    :param data:
    :return:返回的是一个字典 {特征index： 对应的信息增益}
    '''

    # 记录最大的信息增益，并记录是哪一个特征得出的
    information_gain_dict = {}

    # 经验熵
    empirical_entropy = get_empirical_entropy(data)
    # 样本数
    sample_size = len(data)

    # 特征数量，即多少个特征,目前的data给了4个特征
    # 特征取值空间：二维
    feature_number, feature_value_space = get_feature(data)

    # 遍历每个特征，计算他们的增益
    for i in range(feature_number):
        # 经验条件熵
        empirical_conditional_entropy = 0

        # 用来查找在源data中特征取值的index
        data_used_to_find_feature_value_index = data.T[i]

        # 每个特征对应的取值
        feature_value = feature_value_space[i]

        # 选取哪一列
        column_index = [i, -1]

        for value in feature_value:
            index = []
            park = np.where(data_used_to_find_feature_value_index == value)
            # 判断语句其实没有任何必要
            if len(park) != 0:
                index.extend(list(park[0]))

            # 特征的每一个取值 顺带着 类
            # index.append(-1)
            # 选取指定的行，列(我目前只会numpy这种选取)
            every_feature_category_data = data[index]
            every_feature_category_data = every_feature_category_data[:, column_index]

            empirical_conditional_entropy -= get_empirical_conditional_entropy(every_feature_category_data,
                                                                               sample_size)

        # 信息增益
        information_gain = empirical_entropy - empirical_conditional_entropy
        information_gain_dict[i] = information_gain
    return information_gain_dict


# 获取最大信息增益所对应的特征
def get_feature_index_of_max_information_gain(information_gain_dict):
    feature_index = max(information_gain_dict, key=lambda x: information_gain_dict[x])
    return feature_index


# 从原有数据中删除信息增益最大的特征对应的指定列
def remove_feature(data, feature_index):
    data = np.delete(data, feature_index, axis=1)
    return data


# 删除信息增益最大的特征对应的label
def remove_label(labels, feature_index):
    del labels[feature_index]
    return labels


# 从原有数据中删除特征对应的指定行！注意是行，只有一个特征值对应一个类这行才被删除
def delete_data_rows(data, feature_row_index):
    data = np.delete(data, feature_row_index, axis=0)
    return data


# 利用信息增益最大的特征创建节点
def create_node(data, feature_index):
    feature_value_space = np.unique(data[feature_index])


if __name__ == "__main__":
    data = vector_data.get_data()
    # data = np.array([[1], [2], [1]])
    if data.size == 0:
        print '数据集为空，请重新输入'
        sys.exit(0)
    information_gain_dict = get_information_gain_dict(data)
    feature_index = get_feature_index_of_max_information_gain(information_gain_dict)
    # data = remove_feature_and_label(data, feature_index)
    print data
