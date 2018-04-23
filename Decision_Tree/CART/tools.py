# -*- coding: utf-8 -*-
# @Time    : 2018/4/23 10:23
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : tools.py
# @Software: PyCharm
import numpy as np
import collections
import vector_data

'''
用于获取类，类的个数；特征，特征值，特征值个数

'''


# 弃用
def get_category_dict(data):
    category_dict = collections.Counter(data[:, -1])
    return category_dict


# 弃用
# 返回每一个特征对应的特征值以及每个特征值个数
def get_feature_dict(data, feature_index):
    feature_dict = collections.Counter(data[:, feature_index])
    return feature_dict


# 删除特征在数据中的所在行
# 注意是行，只有一个特征值对应一个类这行才被删除
def delete_data_rows(data, index):
    data = np.delete(data, index, axis=0)
    return data


# 删除特征所在列
def delete_data_column(data, feature_index):
    data = np.delete(data, feature_index, axis=1)
    return data


# 删除特征对应的label
def delete_label(labels, index):
    del labels[index]
    return labels


# 1.获取类以及类对应的个数;
# 或者
# 2.返回每一个特征对应的特征值以及每个特征值个数
# 默认获取类以及类对应的个数
def get_category_or_feature_dict(data, index=-1):
    one_of_them_dict = collections.Counter(data[:, index])
    return one_of_them_dict


# 获取特征值对应的类
def get_feature_value_corresponding_class(data, category):
    '''

    :param data: data是已经被特征的每个取值划分后的data
    :param category:类
    :return:
    '''
    # tuple类型，
    return np.where(data[:-1] == category)


if __name__ == "__main__":
    data = vector_data.get_data()
    print get_category_dict(data)
    print get_feature_dict(data, 0)
    print get_category_or_feature_dict(data, -1)
    print get_category_or_feature_dict(data, 0)
