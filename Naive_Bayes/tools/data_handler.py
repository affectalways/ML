# -*- coding:UTF-8 -*-
# @Time    : 2018/4/15 20:09
# @Author  : affectalways
# @Site    :
# @Contact : affectalways@gmail.com
# @File    : data_handler.py
# @Software: PyCharm

import numpy as np
import fractions


def get_vector_data():
    vector_data = np.array([
        [1, 'S', -1],
        [1, 'M', -1],
        [1, 'M', 1],
        [1, 'S', 1],
        [1, 'S', -1],
        [2, 'S', -1],
        [2, 'M', -1],
        [2, 'M', 1],
        [2, 'L', 1],
        [2, 'L', 1],
        [3, 'L', 1],
        [3, 'M', 1],
        [3, 'M', 1],
        [3, 'L', 1],
        [3, 'L', -1],
    ])
    return vector_data


# 获取每一列的不同取值范围,默认y的取值范围
def get_unique_element(data, columns=-1):
    unique_elements = np.unique(data[:, columns])
    return unique_elements


# 获取指定列的指定值的个数
def get_specified_element_count(data, **kwargs):
    # x的索引，y的索引，x，y的取值
    if kwargs is None or len(kwargs) == 0:
        return len(data)

    # 哪一列
    columns_index = []
    # 哪一列的值
    columns_index_value = []

    for key, value in kwargs.items():
        if 'index' in key:
            columns_index.append(value)
        elif 'element' in key:
            columns_index_value.append(value)

    data = data[:, columns_index]
    # print data
    flag_list = []
    for row_data in data:
        flag = (row_data == columns_index_value).all()
        flag_list.append(flag)
        # print data[flag_list]
        # if kwargs.has_key('corrected_parameter'):
        # 可取值个数
        # number_of_values =  get_unique_element(data, columns=)
    return flag_list.count(True)


# 先验概率
def get_prior_probability(data, corrected_parameter=0, **kwargs):
    '''
    先验概率
    :param data:原始数据
    :param kwargs: 包含y的所属哪一列，指定y的取值, 可能含有修正参数
    :return:
    '''
    specified_y_len = get_specified_element_count(data, **kwargs)
    data_len = get_specified_element_count(data)
    result = fractions.Fraction(specified_y_len + corrected_parameter,
                                data_len + len(get_unique_element(data)) * corrected_parameter)

    # if kwargs.has_key('corrected_parameter'):
    # y可取值个数
    # K = get_unique_element(data, columns=-1)
    return result


# 贝叶斯估计条件概率
def get_bayesian_estimation_conditional_probability(data, corrected_parameter=0, **kwargs):
    '''

    :param data:
    :param kwargs:参数key必须要明确写明那个是x_index，x_element那个是y_index,y_element！！！
    :return:
    '''
    numerator = get_specified_element_count(data, **kwargs) + corrected_parameter
    print 'numerator : ' + str(numerator)
    # 获取分母，即y=1或-1有多少个
    x_indexes = []
    for key in kwargs.keys():
        if 'x_index' in key or 'x_element' in key:
            if 'index' in key:
                x_indexes.append(kwargs[key])
            del kwargs[key]

    print 'kwargs : ' + str(kwargs)
    # y=1或-1有多少个
    specified_y_element = get_specified_element_count(data, **kwargs)
    print 'y 有 ' + str(specified_y_element)
    # x在某一维度取值有几个
    print 'x 在哪一个维度 + ' + str(x_indexes[0])
    x_dimension_number_of_value = len(get_unique_element(data, columns=x_indexes[0]))

    denominator = specified_y_element + x_dimension_number_of_value * corrected_parameter
    print '分母为 ' + str(denominator)
    return fractions.Fraction(numerator, denominator)


# 似然估计条件概率
def get_conditional_probability(prior_probability, conditional_probability_distribution):
    '''
    条件概率
    :param prior_probability:先验概率
    :param conditional_probability_distribution:联合分布概率
    :return:条件概率
    '''
    return fractions.Fraction(conditional_probability_distribution, prior_probability)


# 联合分布的条件分布
def get_conditional_probability_distribution(data, **kwargs):
    '''

    :param data:原始data
    :param kwargs: 包含x，y的index和指定值
    :return:
    '''
    unite_xy_len = get_specified_element_count(data, **kwargs)
    data_len = get_specified_element_count(data)
    return fractions.Fraction(unite_xy_len, data_len)
