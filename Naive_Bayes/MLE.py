# -*- coding:UTF-8 -*-
# @Time    : 2018/4/15 20:09
# @Author  : affectalways
# @Site    :
# @Contact : affectalways@gmail.com
# @File    : MLE.py
# @Software: PyCharm
from tools import *
import numpy as np


def mle(unclassify_element, unique_y, data):
    '''
    极大似然估计
    :param unclassify_element:未分类元素，不包含y
    :param unique_y:y的取值范围
    :param data:测试数据集
    :return:返回所属类
    '''
    # 朴素贝叶斯分类器
    classifier = {}
    # 遍历x，获取index和value
    for y in unique_y:
        prior_probability = data_handler.get_prior_probability(data, y_index=-1, y_element=str(y))
        classifier[y] = prior_probability
        for index, value in enumerate(unclassify_element):
            union_probability = data_handler.get_conditional_probability_distribution(data,
                                                                                      x_index=index,
                                                                                      x_element=str(value),
                                                                                      y_index=-1,
                                                                                      y_element=str(y))
            classifier[y] *= data_handler.get_conditional_probability(prior_probability, union_probability)
    print '对于给定的元素 ：' + str(unclassify_element) + ',每一个类的极大似然估计为 ： ' + str(classifier)
    return max(classifier, key=lambda x: classifier[x])


if __name__ == "__main__":
    vector_data = data_handler.get_vector_data()

    unclassify_element = np.array([2, 'S'])
    unique_y = data_handler.get_unique_element(vector_data, -1)
    print str(unclassify_element) + '元素所属类型为 ： ' + str(mle(unclassify_element, unique_y, vector_data))
