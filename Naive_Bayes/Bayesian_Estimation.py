# -*- coding: utf-8 -*-
# @Time    : 2018/4/15 20:55
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : Bayesian_Estimation.py
# @Software: PyCharm
from tools import *
import numpy as np


def bayesian_estimation(unclassify_element, unique_y, data, corrected_parameter=0):
    # 朴素贝叶斯分类器
    classifier = {}
    # 遍历x，获取index和value
    for y in unique_y:
        print 'y 为 ：' + str(y)
        prior_probability = data_handler.get_prior_probability(data, corrected_parameter=corrected_parameter,
                                                               y_index=-1, y_element=str(y))
        classifier[y] = prior_probability
        print 'prior_probability ' + str(prior_probability)
        for index, value in enumerate(unclassify_element):
            union_probability = data_handler.get_bayesian_estimation_conditional_probability(data,
                                                                                             corrected_parameter=corrected_parameter,
                                                                                             x_index=index,
                                                                                             x_element=str(value),
                                                                                             y_index=-1,
                                                                                             y_element=str(y))
            print 'index 为 ' + str(index) + '  ' + 'value 为 ' + str(value)
            print union_probability
            classifier[y] *= union_probability
            print 'classifier[y] ' + str(classifier[y])
    print '对于给定的元素 ：' + str(unclassify_element) + ',每一个类的贝叶斯估计为 ： ' + str(classifier)
    return max(classifier, key=lambda x: classifier[x])


if __name__ == "__main__":
    vector_data = data_handler.get_vector_data()
    unclassify_element = np.array([2, 'S'])
    unique_y = data_handler.get_unique_element(vector_data, -1)
    print str(unclassify_element) + '元素所属类型为 ： ' + str(
        bayesian_estimation(unclassify_element, unique_y, vector_data, corrected_parameter=1))
