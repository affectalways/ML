# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 19:50
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : vector_data.py
# @Software: PyCharm
import pandas as pd
import numpy as np


def get_training_data():
    # training_data = pd.read_csv('testSet.txt', sep="\s+", header=None)
    training_data = pd.read_csv('testSetRBF.txt', sep="\s+", header=None)
    return np.array(training_data)
    # 数据列数
    # training_data_column_count = training_data.shape[1]
    # 为数据每一列命名
    # names = ['x' + str(i) for i in range(training_data_column_count - 1)]
    # y_name = ['category']
    # names.extend(y_name)
    # training_data.columns = names
    # return training_data


if __name__ == "__main__":
    get_training_data()
