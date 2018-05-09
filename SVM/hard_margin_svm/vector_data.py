# -*- coding: utf-8 -*-
# @Time    : 2018/5/2 15:03
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : vector_data.py
# @Software: PyCharm
import numpy as np


# 训练数据
def get_training_data():
    data = np.array([
        [3, 3, 1],
        [4, 3, 1],
        [1, 1, -1]
    ])
    return data

# 测试数据
def get_test_data():
    pass

if __name__ == "__main__":
    print get_training_data()