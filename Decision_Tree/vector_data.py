# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 12:56
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : vector_data.py
# @Software: PyCharm
import numpy as np

labels = ['age', 'job', 'house', 'borrowing situation']

data = np.array([
    [1, 0, 0, 1, 0],
    [1, 0, 0, 2, 0],
    [1, 1, 0, 2, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0],

    [2, 0, 0, 1, 0],
    [2, 0, 0, 2, 0],
    [2, 1, 1, 2, 1],
    [2, 0, 1, 3, 1],
    [2, 0, 1, 3, 1],

    [3, 0, 1, 3, 1],
    [3, 0, 1, 2, 1],
    [3, 1, 0, 2, 1],
    [3, 1, 0, 3, 1],
    [3, 0, 0, 1, 0],

])


def get_data():
    global data
    return data


def get_labels():
    global labels
    return labels


if __name__ == "__main__":
    print get_data()
