# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 12:56
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : vector_data.py
# @Software: PyCharm
import numpy as np
import pandas as pd

# labels = ['age', 'job', 'house', 'borrowing situation']

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
    # global labels
    return ['age', 'job', 'house', 'borrowing situation']


def get_data_and_labels_from_file():
    # buying price, price of the maintenance, number of doors, capacity in terms of persons to carry
    # the size of luggage boot, estimated safety of the car
    '''
        attributes:
            price:   vhigh, high, med, low.
            maintenance:    vhigh, high, med, low.
            car_door:    2, 3, 4, 5more.
            persons:  2, 4, more.
            luggage_boot: small, med, big.
            safety:   low, high.

        category:
            unacc, acc, good, vgood
    :return:
    '''

    labels = ['price', 'maintenance', 'car_door', 'persons', 'luggage_boot', 'safety', 'category']
    data = pd.read_csv('car.txt', sep=',', names=labels)
    # 因为是要二分，所以只选取unacc与good，而删除acc，vgood所在行
    # data = data[data.category.isin(['unacc', 'good'])]
    # price: low, high
    # data = data[data.price.isin(['low', 'high'])]
    # data = data[data.maintenance.isin(['low', 'high'])]
    # data = data[data.maintenance.isin(['low', 'high'])]
    print data
    # 现在需要做的是将每个特征换成数字
    return data, labels


if __name__ == "__main__":
    # print get_data()
    get_data_and_labels_from_file()
    # print get_data_and_labels_from_file()
