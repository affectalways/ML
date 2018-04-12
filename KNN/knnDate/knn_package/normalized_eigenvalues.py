# -*- coding:UTF-8 -*-
import numpy as np


# 对数据进行归一化处理，防止有些数据偏大，导致结果不准确！！！
def auto_norm(data):
    min_value = np.min(data)[0]
    max_value = np.max(data)[0]
    return max_value - min_value


# 获取各轴坐标最大值
def get_coordinate(data):
    return np.max(data)


