# -*- coding:UTF-8 -*-
import numpy as np
import os


# 从txt中读取数据
def read_dating_file():
    dir_path = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__))))
    with open(dir_path + '\datingData.txt', 'r') as f:
        # 读取所有行
        data = f.readlines()

        list_data = []
        for line in data:
            tmp = line.split()
            list_data.append(tmp)

        # 返回numpy的array类型
        numpy_data = np.array(list_data, dtype=np.float64)
        # print numpy_data[:, 0]
        # print numpy_data[:, 0].dtype
        return numpy_data


if __name__ == "__main__":
    read_dating_file()
