# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
from perceptron_implement.raw_mode import Raw_Mode
from perceptron_implement.dual_form import Dual_Form, Tools
from vector_data_package import vector_data
import numpy as np


class PLT_Draw(object):
    def __init__(self):
        fig = plt.figure()
        self.__picture = fig.add_subplot(1, 1, 1)

    @property
    def picture(self):
        return self.__picture


if __name__ == "__main__":
    # acquire data as numpy format
    origin_data = np.array(vector_data.get_data())

    # 原始形式的感知机
    data = origin_data[:, :-1]
    x = data[:, 0]
    y = data[:, -1]
    # generate an instance, and draw scatter
    plt_draw = PLT_Draw()
    plt_draw.picture.scatter(x, y)

    # use raw_mode draw line
    # use w, b to draw line y = wx + b
    Raw_Mode.select_data(origin_data)
    w, b = Raw_Mode.get_w_b_value()

    if w[1] == 0:
        plt_draw.picture.plot(x, 0)
    plt_draw.picture.plot(x, -(float(w[0]) / w[1]) * x - float(b) / w[1])

    # show the paint
    plt.show()
    # print data

    # 对偶形式的感知机
    plt_draw = PLT_Draw()
    plt_draw.picture.scatter(x, y)
    gram_matrix = Tools.gram_matrix(origin_data)

    data_len = len(origin_data)
    Dual_Form.initialize_variables(data_len)
    Tools.yi_multiply(origin_data)

    y_multiply_matrix = Tools.yi_multiply(data=origin_data)

    Dual_Form.classify_false_points(data=origin_data, gram_matrix=gram_matrix, y_multiply_matrix=y_multiply_matrix)

    w, b = Dual_Form.get_w_b(data=origin_data)
    if w[1] == 0:
        plt_draw.picture.plot(x, 0)
    plt_draw.picture.plot(x, -(float(w[0]) / w[1]) * x - float(b) / w[1])

    print 'w 为 ' + str(w)
    print 'b 为 ' + str(b)
    plt.show()
