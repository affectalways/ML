# -*- coding:UTF-8 -*-
import numpy as np
import sys

sys.path.append('../')
from vector_data_package import vector_data

'''
    assume step = 1
'''


class Dual_Form(object):
    @classmethod
    def initialize_variables(cls, data_len):
        # 有多个a
        cls.a = [0] * data_len
        cls.b = 0
        # 学习率
        cls.step = 1

    @classmethod
    def classify_false_points(cls, data, gram_matrix, y_multiply_matrix):
        # is y;
        data_category = data[:, -1]
        while True:
            # 结束while True循环标志
            exit_sign = True
            for row_index in range(len(data)):
                category = data_category[row_index]
                # b * y1 or b * y2 or b * y3
                result = category * cls.b
                for column_index in range(len(data[row_index])):
                    result += cls.a[column_index] * gram_matrix[row_index][column_index] * y_multiply_matrix[row_index][
                        column_index]
                # 判断是否为误fen
                flag = Tools.classify_correct(result)

                # 若为误fen，更新对应的a，b
                if not flag:
                    exit_sign = False
                    Dual_Form.update_a_b(row_index, category)
            if exit_sign:
                # print '程序结束'
                return

    @classmethod
    def update_a_b(cls, row_index, category):
        cls.a[row_index] += 1
        cls.b += category

    @classmethod
    def get_w_b(cls, data):
        category = data[:, -1]
        data = data[:, :-1]

        w_dimension = data.shape[1]
        w = np.zeros(w_dimension)
        for index in range(len(data)):
            w += category[index] * cls.a[index] * data[index]

        return w, cls.b


class Tools(object):
    # 获取gram矩阵
    @staticmethod
    def gram_matrix(data):
        '''
        :param data: data type is list
        :return:
        '''
        # gram_matrix = np.matmul(data, data.T)
        # 若是每一维中，又是多维就gg了
        data = np.array(data[:, :-1])
        gram_matrix = np.dot(data, data.T)
        return gram_matrix

    # 获取y1*y1
    @staticmethod
    def yi_multiply(data):
        y_multiply_matrix = None
        # 一维。
        y_matrix = data[:, -1]

        for i in y_matrix:
            tmp_matrix = []
            for j in y_matrix:
                tmp_matrix.append(i * j)

            if y_multiply_matrix is None or len(y_multiply_matrix) == 0:
                y_multiply_matrix = np.array(tmp_matrix)
            else:
                y_multiply_matrix = np.row_stack((y_multiply_matrix, tmp_matrix))

        return y_multiply_matrix

    @staticmethod
    def classify_correct(result):
        if result <= 0:
            return False
        else:
            return True


if __name__ == "__main__":
    vector_data = np.array(vector_data.get_data())
    # data = np.array(vector_data[:, :-1])
    gram_matrix = Tools.gram_matrix(vector_data)

    data_len = len(vector_data)
    Dual_Form.initialize_variables(data_len)
    Tools.yi_multiply(vector_data)

    y_multiply_matrix = Tools.yi_multiply(data=vector_data)

    Dual_Form.classify_false_points(data=vector_data, gram_matrix=gram_matrix, y_multiply_matrix=y_multiply_matrix)

    w, b = Dual_Form.get_w_b(data=vector_data)
    print 'w 为 ' + str(w)
    print 'b 为 ' + str(b)
