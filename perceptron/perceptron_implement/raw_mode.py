# -*- coding:UTF-8 -*-
import numpy as np
import sys

sys.path.append('../')
from vector_data_package import vector_data

# Use three points in the statistical learning method.
# vector_data_package = [[3, 3, 1], [4, 3, 1], [1, 1, -1]]
data = vector_data.get_data()


class Raw_Mode(object):
    # specified inital w, b
    w = np.array([0, 0])
    b = 0

    '''
        # random select one vector_data_package, if properly classified, then random choose others(can't choose it again,
        # unless after update w & b)
    '''

    # for loop every element, classify category right or not
    # until all elements' categories right
    @classmethod
    def select_data(cls, data):
        # random generate index
        # index = random.randint(0, 2)
        while True:
            flag = True
            for index in range(len(data)):
                selected_data = data[index]
                # matrix multiply
                category = selected_data[-1]
                x_vector = np.array(selected_data[:-1])
                w_multiply_x = cls.w.dot(np.transpose(x_vector))

                result = category * (w_multiply_x + cls.b)

                if result <= 0:
                    flag = False
                    cls.update_w_b(category, x_vector)

            if flag is True:
                print '程序结束'
                print 'w = ' + str(cls.w)
                print 'b = ' + str(cls.b)
                return True

    # not properly classified, update w, b
    @classmethod
    def update_w_b(cls, category, x_vector):
        cls.w += category * x_vector
        cls.b += category
        # print type(x_vector)
        # print type(category)

    @classmethod
    def get_w_b_value(cls):
        return cls.w, cls.b


if __name__ == "__main__":
    Raw_Mode.select_data()
