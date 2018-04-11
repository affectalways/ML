# -*- coding:UTF-8 -*-

import numpy as np

'''
    no useful, because don't know the category, +1 or -1
'''


# vector_data_package
# use numpy generate 200 random vector(x, y)
# single case
class Vector_Data(object):
    __single_case = None

    def __new__(cls, *args, **kwargs):
        if cls.__single_case is None:
            cls.data = np.random.randint(-2, 10, size=(200, 2))
            cls.__single_case = object.__new__(cls)
        return cls.__single_case


# return numpy type
def get_data():
    # return Data().vector_data_package
    return [[3, 3, 1], [4, 3, 1], [1, 1, -1]]


if __name__ == "__main__":
    print get_data()
