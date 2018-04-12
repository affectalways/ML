import numpy as np


def create_dateset():
    group = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    labels = ['A', 'B', 'D', 'B', 'C', 'C']
    return group, labels
