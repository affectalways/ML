# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import normalized_eigenvalues
import readfile


def paint_result(data):
    fig = plt.figure()
    ax = Axes3D(fig)
    # 设置图形坐标范围
    x_max_value = normalized_eigenvalues.get_coordinate(data[:, 0])
    ax.set_xlim3d(0, x_max_value)
    y_max_value = normalized_eigenvalues.get_coordinate(data[:, 1])
    ax.set_ylim3d(0, y_max_value)
    z_max_value = normalized_eigenvalues.get_coordinate(data[:, 2])
    ax.set_zlim3d(0, z_max_value)
    for x, y, z in zip(data[:, 0], data[:, 1], data[:, 2]):
        ax.scatter3D(x, y, z)
    plt.show()


if __name__ == '__main__':
    data = readfile.read_dating_file()
    paint_result(data)
