# -*- coding:UTF-8 -*-
import numpy as np
from knn_package import normalized_eigenvalues, paint, readfile


def test_package():
    data = readfile.read_dating_file()
    print normalized_eigenvalues.get_coordinate(data[:, 0])


if __name__ == "__main__":
    test_package()
