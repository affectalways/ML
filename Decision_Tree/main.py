# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 12:56
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : main.py
# @Software: PyCharm
import decision_tree
import numpy as np
import vector_data
import ID3


def main():
    # decision_tree = {}
    data = vector_data.get_data()
    labels = vector_data.get_labels()
    information_gain_dict = ID3.get_information_gain_dict(data)
    feature_index = ID3.get_feature_index_of_max_information_gain(information_gain_dict)
    decision_tree = decision_tree.create_tree(data, labels, feature_index)
    # 从数据中删除指定特征；labels中删除特征对应的label
    data, labels = ID3.remove_feature_and_label(data, labels, feature_index)
    if len(labels) == 0:
        # 程序结束
        return


if __name__ == "__main__":
    main()
