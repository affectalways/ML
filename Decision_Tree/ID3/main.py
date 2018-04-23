# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 12:56
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : main.py
# @Software: PyCharm
import Decision_Tree.ID3
import Decision_Tree.ID3.vector_data
import Decision_Tree.ID3.decision_tree


def main():
    # decision_tree = {}
    data = Decision_Tree.ID3.vector_data.get_data()
    labels = Decision_Tree.ID3.vector_data.get_labels()
    information_gain_dict = Decision_Tree.ID3.get_information_gain_dict(data)
    feature_index = Decision_Tree.ID3.get_feature_index_of_max_information_gain(information_gain_dict)
    created_tree = Decision_Tree.ID3.decision_tree.create_tree(data, labels, feature_index)
    # 从数据中删除指定特征；labels中删除特征对应的label
    data, labels = Decision_Tree.ID3.remove_feature_and_label(data, labels, feature_index)
    if len(labels) == 0:
        # 程序结束
        return


if __name__ == "__main__":
    main()
