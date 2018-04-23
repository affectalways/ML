# -*- coding: utf-8 -*-
# @Time    : 2018/4/19 15:00
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : decision_tree.py
# @Software: PyCharm
import numpy as np

import Decision_Tree.ID3
import Decision_Tree.ID3.vector_data


def majority_category(data):
    category_dict = Decision_Tree.ID3.get_category_count(data)
    return max(category_dict, key=lambda x: category_dict[x])


# 创建多叉树（我也不知道该叫什么）
def create_tree(data, labels):
    # 若所有实例属于同一类，则属于单节点树，并将类作为该节点的类标记，返回T
    if len(np.unique(data[:, -1])) == 1:
        return np.unique(data[:, -1])[0]

    # 若 无特征集， 返回数据集中 占大多数的类
    if data.shape[1] == 1:
        return majority_category(data)

    # 获取信息增量最大的特征对应的index
    information_gain_dict = Decision_Tree.ID3.get_information_gain_dict(data)
    feature_index = Decision_Tree.ID3.get_feature_index_of_max_information_gain(information_gain_dict)

    # 用字典存储树，数据结构和算法有提过
    # key: feature_index , value 子节点或对应的特征：最大分类
    key = labels[feature_index]
    # 删除对应的label
    labels = Decision_Tree.ID3.remove_label(labels, feature_index)

    decision_tree = {}
    decision_tree[key] = {}

    # 特征uniuqe的值
    feature_value_space = np.unique(data[:, feature_index])
    # category每个类出现的次数
    category_dict = Decision_Tree.ID3.get_category_count(data)

    # 只要feature index不被类的index包含，就False, 被包含就True
    feature_value_inclusion_relation = [item for item in feature_value_space]

    # 存储每个类在data中的index, 类：类所在index
    category_index_of_specified_value_dict = {}
    for category in category_dict.keys():
        # 类 指定值 所在的index
        category_index_of_specified_value = np.where(data[:, -1] == category)[0]
        category_index_of_specified_value_dict[category] = category_index_of_specified_value.tolist()
    for feature_value in feature_value_space:
        # 特征 指定值 所在的index
        feature_index_of_specified_value = np.where(data[:, feature_index] == feature_value)[0].tolist()
        # feature value所在行被category value 所在行包含或者相同，就删除feature value所在行
        for category, value in category_index_of_specified_value_dict.items():
            # 包含关系
            # 判断是否包含,包含为True
            flag = set(feature_index_of_specified_value).issubset(value)
            if flag:
                # feature_value_inclusion_relation_dict[feature_value] = True
                # 删除被包含的
                feature_value_inclusion_relation.remove(feature_value)
                #     删数据喽
                data = Decision_Tree.ID3.delete_data_rows(data, feature_index_of_specified_value)
                # decision_tree字典创建一个新的分支，对应叶节点
                decision_tree[key][feature_value] = category
                # print decision_tree
                break

    # 获取 不被一个类包含的feature value作为key 的value
    for feature_value in feature_value_inclusion_relation:
        decision_tree[key][feature_value] = create_tree(data, labels)
    return decision_tree


if __name__ == "__main__":
    data = Decision_Tree.ID3.vector_data.get_data()
    labels = Decision_Tree.ID3.vector_data.get_labels()
    print create_tree(data, labels)
    # create_tree(data, 1)
