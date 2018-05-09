# -*- coding: utf-8 -*-
# @Time    : 2018/4/23 10:04
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : tree.py
# @Software: PyCharm
import numpy as np
import tools
import gini


# 弃用
class Tree(object):
    def __init__(self, root=None):
        self.root = root

    # 指哪打哪，添加左子节点就是左子节点，添加右子节点就是右子节点
    def add_node(self, node):
        pass


# 返回占大多数的类
def majority_category(data):
    category_dict = tools.get_category_or_feature_dict(data)
    return max(category_dict, key=lambda x: category_dict[x])


# 创建tree
def create_tree(data, labels):
    # if len(data) == 1:
    #     return

    # 若所有实例属于同一类，则属于单节点树，并将类作为该节点的类标记，返回T
    if len(np.unique(data[:, -1])) == 1:
        return np.unique(data[:, -1])[0]

    # 若 无特征集， 返回数据集中 占大多数的类
    if data.shape[1] == 1:
        return majority_category(data)

    # key:label+feature_value， value:基尼指数
    feature_gini_dict = {}

    # 只包含特征的data
    feature_data = data[:, :-1]
    feature_index = 0
    while feature_index < len(feature_data.T):
        feature_dict = tools.get_category_or_feature_dict(data, feature_index)
        # print data
        for feature_value in feature_dict.keys():
            # print labels
            key = labels[feature_index]
            feature_gini_dict[key] = gini.get_feature_gini(data, feature_value, feature_index)
        feature_index += 1
    # 获得最有特征，以及最优切分点
    optimum_feature_best_cut_point = gini.get_optimum_feature_and_best_cut_point(feature_gini_dict)

    # 创建树
    cart_tree = {}
    tmp_values = optimum_feature_best_cut_point.values()[0]
    # 从optimum_feature_best_cut_point中取出feature_index
    feature_index = tmp_values[2]

    # 获取特征所在列的内容
    feature_index_data = data[:, feature_index]
    # 指定key为最优特征，即house
    key = labels[feature_index]
    cart_tree[key] = {}
    # 这是为了for循环
    # 这里出了问题
    feature_value_dict = tools.get_category_or_feature_dict(data, feature_index)
    # 删除特征所在列
    data = tools.delete_data_column(data, feature_index)
    # 删除特征对应的label
    labels = tools.delete_label(labels, feature_index)

    # 只要feature index不被类的index包含，就False, 被包含就True
    # 多创建这个变量的原因是for循环对list进行删除，会产生错误
    feature_value_inclusion_relation = [item for item in feature_value_dict.keys()]
    for feature_value in feature_value_dict.keys():
        # 若tmp_values[1]为True 且 tmp_values[3] 等于 feature_value， 进行删除操作，且对树赋值
        if tmp_values[1] and tmp_values[3] == feature_value:
            # 特征指定的值在data中的index
            feature_value_index = np.where(feature_index_data == feature_value)[0]
            cart_tree[key][feature_value] = tmp_values[-1]
            feature_value_inclusion_relation.remove(feature_value)
            # 删除特征在data对应的行
            data = tools.delete_data_rows(data, feature_value_index)

    for feature_value in feature_value_inclusion_relation:
        cart_tree[key][feature_value] = create_tree(data, labels)

    # 返回创建的树
    return cart_tree


# 识别元素
def classify_element(element, cart_tree, labels):
    for key, value in cart_tree.items():
        # 获取value中key在labels的index
        label_index = labels.index(key)
        specified_value_of_element = element[label_index]
        # 若是一个字典，递归调用该方法
        if type(value).__name__ == 'dict':
            if type(value[specified_value_of_element]).__name__ == 'dict':
                classified_category = classify_element(element, value[specified_value_of_element], labels)
            else:
                classified_category = value[specified_value_of_element]
        else:
            # 若是值，就OK了
            classified_category = value
    return classified_category
