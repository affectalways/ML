# -*- coding: utf-8 -*-
# @Time    : 2018/4/23 10:22
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : gini.py
# @Software: PyCharm
import tools
import numpy as np
import vector_data
import fractions


# 获取gini；是否对应一个类
# 仅仅是每个特征取指定值所取值得gini
def get_feature_gini(data, feature_value, feature_index):
    # 样本大小
    sample_size = data.shape[0]
    # 基尼指数
    gini = 0
    # 指定特征值所对应的data
    specified_feature_value_data = data[np.where(data[:, feature_index] == feature_value)]
    # 非特征值对应的data
    remanent_data = data[np.where(data[:, feature_index] != feature_value)]
    # 暂时存储的dict
    tmp_dict = {'feature_value_data': specified_feature_value_data, 'remanent_data': remanent_data}

    category = None

    for key, value in tmp_dict.items():
        # 特征值对应的类是否都属于同一个类，默认为False
        flag = False
        # 对应的长度
        length = len(value)
        # 对应的category_dict
        category_dict = tools.get_category_or_feature_dict(value)
        # 若是len==1，则对应的类只有一个
        if len(category_dict) == 1:
            # print category_dict
            category = category_dict.keys()[0]
            flag = True
        # 特征值在整个特征中所占的比利
        probability = fractions.Fraction(length, sample_size)
        # yes or not 在每个特征之中所占比例
        category_probability = fractions.Fraction(category_dict[0], length)
        # 计算基尼指数
        gini += probability * 2 * category_probability * (1 - category_probability)

    # 返回 基尼指数, flag, feature_index, feature_value, category
    return gini, flag, feature_index, feature_value, category


# 获得最优特征，最优切分点
def get_optimum_feature_and_best_cut_point(feature_gini_dict):
    min_value = min(feature_gini_dict.values())[0]
    # 查找value对应的key
    # 最优切分点，不止一个
    best_cut_points_dict = {k: v for k, v in feature_gini_dict.items() if v[0] == min_value}
    # 遍历切分点，判断切分点是否只有一个，若只有一个，返回
    # 若是多个，查看所有属性是否只对应一个类，若只对应一个类，便作为最佳切分点
    if len(best_cut_points_dict) == 1:
        # 返回字典
        return best_cut_points_dict
    for key, value in best_cut_points_dict.items():
        if value[1] is True:
            # 返回字典
            return {key: best_cut_points_dict[key]}
    best_cut_point_key, best_cut_point_value = min(feature_gini_dict.items(), key=lambda x: x[1][0])
    # 返回字典
    return {best_cut_point_key, best_cut_point_value}


if __name__ == "__main__":
    data = vector_data.get_data()
    feature_dict = tools.get_category_or_feature_dict(data, 0)
    print get_feature_gini(data, 3, 0)
