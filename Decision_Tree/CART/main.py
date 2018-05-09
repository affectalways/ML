# -*- coding: utf-8 -*-
# @Time    : 2018/4/23 10:18
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : main.py
# @Software: PyCharm
import vector_data
import tree
import fractions
import copy
import paint


# 程序入口
def main():
    # 读取
    origin_labels = vector_data.get_labels()
    data = vector_data.get_data()
    # 深拷贝
    labels = copy.deepcopy(origin_labels)
    # data 为 pandas 类型
    # data, origin_lables = vector_data.get_data_and_labels_from_file()
    # labels = copy.deepcopy(origin_lables)

    # 创建树
    cart_tree = tree.create_tree(data, labels)
    print cart_tree
    sample_size = len(data)
    error_count = 0

    for element in data:
        category = tree.classify_element(element[:-1], cart_tree, origin_labels)
        print '测试用例 ： ' + str(element[:-1]) + ', ' + '测试结果 : ' + str(category) + ', ' + '实际结果 ：' + str(element[-1])
        if category != element[-1]:
            error_count += 1

    print '错误率 ：' + str(fractions.Fraction(error_count, sample_size))
    paint.create_plot(cart_tree)


if __name__ == "__main__":
    main()
