# -*- coding: utf-8 -*-
# @Time    : 2018/4/23 10:02
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : node.py
# @Software: PyCharm
class Node(object):
    def __init__(self, value, dimension=None, left_child_node=None, right_child_node=None):
        self.value = value
        self.dimension = dimension
        self.left_child_node = left_child_node
        self.right_child_node = right_child_node


if __name__ == "__main__":
    pass
