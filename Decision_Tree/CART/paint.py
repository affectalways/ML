# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 13:10
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : paint.py
# @Software: PyCharm
'''
纯属抄袭，请勿当真！！！！！！！！！！！！
'''
import matplotlib.pyplot as plt

# 定义判断节点形态
decision_node = dict(boxstyle="sawtooth", fc="0.8")
# 定义叶节点形态
leaf_node = dict(boxstyle="round4", fc="0.8")
# 定义箭头
arrow_args = dict(arrowstyle="<-")


# def draw_tree(decision_tree):
def plot_node(node_text, center_node, parent_node, node_type):
    # node_text，其实就是注释
    # center_node:就是节点所在位置
    # parent_node:就是父节点
    # node_type:就是定义的三个全局变量
    create_plot.ax1.annotate(node_text, xy=parent_node, xycoords='axes fraction', xytext=center_node,
                             textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                             arrowprops=arrow_args)


def plot_text_between_parent_child_node(child_node_position, parent_node_position, text):
    x_position = (parent_node_position[0] - child_node_position[0]) / 2.0 + child_node_position[0]
    y_position = (parent_node_position[1] - child_node_position[0]) / 2.0 + child_node_position[1]
    print x_position, y_position
    print text
    create_plot.ax1.text(x_position, y_position, text, va="center", ha='center')


# 画树
def draw_tree(decision_tree, parent_node_position, text):
    # 获取叶节点数目
    leaf_node_number = get_leaf_node_number(decision_tree)
    # 获取树的深度
    tree_depth = get_tree_depth(decision_tree)
    first_node_key = list(decision_tree.keys())[0]
    child_node_position = (
        draw_tree.x_off + (1.0 + float(leaf_node_number)) / 2.0 / draw_tree.total_leaf_node_count, draw_tree.y_off)
    plot_text_between_parent_child_node(child_node_position, parent_node_position, text)
    plot_node(first_node_key, child_node_position, parent_node_position, decision_node)
    secondDict = decision_tree[first_node_key]
    draw_tree.y_off = draw_tree.y_off - 1.0 / draw_tree.total_depth
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            draw_tree(secondDict[key], child_node_position, str(key))
        else:
            draw_tree.x_off = draw_tree.x_off + 1.0 / draw_tree.total_leaf_node_count
            plot_node(secondDict[key], (draw_tree.x_off, draw_tree.y_off), child_node_position, leaf_node)
            plot_text_between_parent_child_node((draw_tree.x_off, draw_tree.y_off), child_node_position, str(key))
    draw_tree.y_off = draw_tree.y_off + 1.0 / draw_tree.total_depth


# 获取叶节点数目
def get_leaf_node_number(decision_tree):
    count = 0
    for key, value in decision_tree.items():
        if type(value).__name__ == 'dict':
            count += get_leaf_node_number(value)
        else:
            count += 1
    return count


# 获取决策树深度
def get_tree_depth(decision_tree):
    depth = 0
    for key, value in decision_tree.items():
        if type(value).__name__ == 'dict':
            depth = 1 + get_tree_depth(value)
        else:
            depth = 1

    return depth


def create_plot(decision_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # plot_node('decision_node', (0.5, 0.1), (0.1, 0.5), decision_node)
    # plot_node('leaf_node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    draw_tree.total_leaf_node_count = float(get_leaf_node_number(decision_tree))
    draw_tree.total_depth = float(get_tree_depth(decision_tree))
    draw_tree.x_off = -0.5 / draw_tree.total_leaf_node_count
    draw_tree.y_off = 1.0
    draw_tree(decision_tree, (0.5, 1.0), '')

    plt.show()


if __name__ == "__main__":
    # pass
    create_plot()
