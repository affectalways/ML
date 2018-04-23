# -*- coding:UTF-8 -*-
import numpy as np

import math_calculation


# 返回排序后的sort_kd_tree_node_distance(type为由元组构成的list类型)
def sort_nodes_distance(kd_tree_node_distance):
    # 对其进行正序， 由小到大， 在调用方法处进行排序即可
    sorted_nodes_distance = sorted(kd_tree_node_distance.items(), key=lambda x: x[1], reverse=False)
    return sorted_nodes_distance


# 返回一个字典
def get_k_labels(sort_kd_tree_node_distance, k=6):
    labels_weight = {}

    if len(sort_kd_tree_node_distance) <= k:
        sort_kd_tree_node_distance = sort_kd_tree_node_distance[:len(sort_kd_tree_node_distance)]
    else:
        sort_kd_tree_node_distance = sort_kd_tree_node_distance[:k]

    for item in sort_kd_tree_node_distance:
        print 'item 点为 ' + str(item[0].label) + '   距离为 ' + str(item[1])
        # 用距离的倒数当做权重试试,避免因样本数不同造成误差
        # 权重越大
        # labels_weight = {变量: [个数， 距离]}
        if item[0].label in labels_weight:
            labels_weight[item[0].label][0] += 1
            labels_weight[item[0].label][1] += item[1]
        else:
            labels_weight[item[0].label] = [1, item[1]]

    return labels_weight


def get_middle_node(data, labels):
    if len(data) == 0 or data is None:
        return None
    variance_numpy = math_calculation.get_variance(data=data)
    variance_max = np.max(variance_numpy)
    variance_max_index = np.where(variance_max == variance_numpy)[0][0]
    # print '维度划分 ' + str(variance_max_index)
    split_dimension = variance_max_index
    '''
    # 这一但用numpy转pandas排序就会出错，相当奇怪，实验并没有错，但到实际程序却出错
    '''
    # return sorted_data[row_midst], sorted_labels[row_midst], split_dimension, row_midst, sorted_data, sorted_labels
    # pandas_data = pd.DataFrame(data=data, index=labels, columns=[x for x in range(1024)])
    # sorted_pandas_data = pandas_data.sort_values(by=split_dimension)
    # data = np.array(sorted_pandas_data)
    # print sorted_pandas_data.index.tolist()
    # -------------------------------------------------
    data_label_tuple_list = []
    for data, label in zip(data, labels):
        data_label_tuple_list.append((data, label))
    # data = list(data)
    # data_ = data_label_tuple_list
    data_label_tuple_list.sort(key=lambda x: x[0][split_dimension])
    sorted_data = []
    for i in data_label_tuple_list:
        sorted_data.append(i[0])
    row_midst = len(sorted_data) // 2
    sorted_data = np.array(sorted_data)
    # print sorted_data
    sorted_labels = []
    for i in data_label_tuple_list:
        sorted_labels.append(i[1])
    # print sorted_labels
    # 为了排序labels
    # data_labels_pandas = pd.DataFrame(data=data, index=labels)
    # sorted_data_labels_pandas = data_labels_pandas.sort_values([split_dimension])
    # print sorted_data_labels_pandas[split_dimension]
    # sorted_labels = sorted_data_labels_pandas.index.tolist()
    # print sorted_labels
    # print '行 分裂之   ' + str(row_midst)
    # labels
    # return data[row_midst], None, split_dimension, row_midst, data, labels
    # ================================================
    return sorted_data[row_midst], sorted_labels[row_midst], split_dimension, row_midst, sorted_data, sorted_labels


class Node(object):
    def __init__(self, value, label=None, split_dimension=None, left_child_node=None, right_child_node=None):
        self.value = value
        # label为node对应的数字，是1还是0
        self.label = label
        # split_dimension为分裂维度
        self.split_dimension = split_dimension
        self.left_child_node = left_child_node
        self.right_child_node = right_child_node

    def __str__(self):
        # return '数字为：%s, 节点值：%s, 分裂维度：%s,\n左子节点为：%s,\n右子节点为：%s,\n' % (
        #     self.label, self.value, self.split_dimension, self.left_child_node, self.right_child_node)
        return '数字为：%s, 节点值：%s, 分裂维度：%s' % (
            self.label, self.value, self.split_dimension)


# 这个二叉树kd_tree不能写成是满二叉树，
# 写成满二叉树？
class KDTree(object):
    # kd_tree的根节点
    __kd_tree_root = None
    # 搜索路径，储存node
    __search_path = []
    # 根据k值变换长度的字典，用于储存选中的k个node对应的labels
    __kd_tree_node_distance = {}
    __distance_max_value_list = []

    # 根节点
    @classmethod
    def add_root_into_kd_tree(cls, data, labels):
        cls.__kd_tree_root = cls.generate_node(data, labels)

    # 最大值这块是借鉴的网上的，因为判断出错率有点高（因为点太少）
    # 初始化最大值，返回一个list类型
    @classmethod
    def initializez_distance_max_value_list(cls, k=11):
        cls.__distance_max_value_list = [float('inf')] * k
        # cls.__distance_max_value_list = [float(999999999999999999999999999999)] * k

    @classmethod
    def initializez_search_path(cls):
        cls.__search_path = []

    @classmethod
    def initializez_kd_tree_node_distance(cls):
        cls.__kd_tree_node_distance = {}

    # 将新添加进的距离进行排序，升序排序
    @classmethod
    def sort_distance_max_value_list(cls, k=11):
        cls.__distance_max_value_list.sort()
        # print cls.__distance_max_value_list
        cls.__distance_max_value_list = cls.__distance_max_value_list[:k]
        # print 'sort distance max value lsit 为 '
        # print cls.__distance_max_value_list

    # 递归后者循环，创建kd
    @classmethod
    def generate_node(cls, data, labels):
        # 如果训练集长度为0 且 node父节点（中间节点）为None，退出循环
        # print 'data %d' % len(training_data)
        if len(data) == 0 or data is None:
            # print 'training_data 长度不够'
            return None
        # return median, middle_label, split_dimension, data_line_divisor, data, labels
        median, middle_label, split_dimension, data_line_divisor, sorted_data, sorted_labels = get_middle_node(data,
                                                                                                               labels)
        return Node(median, middle_label, split_dimension,
                    left_child_node=cls.generate_node(sorted_data[:data_line_divisor],
                                                      sorted_labels[:data_line_divisor]),
                    right_child_node=cls.generate_node(sorted_data[data_line_divisor + 1:],
                                                       sorted_labels[data_line_divisor + 1:]),
                    )

    # test ok
    # 先序遍历
    @classmethod
    def preorder_traversal(cls, node=None):
        if node is None:
            # print 'kd tree 为空！'
            return None
        # print '======================'
        print 'node 为 ' + str(node)
        # print '======================'
        if node.left_child_node is not None:
            cls.preorder_traversal(node=node.left_child_node)
        if node.right_child_node is not None:
            cls.preorder_traversal(node=node.right_child_node)

    '''
        搜索路径这块！！！！！！！！！！可能有问题
    '''

    # 构建搜索路径
    @classmethod
    def get_search_path(cls, unclassified_element, node=None):
        if cls.__kd_tree_root is None:
            print 'kd tree为空，无法进行数字识别！\n请添加训练集！'
            return None

        # 若node为空，则为第一次查询，指定从根节点遍历
        node = cls.__kd_tree_root if node is None else node

        print '搜索路径为 ' + str(node)

        # 将经过的node添加进__search_path，构建搜索路径
        cls.__search_path.append(node)

        # for item in cls.__search_path:
        #     print '搜索路径为 ' + str(item)

        # 获取分裂维度
        split_dimension = node.split_dimension

        # 指定分裂维度上，若<=，进入左子树分支，若>,进入右子树分支
        # 既然搜索路径中已经有了node， 所以就无需返回node
        if unclassified_element[split_dimension] <= node.value[split_dimension] and node.left_child_node is not None:
            cls.get_search_path(unclassified_element, node=node.left_child_node)

        elif unclassified_element[split_dimension] > node.value[split_dimension] and node.right_child_node is not None:
            cls.get_search_path(unclassified_element, node=node.right_child_node)

        else:
            return None

    # 回溯搜索路径，查找最近邻相似点
    # 若是现在这个node比最近邻相似点距离element近，那这个点就变成了最近邻点
    @classmethod
    def backtracking(cls, unclassified_element, nearest_node=None, k=11, flag=True):
        '''

        :param unclassified_element:
        :param nearest_node:
        :param flag:用于标志是否为第一次访问
        :return:
        '''
        # 如果为0，就全部初始化，清空，免得对下一次结果造成误差
        if len(cls.__search_path) == 0:
            # cls.initializez_distance_max_value_list(k=k)
            # cls.__kd_tree_root = None
            # cls.__kd_tree_node_distance = {}
            return None

        if nearest_node is None and flag is False:
            return None
        # 每次一开始先对distance_max_value_list进行排序

        # print '一开始就进行排序喽！！！！！！！！！！！！！！！！！！'
        KDTree.sort_distance_max_value_list(k=k)

        current_node_from_search_path = cls.__search_path.pop(-1)

        print '回溯路径为 ' + str(current_node_from_search_path)
        distance_between_unclassified_current = math_calculation.get_distance(unclassified_element,
                                                                              current_node_from_search_path)
        # print '回溯距离 为 ' + str(distance_between_unclassified_current)
        if flag is True and nearest_node is None:
            nearest_node = current_node_from_search_path
        # print '最大值为'
        # print np.max(cls.__distance_max_value_list)

        if distance_between_unclassified_current <= max(cls.__distance_max_value_list):
            # 如果小于distance_max_value_list中的最大值
            # 就将这个 现在这个节点和距离放进__kd_tree_node_distance = {},__distance_max_value_list = []这两个里面
            cls.__distance_max_value_list.append(distance_between_unclassified_current)
            cls.__kd_tree_node_distance[current_node_from_search_path] = distance_between_unclassified_current

            # 获取分裂维度
            split_dimension = current_node_from_search_path.split_dimension

            if distance_between_unclassified_current <= min(cls.__distance_max_value_list):
                # current_node 变为 nearest_node
                nearest_node = current_node_from_search_path

                if unclassified_element[split_dimension] <= current_node_from_search_path.value[split_dimension]:
                    if current_node_from_search_path.right_child_node is not None:
                        cls.get_search_path(unclassified_element=unclassified_element,
                                            node=current_node_from_search_path.right_child_node)
                elif unclassified_element[split_dimension] > current_node_from_search_path.value[split_dimension]:
                    if current_node_from_search_path.left_child_node is not None:
                        cls.get_search_path(unclassified_element=unclassified_element,
                                            node=current_node_from_search_path.left_child_node)

        cls.backtracking(unclassified_element=unclassified_element, nearest_node=nearest_node, flag=False, k=k)

    # 返回kd tree, 列表类型
    @staticmethod
    def return_kd_tree_root():
        return KDTree.__kd_tree_root

    # 返回搜索路径
    @staticmethod
    def return_search_path():
        return KDTree.__search_path

    # 返回长度列表
    @staticmethod
    def return_kd_tree_node_distance():
        return KDTree.__kd_tree_node_distance


# test ok
if __name__ == "__main__":
    # training_data = np.array([[2, 3], [2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    # training_data = np.array([[0, 0], [1, 0], [1, 1], [4, 4], [4, 5], [7, 4]])
    training_data = np.array([[4, 1, 3, 5], [3, 6, 5, 7], [5, 2, 6.5, 5], [4.8, 4.2, 5, 8], [1, 1, 8, 6], [1, 6, 5, 3],
                              [4.1, 3.7, 2, 5], [4.7, 4.1, 5, 9], [2, 4, 6, 8.7]])
    training_labels = ['a', 'a', 'a', 'b', 'a', 'b', 'b', 'a', 'b']
    # training_labels = ['a', 'a', 'a', 'b', 'a', 'b', 'b']

    KDTree.add_root_into_kd_tree(training_data, training_labels)
    KDTree.preorder_traversal(KDTree.return_kd_tree_root())
    # 查找搜索路径
    # KDTree.get_search_path(unclassified_element=[0, 0])
    KDTree.get_search_path(unclassified_element=[4.8, 3.8, 2, 4])
    # 返回搜索路径
    search_path = KDTree.return_search_path()
    # for item in search_path:
    #     print '搜索路径为 ' + str(item)
    # 回溯搜索路径，查找到最近点，并添加到nearest_code_distance
    KDTree.backtracking(unclassified_element=[4.8, 3.8, 2, 4])
    print 'kd tree node is ' + str(KDTree.return_kd_tree_node_distance())
    sorted_nodes_distance = sort_nodes_distance(KDTree.return_kd_tree_node_distance())
    for item in sorted_nodes_distance:
        print item[0], item[1]
        # print labels_count
        # print KDTree.return_search_path()

'''
            if distance < distance_between_unclassified_current:

                cls.__kd_tree_node_distance[current_node_from_search_path] = distance_between_unclassified_current

                cls.backtracking(unclassified_element=unclassified_element,
                                 nearest_node=nearest_node, flag=False)

            elif distance >= distance_between_unclassified_current:
                # 添加进kd_tree_node_distance
                # print '现在最近点更改为 ' + str(current_node_from_search_path)
                cls.__kd_tree_node_distance[current_node_from_search_path] = distance_between_unclassified_current

                # 现在的点变为最近邻点
                # nearest_node = current_node_from_search_path

                # 判断是进左子空间还是右子空间
                split_dimension = current_node_from_search_path.split_dimension

                # <=,>一修改，正确率就提高了
                if unclassified_element[split_dimension] <= current_node_from_search_path.value[split_dimension]:
                    if current_node_from_search_path.right_child_node is not None:
                        # 访问右子空间
                        cls.get_search_path(unclassified_element=unclassified_element,
                                            node=current_node_from_search_path.right_child_node)
                        # cls.backtracking(unclassified_element=unclassified_element,
                        #                  nearest_node=current_node_from_search_path.right_child_node, flag=False)
                elif unclassified_element[split_dimension] > current_node_from_search_path.value[split_dimension]:
                    if current_node_from_search_path.left_child_node is not None:
                        cls.get_search_path(unclassified_element=unclassified_element,
                                            node=current_node_from_search_path.left_child_node)

                cls.backtracking(unclassified_element=unclassified_element,
                                 nearest_node=current_node_from_search_path, flag=False)
'''
