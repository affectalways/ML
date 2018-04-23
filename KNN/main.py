# -*- coding:UTF-8 -*-
from knn import KDTreeNode
from knn import read_file_content


# 程序入口
def program_entry(training_data_filename, unclassified_data_filename, k=8):
    # 训练数据集
    training_data, training_labels = read_file_content.generate_np_array_vector(training_data_filename)

    # training_labels = training_labels
    # print training_data

    # 获取待分类元素数据集
    unclassified_data, unclassified_labels = read_file_content.generate_np_array_vector(unclassified_data_filename)
    # unclassified_labels = unclassified_labels

    # 创建kd_tree
    KDTreeNode.KDTree.add_root_into_kd_tree(data=training_data, labels=training_labels)
    # 获取创建的kd_tree的根节点kd_tree_root
    # kd_tree_root = KDTreeNode.KDTree.return_kd_tree_root()
    # 遍历创建的tree
    # KDTreeNode.KDTree.preorder_traversal(kd_tree_root)

    # 计算错误率
    # 待测数据的总数量
    total_quantity = unclassified_data.shape[0]
    # print total_quantity
    # 错误数
    error_count = 0

    # unclassified_len = len(unclassified_labels)

    for i in range(total_quantity):

        # 距离初始化最大值(全都是无限大)
        KDTreeNode.KDTree.initializez_distance_max_value_list(k=k)
        KDTreeNode.KDTree.initializez_kd_tree_node_distance()
        KDTreeNode.KDTree.initializez_search_path()

        unclassified_element = unclassified_data[i]
        unclassified_label = unclassified_labels[i]
        # 创建搜索路径
        KDTreeNode.KDTree.get_search_path(unclassified_element=unclassified_element)

        # 回溯搜索路径
        KDTreeNode.KDTree.backtracking(unclassified_element=unclassified_element, k=k)

        # 找出最近点与其距离
        # for item, distance in KDTreeNode.KDTree.return_kd_tree_node_distance().items():
        #     print '最近距离 ' + str(item) + '   ' +  str(distance)
        sort_kd_tree_node_distance = KDTreeNode.sort_nodes_distance(KDTreeNode.KDTree.return_kd_tree_node_distance())
        if sort_kd_tree_node_distance is None:
            return None
        # print '最近点为 ' + str(sort_kd_tree_node_distance)

        # labels_weight = {label: [count, total_distance]}
        labels_weight = KDTreeNode.get_k_labels(sort_kd_tree_node_distance=sort_kd_tree_node_distance, k=k)

        un_labels = {}
        for key, value in labels_weight.items():
            item = value[1] / float(value[0])
            un_labels[key] = item

        un_labels = sorted(un_labels.items(), key=lambda x: x[1])
        print un_labels
        classified_label = un_labels[0][0]
        # 选取字典中最大值所对应的类型，也就是所属数字
        # classified_label = max(labels_weight.items(), key=lambda x: x[1])

        print '待测数据 实际类型类型为 %s， 实际测得类型为 %s ' % (unclassified_label, classified_label)
        if unclassified_label != classified_label:
            error_count += 1
    print '错误率为 %f' % (error_count / float(total_quantity))


if __name__ == '__main__':
    # k 16 最优
    program_entry('trainingDigits', 'testDigits', k=16)
    # data, tmp = read_file_content.generate_np_array_vector('trainingDigits')
