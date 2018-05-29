# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 19:51
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : smo.py
# @Software: PyCharm
import random

import numpy as np

import vector_data


def get_data_categories(training_data):
    '''
    获取statistics种类，其实也就是-1,1
    :param training_data:训练数据
    :return: 训练数据数据的所有类型，-1,1， 返回类型 numpy.ndarray
    '''
    # category = training_data.ix[:, -1]
    category = training_data[:, -1]
    return np.unique(category)


def gram(training_data_exclude_category):
    '''
    :param training_data_exclude_category:训练数据集,不包含category
    :return:
    '''
    return np.dot(training_data_exclude_category, training_data_exclude_category.T)


def kernel_function(training_data_exclude_category, kernel_category='linear'):
    '''

    :param training_data_exclude_category:不包含category的完整training_data
    :param kernel_category: 核函数的类型，线性核函数，高斯核函数,默认线性核函数
    :return:
    '''
    if kernel_category == "linear":
        return gram(training_data_exclude_category=training_data_exclude_category)
    elif kernel_category == "rbf":
        training_data_exclude_category_shape = training_data_exclude_category.shape[0]
        Kernel = np.zeros((training_data_exclude_category_shape, training_data_exclude_category_shape))
        for i in range(training_data_exclude_category_shape):
            for j in range(training_data_exclude_category_shape):
                tmp_value = training_data_exclude_category[i] - training_data_exclude_category[j]
                # value = abs(training_data_exclude_category[i] - training_data_exclude_category[j]) ** 2 * (-1)
                value = tmp_value.dot(tmp_value.T) ** 2 * (-1)
                # 前面那个调成-1，所以为正
                Kernel[i][j] = np.exp(value)
        return Kernel


class SMO(object):
    def __init__(self, training_data, training_data_category, C, accuracy, max_iteration=10, decline=0.01,
                 kernel_category="linear"):
        '''
        初始化
        :param training_data:训练数据
        :param training_data_category:训练数据所有的类，unique
        :param C: 惩罚参数
        :param accuracy:松弛变量
        :param max_iteration: 最大迭代次数，就是训练数据集的行数
        :param decline:下降
        :param kernel_category:核函数类型，程序中仅仅写个线性核函数，所以可有可无
        '''
        self.training_data = training_data
        # 训练数据集的行数，列数，包括其所属类
        self.number_of_rows, self.number_of_columns = training_data.shape
        # 仅仅是数据特征的列数
        self.number_of_columns -= 1
        self.training_data_category = training_data_category
        # 惩罚参数
        self.C = C
        # 精确度
        self.accuracy = accuracy
        # 最大迭代次数，采用这个的原因：因为alpha2变量是=随机选取的，争取每个点都能选一次，减小误差
        self.max_iteration = max_iteration
        # 下降
        self.decline = decline
        # 初始化进行了第几次迭代，先试试看，感觉可能呢有问题
        self.iteration_flag = 0

        # 1-精确度，或者1+精确度
        self.margin_substract_accuracy_small = 1 - self.accuracy
        self.margin_substract_accuracy_big = 1 + self.accuracy

        # 核函数
        self.Kernel = kernel_function(training_data_exclude_category=training_data[:, :-1],
                                      kernel_category=kernel_category)
        # 误差列表
        self.PREDICT_ERROR_LIST = np.zeros(self.number_of_rows)
        # 初始化w
        self.original_w = np.zeros(self.number_of_columns)
        # 初始化b
        self.original_b = 0
        # 初始化alpha
        self.original_alpha = np.zeros(self.number_of_rows)
        # print self.Kernel
        self.support_vector_index = []

    def calculate_predicted_error(self, authentic_category, predicted_category, index=-1):
        '''

        :param authentic_category:
        :param predicted_category:
        :param index:
        :return:
        '''
        # 预测与实际误差
        '''
            不加绝对值
            注：
                预测差值一定是 预测值-真实值
        '''
        predict_error = predicted_category - authentic_category
        if index != -1:
            self.PREDICT_ERROR_LIST[index] = predict_error
        return predict_error

    def predicted_category(self, row_index):
        '''

        对选出的alpha1，alpha2所对应的变量预测其分类，
        :param row_index:选中的
        :return:
        '''
        training_data_selected = self.training_data[row_index]
        # 选择的alpha变量
        alpha_selected = float(self.original_alpha[row_index])
        predicted_category_of_selected_training_data = 0
        category_selected = training_data_selected[-1]
        for index, item_from_training_data in enumerate(self.training_data):
            # alpha乘以变量对应的y
            # alpha_multiply_category = np.multiply(self.original_alpha[index], item_from_training_data[-1])
            alpha_multiply_category = self.original_alpha[index] * item_from_training_data[-1]
            '''
                直接从kernel中得到结果，根据i
            '''
            # 选中的结果
            predicted_category_of_selected_training_data += alpha_multiply_category * self.Kernel[row_index, index]
        # 这就是书上说的g(xi)，即变量对应的预测类
        # print 'alpha_selected 为' + str(alpha_selected)
        predicted_category_of_selected_training_data += self.original_b
        return training_data_selected[
               :-1], category_selected, alpha_selected, predicted_category_of_selected_training_data

    def judge_satisfy_kkt_condition(self, alpha, authentic_category, predicted_category):
        '''
            判断是否满足kkt条件
        :param alpha:
        :param authentic_category:
        :param predicted_category:
        :return:
        '''
        # 预测与实际误差
        # predict_error = self.calculate_predicted_error(authentic_category=authentic_category,
        #                                                predicted_category=predicted_category)
        if (alpha > 0) and (alpha < self.C) and (authentic_category * predicted_category <= 1 + self.accuracy) and (
                        authentic_category * predicted_category >= 1 - self.accuracy):
            # 正好位于间隔边界上，即支持向量
            return True
        elif (alpha == 0) and (authentic_category * predicted_category >= 1 - self.accuracy):
            return True
        elif (alpha == self.C) and (authentic_category * predicted_category <= 1 + self.accuracy) and (
                        authentic_category * predicted_category > 0):
            return True
        else:
            return False

    def outer_loop(self):
        # 因为初始化alpha全部为0，所以就相当于直接遍历整个训练集，检验是否满足偏离条件
        '''
        外层循环，选择第一个变量
        :return:
        '''
        # 第几次遍历
        iteration = 0
        # 标志是第一次遍历，即alpha 都被初始化为0
        loop_flag = True

        # 只要alpha没有被改变alpha_changed_flag就为False，而且训练数据被遍历，loop_flag就为False，
        # 意指：所有变量的节都满足此最优化问题的KKT条件
        while (iteration < self.max_iteration) or loop_flag:
            # 标志alpha是否被修改过，与loop_flag一起使用
            # alpha_changed_flag = False
            # 还是照着机器学习那本书写吧
            alpha_changed_flag = 0
            # print '==================='
            # print self.original_alpha
            # 是否为第一次遍历
            if loop_flag:
                alpha_changed_flag += self.calculate_b(range(self.training_data.shape[0]))
                # loop_flag = False
                iteration += 1
            else:
                # 遍历间隔边界上的点，即支持向量
                # 这个支持向量每循环一次，差不多就变一次，所以多增加几个
                self.support_vector_index = np.where((self.original_alpha > 0) & (self.original_alpha < self.C))[0]
                alpha_changed_flag += self.calculate_b(self.support_vector_index)
                iteration += 1

            # print '第%d次遍历， alpha更改次数%d' % (iteration, alpha_changed_flag)
            if loop_flag:
                loop_flag = False
            elif alpha_changed_flag == 0:
                # 如果所有alpha都不改变，将全部数据遍历一次
                # print '将整个数据集遍历一次'
                loop_flag = True
            # print '支持向量共有 ' + str(self.support_vector_index)

    def calculate_b(self, indexes):
        flag = 0
        for first_alpha_index in indexes:
            # 获取training_data, category, alpha, predict_category
            first_training_data, first_category, first_alpha, first_predict_category = self.predicted_category(
                row_index=first_alpha_index)
            # 如果满足，返回True
            kkt_flag = self.judge_satisfy_kkt_condition(alpha=first_alpha, authentic_category=first_category,
                                                        predicted_category=first_predict_category)
            if kkt_flag:
                # 满足，就不需要对其进行修改！
                flag += 0
                continue
                # return True
            else:
                # print '不满足条件'
                # 选取alpha2
                second_alpha_index = self.inner_loop(first_alpha_index=first_alpha_index)
                if second_alpha_index is None:
                    # print "下降度不足，更换第一个点！！！！！！"
                    flag += 0
                    continue

                second_training_data, second_category, second_alpha, second_predict_category = self.predicted_category(
                    row_index=second_alpha_index)
                # 预测误差
                first_predict_error = self.calculate_predicted_error(first_category, first_predict_category,
                                                                     index=first_alpha_index)
                second_predict_error = self.calculate_predicted_error(second_category, second_predict_category,
                                                                      index=second_alpha_index)
                if first_category == second_category:
                    L = max(0.0, second_alpha + first_alpha - self.C)
                    H = min(self.C, second_alpha + first_alpha)
                else:
                    L = max(0.0, second_alpha - first_alpha)
                    H = min(self.C, self.C + second_alpha - first_alpha)

                # 分母
                K11 = self.Kernel[first_alpha_index, first_alpha_index]
                K12 = self.Kernel[first_alpha_index, second_alpha_index]
                K21 = self.Kernel[second_alpha_index, first_alpha_index]
                K22 = self.Kernel[second_alpha_index, second_alpha_index]

                denominator_K = K11 + K22 - 2 * K12

                second_alpha_middle = second_alpha + second_category * (
                    first_predict_error - second_predict_error) / denominator_K

                second_alpha_result = self.estimate_alpha(second_alpha_middle, L, H)

                first_alpha_result = first_alpha + first_category * second_category * (
                    second_alpha - second_alpha_result)

                # alpha更改的标志
                if first_alpha_result == first_alpha and second_alpha_result == second_alpha:
                    flag += 0
                    continue
                else:
                    flag += 1

                # 更新alpha
                self.original_alpha[first_alpha_index], self.original_alpha[
                    second_alpha_index] = first_alpha_result, second_alpha_result

                # print '更新alpha'
                # print '第一个alpha ' + str(first_alpha_index) + '  ' + str(first_alpha_result)
                # print '第二个alpha ' + str(second_alpha_index) + '  ' + str(second_alpha_result)

                first_b_new = (-1) * first_predict_error - first_category * K11 * (
                    first_alpha_result - first_alpha) - second_alpha * K21 * (
                    second_alpha_result - second_alpha) + self.original_b
                second_b_new = (-1) * second_predict_error - first_alpha * K12 * (
                    first_alpha_result - first_alpha) - second_alpha * K22 * (
                    second_alpha_result - second_alpha) + self.original_b

                if (first_alpha_result > 0) and (first_alpha_result < self.C):
                    self.original_b = first_b_new
                elif (second_alpha_result > 0) and (second_alpha_result < self.C):
                    self.original_b = second_b_new
                else:
                    self.original_b = (first_b_new + second_b_new) / 2.0
        return flag

    def get_final_alpha_b(self):
        '''
            获取最终的alpha，b
        :return:
        '''
        return self.original_alpha, self.original_b

    def inner_loop(self, first_alpha_index):
        '''
        内层循环，选择第二个变量
        :param iteration_flag: 第一个变量alpha对应的值
        :return:返回选取的变量的下标
        '''
        first_predicted_error = self.PREDICT_ERROR_LIST[first_alpha_index]
        if (self.PREDICT_ERROR_LIST == 0).all():
            # 如果误差列表所有误差都为0，随机选取一个下标不一样的
            # 选取下标不一样的
            while True:
                alpha_second_index = random.randint(0, self.number_of_rows - 1)
                if alpha_second_index != first_alpha_index:
                    return alpha_second_index
                else:
                    continue
        else:
            # 如果first_predicted_error为正数，选取最小的误差
            # 记录遍历次数，如果超过self.alpha的大小，就gg
            count = 0
            while True:
                count += 1
                alpha_second_index = list(self.PREDICT_ERROR_LIST).index(
                    min(self.PREDICT_ERROR_LIST)) if first_predicted_error > 0 else list(self.PREDICT_ERROR_LIST).index(
                    max(self.PREDICT_ERROR_LIST))
                if alpha_second_index != first_alpha_index:
                    difference = abs(first_predicted_error - self.PREDICT_ERROR_LIST[alpha_second_index])
                    # 如果下降不足，因为已经差值已经是最大的了，所以直接放弃第一个alpha，重新获取第一个alpha
                    if difference >= 0.0001:
                        return alpha_second_index
                    else:
                        # 足够的下降度
                        # print '下降度不足，更换节点！'
                        return None
                if count == self.original_alpha.shape[0]:
                    return None

    def estimate_alpha(self, alpha_middle, L, H):
        if alpha_middle > H:
            return H
        elif (alpha_middle <= H) and (alpha_middle >= L):
            return alpha_middle
        else:
            return L

    # abandon
    def get_K(self, first_training_data, second_training_data):
        '''
             获取K11+K22-2K12,当分母
        :param selected_training_data:包含y
        :return:K11，K12,K22
        '''
        return np.dot(first_training_data[:-1], second_training_data[:-1])

    def calculate_w(self):
        '''
        计算w
        :return:
        '''
        # alpha此时已经被计算完毕
        # self.original_alpha
        alpha_not_zero_index = np.where(self.original_alpha != 0)[0]
        # print self.original_alpha
        # print self.original_w.shape
        # print alpha_not_zero_index
        for index in alpha_not_zero_index:
            alpha_multiply_category = self.original_alpha[index] * self.training_data[index, -1]
            alpha_multiply_category_x = alpha_multiply_category * self.training_data[index][:-1]
            self.original_w += alpha_multiply_category_x

        return self.original_w

    # 测试线性kernel
    # def estimate_linear_kernel_test_data_category(test_data, w, b):
    #     all_data_count = test_data.shape[0]
    #     error_count = 0
    #     for item in test_data:
    #         data = item[:-1]
    #         category = item[-1]
    #
    #         predicted_category = np.dot(w, data) + b
    #         if (predicted_category > 0 and category == 1) or (predicted_category < 0 and category == -1):
    #             pass
    #         else:
    #             error_count += 1
    #         print "预测类为 ：" + str(predicted_category) + ' , ' + '实际类为 ：' + str(category)
    #     error_probability = error_count / (all_data_count * 1.0)
    #     print '错误率为 ：' + str(error_probability)


    # 测试rbf高斯核函数
    def estimate_rbf_kernel_category(self, test_data):
        # 测试高斯核函数
        all_data_count = test_data.shape[0]

        error_count = 0

        for j in range(self.number_of_rows):
            predicted_category = 0
            # authentic_category = self.training_data[j:, -1]
            for i in self.support_vector_index:
                support_alpha_multiply_category = self.original_alpha[i] * self.training_data[j, -1]
                predicted_category += support_alpha_multiply_category * self.Kernel[i][j]
            predicted_category += self.original_b
            # print '============================'
            # 仅支持向量
            # 反正测试数据与训练数据相同，所以直接用self.Kernel
            if np.sign(predicted_category) != np.sign(self.training_data[j, -1]):
                error_count += 1
            # print "预测类为 ：" + str(predicted_category) + ' , ' + '实际类为 ：' + str(self.training_data[j, -1])

        error_probability = error_count / (all_data_count * 1.0)
        print '错误率为 ：' + str(error_probability)

    def estimate_linear_kernel_category(self, test_data):
        # 测试线性核函数
        all_data_count = test_data.shape[0]
        error_count = 0
        for item in test_data:
            data = item[:-1]
            category = item[-1]

            predicted_category = np.dot(self.original_w, data) + self.original_b
            if np.sign(predicted_category) != np.sign(item[-1]):
                error_count += 1
            print "预测类为 ：" + str(predicted_category) + ' , ' + '实际类为 ：' + str(category)

        error_probability = error_count / (all_data_count * 1.0)
        print '错误率为 ：' + str(error_probability)

    def estimate_category(self, test_data, kernel):
        '''
        根据kernel调用对用的方法
        :param test_data:
        :param kernel:
        :return:
        '''
        if kernel == 'rbf':
            self.estimate_rbf_kernel_category(test_data=test_data)
        elif kernel == 'linear':
            self.estimate_linear_kernel_category(test_data=test_data)


def main():
    # 获取数据
    training_data = vector_data.get_training_data()
    # 获取数据所有类别，即1，-1
    training_data_category = get_data_categories(training_data)
    '''
        max_iteration：最大迭代次数，与training_data有多少行，多少列无关
    '''
    # kernel_category = 'linear'
    kernel_category = 'rbf'
    smo_instance = SMO(training_data=training_data, training_data_category=training_data_category, C=6,
                       accuracy=0.001,
                       max_iteration=100, decline=0.01, kernel_category=kernel_category)
    # smo_instance = SMO(training_data=training_data, training_data_category=training_data_category, C=6, accuracy=0.001,
    #                    max_iteration=100, decline=0.01, kernel_category='rbf')
    # 选取第一个alpha
    smo_instance.outer_loop()
    final_alpha, final_b = smo_instance.get_final_alpha_b()
    # print final_alpha

    w = smo_instance.calculate_w()
    # print w
    # print final_b
    # 实例化SMO时，不想再写一个方法，直接测试看看，程序应该有错
    smo_instance.estimate_category(test_data=training_data, kernel=kernel_category)


if __name__ == "__main__":
    while True:
        main()
