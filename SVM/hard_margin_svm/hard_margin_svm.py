# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 16:05
# @Author  : affectalways
# @Site    : 
# @Contact : affectalways@gmail.com
# @File    : soft_margin_svm.py
# @Software: PyCharm
import vector_data
import numpy as np
import sympy
import random
import copy


# 是否满足约束条件，即alpha i >=0
def satisfy_the_constraint(alpha_result):
    # 用这个判断是否都为True，若都未True，all（）返回True，否则返回False
    # print (alpha_result_values >= 0).all()
    alpha_result_values = np.array(alpha_result.values())
    # 不满足约束，返回False
    if not (alpha_result_values >= 0).all():
        return False
    else:
        # 满足约束条件，返回True
        return True


# 调用方法，查看是否满足所有alpha>=0,若不满足就不是极值点，在边界上取极值
# 检查是否为极值点，必须满足所有alpha>=0
# 若不满足，边界取极值点
# 对应的是一个alpha被其与alpha代替所求的最小值点，并不是所有的
def is_the_extreme_point(alpha_result, alpha_copy, partial_derivative_dict, subject_to, dual_problem):
    if not satisfy_the_constraint(alpha_result):
        # 不是极值点，所以在边界取
        print str(alpha_result) + '不是极值点'
        alpha_point, min_value = boundary_take_the_extreme_point(alpha_result.keys(), alpha_copy,
                                                                 partial_derivative_dict, subject_to, dual_problem)
    else:
        print str(alpha_result) + '是极值点'
        # 带入solve_for_the_equation
        alpha_point, min_value = solve_for_the_equation(dual_problem, [alpha_result])
        # 该点是极值点，返回
        # return alpha_result,
    return alpha_point, min_value


# 边界取极值点，直接带入偏导alpha=0带入alpha2的偏导
def boundary_take_the_extreme_point(alpha_result_key, alpha_copy, partial_derivative_dict, subject_to, dual_problem):
    alpha_extreme_point_list = []

    # 因为是硬间隔，必须满足条件alpha>=0,在边界，就是每个alpha取0，然后求就行了
    for key in alpha_copy:
        # print 'key wei'
        # print key
        # key对应的极值点
        partical_derivative_dict_copy = copy.deepcopy(partial_derivative_dict)
        # 删除这个键对应的偏导数
        del partical_derivative_dict_copy[key]
        key_corresponding_extreme_point = solve_alpha(partical_derivative_dict_copy, alpha_result_key, key)
        # print key_corresponding_extreme_point
        # 求出每个alpha的值
        alpha_value_dict = st_alpha_value(key_corresponding_extreme_point, alpha_result_key, subject_to)
        # 判断这个极值点是否都>0
        flag = satisfy_the_constraint(alpha_value_dict)
        if flag is True:
            # 若满足约束条件， 即alpha都>=0
            alpha_extreme_point_list.append(alpha_value_dict)
    # 将alpha带入方程，求方程的值
    alpha_point, min_value = solve_for_the_equation(dual_problem, alpha_extreme_point_list)
    return alpha_point, min_value


# 将alpha带入方程，求方程的值
def solve_for_the_equation(dual_problem, alpha_extreme_point_list):
    # 开始带入dual_problem, 求解
    min_equation_value = None
    alpha_point = None
    for alpha_extreme_point in alpha_extreme_point_list:
        value = dual_problem.subs(alpha_extreme_point)
        # print str(alpha_extreme_point) + ' ' + str(value)
        if min_equation_value is None or min_equation_value > value:
            min_equation_value = value
            alpha_point = alpha_extreme_point

    # 选取最小的value
    # print min_equation_value
    return alpha_point, min_equation_value


# 根据约束求每个alpha的值
# alpha_result_key必须是完整的alpha
def st_alpha_value(key_corresponding_extreme_point, alpha_result_key, subject_to):
    # 等式
    equation_list = []
    for key, value in key_corresponding_extreme_point.items():
        equation_list.append(sympy.Eq(key, value))
        # str_expr = str(key) + " - " + str(value)
        # equation_list.append(sympy.sympify(str_expr))
    equation_list.append(subject_to)
    # 返回dict
    return sympy.solve(equation_list, alpha_result_key)


# 将其中一个alpha用其他alpha代替
def replace_alpha_by_others(alpha_copy, alpha_element, dual_problem, subject_to):
    # alpha_element = random.choice(alpha)
    # 移除
    alpha_copy.remove(alpha_element)
    # 被移除的alpha用其他的alpha代替
    random_alpha_replace_by_others = sympy.solve(sympy.Eq(subject_to, 0), alpha_element)[0]
    # 用random_alpha_replace_by_others替换对偶问题中的random_alpha
    dual_problem = dual_problem.replace(alpha_element, random_alpha_replace_by_others)
    return dual_problem, alpha_copy


# 求偏导
def get_partial_derivative(dual_problem, alpha):
    # 保存求偏导之后的方程
    partial_derivative_dict = {}
    for element in alpha:
        partial_derivative_dict[element] = sympy.diff(dual_problem, element)
    return partial_derivative_dict


# 利用sympy.nonlinsolve，求解alpha
# args约束条件
def solve_alpha(partial_derivative_dict, alpha, *args):
    partial_derivative_list = []
    partial_derivative_list.extend(partial_derivative_dict.values())
    # 将约束条件也加进来，当然不包含alpha>=0
    partial_derivative_list.extend(args)
    # 字典类型
    return sympy.solve(partial_derivative_list, alpha)


# 获取对偶问题的约束条件
def get_dual_problem_st(training_data, alpha):
    count = alpha.shape[0]
    subject_to = 0
    for i in range(count):
        subject_to += training_data[i, -1] * alpha[i]
    return subject_to


# 获取拉格朗日对偶公式
# 获取对偶公式
# shape写死了，很不好
def get_dual_problem(training_data, alpha):
    # 数据有多少行
    row_count = training_data.shape[0]
    dual_problem = 0
    # 两层循环，对偶问题，ij
    for i in range(row_count):
        alphai = alpha[i]
        yi = training_data[i, -1]
        xi = training_data[i, :-1].T
        for j in range(row_count):
            dual_problem += alphai * alpha[j] * yi * training_data[j, -1] * np.dot(xi, training_data[j, :-1]) / 2

    dual_problem = dual_problem

    for i in range(row_count):
        dual_problem -= alpha[i]

    # 返回对偶问题公式
    # print dual_problem
    return dual_problem


# 初始化alpha
def initialize_alpha(training_data):
    # w的个数，跟有多少数据对应对应
    w_count = training_data.shape[0]
    alpha = []
    for i in range(w_count):
        element = 'alpha' + str(i)
        alpha.append(sympy.Symbol(element))
    return np.array(alpha)


# 计算w
# 完整的alpha，取值
# 训练数据的x，y
def calculate_w(training_data, alpha_values):
    # w = np.array(training_data.shape[0])
    w = np.zeros(training_data[0][:-1].shape[0])
    # 遍历每行
    for index, element in enumerate(training_data):
        # w += alphai*xi*yi
        # <class 'sympy.core.numbers.Rational>,所以需要强制转换
        alphai_yi = float(alpha_values[index]) * element[-1]
        w += element[:-1] * alphai_yi
    return w


# 计算b
# 随机选取一个正交分量alpha>0的实例
def calculate_b(training_data, alpha_values):
    alpha_value_greater_than_zero_dict = {index: alpha for index, alpha in enumerate(alpha_values) if alpha > 0}
    # print alpha_value_greater_than_zero_dict
    # 随机选取字典的键
    alpha_random_value_index = random.choice(alpha_value_greater_than_zero_dict.keys())
    # 根据选择的字典的键选取训练数据中对应的x，y
    print alpha_values[alpha_random_value_index]
    selected_x_y = training_data[alpha_random_value_index]
    # b = yj
    b = selected_x_y[-1]

    for index, element in enumerate(training_data):
        # alpha i * y i * x i * x j
        b -= alpha_values[index] * training_data[index, -1] * training_data[index, :-1].dot(selected_x_y[:-1].T)
    return b


# 创建分离超平面
def create_seperate_hyperplane(w, b):
    seperate_hyperplane = ''
    for index, w_element in enumerate(w):
        x = "x" + str(index)
        # 添加加号
        if index != 0:
            add_sign = "+"
        else:
            add_sign = ''
        seperate_hyperplane += add_sign + str(w_element) + "*" + x
    if b < 0:
        seperate_hyperplane += str(b)
    elif b > 0:
        seperate_hyperplane += add_sign + str(b)
    print "分离超平面为 ：" + seperate_hyperplane
    # seperate_hyperplane = None
    # for index, w_element in enumerate(w[0]):
    #     x = "x" + str(index)
    #     if seperate_hyperplane is None:
    #         seperate_hyperplane = w_element * sympy.Symbol(x)
    #     else:
    #         seperate_hyperplane += w_element * sympy.Symbol(x)
    # seperate_hyperplane += b
    # print "分离超平面为 ：" + seperate_hyperplane


# 分类决策函数
def create_category_decision_function(w, b, x):
    if (w.dot(x) + b) > 0:
        return 1
    elif (w.dot(x) + b) < 0:
        return -1
    else:
        # =0的情况，正好位于分离超平面
        return 0


# 训练
def training():
    training_data = vector_data.get_training_data()
    # 创建alpha
    alpha = initialize_alpha(training_data)
    # 获取对偶问题
    dual_problem = get_dual_problem(training_data, alpha)
    # print type(dual_problem)
    # 获取对偶问题的约束条件
    subject_to = get_dual_problem_st(training_data, alpha)
    # 根据subject_to令其中一个alpha = 其余的alpha,
    # 直接带入alpha出错，怎么个情况
    # 随机从alpha中选取一个alpha
    # 将alpha搞定，替换一个
    # alpha_copy = list(copy.deepcopy(alpha))
    '''

        每个alpha都用其他的alpha表示
        原因：统计学习方法p108，这个式子本身就有问题，若是三者一起联立，求不出结果
        alpha3=alpha1+alpha2 极值点（1/4,0,1/4）
        alpha1=alpha3-alpha2极值点（2/13， 0， 2/13）
        alpha2=alpha3-alpha1极值点（0， 2/13， 2/13）
        不一样，最小值也不一样，gg
    '''
    # 调用方法，使其中一个alpha被其他alpha表示
    # 此时alpha为list类型
    # 循环遍历完整的alpha
    extreme_dict = {}
    for alpha_element in alpha:
        # 复制一个alpha， 在方法中修改，而又不影响原有的alpha
        alpha_copy = list(copy.deepcopy(alpha))
        dual_problem, alpha_copy = replace_alpha_by_others(alpha_copy, alpha_element, dual_problem, subject_to)

        # 求偏导，若alpha被替换，只对那些替换它的alpha进行偏导
        partial_derivative_dict = get_partial_derivative(dual_problem, alpha_copy)

        # 后面传入的参数只能是deepcopy
        # 求解alpha，是将所有的alpha求解偏导
        alpha_result = solve_alpha(partial_derivative_dict, tuple(alpha), subject_to)
        # 调用方法，查看是否满足所有alpha>=0,若不满足就不是极值点，在边界上取极值
        # alpha_copy是不完整的alpha，即alpha中有被替换的

        extreme_dict[alpha_element] = is_the_extreme_point(alpha_result, alpha_copy, partial_derivative_dict,
                                                           subject_to,
                                                           dual_problem)

    extreme_point_corresponding_key = min(extreme_dict, key=lambda x: extreme_dict[x][-1])
    # alpha取值， 对偶问题的最小值
    alpha_value, dual_problem_min_value = extreme_dict[extreme_point_corresponding_key]

    print '极值点为 ： ' + str(alpha_value)
    print '对偶问题最小值为 ：' + str(dual_problem_min_value)

    # 支持向量，随机选取一个正分量alpha>0的
    # 还是先判断alpha_value 是否全为0，若全为0，直接判断无法划分，因为b算不出来呦（没有支持向量）
    flag = np.array(alpha_value.values()).any() == 0
    if flag:
        print '///////////////'
        print '无支持向量，无法判断，数据有误！！！'
    # 计算w
    # 这个alpha_value在计算w之前需要排列的与training_data中的数据一样，但是之前求解alpha就是按这个顺序排列的
    # ，所以在此放过排序
    # alpha_value.values()作为参数，而不是alpha_value字典
    # 原因：不用担心顺序
    w = calculate_w(training_data, alpha_value.values())
    # 计算b
    b = calculate_b(training_data, alpha_value.values())
    # 创建分离超平面
    create_seperate_hyperplane(w, b)
    return w, b


# 测试
def test(w, b):
    test_data = vector_data.get_training_data()
    test_data = np.array([[0, 0, -1], [9, 3, 1], [4, 5, 1], [2, 2, 1]])
    print test_data
    print '===' * 20
    count = 0
    all_data_count = len(test_data)
    for element in test_data:
        y = element[-1]
        x = element[:-1]
        analytical_y = create_category_decision_function(w, b, x)
        print '数据点为 ' + str(element[:-1]) + ', ' + '类别为 : ' + str(y) + ', ' + '程序分析类别为：' + str(analytical_y)
        if analytical_y != y and analytical_y != 0:
            count += 1
    print '总数据量 ： ' + str(all_data_count) + ",  出错次数为：" + str(count) + ',  错误率为 ： ' + str(count / float(all_data_count))


    # 程序入口


def main():
    w, b = training()
    test(w, b)


if __name__ == "__main__":
    main()
