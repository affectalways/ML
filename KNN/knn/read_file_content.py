# -*- coding:UTF-8 -*-
import os
import numpy as np


# 将图像文本转成list，已经写死了，32*32==》1*1024
def image_transform_list(filename):
    result_vector = np.zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        file_line_content = f.readline()
        for j in range(32):
            result_vector[0, 32 * i + j] = int(file_line_content[j])
    return result_vector


# 将获取到的list转成np.array, 获取文件名（其实是获取数字作为对应的labels）
def generate_np_array_vector(file_path):
    # temp_file_content = []
    # 获取上级目录
    # os.getcwd()获取当前工作目录，谁调用这个函数，就获取谁的当前工作目录
    # parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    parent_dir = os.path.abspath(os.getcwd())

    # 获取上层目录指定的trainingDigits下的所有文件名
    # training_digits_path = parent_dir + '\\trainingDigits'
    training_digits_path = parent_dir + '\\' + file_path
    file_name_list = os.listdir(training_digits_path)

    # 文件数量，就创建多少行的数据
    file_count = len(file_name_list)

    labels_list = []
    file_content_np_array = np.zeros((file_count, 1024))

    # 根据文件数量创建对应的np.array((file_count, 1024))
    for i in range(file_count):
        # 数字
        number = int(file_name_list[i].split('_')[0])
        labels_list.append(number)
        # temp_file_content.append(image_transform_list(training_digits_path + '/' + file_name))
        file_content_np_array[i, :] = image_transform_list(training_digits_path + '/' + file_name_list[i])

    file_content_np_array = np.array(file_content_np_array, dtype=np.float64)
    return file_content_np_array, labels_list


if __name__ == "__main__":
    training_data, training_labels = generate_np_array_vector('trainingDigits')
    for i in training_data:
        for j in i:
            print j,
        print
