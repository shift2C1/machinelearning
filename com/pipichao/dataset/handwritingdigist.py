'''
手写数字
'''

import numpy as np
import gzip
import struct


def read_test_set():
    root_dir = "../../../../data/digit/"
    test__input_path = "t10k-images-idx3-ubyte.gz"
    test_target_path = "t10k-labels-idx1-ubyte.gz"
    # 训练集数据
    test_input_set = read_gzip_img(root_dir +test__input_path)
    # print("样本输入：\n", train_input_set.shape)
    test_target_set = read_gzip_lable(root_dir +test_target_path)
    # print("样本目标：\n", train_target_set.shape)
    return (test_input_set,test_target_set)

def read_train_set():
    root_dir = "../../../../data/digit/"
    train_input_path = "train-images-idx3-ubyte.gz"
    train_target_path = "train-labels-idx1-ubyte.gz"
    # 训练集数据
    train_input_set = read_gzip_img(root_dir + train_input_path)
    # print("测试输入：\n", test_input_set.shape)
    train_target_set = read_gzip_lable(root_dir + train_target_path)
    # print("测试目标：\n", test_target_set.shape)
    return (train_input_set, train_target_set)



def read_gzip_img(file_path):
    gzip_file = gzip.GzipFile(file_path)
    data = gzip_file.read()
    magic, num, rows, columns, = struct.unpack(">iiii", data[:16])
    dimession = rows * columns
    X = np.zeros((num, rows, columns), dtype="uint8")
    # print(X.shape)
    offset = 16
    for i in range(num):
        a = np.frombuffer(data, dtype=np.int8, count=dimession, offset=offset)
        X[i] = a.reshape((rows, columns))
        offset = offset + dimession
    print(X[1].shape)
    return X


def read_gzip_lable(file_path):
    gzip_file = gzip.GzipFile( file_path)
    data = gzip_file.read()
    magic, num = struct.unpack(">ii", data[:8])
    target = np.frombuffer(data, dtype=np.int8, count=num, offset=8)
    # print(d)
    return target
