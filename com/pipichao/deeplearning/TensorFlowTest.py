import tensorflow as tf
# from  tensorflow.core.examples.tutorials.mnist import input_data
import os
import gzip

os.environ["TF_APP_MIN_LOG_LEVEL"] = "2"
tf = tf.compat.v1  # 2.0版本如何调用，先不管
tf.disable_eager_execution()


def test_tf():
    # tf.compat.v1.disable_eager_execution()
    # 定义图
    gragh = tf.Graph()
    with gragh.as_default():
        a = tf.constant(2)
        b = tf.constant(3)
        c = a + b
        print(c)

    # 打印结果：tf.Tensor(5, shape=(), dtype=int32)

    # 开启会话执行
    with tf.Session(graph=gragh,
                    config=tf.ConfigProto(log_device_placement=True)) as session:
        c_val = session.run(c)
        print(c_val)
    print(session.graph)  # <tensorflow.python.framework.ops.Graph object at 0x0000026356577FD0>

    # 可视化图
    # tensorbourd

    return None


# 求出 y=0.8x+0.7
def linear_regression():
    # 定义变量,要学习的参数，
    weight_ml = tf.Variable(initial_value=0.5, name="weight_ml initial_value")
    bias_ml = tf.Variable(initial_value=0.5)
    # 训练集,正态分布随机产生100数字
    x_train = tf.random.normal([100, 1])
    # 训练集，y
    y_train = tf.matmul(x_train, [[weight_ml]]) + bias_ml

    # 目标值，真实值
    y_target = tf.matmul(x_train, [[0.8]]) + 0.7
    print(y_target)
    print(x_train)
    error = tf.subtract(y_target, y_train)
    # print(error)
    square_error = tf.square(error)

    # 构造了均值平方误差
    mean_square_error = tf.reduce_mean(square_error)
    # print(mean_square_error)

    # 使用梯度下降方法 LMS???
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(mean_square_error)
    # print(optimizer.values())

    # 收集运行过程的信息
    tf.summary.scalar("mean_square_error", mean_square_error)
    tf.summary.histogram("weight_ml", weight_ml)
    tf.summary.histogram("bias_ml", bias_ml)
    all_summary_info = tf.summary.merge_all()

    # 初始化变量
    init = tf.initialize_variables([weight_ml, bias_ml])
    # 开始运行
    with tf.Session() as session:
        # 初始化变量
        session.run(init)
        print("初始值：weight_ml:%f,bias_ml:%f" % (weight_ml.eval(), bias_ml.eval()))

        # 输出到文件

        file_writer = tf.summary.FileWriter("../../../tmp/lr", tf.get_default_graph())

        for i in range(100):
            session.run(optimizer)
            print("第%d次迭代的值：weight_ml:%f,bias_ml:%f" % (i + 1, weight_ml.eval(), bias_ml.eval()))

            # 收集到的信息输出到文件
            info = session.run(all_summary_info)
            file_writer.add_summary(info, i)
    #         tensorboard --logdir=./tmp/lr 控制台执行命令可以查看输出的信息，访问 http://localhost:6006/
    return None


# 手写数字识别
def hand_writen_digits_recogenition():
    train__input_path="t10k-images-idx3-ubyte.gz"
    train_target_path="t10k-labels-idx1-ubyte.gz"
    test_input_path="train-images-idx3-ubyte.gz"
    test_target_path = "train-labels-idx1-ubyte.gz"
    # 训练集数据
    train_input_set=read_gzip_img(train__input_path)
    print(train_input_set.shape)
    train_target_set=read_gzip_lable(train_target_path)
    print(train_target_set.shape)

    return None


import struct
import numpy as np


def read_gzip_img(file_path):
    root_dir = "../../../data/digit/"
    gzip_file = gzip.GzipFile(root_dir + file_path)
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
    # print(X[1].shape)
    return X


def read_gzip_lable(file_path):
    root_dir = "../../../data/digit/"
    gzip_file = gzip.GzipFile(root_dir + file_path)
    data = gzip_file.read()
    magic, num = struct.unpack(">ii", data[:8])
    target = np.frombuffer(data, dtype=np.int8, count=num, offset=8)
    # print(d)
    return target


if __name__ == '__main__':
    # test_tf()
    # linear_regression()
    # read_gzip_img("data/digit/t10k-images-idx3-ubyte.gz")
    # read_gzip_lable("data/digit/t10k-labels-idx1-ubyte.gz")
    hand_writen_digits_recogenition()