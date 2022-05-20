import tensorflow as tf
import os

os.environ["TF_APP_MIN_LOG_LEVEL"] = "2"
tf = tf.compat.v1  # 2.0版本如何调用，先不管


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


if __name__ == '__main__':
    test_tf()
