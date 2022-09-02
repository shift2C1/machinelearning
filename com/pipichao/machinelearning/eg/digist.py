import com.pipichao.dataset.handwritingdigist as digst
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 读取训练集数据
    train_set = digst.read_train_set()
    # 输入
    train_input = train_set[0]
    plt.imshow(train_input[0],cmap=plt.cm.binary,interpolation="nearest")
    plt.axis("off")
    plt.show()

    # 目标
    train_target = train_set[1]

    # 测试机数据
    digst.read_test_set()
