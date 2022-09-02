import com.pipichao.dataset.handwritingdigist as digst
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from numpy import array
from numpy import empty
from sklearn.metrics import mean_squared_error


def transform_2d_array_to_1d_vector(img_array):
    print(img_array.shape)
    # 创建一个空的ndarray
    # 转换成784*1 的列向量，最后一维不用写1
    vector_list=empty(shape=[len(img_array),784,])
    for i in range(len(img_array)):

        vector=img_array[i].reshape([784,])
        vector_list[i]=vector
    print(vector_list.shape)
    return array(vector_list)

if __name__ == '__main__':
    # 读取训练集数据
    train_set = digst.read_train_set()
    # 输入
    train_input = train_set[0]
    # 看一个图
    # plt.imshow(train_input[1],cmap=plt.cm.binary,interpolation="nearest")
    # plt.axis("off")
    # plt.show()
    # 看多个图 3行4列
    # for i in range(12):
    #     plt.subplot(3,4,i+1)
    #     plt.imshow(train_input[i],cmap=plt.cm.binary,interpolation="nearest")
    #     plt.axis("off")
    # plt.show()

    # 目标
    train_target = train_set[1]


    '''
    先分类5和非5
    '''
    img_5target=(train_target==5)
    # 布尔索引
    img_5input=train_input[img_5target]
    # for i in range(12):
    #     plt.subplot(3,4,i+1)
    #     plt.imshow(img_5input[i],cmap=plt.cm.binary,interpolation="nearest")
    #     plt.axis("off")
    # plt.show()


    train_input_vector=transform_2d_array_to_1d_vector(train_input)
    # 随机梯度下降
    sGDClassifier=SGDClassifier(random_state=42)
    sGDClassifier.fit(train_input_vector,img_5target)

    # 测试机数据
    test_input,test_target=digst.read_test_set()
    output=sGDClassifier.predict(transform_2d_array_to_1d_vector(test_input))
    img_5test_target=(test_target==5)
    mse=mean_squared_error(test_target,img_5test_target)
    print(mse)


