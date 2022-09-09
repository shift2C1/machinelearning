import com.pipichao.dataset.handwritingdigist as digst
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from numpy import array
from numpy import empty
from sklearn.metrics import mean_squared_error
import numpy as np
from  sklearn.model_selection import StratifiedKFold
from sklearn import  clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import  cross_val_predict
from sklearn.metrics import confusion_matrix

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
def implementing_cross_validation(train_input_vector,img_5target):
    # 实现一个交叉验证

    # 分层抽样
    skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    for train_indices, test_indices in skf.split(train_input_vector, img_5target):
        # 返回所拆分的测试集和验证集索引
        print(train_indices)
        print(test_indices)
        print("///////////////////")
        x_train_folds = train_input_vector[train_indices]
        y_train_folds = img_5target[train_indices]

        x_test_folds = train_input_vector[test_indices]
        y_test_folds = img_5target[test_indices]

        # 同样使用交叉验证
        copy_sgd = clone(sGDClassifier)
        copy_sgd.fit(x_train_folds, y_train_folds)

        y_pred = copy_sgd.predict(x_test_folds)

        # 正确的数量
        num_correct = sum(y_pred == y_test_folds)

        accurecy = num_correct / len(y_pred)
        print("精度:", accurecy)

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

    '''
    评估模型的性能
    '''
    # 自己实现一个交叉验证
    # implementing_cross_validation(train_input_vector,img_5target)

    # 使用自带的交叉验证
    # score_array=cross_val_score(estimator=sGDClassifier,X=train_input_vector,y=img_5target,cv=3,scoring="accuracy")
    # print("精度：",score_array)

    # 返回交叉验证的预测值，
    cross_pre=cross_val_predict(estimator=sGDClassifier,X=train_input_vector,y=img_5target,cv=3)

    # 误差矩阵（混淆矩阵）
    error_matrix=confusion_matrix(y_true=img_5target,y_pred=cross_pre)
    print(error_matrix)
    # [[53892   687]
    #  [1891  3530]]