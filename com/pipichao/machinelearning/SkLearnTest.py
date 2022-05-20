import numpy as np
import matplotlib.pyplot as plt
import math
# import tensorflow as tf
# import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import jieba


# var = np.array([[1,2,3],
#                 [4,5,6],
#                 [7,7,9]])
# print(var)
# inverse=np.linalg.inv(var)
# print(inverse)
# # eigenvalues and eigenvectors
# eig=np.linalg.eig(var)
# print("val:")
# print(eig[0])
# print("vec:")
# print(eig[1])
# x = np.arange(0, 2 * math.pi, 0.1)
# print(math.e)
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()
# a=("a","b","c")
# print(a)
# b={
#     "key": 1001
# }
# print(b)
# print(b["key"])

# iris = sk.datasets.load_iris()
# print(iris)
def getData():
    # 加载数据
    iris = load_iris()
    # print(iris)
    print(iris["DESCR"])
    # 拆分训练集和测试集: 训练集数据，测试集数据，训练集目标，测试集目标
    x_train, x_test, y_train, y_test = train_test_split(iris["data"], iris["target"], test_size=0.2)
    return None


def feature_extraction_fun():
    # 中文分词
    data = [
        "毛主席万岁",
        "习主席万岁"
    ]
    splite_data = []
    for string in data:
        # 中文分词
        generator = jieba.cut(string)
        a = list(generator)
        # print(a)
        b = " ".join(a)
        # print(b)
        splite_data.append(b)
    print(splite_data)
    # 向量化
    count_vectorizer = CountVectorizer()
    a = count_vectorizer.fit_transform(splite_data)
    print("特征名字：\n",count_vectorizer.get_feature_names())
    print(a)
    print(a.toarray())
    return None

# 归一化
def normalize_(data):
    # 根据最大最小值
    minmax=MinMaxScaler(feature_range=(0,1))
    minmax.fit_transform(data,)
    # 根据均值和方差
    stander=StandardScaler()
    stander.fit_transform(data)
    return None


if __name__ == '__main__':
    # getData()
    feature_extraction_fun()
