import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# 读取cvs的数据
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    # 返回DataFrame 对象
    return pd.read_csv(csv_path)


# 把数据分成测试集和验证集
def split_train_test(data, test_ratio):
    # 数据打乱，返回乱序的数据索引
    np.random.seed(seed=42) #设置随机种子，不然每次都是不同的测试集
    disorder_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)

    # 前size个作为测试集
    test_set_indices = disorder_indices[0:test_set_size]

    # 剩下的作为训练集
    train_set_indices = disorder_indices[test_set_size:]

    # indexlocations
    test_set = data.iloc[test_set_indices]
    train_set = data.iloc[train_set_indices]


    # 用sklearn 可以实现同样的方式
    result_with_sklearn=ms.train_test_split(data,test_size=0.2, random_state=42)
    return (test_set, train_set)


if __name__ == '__main__':
    # fetch_housing_data()
    cvsdata = load_housing_data()
    # print(type(cvsdata))

    '''
    简单的查看一下数据本基本情况
    '''
    # 查看表头信息
    # print(cvsdata.info())
    #
    # # 查看前几行数据
    # # print(cvsdata.head(1))
    #
    # # 其他列都是数字类型，只有这个是对象
    # ocean_proximity = cvsdata["ocean_proximity"]
    #
    # # 相同的值分组，看有多少数量
    # print(ocean_proximity.value_counts())
    #
    # # 查看数据的基本统计信息，均值，方差，低于该值百分比等
    # print(cvsdata.describe())
    #
    # # 直方图
    # cvsdata.hist(bins=50, figsize=(20, 15))
    #
    # plt.show()

    test_train = split_train_test(cvsdata, 0.2)
    test_set = t = test_train[0]
    train_set = test_train[1]
    print(np.shape(test_set))
    print(np.shape(train_set))
