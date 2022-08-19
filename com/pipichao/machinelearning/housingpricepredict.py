import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.model_selection as ms
import pandas.plotting as pandas_plotting

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
    np.random.seed(seed=42)  # 设置随机种子，不然每次都是不同的测试集
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
    # result_with_sklearn=ms.train_test_split(data,test_size=0.2, random_state=42)
    return (test_set, train_set)


def take_glance_at_data(cvsdata):
    '''
        简单的查看一下数据本基本情况
        '''
    # 查看表头信息
    print(cvsdata.info())
    #
    # # 查看前几行数据
    print(cvsdata.head(1))
    #
    # # 其他列都是数字类型，只有这个是对象
    ocean_proximity = cvsdata["ocean_proximity"]
    #
    # # 相同的值分组，看有多少数量
    print(ocean_proximity.value_counts())
    #
    # # 查看数据的基本统计信息，均值，方差，低于该值百分比等
    print(cvsdata.describe())
    #
    # # 直方图
    cvsdata.hist(bins=50, figsize=(20, 15))
    #
    plt.show()


def create_test_set(cvsdata):
    test_train = split_train_test(cvsdata, 0.2)
    test_set = test_train[0]
    train_set = test_train[1]
    print(np.shape(test_set))
    print(np.shape(train_set))
    median_income = cvsdata["median_income"]
    median_income.hist()
    plt.show()
    # median_income 是个连续变量，采样分成五类
    income_cat = pd.cut(median_income,
                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                        labels=[1, 2, 3, 4, 5])
    cvsdata["income_cat"] = income_cat
    cvsdata["income_cat"].hist()
    plt.show()
    # print(income_cut)

    # sklearn的方式分割 : 分层搅乱分割器对象
    spliter = ms.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # # 复制出一份数据，防止原始数据被破坏
    copy = cvsdata.copy()
    train_test_indices = spliter.split(copy, copy["income_cat"])
    for train_index, test_index in train_test_indices:
        stratified_train_set = copy.loc[train_index]
        stratified_test_set = copy.loc[test_index]
    return None


def gain_insight_of_data(cvsdata):
    # 查看经纬度图像
    # alpha=0.1 图像更模糊
    cvsdata.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show()

    # 价格和人口密度 靠近海岸线有关。从图上来看
    cvsdata.plot(
        kind="scatter", alpha=0.4, x="longitude", y="latitude",
        s=cvsdata["population"] / 100, label="population", figsize=(10, 7),
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
    )
    plt.legend()
    plt.show()

    # 因此查看一下各个变量之间的协关系 -1 到 1 之间 越接近1 越正相关，越接近-1 越负相关
    corr_matrix = cvsdata.corr()  # 标准差
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # pandas 的散点图函数可以画出各个变量的协关系图
    attribute = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    pandas_plotting.scatter_matrix(cvsdata[attribute], figsize=(12, 8))
    plt.show()

    cvsdata.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.show()


if __name__ == '__main__':
    # fetch_housing_data()
    cvsdata = load_housing_data()
    # print(type(cvsdata))

    # take_glance_at_data(cvsdata)

    # create_test_set(cvsdata)

    # gain_insight_of_data(cvsdata)
