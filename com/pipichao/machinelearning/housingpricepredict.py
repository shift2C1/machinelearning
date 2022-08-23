import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.model_selection as ms
import pandas.plotting as pandas_plotting
import sklearn.impute as im
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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


def prepare_data(cvsdata):
    '''
    为后续的算法准备数据，
    :return:
    '''
    print("数据清洗")
    # 数据清洗：填充缺失的值，字符串数字化
    # 训练集

    # 方法1：去除total_bedrooms 字段为所对应的数据
    remove_null_record = cvsdata.copy().dropna(subset=["total_bedrooms"])
    remove_null_record.info()  # 可以看到total_bedrooms为空的数据被去除了，数据的总数量去掉了

    # 方法2：去除该列，不要这个维度的数据
    remove_attr = cvsdata.copy().drop("total_bedrooms", axis=1)  # axis : {0 or 'index', 1 or 'columns'}, default 0
    remove_attr.info()  # 可以看到少了一列数据

    # 方法3：填充值，用中值，均值等
    data_copy = cvsdata.copy()
    median = data_copy["total_bedrooms"].median()
    data_copy["total_bedrooms"].fillna(median, inplace=True)
    data_copy.info()  # 可以看到total_bedrooms这一列的数据不再缺失 和其他维度的数据总数都一样

    print("使用sklearn的方式数据清洗")
    # 使用sklearn的方式数据清洗
    simple_imputer = im.SimpleImputer(strategy="median")
    # SimpleImputer 只能操作数值类型的数据，得把非数值类型的去除 比如这批数据里的 ocean_proximity字段
    cvsdata_pure_num = cvsdata.drop("ocean_proximity", axis=1)
    simple_imputer.fit(cvsdata_pure_num)
    print(simple_imputer.statistics_)
    print(cvsdata_pure_num.median().values)  # 可以看到 中值完全相等
    cvsdata_pure_num.info()  # 这个地方有点问题 为什么填充完值total_bedrooms 还是20433个非空，一共20640个

    X = simple_imputer.transform(cvsdata_pure_num)
    print(type(X))  # 转换成了一个numpy的ndarray <class 'numpy.ndarray'>
    print(X.shape)  # (20640, 9)

    # ndarray 转换成pandas的 DataFrame
    print(cvsdata_pure_num.columns)
    cvs_dat_tr = pd.DataFrame(X, columns=cvsdata_pure_num.columns)
    print(type(cvs_dat_tr))

    return None


def handling_text_and_categorical_attribute(cvsdata):
    # 处理文本类型和 类别属性的字段
    cvs_cat=cvsdata[["ocean_proximity"]]
    print(cvs_cat.head(5))
    # 有多少类型，每个类型有多少条数据，类似于group by
    print(cvs_cat.value_counts())

    #
    print("使用顺序编码器")
    ordinal_encoder=OrdinalEncoder()
    # 字符串编码为数值类型的属性
    cvs_cat_encoded=ordinal_encoder.fit_transform(cvs_cat)
    print(type(cvs_cat_encoded))
    # print(cvs_cat_encoded[50:55])
    # 查看包含的类型
    print(ordinal_encoder.categories_)

    print("使用one hot编码器")
    one_hot_encoder=OneHotEncoder(sparse=True)
    one_hot_code=one_hot_encoder.fit_transform(cvs_cat)
    print(one_hot_code[0:3])
    print(one_hot_encoder.categories_)
    return None

def scaling_feature(cvsdata):
    print("特征缩放")
    data_num=cvsdata.drop("ocean_proximity",axis=1)
    # 特征缩放，将所有的特征缩放到同一个范围，不然不具备可比性


    # 归一化，将特征缩放至 0-1
    '''
    (x-min)/(max-min)
    
    '''
    print("归一化")
    min_max_scaler=MinMaxScaler()
    normalized_data=min_max_scaler.fit_transform(data_num)
    print(type(normalized_data)) # <class 'numpy.ndarray'>
    print(normalized_data[0:3])

    # 标准化 不在0-1之间
    '''
    (x-mean)/ standard deviation
    '''
    print("标准化")
    standard_scaler=StandardScaler()
    standardized_data=standard_scaler.fit_transform(data_num)
    print(type(standardized_data))
    if isinstance(standardized_data,type(standardized_data)):
        print(standardized_data[0:3])
    return None


if __name__ == '__main__':
    # fetch_housing_data()
    cvsdata = load_housing_data()
    # print(type(cvsdata))

    # take_glance_at_data(cvsdata)

    # create_test_set(cvsdata)

    # gain_insight_of_data(cvsdata)

    # prepare_data(cvsdata)

    # handling_text_and_categorical_attribute(cvsdata)

    scaling_feature(cvsdata)

