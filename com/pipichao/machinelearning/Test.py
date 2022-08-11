import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt

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


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    # 返回DataFrame 对象
    return pd.read_csv(csv_path)


if __name__ == '__main__':
    # fetch_housing_data()
    cvsdata=load_housing_data()
    # print(type(cvsdata))

    # 查看表头信息
    print(cvsdata.info())

    # 查看前几行数据
    # print(cvsdata.head(1))

    # 其他列都是数字类型，只有这个是对象
    ocean_proximity=cvsdata["ocean_proximity"]

    # 相同的值分组，看有多少数量
    print(ocean_proximity.value_counts())

    # 查看数据的基本统计信息，均值，方差，低于该值百分比等
    print(cvsdata.describe())


    # 直方图
    cvsdata.hist(bins=50,figsize=(20,15))

    plt.show()
