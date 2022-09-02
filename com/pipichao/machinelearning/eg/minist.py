from sklearn.datasets import fetch_openml
from scipy.io.arff import loadarff
from pandas import DataFrame
import matplotlib.pyplot as  plt

from sklearn.datasets._base import Bunch

minist_download_url="https://www.openml.org/search?type=data&sort=runs&id=554&status=active"

if __name__ == '__main__':

    # 在线拉去太慢了，下载下来
    # minist_dataset=fetch_openml("mnist_784",version=1)
    # print(minist_dataset.keys())
    # print(minist_dataset["url"])

    # 返回一个元组，第一个是数据，第二个是描述信息
    arff=loadarff("F:\迅雷下载\mnist_784.arff")
    info=arff[1]
    # print(info)
    # 转成pandas的数据格式
    minist_dataset=DataFrame(data=arff[0])
    # print(minist_dataset.info())
    # print(minist_dataset.head(5))
    # print('///////////////')
    # print(minist_dataset.iloc[0])
    # minist_dataset.to_csv("minist_784.csv",sep="\t")

