from pandas import read_csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import  os



housing_path = os.path.join("datasets", "housing")


if __name__ == '__main__':

    csv_path = "D:\code\machinelearning\com\pipichao\machinelearning\datasets\housing\housing.csv"
    housing_data_frame=read_csv(csv_path)
    housing_data_frame_without_null_value =housing_data_frame.dropna(subset=["total_bedrooms"])
    ocean_proximity=housing_data_frame_without_null_value[["ocean_proximity"]]

    one_hot_encoder=OrdinalEncoder()

    ocean_proximity_code=one_hot_encoder.fit_transform(ocean_proximity)

    # 添加该列
    housing_data_frame_without_null_value.insert(9,"ocean_proximity_code",ocean_proximity_code)
    # 去除该列
    pure_num_data=housing_data_frame_without_null_value.drop("ocean_proximity",axis=1)

    print(pure_num_data.info())

    min_max_scaler = MinMaxScaler()


    # 缩放特征
    prepared_data=min_max_scaler.fit_transform(pure_num_data)

    print(type(prepared_data))
    print(prepared_data[0:3])

    train_set,test_set=train_test_split(prepared_data,test_size=0.2,random_state=42)
    print("训练集")
    print(train_set[0:3]) # numpy的切片 ：默认按照行取，第一维代表行索引，第二位代表列索引
    print(test_set.shape)

    train_target=train_set[:,8] #取第八列的 房价（median_house_value） 为目标值
    print("训练集目标")
    print(train_target[0:3])
    train_input=train_set[:,(0,1,2,3,4,5,6,7,9)] #取除了第8列之外的数据
    print("训练集输入")
    print(train_input[0:3])

#     test 集
    test_set_input=test_set[:,(0,1,2,3,4,5,6,7,9)]
    test_set_target=test_set[:,8]


# 使用线性回归先试一下
    lr=LinearRegression()
    lr.fit(train_input,train_target)
    print("训练模型后")
    output=lr.predict(test_set_input)
    print("模型输出：",output[0:10])
    print("实际目标：",test_set_target[0:10])

    mse=mean_squared_error(test_set_target,output)
    print("均方误差的开平发：",np.sqrt(mse))
