from pandas import read_csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
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

    min_max_scaler = MinMaxScaler()

    # 缩放特征
    prepared_data=min_max_scaler.fit_transform(pure_num_data)

    print(type(prepared_data))
    print(prepared_data[0:5])
