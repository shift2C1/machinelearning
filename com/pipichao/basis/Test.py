import json


class test:
    name = "hahha"

    def __set__(self, instance, value):
        self.name = value


if __name__ == '__main__':
    file = open('D:\\code\\machinelearning\\1.txt', mode="r")
    input_stream = file.read()

    # print(input_stream)
    file.close()
    file1 = open("D:\\code\\machinelearning\\2.txt", mode="w") # 直接覆盖之前的文件内容
    # file1 = open("D:\\code\\machinelearning\\2.txt", mode="a") # 原文件中拼接这次的内容
    file1.write("hahahh\n")
    file1.close()

