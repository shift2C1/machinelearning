import scipy.integrate as integrate
import scipy.optimize as optimize
import math
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

pi = math.pi


# scipy:科学计算的库

# 积分
def integrate_one_variable():
    # 计算给定函数的，单变量积分（一重积分）
    # 返回一个元组：第一个元素是积分值，第二个元素为误差
    # result = integrate.quad(lambda x: 0.5 * x, 1, 2)  # (0.75, 8.326672684688674e-15)
    # print(result)
    # print(result[0])

    # 带参数的
    a = 2
    b = 1
    result = integrate.quad(integrate_fun_with_param, 0, 1, args=(a, b))
    print(result[0])

    return None


def integrate_fun_with_param(x, a, b):
    y = a * x ** 2 + b
    return y


def exp_int(n, x):
    result = integrate.quad(intergrate_func, 1, np.inf, args=(n, x))
    return result[0]


def intergrate_func(t, n, x):
    y = np.exp(-x * t) / t ** n
    return y


def integrate_multiple_variable():
    return None


# 需要优化的函数
def rosenblock_function(
        # x1, x2
        x
):
    # rosenblock 函数： 一个非凸函数 全部变量为1时候函数有最小值0
    y=sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    # y = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    return y


# 用不同的方法去最小化 函数
def minimize_function_with_diffrent_method():
    init_search_location = np.array([3, -1])
    res = optimize.minimize(fun=rosenblock_function,
                            x0=init_search_location,
                            method='nelder-mead',
                            )
    print(res)
    pass


def plot_2d_rosenblock_funtion():
    # fig = plt.figure()
    # ax = plt.gca(projection="3d")
    # x1 = np.arange(-5, 5, 0.1)
    # x2 = np.arange(-5, 5, 0.1)
    # x, y = np.meshgrid(x1, x2)
    # z = rosenblock_function(x, y)
    # ax.plot_surface(x, y, z)
    # plt.show()
    pass


if __name__ == '__main__':
    # integrate_one_variable()
    # integrate_multiple_variable()
    # plot_2d_rosenblock_funtion()
    minimize_function_with_diffrent_method()
    pass
