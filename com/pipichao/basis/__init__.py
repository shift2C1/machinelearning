# 基础
# 获取类型
# type()
# 强转
# str()
# list()
# tuple()
# int()
# float()
# eval()

# 格式化输出 ： %d, %f, %s


# 三目运算符 条件成立 条件 条件不成立
# a = 10
# b = 15
# c = a if a > b else b


# 循环 分支 while for if


# 字符串
"""
下标
a[5], 从0开始
切片
a[2:5:1] ,[开始索引：结束索引：步长] 左闭右开
默认步长为1 默认开始索引0 结束索引默认为最后的索引

字符串的常用函数：
string.function()
find() index() count()
replace() upper() lower() split() join()
"""

# 列表
"""
in ,not in 
新增：append(), insert() , extent()
删除: del ,del() ,pop() ,remove() ,clear()
修改： 通过下标修改
遍历： for， while
"""

# 元组：
"""
定义： 单个元组也需要加“，”
"""

#字典
"""
创建：dic(), { },{ key:value,key:val}
新增：var["key"]=val
删除：del, del(), clear()
遍历： get() , keys() , values() ,items()
"""

# 集合
"""
set()
"""

# 通用运算符,函数
"""
+ : 合并 str list tuple
* ： 复制
in 
not in

len() ,del ,del()
range() : 生成一个迭代对象 去遍历
enumerate(): 生成一个迭代对象 去遍历
"""
# 函数
"""
# 变量作用域：
        函数内修改全局变量 global var
函数 定义参数 ：
            位置参数 ，
            关键字参数：调用时 key= val，
            缺省（默认参数）：定义时候key=val
            不定长参数：不定长位置：fun_name(*args) ： 参数被打包成一个元组
                        不定长关键字：fun_name(**key)：参数被打包成一个字典
返回 ：一次可以返回多个变量，默认元组 接收时候按照返回值顺序接收，也可以打包成字典返回

引用：类似于指针 id():获取内存地址，引用传参：变量名传参
可变类型，不可变类型

lamda表达式：

高阶函数: 函数式编程，把一个函数当作参数传递
            常用内置高阶函数：map(),filter() reduce()

"""

# 文件 i/o
"""

"""
# 面向对象
"""
定义类： class（ 继承的类）
self ： 调用的对象，相当于this

魔法方法： __func_name__ :有特殊的功能
            __init(self)__ :构造器
            __str()__ : toString
            __del()__: 对象销毁的时候会被调用

继承：定义类时候在 括号里写需要继承的类，默认是object，多继承用 “,” 隔开
    如果继承了多个类有同名字的方法，则调用时默认执行第一个所继承的类的该方法
定义私有属性： __属性名字:通过get set 方法访问
               方法： __方法名字

类属性：所有实例都公用这个属性，直接写在类定义内，不用写在构造器内？
类方法：@classmethod 装饰器修饰 方法参数自动提示: cls
静态方法：@staticmethod ，不需要传递self 或者cls参数


"""
# 模块 包
"""
模块：一个模块就是一个py文件，import就是导入这个python文件
导入模块:                                   调用模块：
import 模块名字                            模块.功能
from 模块名字 import 功能                     直接调用 功能

as 给模块名字或者 功能名字起别名

__name__ ：模块名字，

模块的加载顺序： 由近导远

__all__ = ['功能'] ；控制时候全部导出导模块

包：文件夹：包含多个模块 
    __init__.py 文件：控制包的导入行为 这个文件夹下的模块必须注册到 __init__.py 通过 __all__=["模块1","模块2"]，否则引用不到

"""
