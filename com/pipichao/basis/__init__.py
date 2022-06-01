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

