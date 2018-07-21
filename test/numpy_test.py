#!/usr/bin/env python
# _*_ UTF-8 _*_

import numpy as np
# numpy与scipy一起形成对数组、矩阵的处理；
# 标准python库中存在用list来存储数据，但是列表作为对象是将指针存储在内存中，影响数据的运行效率，因此numpy应运而生，产生了对数组矩阵的操作，
# numpy主要有两种格式：
# 1）ndarray（n维数组对象，存储单一数据类型的n维数组）；
# 2）ufunc（通用函数对象，对数组进行处理的函数）。
# 举例为：
# 1）数组的构建
a = np.array([1,2,3,4,5,6])
b = np.array([[3,4,5,6,7,8],[4,6,8,9,3,2]])
g = np.random.rand(10)
# a.shape:查看数组形状；
print(a.shape)
print(b.shape)
b.shape=6,2 #对数组的变形，需要考虑数组是否能变形成功，并没有对数组转置，只是改变了数组的大小，元素在内存中的位置没有变化，第二个参数为-1时，系统会自动计算第二个参数；
print(b)
# a.reshape((2,2)):返回一个改变了尺寸的新数组，原数组不变;
print(a.reshape((2,3)))
print(b.reshape((3,4)))
# dtype:定义数组类型：
c = np.array([[1,2,3],[4,5,6],[7,9,8]], dtype=np.float)
print(c)
# 构建数组的其他方式：arange(开始值，终值，步长)
d = np.arange(0,1,0.1)
print(d)
# 构建数组的其他方式：linspace(开始值，终值，元素个数)
e = np.logspace(0, 2, 10)
print(e)
# 构建数组的其他方式：fromstring(s, dtype=np.int8)
s = '123456789'
f = np.fromstring(s, dtype=np.int8)
print(f)
# 计算每个数组元素的函数，即将数据用函数进行运算，然后返回数组形式的运算结果：
def func2(i, j):
    return (i+1)*(j+1)
print(np.fromfunction(func2, (9,9)))
# 2）数据的存取：
print(a[1:5])
a[3]=7
print(a)
# ndarray的数据结构为：dtype（数组类型）、dim_count（数组维数）、dimensions（数组结构shape）、strides（存储区的字节变化数）、data（存储的数据）
# 得到（8,4）：即第0轴下标每增加1，则地址增加8个字节；第1轴下标每增加1，则地址增加4个字节；
print(b.strides) 
# 追加数据：
a5 = [1,2,3]
print(a5.append(4))
# 删除数据：
print(a5.remove(4))
# 判断数组中是否存在某元素：
result = "shanghai" not in a5
result2 = "shanghai" in a5
print(result)
print(result2)
# 获取指定元素的个数：
print(a5.count(3))
# 指定元素所在的位置：
print(a5.index(3))
# 数组排序：
print(a5.sort())
# 反转：
print(a5.reverse())
# 3）数据的ufunc函数：
# sin()\add()\subtract()\multiply()\divide()\power()\remiander(取余)
print(np.add.reduce([1,2,3]))
# add.redcue():求和，不保存所有的计算结果；
# add.accumulate():求和，保存所有的计算结果；
# add.reduceat():求和，通过indices指定起始、终止位置；
z = np.array([1,2,3,4])
result = np.add.reduceat(a, indices=[0,1])  #[1,2,1,2,3,1,4,1]
# 如果indices[i-1]<indices[i]:保留此位置的数据；
# 如果indices[i-1]>indices[i]:将前面的数值相加；
print(result)
print(np.multiply.outer([1,2,3,4,5],[2,3,4])) # 类似于广播的成绩形式；
# 4）广播：
# ufunc函数对两个数组进行计算时，需要保证两个数组的结构一致，如果两个数组结构不一致，shape中的不足之处都通过在前面加1补齐；
# 输出数组的shape是输入数组的shape的各个轴上的最大值；
# 当输入数值的某个值得长度为1时，沿此轴运算时都用此轴上的第一组值。
c1 = np.arange(0, 60, 10).reshape(-1, 1)
c2 = np.arange(0, 5)
c = a + b
print(c)
# 5）矩阵：
a1 = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
print(a1)
# print(a**-1)
# dot():返回的是两个数组的点积，即计算两个矩阵的乘积；
b1 = np.arange(1, 5).reshape(2,2)
b2 = np.arange(5, 9).reshape(2,2)
print(np.dot(b1, b2))

# 一：用numpy写的一个神经网络流程：
# 数据集为：
# 001  0
# 111  1
# 101  1
# 011  0
# 前三个维度为x，第四个维度为y；
# 输入神经元有三个维度，得到一个y值：f(w1*x1+w2*x2+w3*x3)=y
# 激活函数为：sigmod函数；
# 主要的计算过程为：1）前向传播求损失；2）后向传播求权值；
# 一个主要的求导公式为：dy/dx=y(1-y)
# 构建逻辑回归函数：
def nonlin(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
# 输入值：
x = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
# 输出值：
y = np.array([0,1,1,1]).T
# 初始化权重syn0：
# seed(num)是产生随机数：如果num的数据一致，则产生的随机数一样；如果不一致，则产生的随机数不一样；如果括号中为空，则seed()随机产生随机数；
np.random.seed(1)
syn0 = 2*np.random.random((3,1))-1
# 迭代系统：
for iter in range(10000):
    l0 = x
    # l1:是运算结果；
    l1 = nonlin(np.dot(l0, syn0))
    y1 = [[0 for i in range(1)] for i in range(len(y))]
    for i in range(len(y)):
        y1[i][0] = y[i]
    # 运算误差
    l1_error = y1 - l1
    # l1_delta:权重变化值；
    # nonlin(l1, True):每一组的导数值；
    l1_delta =l1_error*nonlin(l1, True)
    syn0 += np.dot(l0.T, l1_delta)
#     print(syn0)
print("output after training")
print(l1)
# l1:是经过四组数据训练后的最终结果；四次训练后会得到一个稳定的权重值，然后用这组权重值syn0进行分类；





























