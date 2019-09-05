---
layout: post
title:  "python中的赋值"
date:   2019-09-05 10:15:00 +0800
categories: python_tips
tags: python 赋值
author: ac酱
mathjax: true
---

* content
{:toc}
在刷题过程中发现python在多元赋值上和c等其他语言有着很大的不同，查阅了一些资料，特此记录。



## 问题

在leetcode的题解中看到一行代码：
```python
pre.right, node.left, node = node, None, node.left
```
按照之前的习惯，认为拆分后应该等价为：
```python
pre.right = node
node.left = None
node = node.left
```
但是经过修改后程序不是报错，就是运行结果与预期不相符。经过尝试后发现，实际上拆分后应等价为：
```python
temp = node.left
pre.right = node
node.left = None
node = temp 
```
似乎在不知不觉中，python为你新增了一个临时变量，暂存了运算结果，在赋值符号右边的运算式计算完毕后，再一次赋值给赋值符号左边的变量名。

那么python在赋值时到底做了些什么呢？

## python中的赋值

我们都知道，C语言中，在给变量赋值时，需要先指定变量的数据类型，根据数据的类型分配一块内存区域进行数值存储。第一次声明一个变量的类型之后，他的所占的内存地址就不会改变了，之后多次进行赋值，改变的也只有该段内存中存储的值。

把一个变量a赋值给另一个变量b时，相当于把变量a的值拷贝一份传递给变量b，变量a与变量b的地址不会发生任何改变。
```c
int a;
printf("%d\n",&a);
a = 1;
printf("%d\n",&a);
a = 2;
printf("%d\n",&a);

int b;
printf("%d\n",&b);
b = a;
printf("%d\n",b);
printf("a:%d,b:%d\n",&a,&b);
```
```
6487580
6487580
6487580
6487576
2
a:6487580,b:6487576
```
而python就完全不同了。python中，“变量”的严格叫法是名字（name），给变量赋值相当于给对象贴标签，变量本身是没有任何意义的。
例如：
```python
a = 1
```
python内部首先将分配一块内存空间用于创建整数对象1，然后给整数对象1贴上名为a的标签。之后执行：
```python
a = 2
```
会在另一块内存空间中创建整数对象2，然后把标签a从对象1上撕下来贴在2身上，之后我们无法通过a来访问1这个对象了。之后把名字a赋值给名字b：
```python
b = a
```
相当于在刚才的对象2身上又贴了一个新的标签b，我们既能通过名字a也能通过名字b访问对象2，访问的对象是相同的。

再看看python中函数参数的传递：
```python
def fun_a(a):
    a += 4

g = 0
fun_a(g)
g
```
```
0
```
全局变量g传递给函数fun_a时，相当于函数中的参数a也被作为标签贴在了对象0上，之后a被重新赋值(a+=4)，相当于从对象0撕下标签a贴到对象4上，但是g依然是对象0上的标签。

如果传入函数的参数是一个列表对象：
```python
def fun_b(names):
    names[0] = ['x', 'y']

n_list = ['a', 'b', 'c']
fun_b(n_list)
n_list
```
```
[['x', 'y'], 'b', 'c']
```
和之前类似，names和n_list都是['a', 'b', 'c']上的标签，只是列表对象中的第0个元素被重新赋值了，但是两个标签依然贴在这个列表对象上，虽然列表对象的值更新了，但是对象依然是原来的对象。

## python中的“多元”赋值

python中存在另一种将多个变量同时赋值的方法，我们称为多元赋值（multuple，将 "mul-tuple"连在一起自造的)。因为采用这种方式赋值时，等号两边的对象都是元组。
```python
x, y, z = 1, 2, 'a string'
# 等同于 (x, y, z) = (1, 2, 'a string')
```

这样一来，问题也能得到解释了。
```python
pre.right, node.left, node = node, None, node.left
```
首先先对赋值符号右边的序列封装成元组，之后再将该元组进行拆分，对应地赋值给左边的变量。由于封装元组的操作在赋值之前，因此赋值时进行无论变量如何改变，都不会影响到已经封装好的元组。

利用这种特性，我们能将常用的变量交换的操作，从原来的3行代码压缩到1行：
```python
# 交换a与b的值
temp = a
a = b
b = temp

# 等价于
a, b = b, a
```

## python中的连续赋值

```python
foo = [0]
bar = foo
foo[0] = foo = [1]

print(foo)
print(bar)
```
这段代码如果按照C语言的思想，以foo[0]=(foo=[1])的循序执行的话，将会得到：
```
[[1]]
[0]
```
但是在python中，运行的结果是：
```
[1]
[[1]]
```
忘掉C语言的思想，我们用python的思想执行一遍：

    1. 首先foo和bar都指向[0]；
    2. 之后foo[0]的值发生变化，变为[1]，由于foo和bar指向的是同一个对象，因此foo和bar此时都为[[1]]；
    3. 最后foo又被指向[1]。

以这样的顺序，最终得到的上面的结果。由此可见，在连续赋值的过程中，赋值的顺序是：

    1. foo[0] = [1]
    2. bar = [1]

根据查阅到的资料，使用dis模块查看连续赋值的过程，可以得到连续赋值的执行顺序：

    首先构建要赋值的对象
    将对象在栈顶进行一份复制，然后将复制的值赋给第一个变量
    将对象在栈顶进行一份复制，然后将复制的值赋给第二个变量
    ……
    将对象赋值给最后一个变量

这和我们推导得到的结论是一致的。

**ac酱**

**完成于2019-09-05 中午**

> 参考资料：
* [图解Python变量与赋值](https://foofish.net/python-variable.html)
* [python——赋值与深浅拷贝](https://www.cnblogs.com/Eva-J/p/5534037.html)
* [python 关于连续赋值的简单工作原理？ - yonggege的回答 - 知乎](https://www.zhihu.com/question/46505057/answer/227007709)
* [Python连续赋值的内部原理](https://imliyan.com/blogs/article/Python%E8%BF%9E%E7%BB%AD%E8%B5%8B%E5%80%BC%E7%9A%84%E5%86%85%E9%83%A8%E5%8E%9F%E7%90%86/)