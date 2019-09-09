---
layout: post
title:  "python中的“self”"
date:   2019-09-09 10:05:00 +0800
categories: python_tips
tags: python self
author: ac酱
mathjax: true
---

* content
{:toc}
在python中，定义一个类和他的实例方法时，往往少不了与self这个参数打交道，却一直都没有去真正的探究过self到底是个什么东西。今天查阅了一些资料，特此记录。



## 问题
刚开始学习python时，看到教程上在定义实例方法时，传入了一个名为self的参数。但是在调用时，却没有传入self，感觉很不解。

所以这个self到底是干啥的？

## 实例方法中的self
首先我们知道，只有在实例方法中才需要传入参数self，但是在调用该类方法的时候却不用传入self。这个self到底是什么呢？

### self代表的是类的实例
```python
class Test:
    def prt(self):
        print(self)
        print(self.__class__)

t = Test()
t.prt()
```
```
<__main__.Test object at 0x000001C3F3BC3128>
<class '__main__.Test'>
```
从这个例子中可以明显看出self代表的是类的实例。

### self可以省略不写吗
```python
class Test:
    def prt():
        print(self)

t = Test()
t.prt()
```
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: prt() takes 0 positional arguments but 1 was given
```
在定义类中的实例方法时如果将self省略不写，则会报错，提示我们多传了一个参数。看起来当通过`实例.方法`的方式调用实例方法时，实际上是将实例作为该方法的第一个参数传入方法中。

再看看类中方法的相互调用吧。
```python
class Test:
    def __init__(self):
        self.prt()
    def prt(self):
        print(self)

t = Test()
```
```
<__main__.Test object at 0x0000023CA58930B8>
```
这是正常的类中方法的相互调用。

我们尝试省略部分self。首先将被调用的方法的参数self省略：
```python
class Test:
    def __init__(self):
        self.prt()
    def prt():
        print(self)

t = Test()
```
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in __init__
TypeError: prt() takes 0 positional arguments but 1 was given
```
和上面的通过实例调用方法的报错一致，印证了我们的猜想。

若是将调用时的self去掉：
```python
class Test:
    def __init__(self):
        prt()
    def prt(self):
        print(self)

t = Test()
```
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in __init__
NameError: name 'prt' is not defined
```
则会报错提示函数未定义。看起来像是把prt()当成静态方法了。
```python
def prt():
    print("test")

class Test:
    def __init__(self):
        prt()

t = Test()
```
```
test
```
将prt()定义成静态方法后果然能够正确运行了。

可见，当在类中定义实例方法时，参数self是不能被省略的。在类中调用实例方法时，说明实例的self也是不能被省略的。

## 类方法
通过上面的几段代码的测试，我们知道在与类中的实例方法相关时，所以的self都不能被省略，self.方法实际上是将self作为方法的第一个参数传入。

当我们在定义类中方法和调用类中方法时都不传递self时，就构成了类方法。
```python
class Test:
    def prt():  
        print(__class__)  

Test.prt()  
```
```
<class '__main__.Test'>
```

## 继承
在继承时，self又代表什么呢？
```python
class Parent:
    def pprt(self):
        print(self)

class Child(Parent):
    def cprt(self):
        print(self)

c = Child()
c.cprt()
c.pprt()
p = Parent()
p.pprt()
```
```
<__main__.Child object at 0x00000202D0B94128>
<__main__.Child object at 0x00000202D0B94128>
<__main__.Parent object at 0x00000202D0B940F0>
```
运行c.cprt()时应该没有理解问题，指的是Child类的实例。

但是在运行c.pprt()时，等同于Child.pprt(c)，所以self指的依然是Child类的实例，由于self中没有定义pprt()方法，所以沿着继承树往上找，发现在父类Parent中定义了pprt()方法，所以就会成功调用。

可见之前理解的将`self.方法`的self作为方法的第一个参数传入，在继承的场景下依然成立。

## “self”一定要叫做“self”吗
```python
class Test:
    def prt(this):
        print(this)
        print(this.__class__)

t = Test()
t.prt()
```
```
<__main__.Test object at 0x000002735B1E3198>
<class '__main__.Test'>
```
说明“self”不一定要叫做“self”，这只是约定俗成的名称，但是为了避免歧义，还是不要随便给他改名比较好。

## 总结

* 1.实例方法中的self，无论在定义或是调用时都不能省略。调用时通过`实例.方法`将实例作为第一个参数传入方法中。
* 2.self只是约定俗成的名称，实际上可以改成任何其他的名称。
* 3.谁调用了实例方法，该实例方法中的self就指代谁。
* 4.在定义类中方法和调用类中方法时都不传递self时，就构成了类方法。类方法不需要通过实例调用。

**ac酱**

**完成于2019-09-09 中午**

> 参考资料：
* [Python中self的含义](https://www.iteye.com/blog/uule-2353480)
* [全面理解python中self的用法](https://www.cnblogs.com/wangjian941118/p/9360471.html)
* [Python3 面向对象](https://www.runoob.com/python3/python3-class.html)
