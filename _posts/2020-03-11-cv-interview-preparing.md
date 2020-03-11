---
layout: post
title:  "CV方向的面试准备"
date:   2020-03-11 18:11:00 +0800
categories: CV面试准备
tags: interview ComputerVision
author: ac酱
mathjax: true
---

* content
{:toc}
CV方向知识点汇总




## 思维导图

## BackBone

### AlexNet(2012)
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/4.jpg" />
<div>AlexNet</div>
</center>
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/5.jpg" />
<div>AlexNet</div>
</center>
网络的输入是227×227×3的图像，网络一共有八层，其中前五个是卷基层，后3个是全连接层。

可能由于当时GPU连接间的处理限制， AlexNet使用两个单独的GPU在 ImageNet数据库上执行训练，因此常常能看到将其拆分为两个网络的结构示意图，每一部分的kernel数量为实际kernel数量的一半。

Conv1阶段:
卷积：
输入：227×227×3，卷积核：11×11×3，卷积核个数：96，步长：4，输出：55×55×96

激活函数：relu

归一化：LRN（局部响应归一化层，Local Response Normalization Layer），local_size=5

池化：
类型：max pooling，池化尺度：3×3，步长：2，输出：27×27×96

Conv2阶段：
卷积：
输入：27×27×96，卷积核：5×5×96，卷积核个数：256，步长：1，padding：same（相同补白，此处为2×2，使得卷积后图像大小不变），输出：27×27×256

激活函数：relu

归一化：LRN，local_size=5

池化：
类型：max pooling，池化尺度：3×3，步长：2，输出：13×13×256

Conv3阶段：
卷积：
输入：13×13×256，卷积核：3×3×256，卷积核个数：384，步长：1，padding：same（1×1），输出：13×13×384

激活函数：relu

Conv4阶段：
卷积：
输入：13×13×384，卷积核：3×3×384，卷积核个数：384，步长：1，padding：same（1×1），输出：13×13×384

激活函数：relu

Conv5阶段：
卷积：
输入：13×13×384，卷积核：3×3×384，卷积核个数：256，步长：1，padding：same（1×1），输出：13×13×256

激活函数：relu

池化：
类型：max pooling，池化尺度：3×3，步长：2，输出：6×6×256

FC6阶段：
输入：6×6×256，




使用ReLU激活函数和0.5概率的 dropout来对抗过拟合。
### VGG
### GoogleNet
### Xception
### Inception
### SqueezeNet
### ShuffleNet
### ResNet
### ResNext
### Residual Attention Moudle
### DenseNet
### MobileNet
### SENet
### Stacked Hourglass Networks
### DetNet 
### Deformable convolution Networks


## 其他经典网络
### LeNet-5(1998)
LeNet-5是一个7层卷积神经网络（一般输入层不计，也许有人会问，这个网络的名字里面为什么有个5，其实这个网络的主干就是5层，两个卷积层+两个全连接层+输出层）。网络输入是一个32×32×1的灰度图像。
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/1.jpg" />
<div>LeNet-5</div>
</center>
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/2.jpg" />
<div>LeNet-5</div>
</center>
LeNet-5是一个7层卷积神经网络，总共有约6万（60790）个参数。

随着网络越来越深，图像的高度和宽度在缩小，与此同时，图像的channel数量一直在增加。

LeNet中选取的激活函数为Sigmoid。

注：
LeNet有一个很有趣的地方，就是S2层与C3层的连接方式。在原文里，这个方式称为“Locally Connect”。
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/3.jpg" />
<div>locally connect in LeNet-5</div>
</center>
规定左上角为(0,0)，右下角为(5,15)，那么在(n,m)位置的“X”表示S2层的第n个feature map与C3层的第m个kernel进行卷积操作。例如说，C3层的第0个kernel只与S2层的前三个feature map有连接，与其余三个feature map是没有连接的；C3层的第15个kernel与S2层的所有feature map都有连接。这难道不就是ShuffleNet？



## 目标检测发展历程

## 实例分割

## 目标识别

## 网络中的各种细节
### 感受野计算
### FLOPs计算
### 参数量计算
### 输入输出尺寸计算
输出_w=(输入_w-kernel_w+padding_l+padding_r)/stride_w+1

输出_h=(输入_h-kernel_h+padding_t+padding_b)/stride_h+1

池化卷积均可用此公式计算，注意横向、纵向有所区别时需各自计算，有时四边padding不同也需注意。
### 参数初始化方法
### 激活函数
### 优化器
### 预处理方法

## 项目相关

## 竞赛相关

## 智力题

**ac酱**

**更新于2020-03-11 晚上**

> 参考资料：
* [](https://blog.csdn.net/kuweicai/article/details/93359992)
* [](https://zhuanlan.zhihu.com/p/31006686)

* [](https://zhuanlan.zhihu.com/p/93069133)
* [](https://zhuanlan.zhihu.com/p/22659166)
* [](https://zhuanlan.zhihu.com/p/47391705)
* [](https://zhuanlan.zhihu.com/p/73688224)
* [](https://zhuanlan.zhihu.com/p/86447716)
* [](https://blog.csdn.net/kuweicai/article/details/102789420)
