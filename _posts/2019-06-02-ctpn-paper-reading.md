---
layout: post
title:  "CTPN论文的阅读与学习"
date:   2019-06-02 12:00:00 +0800
categories: cv
tags: gitment
author: ac酱
mathjax: true
---

* content
{:toc}
最近在项目中使用了CTPN及CRNN进行中文OCR系统的构造，因此很有必要对这两个网络进行更进一步的了解与学习。本文将记录与CTPN相关的内容。



#CTPN流程概括
![test](_posts/res/2019-06-02-ctpn-paper-reading/architecture.jpg)

* 以VGG16为预训练模型，用于输入图片的特征提取，使用经过其最后一个卷积层（VGG论文中conv3-512）得到的特征图，大小为W*H*C
* 在特征图的每个像素点上使用3*3*C的滑动窗口提取进一步特征，用于RPN
* 每一行像素通过滑动窗口处理后得到的特征（W*3*3*C）作为256维BLSTM（包含两个128维的LSTM）的输入
* 将BLSTM的输出作为512维全连接层的输入
* 全连接层之后紧接着输出层，输出层包含三个分类或是回归。第二个2k scores表示的是k个anchor的类别信息（是否是字符）。第一个2k vertical coordinates（表示bounding box的高度和中心的y轴坐标，可决定上下边界）和第三个k side-refinement（表示bounding box的水平平移量，但在这里默认的宽度是固定的16像素）用来回归k个anchor的位置信息
* 通过简单的文本线构造算法，把分类得到的文字的proposal合并成文本线
