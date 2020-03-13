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
可能由于当时GPU连接间的处理限制， AlexNet使用两个单独的GPU在 ImageNet数据库上执行训练，因此常常能看到将其拆分为两个网络的结构示意图，每一部分的kernel数量为实际kernel数量的一半。第二，第四和第五个卷积层的内核仅与上一层存放在同一GPU上的内核映射相连。第三个卷积层的内核连接到第二层中的所有内核映射。全连接层中的神经元连接到前一层中的所有神经元。

Conv1阶段:  
卷积：  
输入：227×227×3，卷积核：11×11×3，卷积核个数：96，步长：4，输出：55×55×96  
激活函数：relu  
归一化：LRN（局部响应归一化层，Local Response Normalization Layer），local_size=5  
池化：  
类型：max pooling，池化窗口：3×3，步长：2，输出：27×27×96

Conv2阶段：  
卷积：  
输入：27×27×96，卷积核：5×5×96，卷积核个数：256，步长：1，padding：same（相同补白，此处为2×2，使得卷积后图像大小不变），输出：27×27×256  
激活函数：relu  
归一化：LRN，local_size=5  
池化：  
类型：max pooling，池化窗口：3×3，步长：2，输出：13×13×256

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
类型：max pooling，池化窗口：3×3，步长：2，输出：6×6×256

FC6阶段：  
输入：6×6×256，flatten:[-1,9216]，输出：4096  
激活函数：relu  
dropout：0.5

> 一说：  
第6层采用6\*6\*256尺寸的滤波器对输入数据进行卷积运算；每个6\*6\*256尺寸的滤波器对第六层的输入数据进行卷积运算生成一个运算结果，通过一个神经元输出这个运算结果；共有4096个6\*6\*256尺寸的滤波器对输入数据进行卷积，通过4096个神经元的输出运算结果；然后通过ReLU激活函数以及dropout运算输出4096个本层的输出结果值。  
很明显在第6层中，采用的滤波器的尺寸（6\*6\*256）和待处理的feature map的尺寸（6\*6\*256）相同，即滤波器中的每个系数只与feature map中的一个像素值相乘；而采用的滤波器的尺寸和待处理的feature map的尺寸不相同，每个滤波器的系数都会与多个feature map中像素相乘。因此第6层被称为全连接层。

>但是现在找到的源码基本都是直接将最后一个卷积层的输出降维，作为全连接层的输入，因此这一说法有待考证。

FC7阶段：
输入：4096，输出：4096  
激活函数：relu  
dropout：0.5

FC8阶段：
输入：4096，输出：1000  
损失函数:softmax-crossentropy

结构上重要的改进：  
* 使用ReLU激活函数：ReLU(x)=max(x,0)，具体在[激活函数](#激活函数)部分讲解。  
* 0.5概率的dropout来对抗过拟合，具体在[Dropout](#Dropout)部分讲解。  
* LRN（局部响应归一化层，Local Response Normalization Layer）  
    LRN模拟神经生物学上一个叫做 侧抑制（lateral inhibitio）的功能，侧抑制指的是被激活的神经元会抑制相邻的神经元。  
    引入这一层的主要目的，主要是为了防止过拟合，增加模型的泛化能力。  
    在神经网络中，我们用激活函数将神经元的输出做一个非线性映射，但是tanh和sigmoid这些传统的激活函数的值域都是有范围的（即他们自带归一化功能），但是ReLU激活函数得到的值域没有一个区间，所以要对ReLU得到的结果进行归一化。  
    <center>
    <img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/6.jpg" />
    <div>LRN</div>
    </center>

    其中a代表在feature map中第i个卷积核(x,y)坐标经过了ReLU激活函数的输出，n表示相邻的几个卷积核。N表示这一层总的卷积核数量。  
    k, n, α和β是hyper-parameters，他们的值是在验证集上实验得到的，其中k = 2，n = 5，α = 0.0001，β = 0.75。  
    具体方法是在某一确定位置(x,y)将前后各2/n个feature map求和作为下一层的输入。但是存在争论，说LRN Layer其实并没有什么效果，在这里不讨论。
* Overlapping Pooling  
    传统的卷积层中，相邻的池化单元是不重叠的，即步长等于池化窗口的尺寸。AlexNet中由于步长小于池化窗口的尺寸，因此相邻的池化窗口间存在重叠的部分。实验结果显示这样的池化方法相比于传统池化方法对准确率有略微的提升。


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
### Dropout
在机器学习的模型中，如果模型的参数太多，而训练样本又太少，训练出来的模型很容易产生过拟合的现象。  
Dropout可以比较有效的缓解过拟合的发生，在一定程度上达到正则化的效果。  
Dropout在每个训练批次中，通过忽略一部分的特征检测器（让一部分的隐层节点值为0），可以明显地减少过拟合现象。这种方式可以减少特征检测器（隐层节点）间的相互作用，检测器相互作用是指某些检测器依赖其他检测器才能发挥作用。  
Dropout简单的说就是：在前向传播的时候，让某个神经元的激活值以一定的（丢弃）概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征。
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/8.jpg" />
<div>Dropout</div>
</center>

具体流程：  
1. 首先根据Dropout的（丢弃）概率p随机临时地删除网络中的部分隐藏神经元，即节点值暂时置为0。  
2. 然后将输入x送入到经过Dropout处理后的网络进行前向传播，然后把得到的损失结果在处理后的网络中反向传播。一小批训练样本执行完这个过程后，在没有被删除的神经元上按照随机梯度下降法更新对应的参数（w，b）。  
3. 恢复被删掉的神经元（此时被删除的神经元保持原样，而没有被删除的神经元已经有所更新）。  
不断重复上述过程。

显然，被Dropout丢弃的神经元在该批次的学习中不参与前向传播和后向传播，在该批次学习结束后恢复原值，进入下一个学习批次。

在实践中通常有两种处理方式，一种是在训练时根据Dropout概率p随机丢弃节点（实际上也可丢弃节点对应的输入值），并对输入值乘以1/(1-p)进行放大；另一种是在训练时根据Dropout概率p随机丢弃节点，但不对输入进行缩放，而是在预测时对所有节点的权重乘以p进行缩小。前者将数值运算放在训练阶段，能够减小测试阶段的计算量，提升速度，称为inverted dropout。

训练时处理：
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/9.jpg" />
<div>添加Dropout后计算流程对比</div>
</center>

<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/10.jpg" />
<div>无Dropout的计算公式</div>
</center>

<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/11.jpg" />
<div>添加Dropout的计算公式</div>
</center>

其中Bernoulli函数即为0-1分布发生器，随机生成一个0、1向量。假设将Dropout的概率p设置为0.4，即有40%的神经元的值置为0（丢弃40%的神经元），那么得到的z的值将为原来的60%（剩余60%的神经元）。因此我们通常需要对y进行放大，乘以1/(1-p)，这里即为乘以5/3。如果未在训练时对y进行放大，那么在测试时就需要对权重进行缩小。

测试时处理：
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/12.jpg" />
<div>对每一个神经元权重乘以p进行缩放</div>
</center>

防止过拟合的原理：
1. 取平均的作用  
Dropout过程就相当于对很多个不同的神经网络取平均。  
我们用相同的训练数据去训练5个不同的神经网络，一般会得到5个不同的结果，此时我们可以采用 “5个结果取均值”或者“多数取胜的投票策略”去决定最终结果。
2. 减少神经元之间复杂的共适应关系  
Dropout导致两个神经元不一定每次都在一个Dropout网络中出现。这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，迫使网络去学习更加鲁棒的特征。
3. 相当于对样本增加噪声  
观点十分明确，就是对于每一个Dropout后的网络，进行训练时，相当于做了Data Augmentation，因为，总可以找到一个样本，使得在原始的网络上也能达到Dropout单元后的效果。比如，对于某一层，Dropout一些单元后，形成的结果是(1.5,0,2.5,0,1,2,0)，其中0是被drop的单元，那么总能找到一个样本与Dropout后的结果相同。这样，每一次Dropout其实都相当于增加了样本。

总的来说，由于Dropout是随机丢弃，故而相当于每一个mini-batch都在训练不同的网络，可以有效防止模型过拟合，让网络泛化能力更强，同时由于减少了网络复杂度，加快了运算速度。还有一种观点认为Dropout有效的原因是对样本增加来噪声，变相增加了训练样本。  
Dropout通常用于全连接层中和输入层中，很少见到卷积层后接Dropout，原因主要是卷积参数少，不易过拟合。


### 激活函数
#### ReLU
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/7.jpg" />
<div>ReLU</div>
</center>

\begin{equation} ReLU(x)=max{0,x} \end{equation}

这个激活函数应该是在实际应用中最广泛的一个。

优点：  
* x大于0时，其导数恒为1，这样就不会存在梯度消失（如在sigmoid接近饱和区时，变换太缓慢，导数趋于0，这种情况会造成信息丢失）的问题。  
* 计算导数非常快，只需要判断x是大于0，还是小于0。  
* 收敛速度远远快于Sigmoid和Tanh函数。  
* Relu会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。

缺点：  
* none-zero-centered  
* Dead ReLU Problem，指的是某些神经元可能永远不会被激活，导致相应的参数永远不能被更新。因为当x小于等于0时输出恒为0，如果某个神经元的输出总是满足小于等于0的话，那么它将无法进入计算。  
有两个主要原因可能导致这种情况产生: (1) 非常不幸的参数初始化，这种情况比较少见 (2) learning rate太高导致在训练过程中参数更新太大，不幸使网络进入这种状态。  
解决方法是可以采用MSRA初始化方法，以及避免将learning rate设置太大或使用adagrad等自动调节learning rate的算法。

#### Softmax

### 损失函数
### 优化器
### 预处理方法

## 项目相关

## 竞赛相关

## 智力题

**ac酱**

**更新于2020-03-12 中午**

> 参考资料：
* [](https://blog.csdn.net/kuweicai/article/details/93359992)
* [](https://zhuanlan.zhihu.com/p/31006686)

* [](https://zhuanlan.zhihu.com/p/93069133)
* [](https://zhuanlan.zhihu.com/p/22659166)
* [](https://zhuanlan.zhihu.com/p/47391705)
* [](https://zhuanlan.zhihu.com/p/73688224)
* [](https://zhuanlan.zhihu.com/p/86447716)
* [](https://blog.csdn.net/kuweicai/article/details/102789420)

* [](https://blog.csdn.net/kuweicai/article/details/93926393)
* [](https://blog.csdn.net/program_developer/article/details/80737724)



https://blog.csdn.net/program_developer/article/details/80737724
https://blog.csdn.net/kuweicai/article/details/93926393
https://blog.csdn.net/kuweicai/article/details/102789420
https://zhuanlan.zhihu.com/p/88946608
https://zhuanlan.zhihu.com/p/31006686
https://zhuanlan.zhihu.com/p/73688224
https://zhuanlan.zhihu.com/p/47391705
https://zhuanlan.zhihu.com/p/22659166
https://zhuanlan.zhihu.com/p/93069133
