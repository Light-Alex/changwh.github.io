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

网络的输入是227×227×3（论文中为224×224×3，但由于是通过裁剪，不然直接裁剪为227×227×3，免去后续的补边，Caffe也是如此修改的）的图像，网络一共有八层，其中前五个是卷基层，后3个是全连接层。  
可能由于当时GPU连接间的处理限制， AlexNet使用两个单独的GPU在 ImageNet数据库上执行训练，因此常常能看到将其拆分为两个网络的结构示意图，每一部分的kernel数量为实际kernel数量的一半。第二，第四和第五个卷积层的内核仅与上一层存放在同一GPU上的内核映射相连。第三个卷积层的内核连接到第二层中的所有内核映射。全连接层中的神经元连接到前一层中的所有神经元。

Conv1阶段:  
卷积：  
输入：227×227×3，卷积核：11×11×3，卷积核个数：96，步长：4，输出：55×55×96  
激活函数：ReLU  
归一化：LRN（局部响应归一化层，Local Response Normalization Layer），local_size=5  
池化：  
类型：max pooling，池化窗口：3×3，步长：2，输出：27×27×96

Conv2阶段：  
卷积：  
输入：27×27×96，卷积核：5×5×96，卷积核个数：256，步长：1，padding：same（相同补白，此处为2×2，使得卷积后图像大小不变），输出：27×27×256  
激活函数：ReLU  
归一化：LRN，local_size=5  
池化：  
类型：max pooling，池化窗口：3×3，步长：2，输出：13×13×256

Conv3阶段：  
卷积：  
输入：13×13×256，卷积核：3×3×256，卷积核个数：384，步长：1，padding：same（1×1），输出：13×13×384  
激活函数：ReLU

Conv4阶段：  
卷积：  
输入：13×13×384，卷积核：3×3×384，卷积核个数：384，步长：1，padding：same（1×1），输出：13×13×384  
激活函数：ReLU

Conv5阶段：  
卷积：  
输入：13×13×384，卷积核：3×3×384，卷积核个数：256，步长：1，padding：same（1×1），输出：13×13×256  
激活函数：ReLU  
池化：  
类型：max pooling，池化窗口：3×3，步长：2，输出：6×6×256

FC6阶段：  
输入：6×6×256，flatten:[-1,9216]，输出：4096  
激活函数：ReLU  
Dropout：0.5

> 一说：  
第6层采用6\*6\*256尺寸的滤波器对输入数据进行卷积运算；每个6\*6\*256尺寸的滤波器对第六层的输入数据进行卷积运算生成一个运算结果，通过一个神经元输出这个运算结果；共有4096个6\*6\*256尺寸的滤波器对输入数据进行卷积，通过4096个神经元的输出运算结果；然后通过ReLU激活函数以及Dropout运算输出4096个本层的输出结果值。  
很明显在第6层中，采用的滤波器的尺寸（6\*6\*256）和待处理的feature map的尺寸（6\*6\*256）相同，即滤波器中的每个系数只与feature map中的一个像素值相乘；而采用的滤波器的尺寸和待处理的feature map的尺寸不相同，每个滤波器的系数都会与多个feature map中像素相乘。因此第6层被称为全连接层。

>但是现在找到的源码基本都是直接将最后一个卷积层的输出降维，作为全连接层的输入，因此这一说法有待考证。

FC7阶段：
输入：4096，输出：4096  
激活函数：ReLU  
Dropout：0.5

FC8阶段：
输入：4096，输出：1000  
损失函数:Softmax-crossentropy

结构上重要的改进：  
* 使用ReLU激活函数：ReLU(x)=max(x,0)，具体在[激活函数](#激活函数)部分讲解。  
* 0.5概率的Dropout来对抗过拟合，具体在[Dropout](#dropout)部分讲解。  
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

### VGG(2014)
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/13.jpg" />
<div>VGG</div>
</center>

<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/14.jpg" />
<div>VGG</div>
</center>

VGG网络是2014年由牛津大学Visual Geometry Group提出的，是迄今为止最为经典的卷积神经网络，即使到了今天，也常使用其前半部分结构用于特征的提取。他不仅提出了使用基础块代替网络层的思想，使得网络结构看起来更为简单优雅，而且证明了增加网络的深度能够在一定程度上影响网络的最终性能。  
论文中一共提出了6种网络结构，每种结构名称后的数字代表他们具有的网络层数，他们之间的主要差别也就在于此。其中最为重要的是VGG16这一模型，VGG16和VGG19的效果相当，但是VGG16的结构更简洁美观，因此VGG16应用的更为广泛一些。我们就以VGG16为例进行介绍。

VGG16由13个卷积层和3个全连接层（5个池化层不计算在内）组成，其输入为224×224×3的图像。其他的VGG网络也仅在卷积层数量上有所差异，其余结构均相同。  
整体上看，VGG还是遵循着input -> n*(Conv->ReLU->MaxPool) -> 3*fc -> Softmax -> output的结构。  
正如前面所说，VGG使用了块代替层的思想，具体的来说，它提出了构建基础的卷积块和全连接块来替代卷积层和全连接层，而这里的块是由多个输出通道相同的层组成。VGG16中的13个卷积层被5个池化层分割为5个卷积块，每个卷积块中输出通道数相同，即使在不同的模型中，同层的卷积块中的输出通道数也相同。  
相比于LeNet和AlexNet较为复杂的卷积核、步长设置，VGGNet的卷积块中统一采用的卷积核大小为3×3，步长为1，padding为1，池化窗口大小为2×2，步长为2，padding为0。这种特性使得我们在将网络迁移至其他任务中时（输入网络的图片尺寸可能发生变化），不需要将过多的精力花费在这些超参数的设计上，只需要关注每一层输入输出的通道数即可。

卷积块1  
包含2个通道数为64，激活函数为ReLU的卷积层，输入是224×224×3的图像，输出为224×224×64。

池化层1
类型：max pooling，输入为224×224×64，输出为112×112×64。

卷积块2
包含2个通道数为128，激活函数为ReLU的卷积层，输入是112×112×64，输出为112×112×128。

最大池化层2
类型：max pooling，输入为112×112×128，输出为56×56×128。

卷积块3
包含3个通道数为256，激活函数为ReLU的卷积层，输入是56×56×128，输出为56×56×256。

最大池化层3
类型：max pooling，输入为56×56×256，输出为28×28×256。

卷积块4
包含3个通道数为512，激活函数为ReLU的卷积层，输入是28×28×256，输出为28×28×512。

最大池化层4
类型：max pooling，输入为28×28×512，输出为14×14×512。

卷积块5
包含3个通道数为512，激活函数为ReLU的卷积层，输入是14×14×512，输出为14×14×512。

最大池化层5
类型：max pooling，输入为14×14×512，输出为7×7×512。

全连接层1  
输入：7×7×512 -> 25088，输出：4096，激活函数：ReLU

全连接层2  
输入：4096，输出：4096，激活函数：ReLU

全连接层3  
输入：4096，输出：1000，激活函数：Softmax

卷积层的堆叠作用  
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/16.jpg" />
<div>卷积层的堆叠作用</div>
</center>

直观上我们会觉得大的卷积核更好，因为它可以提取到更大区域内的信息。但是实际上，大卷积核可以用多个小卷积核进行代替。例如，一个5×5的卷积核就可以用两个串联的3×3卷积核来代替，一个7×7的卷积核就可以用三个串联的3×3卷积核来代替。这样的替代方式有两点好处：

1. 减少了参数个数：  
假设输入与输出的channel数都为C。  
两个串联的小卷积核需要3×3×C×C×2=18C^2个参数，一个5×5的卷积核则有25C^2个参数。  
三个串联的小卷积核需要3×3×C×C×3=27C^2个参数，一个7×7的卷积核则有49C^2个参数。  
通过小卷积核的堆叠实现与大卷积核相同的感受野，从而大大减少了参数的数量。

2. 引入了更多的非线性：  
多少个串联的小卷积核就对应着多少次激活(activation)的过程，而一个大的卷积核就只有一次激活的过程。引入了更多的非线性变换，也就意味着模型的表达能力会更强，可以去拟合更高维的分布。  
值得一提的是，VGGNet结构C里还用到了1×1的卷积核。但是这里对这种卷积核的使用并不是像Inception里面拿来对通道进行整合，模拟升维和降维，这里并没有改变通道数，所以可以理解为是进一步的引入非线性。

总地来说，VGGNet的出现让我们知道CNN的潜力无穷，并且越深的网络在分类问题上表现出来的性能越好，并不是越大的卷积核就越好，也不是越小的就越好，就VGGNet来看， 3x3卷积核是最合理的。

作者对照实验组说明
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/15.jpg" />
<div>VGG</div>
</center>

1. A和A-LRN对比：分析LRN在网络中的效果  
A和A-LRN对比：精度提高0.1%，可以认为精度变化不大，但是LRN操作会增大计算量，所以作者认为在网络中添加LRN意义不大。

2. A和B对比：分析在网络靠近输入部分增加卷积层数的效果  
A和B对比：top-1提高0.9%，说明在靠近输入部分增加深度可以提高精度。

3. B和C对比：分析在网络靠近输出部分增加卷积层数的效果  
B和C对比：top-1提高0.6%，说明在靠近输出部分增加深度也可以提高精度。

4. C和D对比：分析1X1卷积核和3X3卷积核的对比效果  
C和D对比：top-1提高1.1%，说明3X3卷积核的效果要明显由于1X1卷积核的效果。

5. D和E对比：分析在网络靠近输出部分增加卷积层数的效果（这个和3）的作用有点像，只是网络进一步加深）  
D和E对比：top-1反而下降0.3%，说明深度增加到一定程度后，精度不再明显提升。

6. 总结论：  
网络深度增加可以提高精度，但是增加到一定程度之后就不适合再增加，增加3X3的卷积核比1X1的效果好。

multi-crop（多裁剪评估）和dense evaluation（密集评估）

作者在论文中提出了一种策略，即使用卷积层代替全连接层(具体理解可参考FCN网络)，这种策略不限制输入图片的大小，最终输出结果是一个w×h×n的score map。其中，w和h与输入图片大小有关，而n为类别数。而将w×h个值进行sum pool（对每一个channel进行求和），即可得到在某一类上的概率。这种方法叫做dense evaluation。  
另一种策略就是经常使用的卷积层+全连接层。通过将测试图片缩放到不同大小Q，Q可以不等于S(训练时图片大小)。在Q×Q图片上裁剪出多个S×S的图像块，将这些图像块进行测试，得到多个1×n维的向量。通过对这些向量每一纬求平均，得到一个1×n维的向量，从而得到在某一类上的概率。这种方法叫做multi-crop。  
作者认为，这两种方法的差别在于convolution boundary condition不同：dense由于不限制图片大小，可以利用像素点及其周围像素的信息（来自于卷积和池化），包含了一些上下文信息，增大了感受野，因此提高分类的准确度；而multi-crop由于从图片上裁剪再输网络，需要对图像进行padding，因此增加了噪声（个人理解应为裁剪的区域不同，相当于训练或检测的样本不同，等于引入了噪声）。但是由于multi-crop需要裁剪出大量图片，每张图片都要单独计算，增加了计算量，并且准确率提升不大，multi-crop比dense在top-1错误率上提升只有0.2。

全连接层->全局平均池化层（GAP）

此概念首先在NIN（Network In Network）中提出。全局池化（global pooling）指滑动窗口的大小与整个feature map的大小一样，这样一整张feature map只产生一个值。比如一个4×4的feature map使用传统的池化方法（2×2，2s），那么最终产生的feature map大小为2×2。而如果使用全局池化的话（4×4，1s，即大小与feature map相同），一个feature map只产生一个值，即输出为1×1。如果前一层有多个feature map的话，只需要把经过全局池化的结果堆叠起来即可。如果使用Average 池化方法，那么就成为Global Average Pooling，即GAP。  
从而可以总结出，如果输入feature map为W×H×C，那么经过全局池化之后的输出就为1×1×C。

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
### 神经元数量计算
输出维度是多少，神经元就有多少。如AlexNet的第一个卷积层输出为55×55×96，那么就有55×55×96=290400个神经元。

### 输入输出尺寸计算
输出_w=(输入_w-kernel_w+padding_l+padding_r)/stride_w+1  
输出_h=(输入_h-kernel_h+padding_t+padding_b)/stride_h+1

池化卷积均可用此公式计算，注意横向、纵向有所区别时需各自计算，有时四边padding不同也需注意。

### 参数量计算
* 卷积网络

    参数量 = (卷积核_w × 卷积核_h × 输入channel数 + 1) × 输出channel数
    
    括号内的为一个卷积核的参数量，+1表示bias，使用Batch Normalization时不需要bias，此时计算式中的+1项去除。

* 全连接网络

    参数量 = (输入神经元数量 + 1) × 输出神经元数量

    每一个输出神经元连接着所有输入神经元，且每个输出神经元还要加一个bias。

### FLOPs（floating point of operations，浮点运算数）计算
* 卷积网络

    FLOPs = [(输入channel数 × 卷积核_w × 卷积核_h) + (输入channel数 × 卷积核_w × 卷积核_h - 1) + 1] × 输出_w × 输出_h × 输出channel数  
    = 2 × 输入channel数 × 卷积核_w × 卷积核_h × 输出_w × 输出_h × 输出channel数

    (输入channel数 × 卷积核_w × 卷积核_h)表示一次卷积操作中的乘法运算量，(输入channel数 × 卷积核_w × 卷积核_h - 1)表示一次卷积操作中的加法运算量，+1表示bias，每个输出的神经元都对应一次卷积计算，因此需要乘于输出_w × 输出_h × 输出channel数。  
    在计算机视觉论文中，常常将一个‘乘-加’组合视为一次浮点运算，英文表述为'Multi-Add'，运算量正好是上面的算法减半，此时的运算量为：

    FLOPs = 输入channel数 × 卷积核_w × 卷积核_h × 输出_w × 输出_h × 输出channel数

* 全连接网络

    FLOPs = [输入神经元数量 + (输入神经元数量 - 1) + 1] × 输出神经元数量

    中括号的值表示计算出一个神经元所需的运算量：要得到一个输出神经元，需要对每个输入神经元做一次乘法，再将乘法得到的结果进行求和，即(输入神经元数量 - 1)次加法运算，+1表示bias。

### 链接数计算
* 卷积网络

    链接数 = 局部连接的输入层神经元数（输入channel数 × 卷积核_w × 卷积核_h + 1） × 卷积层神经元数（输出_w × 输出_h × 输出channel数）

* 全连接网络

    链接数 = (输入神经元数量 + 1) × 输出神经元数量

### 感受野计算
### 参数初始化方法
### 归一化
### 模拟退火
### 模型蒸馏
### 分组卷积
### 深度可分离卷积
### 空洞卷积
### 转置卷积
### 反卷积
### 1×1卷积(NIN)
### 可变形卷积
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
* none-zero-centered（非零均值， ReLU函数的输出值恒大于等于0），会使训练出现zig-zagging dynamics现象，使得收敛速度变慢。  
* Dead ReLU Problem，指的是某些神经元可能永远不会被激活，导致相应的参数永远不能被更新。因为当x小于等于0时输出恒为0，如果某个神经元的输出总是满足小于等于0的话，那么它将无法进入计算。  
有两个主要原因可能导致这种情况产生: (1) 非常不幸的参数初始化，这种情况比较少见 (2) learning rate太高导致在训练过程中参数更新太大，不幸使网络进入这种状态。  
解决方法是可以采用MSRA初始化方法，以及避免将learning rate设置太大或使用adagrad等自动调节learning rate的算法。

#### Softmax

### 损失函数
### 优化器
### 预处理方法

## 项目相关

## 竞赛相关
### Cascade rcnn
### DCN
### FPN
### RPN
### NAS
### ROI Align
### Smooth L1 loss
### Focal loss
### OHEM
### Soft-NMS
### MMDetection

## 智力题

**ac酱**

**更新于2020-03-19 凌晨**

> 参考资料：

LeNet  
* [](https://blog.csdn.net/kuweicai/article/details/93359992)
* [](https://zhuanlan.zhihu.com/p/31006686)

AlexNet  
* [](https://zhuanlan.zhihu.com/p/93069133)
* [](https://zhuanlan.zhihu.com/p/22659166)
* [](https://zhuanlan.zhihu.com/p/47391705)
* [](https://zhuanlan.zhihu.com/p/73688224)
* [](https://zhuanlan.zhihu.com/p/86447716)
* [](https://blog.csdn.net/kuweicai/article/details/102789420)
* [](https://zhuanlan.zhihu.com/p/20324656)

ReLU  
* [](https://blog.csdn.net/kuweicai/article/details/93926393)

Dropout  
* [](https://blog.csdn.net/program_developer/article/details/80737724)
* [](https://blog.csdn.net/GreatXiang888/article/details/99310164)

VGGNet  
* [](https://blog.csdn.net/kuweicai/article/details/102789420)
* [](https://zhuanlan.zhihu.com/p/88946608)
* [](https://zhuanlan.zhihu.com/p/31006686)
* [](https://zhuanlan.zhihu.com/p/73794404)
* [](https://zhuanlan.zhihu.com/p/47391705)
* [](https://zhuanlan.zhihu.com/p/23518167)
* [](https://www.zhihu.com/question/270988169)
* [](https://blog.csdn.net/hjimce/article/details/50187881)
* [](https://zhuanlan.zhihu.com/p/42233779)

各种细节
* [](https://www.jianshu.com/p/d4db25322435)
* [](https://blog.csdn.net/u013793650/article/details/78250152)
* [](https://blog.csdn.net/weixin_43200669/article/details/101063068)



后续备用
https://www.jianshu.com/p/7967556bcf75
https://blog.csdn.net/kuweicai/article/details/93926393
https://blog.csdn.net/weixin_30444105/article/details/98423768
https://blog.csdn.net/GreatXiang888/article/details/99296607
https://blog.csdn.net/GreatXiang888/article/details/99310164
https://blog.csdn.net/GreatXiang888/article/details/99293507
https://blog.csdn.net/GreatXiang888/article/details/99221246
https://www.jianshu.com/p/a936b7bc54e3
https://www.jianshu.com/p/26a7dbc15246
https://www.jianshu.com/p/491c7bc0e87c
https://www.jianshu.com/p/11bcb28ca0f0
https://zhuanlan.zhihu.com/p/22038289
https://tianchi.aliyun.com/forum/postDetail?postId=62131



临时
https://blog.csdn.net/kuweicai/article/details/102789420
https://zhuanlan.zhihu.com/p/92263138
https://zhuanlan.zhihu.com/p/31006686
https://zhuanlan.zhihu.com/p/73857137
https://zhuanlan.zhihu.com/p/73876718
https://zhuanlan.zhihu.com/p/73879583
https://zhuanlan.zhihu.com/p/73915627
https://zhuanlan.zhihu.com/p/47391705
https://zhuanlan.zhihu.com/p/22817228
https://zhuanlan.zhihu.com/p/93069133
https://zhuanlan.zhihu.com/p/42124583


resnet
https://zhuanlan.zhihu.com/p/23518167
