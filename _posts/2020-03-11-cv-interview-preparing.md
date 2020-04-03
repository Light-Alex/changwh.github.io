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
类型：max pooling，输入为224×224×64，池化窗口：2×2，步长：2×2，输出为112×112×64。

卷积块2
包含2个通道数为128，激活函数为ReLU的卷积层，输入是112×112×64，输出为112×112×128。

最大池化层2
类型：max pooling，输入为112×112×128，池化窗口：2×2，步长：2×2，输出为56×56×128。

卷积块3
包含3个通道数为256，激活函数为ReLU的卷积层，输入是56×56×128，输出为56×56×256。

最大池化层3
类型：max pooling，输入为56×56×256，池化窗口：2×2，步长：2×2，输出为28×28×256。

卷积块4
包含3个通道数为512，激活函数为ReLU的卷积层，输入是28×28×256，输出为28×28×512。

最大池化层4
类型：max pooling，输入为28×28×512，池化窗口：2×2，步长：2×2，输出为14×14×512。

卷积块5
包含3个通道数为512，激活函数为ReLU的卷积层，输入是14×14×512，输出为14×14×512。

最大池化层5
类型：max pooling，输入为14×14×512，池化窗口：2×2，步长：2×2，输出为7×7×512。

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
<div>对照实验结果</div>
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

<a id="globalavgpooling"></a>全连接层->全局平均池化层（GAP）

此概念首先在NIN（Network In Network）中提出。全局池化（global pooling）指滑动窗口的大小与整个feature map的大小一样，这样一整张feature map只产生一个值。比如一个4×4的feature map使用传统的池化方法（2×2，2s），那么最终产生的feature map大小为2×2。而如果使用全局池化的话（4×4，1s，即大小与feature map相同），一个feature map只产生一个值，即输出为1×1。如果前一层有多个feature map的话，只需要把经过全局池化的结果堆叠起来即可。如果使用Average池化方法，那么就成为Global Average Pooling，即GAP。  
从而可以总结出，如果输入feature map为W×H×C，那么经过全局池化之后的输出就为1×1×C。  
通常在最后一层卷积层中将输出的feature map数设置与类别数相同，可使用GAP对每个feature map求全图均值，通过Softmax得到每个类别的概率，这样做等效于卷积层后直接添加Softmax层。即GAP在减少参数量的同时，强行引导网络把最后的feature map学习成对应类别的confidence map。如果需要替代多个全连接层，就将多个1×1卷积的卷积核数量设置为对应全连接的单元数。
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/19.jpg" />
<div>FC等效于GAP处理</div>
</center>

### GoogLeNet（Inception v1，2014）
要理解GoogLeNet，首先需要先了解NIN（Network In Network）提出的MLPConv结构和通过GAP替代全连接层减少参数量的思想。

MLPConv这一结构使用了小的多层全连接网络替换掉卷积操作。但在实际上，是通过几个1×1卷积实现的，这是因为全连接层能够转化为卷积层，通过1×1卷积核实现了不同feature map间的信息交流。
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/20.jpg" />
<div>MLPConv</div>
</center>

CNN提取特征一般经过卷积/池化/激活三个步骤。其中，CNN卷积filter是一种广义线性模型（GLM），仅仅是将输入（局部感受野中的元素）进行线性组合，因此其抽象能力是比较低的。所提取的特征高度非线性时，我们需要更加多的filters来提取各种潜在的特征，但filters太多，导致网络参数太多，网络过于复杂，计算压力太大。为了提取更抽象的深层特征，提出用多层感知机（Muti-layer perception）对输入（局部感受野中的元素）进行更加复杂的运算，提高抽象表达能力。  
MLPConv层实际上是Conv+MLP（多层感知器），因为Conv是线性的，而MLP是非线性的，后者能够得到更高的抽象，泛化能力更强。  
在跨通道（cross channel，cross feature map）情况下，MLPConv等价于卷积层+1×1卷积层×2，所以此时MLPConv层也叫CCCP层（Cascaded Cross Channel Parametric Pooling)。  

关于GAP替代全连接层，可以参考VGG中[全连接层->全局平均池化层（GAP）](#globalavgpooling)部分。通过这样的操作，使得原来占有总参数量中极大比例的全连接层被池化层替代，同时降低了过拟合的风险。

<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/17.jpg" />
<div>GoogLeNet</div>
</center>
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/18.jpg" />
<div>GoogLeNet</div>
</center>

与VGG网络相同，GoogLeNet也使用了基础块的思想，它引入了Inception块替代一般的卷积层，主要通过增加网络的宽度和深度提高整个网络的精度。同时借助了NIN的思想，将全连接层变为了GAP+1×1卷积层，减少量全连接层带来的参数量过多、容易过拟合的问题。

GoogLeNet是一个输入为224×224×3图像的具有22层含有参数的网络（算上没有参数的pooling层则有27层），以池化层为分界，可以将整个网络结构分为5个模块。

模块1  
卷积层1：输入：224×224×3，卷积核：7×7，卷积核个数：64，步长：2，padding：same，激活函数：ReLU，输出：112×112×64  
池化层1：类型：max pooling，输入：112×112×64，池化窗口：3×3，步长：2，padding：same，输出：56×56×64

模块2  
卷积层2_3x3_reduce：输入：56×56×64，卷积核：1×1，卷积核个数：64，步长：1，padding：valid，激活函数：ReLU，输出：56×56×64
卷积层2_3x3：输入：56×56×64，卷积核：3×3，卷积核个数：192，步长：1，padding：same，激活函数：ReLU，输出：56×56×192
池化层2：类型：max pooling，输入：56×56×192，池化窗口：3×3，步长：2，padding：same，输出：28×28×192

模块3_a  
线路1  
卷积层3a_1×1：输入28×28×192，卷积核：1×1，卷积核个数：64，步长：1，padding：same，激活函数：ReLU，输出：28×28×64  
线路2  
卷积层3a_3x3_reduce：输入28×28×192，卷积核：1×1，卷积核个数：96，步长：1，padding：same，激活函数：ReLU，输出：28×28×96  
卷积层3a_3x3：输入28×28×96，卷积核：3×3，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：28×28×128  
线路3  
卷积层3a_5x5_reduce：输入28×28×192，卷积核：1×1，卷积核个数：16，步长：1，padding：same，激活函数：ReLU，输出：28×28×16  
卷积层3a_5x5：输入28×28×96，卷积核：5×5，卷积核个数：32，步长：1，padding：same，激活函数：ReLU，输出：28×28×32  
路线4  
池化层3a_pool：类型：max pooling，输入28×28×192，池化窗口：3×3，步长：1，padding：same，输出：28×28×192  
卷积层3a_pool_proj：输入28×28×192，卷积核：1×1，卷积核个数：32，步长：1，padding：same，激活函数：ReLU，输出：28×28×32  
合并层：输出：28×28×64+128+32+32=28×28×256

模块3_b  
线路1  
卷积层3b_1×1：输入28×28×256，卷积核：1×1，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：28×28×128  
线路2  
卷积层3b_3x3_reduce：输入28×28×256，卷积核：1×1，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：28×28×128  
卷积层3b_3x3：输入28×28×128，卷积核：3×3，卷积核个数：192，步长：1，padding：same，激活函数：ReLU，输出：28×28×192  
线路3  
卷积层3b_5x5_reduce：输入28×28×256，卷积核：1×1，卷积核个数：32，步长：1，padding：same，激活函数：ReLU，输出：28×28×32  
卷积层3b_5x5：输入28×28×32，卷积核：5×5，卷积核个数：96，步长：1，padding：same，激活函数：ReLU，输出：28×28×96  
线路4  
池化层3b_pool：类型：max pooling，输入28×28×256，池化窗口：3×3，步长：1，padding：same，输出：28×28×256  
卷积层3b_pool_proj：输入28×28×256，卷积核：1×1，卷积核个数：64，步长：1，padding：same，激活函数：ReLU，输出：28×28×64  
合并层：输出：28×28×128+192+96+64=28×28×480  
池化层3：类型：max pooling，输入：28×28×480，池化窗口：3×3，步长：2，padding：same，输出：14×14×480

模块4_a  
线路1  
卷积层4a_1x1：输入14×14×480，卷积核：1×1，卷积核个数：192，步长：1，padding：same，激活函数：ReLU，输出：14×14×192  
线路2  
卷积层4a_3x3_reduce：输入14×14×480，卷积核：1×1，卷积核个数：96，步长：1，padding：same，激活函数：ReLU，输出：14×14×96  
卷积层4a_3x3：输入14×14×96，卷积核：3×3，卷积核个数：208，步长：1，padding：same，激活函数：ReLU，输出：14×14×208  
线路3  
卷积层4a_5x5_reduce：输入14×14×480，卷积核：1×1，卷积核个数：16，步长：1，padding：same，激活函数：ReLU，输出：14×14×16  
卷积层4a_5x5：输入14×14×16，卷积核：5×5，卷积核个数：48，步长：1，padding：same，激活函数：ReLU，输出：14×14×48  
路线4  
池化层4a_pool：类型：max pooling，输入14×14×480，池化窗口：3×3，步长：1，padding：same，输出：14×14×480  
卷积层4a_pool_proj：输入14×14×480，卷积核：1×1，卷积核个数：64，步长：1，padding：same，激活函数：ReLU，输出：14×14×64  
合并层：输出：14×14×192+208+48+64=14×14×512

模块4_b  
线路1  
卷积层4b_1×1：输入14×14×512，卷积核：1×1，卷积核个数：160，步长：1，padding：same，激活函数：ReLU，输出：14×14×160  
线路2  
卷积层4b_3x3_reduce：输入14×14×512，卷积核：1×1，卷积核个数：112，步长：1，padding：same，激活函数：ReLU，输出：14×14×112  
卷积层4b_3x3：输入14×14×112，卷积核：3×3，卷积核个数：224，步长：1，padding：same，激活函数：ReLU，输出：14×14×224  
线路3  
卷积层4b_5x5_reduce：输入14×14×512，卷积核：1×1，卷积核个数：24，步长：1，padding：same，激活函数：ReLU，输出：14×14×24  
卷积层4b_5x5：输入14×14×24，卷积核：5×5，卷积核个数：64，步长：1，padding：same，激活函数：ReLU，输出：14×14×64  
线路4  
池化层4b_pool：类型：max pooling，输入14×14×512，池化窗口：3×3，步长：1，padding：same，输出：14×14×512  
卷积层4b_pool_proj：输入14×14×512，卷积核：1×1，卷积核个数：64，步长：1，padding：same，激活函数：ReLU，输出：14×14×64  
合并层：输出：14×14×160+224+64+64=14×14×512  

模块4_c  
线路1  
卷积层4c_1×1：输入14×14×512，卷积核：1×1，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：14×14×128  
线路2  
卷积层4c_3x3_reduce：输入14×14×512，卷积核：1×1，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：14×14×128  
卷积层4c_3x3：输入14×14×128，卷积核：3×3，卷积核个数：256，步长：1，padding：same，激活函数：ReLU，输出：14×14×256  
线路3  
卷积层4c_5x5_reduce：输入14×14×512，卷积核：1×1，卷积核个数：24，步长：1，padding：same，激活函数：ReLU，输出：14×14×24  
卷积层4c_5x5：输入14×14×24，卷积核：5×5，卷积核个数：64，步长：1，padding：same，激活函数：ReLU，输出：14×14×64  
线路4  
池化层4c_pool：类型：max pooling，输入14×14×512，池化窗口：3×3，步长：1，padding：same，输出：14×14×512  
卷积层4c_pool_proj：输入14×14×512，卷积核：1×1，卷积核个数：64，步长：1，padding：same，激活函数：ReLU，输出：14×14×64  
合并层：输出：14×14×128+256+64+64=14×14×512  

模块4_d  
线路1  
卷积层4d_1×1：输入14×14×512，卷积核：1×1，卷积核个数：112，步长：1，padding：same，激活函数：ReLU，输出：14×14×112  
线路2  
卷积层4d_3x3_reduce：输入14×14×512，卷积核：1×1，卷积核个数：144，步长：1，padding：same，激活函数：ReLU，输出：14×14×144  
卷积层4d_3x3：输入14×14×144，卷积核：3×3，卷积核个数：288，步长：1，padding：same，激活函数：ReLU，输出：14×14×288  
线路3  
卷积层4d_5x5_reduce：输入14×14×512，卷积核：1×1，卷积核个数：32，步长：1，padding：same，激活函数：ReLU，输出：14×14×32  
卷积层4d_5x5：输入14×14×32，卷积核：5×5，卷积核个数：64，步长：1，padding：same，激活函数：ReLU，输出：14×14×64  
线路4  
池化层4d_pool：类型：max pooling，输入14×14×512，池化窗口：3×3，步长：1，padding：same，输出：14×14×512  
卷积层4d_pool_proj：输入14×14×512，卷积核：1×1，卷积核个数：64，步长：1，padding：same，激活函数：ReLU，输出：14×14×64  
合并层：输出：14×14×112+288+64+64=14×14×528  

模块4_e  
线路1  
卷积层4e_1×1：输入14×14×512，卷积核：1×1，卷积核个数：256，步长：1，padding：same，激活函数：ReLU，输出：14×14×256  
线路2  
卷积层4e_3x3_reduce：输入14×14×512，卷积核：1×1，卷积核个数：160，步长：1，padding：same，激活函数：ReLU，输出：14×14×160  
卷积层4e_3x3：输入14×14×160，卷积核：3×3，卷积核个数：320，步长：1，padding：same，激活函数：ReLU，输出：14×14×320  
线路3  
卷积层4e_5x5_reduce：输入14×14×512，卷积核：1×1，卷积核个数：32，步长：1，padding：same，激活函数：ReLU，输出：14×14×32  
卷积层4e_5x5：输入14×14×32，卷积核：5×5，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：14×14×128  
线路4  
池化层4e_pool：类型：max pooling，输入14×14×512，池化窗口：3×3，步长：1，padding：same，输出：14×14×512  
卷积层4e_pool_proj：输入14×14×512，卷积核：1×1，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：14×14×128  
合并层：输出：14×14×256+320+128+128=14×14×832  
池化层4：类型：max pooling，输入：14×14×832，池化窗口：3×3，步长：2，padding：same，输出：7×7×832

模块5_a  
线路1  
卷积层5a_1x1：输入7×7×832，卷积核：1×1，卷积核个数：256，步长：1，padding：same，激活函数：ReLU，输出：7×7×256  
线路2  
卷积层5a_3x3_reduce：输入7×7×832，卷积核：1×1，卷积核个数：160，步长：1，padding：same，激活函数：ReLU，输出：7×7×160  
卷积层5a_3x3：输入7×7×160，卷积核：3×3，卷积核个数：320，步长：1，padding：same，激活函数：ReLU，输出：7×7×320  
线路3  
卷积层5a_5x5_reduce：输入7×7×832，卷积核：1×1，卷积核个数：32，步长：1，padding：same，激活函数：ReLU，输出：7×7×32  
卷积层5a_5x5：输入7×7×32，卷积核：5×5，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：7×7×128  
路线4  
池化层5a_pool：类型：max pooling，输入7×7×832，池化窗口：3×3，步长：1，padding：same，输出：7×7×832  
卷积层5a_pool_proj：输入7×7×832，卷积核：1×1，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：7×7×128  
合并层：输出：7×7×256+320+128+128=7×7×832

模块5_b  
线路1  
卷积层5b_1x1：输入7×7×832，卷积核：1×1，卷积核个数：384，步长：1，padding：same，激活函数：ReLU，输出：7×7×384  
线路2  
卷积层5b_3x3_reduce：输入7×7×832，卷积核：1×1，卷积核个数：192，步长：1，padding：same，激活函数：ReLU，输出：7×7×192  
卷积层5b_3x3：输入7×7×192，卷积核：3×3，卷积核个数：384，步长：1，padding：same，激活函数：ReLU，输出：7×7×384  
线路3  
卷积层5b_5x5_reduce：输入7×7×832，卷积核：1×1，卷积核个数：48，步长：1，padding：same，激活函数：ReLU，输出：7×7×48  
卷积层5b_5x5：输入7×7×48，卷积核：5×5，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：7×7×128  
路线4  
池化层5b_pool：类型：max pooling，输入7×7×832，池化窗口：3×3，步长：1，padding：same，输出：7×7×832  
卷积层5b_pool_proj：输入7×7×832，卷积核：1×1，卷积核个数：128，步长：1，padding：same，激活函数：ReLU，输出：7×7×128  
合并层：输出：7×7×384+384+128+128=7×7×1024  
池化层5：类型：avg pooling，输入：7×7×1024，池化窗口：7×7，步长：1，padding：valid，输出：1×1×1024

全连接层1：输入：1024，输出：1000，dropout：40%，激活函数：Softmax

对于GoogLeNet而言，重点是名为Inception块的结构。
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/21.jpg" />
<div>Inception块</div>
</center>

上图为Inception的结构，分布为简化版和降维版。两者的主要区别在于降维版在第2-4条线路上增加了1×1卷积来减少通道维度，以减小模型复杂度。

Inception包含4条并行线路，其中，第1、2、3条线路分别采用了1×1、3×3、5×5，不同的卷积核大小来对输入图像进行特征提取，使用不同大小卷积核能够充分提取图像特征。其中，第2、3两条线路都加入了1×1的卷积层，这两条线路的1×1与第1条线路1×1的卷积层的功能不同，第1条线路是用于特征提取，而第2、3条线路的目的是降低模型复杂度。第4条线路采用的不是卷积层，而是3×3的池化层，之后再加上1×1的卷积层。注意，第4条线路1×1卷积层的位置与第2、3条线路有所区别。  
为什么在第4条线路上先进行池化，之后再进行1×1卷积呢？`我的理解是`由于池化后，虽然特征图的尺寸和深度没有发生变化，但是特征图包含是信息量实际上是减少的（max pooling的下采样作用）。之后连接的1×1卷积，一方面实现了降维，以免该Inception块输出的维度过大，另一方面是为了将不同特征图在同一像素上的信息进行合并，重新将被max pooling去除掉的细节部分还原回来。  
最后，这4条线路的输出保持相同的尺寸，经过Filter Concatenation在维度上进行拼接，得到Inception块的输出。

值得注意的是，网络中有三个softmax，这是为了减轻在深层网络反向传播时梯度消失的影响，也就是说，整个网络的loss是由三个softmax共同组成的，这样在反向传播的时候，即使最后一个softmax传播回来的梯度消失了，还有前两个softmax传播回来的梯度进行辅助，通过两个辅助 softmax 分类器向模型低层次注入梯度，同时也提供了额外的正则化，也让底层网络提取的特征更具判别性。文中也提到这两个辅助 softmax 分类器的损失函数在计算总的损失是需要添加一个衰减系数，文中给出的是0.3。在对网络进行测试的时候，这两个额外的softmax将会被拿掉。这样不仅仅减轻了梯度消失的影响，而且加速了网络的收敛。

### Inception
### Xception
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

## 实例分割

## 目标识别
### 目标识别发展历程
### R-CNN
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/23.jpg" />
<div>R-CNN</div>
</center>

训练过程：
1. 用SS（Selective Search，选择性搜索）提取候选区域（2000个）  
穷举法或滑动窗口缺点：复杂度高，冗余候选区域；不可能顾及到每个尺度（指数级个数）。  
SS：  
   * 利用基于图的图像分割的方法初始化原始区域；  
   * 计算每两个相邻区域的相似度：
     * 保持多样性:三种多样性策略:多种颜色空间，考虑RGB、灰度、HSV及其变种等;多种相似度度量标准，既考虑颜色相似度，又考虑纹理、大小、重叠情况等;通过改变阈值初始化原始区域，阈值越大，分割的区域越少。
     * 给区域打分：不是每个区域作为目标的可能性都是相同的，因此需要衡量可能性，可以根据需要筛选区域建议个数。给予最先合并的图片块较大的权重，比如最后一块完整图像权重为1，倒数第二次合并的区域权重为2以此类推。但是当我们策略很多，多样性很多的时候呢，这个权重就会有太多的重合了，不好排序。文章做法是给他们乘以一个[0,1]的随机数，然后对于相同的区域多次出现的也叠加下权重，毕竟多个策略都说你是目标，也是有理由的嘛。这样我就得到了所有区域的目标分数，也就可以根据自己的需要选择需要多少个区域了。  
   * 合并所有相邻区域中最相似的两块，直到合并完整（合并成为整张图片）；  
   * 保存每次合并的结果，得到图片的分层表示。  

    ```
    输入: 一张图片
    输出：候选的目标位置集合L

    算法：
    1: 利用切分方法得到候选的区域集合R = {r1,r2,…,rn}
    2: 初始化相似集合S = ϕ
    3: foreach 遍历邻居区域对(ri,rj) do
    4:     计算相似度s(ri,rj)
    5:     S = S  ∪ s(ri,rj)
    6: while S not=ϕ do
    7:     从S中得到最大的相似度s(ri,rj)=max(S)
    8:     合并对应的区域rt = ri ∪ rj
    9:     移除ri对应的所有相似度：S = S\s(ri,r*)
    10:    移除rj对应的所有相似度：S = S\s(r*,rj)
    11:    计算rt对应的相似度集合St
    12:    S = S ∪ St
    13:    R = R ∪ rt
    14: L = R中所有区域对应的边框
    ```

2. 用CNN提取区域特征  
AlexNet，finetune最后一层（Softmax，21类，其中1类为背景，激活函数：Log loss），scale：227×227`（在原始图片上截取正方形（猜测是向短边两边延展），向四面再各取16像素作为填充，超过的部分（短边两侧延展和四面填充）用均值，再直接变形为227×227的大小）`，对每个建议框得到4096维特征（FC7）。对待所有的推荐区域，如果其和Ground Truth的IoU>=0.5就认为是正例`（同时确定类别？）`，否则就是负例。每轮SGD迭代（lr=0.001，预训练的十分之一），统一使用32个正例窗口（跨所有类别）和96个背景窗口，即每个mini-batch的大小是128。另外我们倾向于采样正例窗口，因为和背景相比他们很稀少。
3. 对区域进行SVM分类  
使用每个建议框的4096维特征（FC7输出）对每个目标类别训练SVM分类器（二分类，一对其余），识别该区域是否包含目标，共20个`（存疑，论文没有明确，其他博文中有说21个，包含背景）`。对于每个推荐区域的特征，将所有Ground Truth区域作为正例，与Ground Truth的IoU<0.3的推荐区域（包含背景，当前类别的部分，其他类别的正例）作为负例。  
相对CNN，SVM可以采用少量样本训练得到较好的效果。`（存疑）`  
使用NMS（IoU>=0.5）获取无冗余的区域子集。所有区域按分值从大到小排序，NMS剔除冗余后，剩余区域作为新的建议框子集。
4. 边框校准（可选）  
之后训练回归器，输入为每个建议框（与Ground Truth重合的IoU>0.6）通过CNN得到的特征图(Conv5输出)，修正候选区域中目标的位置，对于每个类都训练一个线性回归模型判断当前框位置准确性。

缺点：  
是一个多分步训练的过程。  
训练的时间（18（finetune）+63（特征提取）+3（SVM/Bbox训练）=84小时）和空间开销大，提取的特征需要存放在硬盘上。  
目标检测速度慢（47s/image（gpu））。

值得注意的点：  
mAP，Bounding-box Regression，Hard Negative Mining（难分样本挖掘），Ablation study（切除研究法），非极大值抑制（non-maximum suppression NMS），输入矩形区域的处理  
请参考[RCNN 论文阅读记录](https://zhuanlan.zhihu.com/p/42643788)

### SPP-Net
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/22.jpg" />
<div>SPP-Net</div>
</center>

相较于R-CNN，做了两点改进：
1. 直接输入整张图片进行特征提取（Conv5输出），所有区域共享卷积计算。在卷积结果上提取selective search得到建议框的特征图。
2. 引入空间金字塔池化，为不同尺寸的区域在Conv5输出上提取特征，以映射到尺寸固定的全连接层上。

SPP层的位置位于Conv5之后，替代Conv5的Pooling层，分为3个level共21个Bin（4×4，2×2，1×1），每个Bin内使用Max Pooling。

为什么可以在提取特征之后提取SS建议框的特征图？因为对于输入图片，特征的相对位置是不变的，因此对于某个ROI，只需要对特征图的相应位置进行特征提取即可。

训练过程：
1. 在ImageNet上对CNN模型进行pre-train。（与R-CNN相同）
2. 使用SS得到建议区域，对CNN模型finetune fc6、fc7、fc8层`（因为（感受野太大）计算困难，效率低，所以不对SPP层之前的卷积网络进行finetune）`，并得到所有SS区域的SPP层特征和fc7层特征。
    
    * 如何将SS的建议框映射到特征图中？  
  
        假设每一层的padding都是p/2，p为卷积核大小。对于feature map的一个像素（x',y'），其实际感受野为：（Sx'，Sy'），S为特征图前所有层中步长的乘积。  
        然后对于region proposal的位置，左上角的映射：($\lfloor$ $\rfloor$向下取整， $\lceil$ $\rceil$向上取整)  
        x' = $\lfloor$x / S$\rfloor$ + 1  
        右下角的映射：  
        x' = $\lceil$x / S$\rceil$ - 1  
        如果padding大小不一致，那么就需要计算相应的偏移值。

        另有说法：  
        spatial_scale = feature_map_size / input_img_size  
        roi_point = round(point × spatial_scale)  

        感觉这里的round()不精确，参考：https://www.runoob.com/w3cnote/python-round-func-note.html  
        `个人感觉和上面的说法类似，只需要参照上面的取整方法修改roi_point的计算公式即可。

    * SPP如何实现？  

        设卷积conv5输出的特征图尺寸为a×a，当前金字塔层的bins为n×n，则有：  
        windows_size = $\lceil$a / n$\rceil$  
        stride = $\lfloor$a / n$\rfloor$

3. 使用fc7层特征训练SVM分类器。（与R-CNN相同）
4. 使用SPP层特征训练Bounding box回归模型。

缺点：
是一个多分步训练的过程。  
训练的时间（16（finetune）+5.5（特征提取）+4（SVM/Bbox训练）=25.5小时）和空间开销大，提取的特征需要存放在硬盘上。  
新问题：不能finetune SPP层之前的所有卷积层参数。

### Fast R-CNN
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/24.jpg" />
<div>Fast R-CNN</div>
</center>

<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/25.jpg" />
<div>Fast R-CNN</div>
</center>

与上述两个网络相比更快的训练测试速度，更高的mAP，训练过程是端到端单阶段的（多任务损失函数，multi-task loss），所有层的参数都可finetune（取消SPP层），不需要离线存储特征。

在SPP-Net的基础上，Fast R-CNN引入了两个新技术：  
感兴趣区域池化层（RoI pooling layer）：取代SPP层；  
多任务损失函数（Multi-task loss）：SVM与Bbox Regression两个头合二为一，用一个统一的损失函数（Log loss+Smooth L1 loss）进行梯度下降。

RoI pooling:  
SPP的单层特例，将RoI区域的卷积特征拆分成H×W网格（例如VGG为7×7，因为原始的VGG在conv5后的max pooling得到的也是7×7的图像，这样做和原始vgg的fc1是兼容的），在每个Bin中进行Max pooling。如果RoI在特征图上映射的大小无法被bin在行和列上的个数整除，则有两种方法，一种是将多余的特征图上的点舍去，另一种是将余出的点放在行和列上的最后一个bin中。  
舍去多余的点将造成很大的误差，因此后续有[RoI align](#roi-align)进行改进。
 
`由于RoI之间可能存在重叠区域，反向传播过程中，为偏导之和。`???  
参考：https://zhuanlan.zhihu.com/p/59692298、max pooling的反向传播

Multi-task loss：  
同时考虑分类问题和定位问题，通过$\lambda$（文中为1）协调两个任务的权重。
$$L(p,u,t^u,v)=L_{cls}(p,u)+\lambda[u\geq1]L_{loc}(t^u,v)$$

分类器loss：  
$$L_{cls}(p,u)=-logp_u$$
$u$为Ground Truth类别  
对每个RoI，分类器输出的概率分布$p=(p_0,...,p_K)$（共有K+1类）  
若$p_u$无限接近于1，即分类器的预测结果为$u$，$L_{cls}(p,u)$则无限接近于0（$p_u$越大，$L_{cls}越小$）。

Bbox回归Smooth L1 loss：  
指示函数$[u\geq1]$：  
物体类别($u\geq$ 1)：1，有回归loss；  
背景类别($u=0$)：0，没有回归loss  
$$L_{loc}(t^u,v)=\sum_{i\in\{x,y,w,h\}}smooth_{L_1}(t^u_i-v_i)$$

$$smooth_{L_1}(x)=\begin{cases}
    0.5x^2 & if|x|<1 \\
    |x|-0.5 & otherwise.
\end{cases}$$

smooth L1 的分段是为了避免出现离群点使x过大，导致loss值过大。

$t^u=(t_x^u,t_y^u,t_w^u,t_h^u)$表示预测的x，y，w，h与Ground Truth的x，y，w，h经过$t_x=(G_x-P_x)/P_w$（y类似）和$t_w=log(G_w/P_w)$（h类似）的转换。这里有点归一化的意味。  
$v=(v_x,v_y,v_w,v_h)$表示实际上需要进行的偏移，例如$v_y=(G_y-B_y)/B_h$，$B_y$指SS得到的Bbox的中心点的y坐标。（待考证）

Mini-batch sampling:  
分层抽样（Hierarchical sampling）：  
Batch_size(128)=images(2)*RoIs(64)  
即每个batch取两张完整的图片，再从每张图片中取64个RoI。128个RoI中，正例（包含物体，与Ground Truth的IoU>=0.5）占25%，负例（包含背景，与Ground Truth的IoU在[0.1, 0.5)）占75%。这样做的目的是为了限制负例数量。 

训练时间9.5小时，单图测试0.32s。

### Faster R-CNN
<center>
<img src="https://raw.githubusercontent.com/changwh/changwh.github.io/master/_posts/res/2020-03-11-cv-interview-preparing/26.jpg" />
<div>Faster R-CNN</div>
</center>

Faster R-CNN = Fast R-CNN + RPN（区域建议网络）  
使用RPN取代离线的Selective Search，解决性能瓶颈，提供量少(约300)质优（高precision，高recall）的Region proposals。

RPN：  
输入：conv5输出的特征图  
1) 3×3卷积，卷积核个数：256，步长：1？，激活函数：ReLU  
2.1) Region proposals部分；1×1卷积，卷积核个数：4k，输出k组proposals的offsets（r,c,w,h）  
2.2) Classification部分：1×1卷积，卷积核个数：2k，输出k组（object score，non-object score）  
k表示anchor box类型数，通常情况下为9（3×3）。  
3种原始图片上的尺度（scale）：128，256，512  
3种宽高比（ratio）：1:1，1:2，2:1  
3种尺度和3种宽高比相互组合，形成9种anchor。  
anchor总数量为W×H×k，表示conv5特征图（W×H）的每个点上有k个anchor。  
RPN的loss：  
$$L(\{p_i\},\{t_i\})=\frac{1}{N_{cls}}\sum_iL_{cls}(p_i,p_i^*)+\lambda\frac{1}{N_{reg}}\sum_ip_i^*L_{reg}(t_i,t_i^*)$$

除于$N_{cls}$和$N_{reg}$（进行分类和回归的anchor个数）是为了进一步平衡两个loss。$\lambda$取值，$p_i^{*}$指示函数`？？？`

Classification部分：分为object和non-object两类。

Regression部分：使用Smooth L1 loss:  
$t_x=(x-x_a)/w_a,t_y=(y-y_a)/h_a$  
$t_w=log(w/w_a),t_h=log(h/h_a)$  
$t_x^{*}=(x^{*}-x_a)/w_a,t_y^{*}=(y^{*}-y_a)/h_a$  
$t_w^{*}=log(w^{*}/w_a),t_h^{*}=log(h^{*}/h_a)$  
$x，x_a，x^{*}$分别对应预测框，anchor框，ground truth框的中心点x坐标。y，w，h类似。

训练rpn的mini-batch：  
单张图片  
128个正样本：IoU>0.7的anchor框（或最大的IoU，因为有可能不存在IoU>0.7的anchor框）  
128个负样本：IoU<0.3的anchor框

训练流程：  
1. 训练RPN：  
预训练模型初始化，通过训练RPN得到Region proposals。
2. 训练Fast R-CNN（除RPN以外的部分称为Fast R-CNN）  
预训练模型初始化，与训练RPN时的卷积层不共享，由训练好的RPN提供Region proposals。
3. 调优RPN  
使用训练Fast R-CNN得到的卷积层参数对其初始化，固定卷积层参数，finetune剩余层。得到更精细的Region proposals。
4. 调优Fast R-CNN  
与调优RPN时的卷积层共享，固定卷积层参数，finetune剩余层。由调优好的RPN提供更精细的Region proposals。

### R-FCN
### Mask R-CNN
### U-NET
### YOLO
### YOLO v2
### YOLO v3
### YOLO v3-tiny
### YOLt
### SSD

## 网络中的各种细节
### 反向传播的推导

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
对于单通道（feature map数量为1）输入，单个1×1卷积核仅仅是将每个元素进行缩放，对特征来说没有意义。  
不过卷积网络的输入（上一层的feature map）一般都是多通道的，在这里1×1卷积核的作用非常强大：
1. 融合多个通道的特征。
2. 对通道数进行降维/升维。

通过1×1卷积核的作用之后，feature map的长宽不变，但是通道数会改变（使用多少个1×1卷积核就输出多少通道数的feature）。 
在GAP之前，要生成对应类别数的feature map，就要先用1×1卷积核进行卷积（10个类，就用10个1v1卷积核，这里有10个weight参数+10个bias参数）得到对应类别的特征图。

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

**更新于2020-04-03 下午**

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

GoogLenet
* [](https://www.jianshu.com/p/d42d67cbec40)
* [](https://www.cnblogs.com/shine-lee/p/11655836.html)
* [](https://blog.csdn.net/kuweicai/article/details/102789420)
* [](https://zhuanlan.zhihu.com/p/92263138)
* [](https://zhuanlan.zhihu.com/p/31006686)
* [](https://zhuanlan.zhihu.com/p/73857137)
* [](https://zhuanlan.zhihu.com/p/47391705)
* [](https://zhuanlan.zhihu.com/p/22817228)
* [](https://zhuanlan.zhihu.com/p/42124583)
* [](https://zhuanlan.zhihu.com/p/93069133)
* [](https://my.oschina.net/u/876354/blog/1637819)
* [](https://www.zhihu.com/question/325416643)

各种细节
* [](https://www.jianshu.com/p/d4db25322435)
* [](https://blog.csdn.net/u013793650/article/details/78250152)
* [](https://blog.csdn.net/weixin_43200669/article/details/101063068)


后续备用
https://www.jianshu.com/p/7967556bcf75
https://blog.csdn.net/kuweicai/article/details/93926393
https://blog.csdn.net/weixin_30444105/article/details/98423768
https://www.zhihu.com/question/57194292
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

https://www.zhihu.com/question/56024942/answer/498132341

inception v2+
https://my.oschina.net/u/876354/blog/1637819
https://zhuanlan.zhihu.com/p/73876718
https://zhuanlan.zhihu.com/p/73879583
https://zhuanlan.zhihu.com/p/73915627


resnet
https://blog.csdn.net/kuweicai/article/details/102789420
https://zhuanlan.zhihu.com/p/31006686
https://zhuanlan.zhihu.com/p/47391705
https://zhuanlan.zhihu.com/p/23518167
https://zhuanlan.zhihu.com/p/42440883
https://zhuanlan.zhihu.com/p/93069133



视频
https://www.bilibili.com/video/BV1e7411R7DS?p=3
https://www.bilibili.com/video/BV1R7411a7dL?p=12
萌萌站起来，本地视频
https://www.bilibili.com/video/BV1iJ411V7pM
https://www.bilibili.com/video/BV1iJ411V7A2
bilibili收藏夹 最新两个
https://www.bilibili.com/video/BV1S4411N7Nw?p=7
rcnn
https://zhuanlan.zhihu.com/p/42643788
https://cloud.tencent.com/developer/article/1495383
spp
https://zhuanlan.zhihu.com/p/27485018
https://blog.csdn.net/weixin_33672400/article/details/85943513
https://www.runoob.com/w3cnote/python-round-func-note.html
fast rcnn
https://zhuanlan.zhihu.com/p/43037119
https://zhuanlan.zhihu.com/p/42738847
faster rcnn
https://zhuanlan.zhihu.com/p/43812909
https://zhuanlan.zhihu.com/p/44612080
cascade rcnn
https://blog.csdn.net/u014380165/article/details/80602027
https://zhuanlan.zhihu.com/p/42553957
https://zhuanlan.zhihu.com/p/45036212
https://blog.csdn.net/Chunfengyanyulove/article/details/86414810
https://zhuanlan.zhihu.com/p/112828052
https://juejin.im/post/5b89377451882542d14da67e