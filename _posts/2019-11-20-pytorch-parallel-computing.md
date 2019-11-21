---
layout: post
title:  "pytorch的多gpu并行计算"
date:   2019-11-20 15:30:00 +0800
categories: pytorch
tags: pytorch parallel multi-gpu
author: ac酱
mathjax: true
---

* content
{:toc}
最近弄了台多显卡的服务器，因此就想体验一下在多gpu的环境下训练模型，正好手上有pytorch版的crnn训练代码，于是就拿来稍加修改，最后运行成功并能顺利训练，特此记录一下过程中遇到的问题。注意整个实验的环境为GTX2080TI*2+pytorch 1.3.0，显卡的数量及pytorch版本变更可能使实验的结果发生改变。



## 官方文档中的论述

官方文档将训练的过程全部模拟了一遍，包括参数定义，虚拟数据集构建、简单模型构建等。其中值得一提的是，在模型的forward()中打印模型的输入和输出的形状，这样将模型分发至不同的gpu时，每个模型各自的输入和输出便一目了然：
```python
def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
```
之后就是关键的运行模型部分。需要注意的是官方定义了device，后文中只有在数据和模型的分发上使用了这一变量。
```python
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
```
可以明确的看到，在gpu可用的情况下，device被定义为了"cuda:0"，后续模型与数据的分发似乎也只是分发到这个device上。那么问题就来了，为什么`只需要将模型与数据分发至一个gpu上就能实现并行计算`？这个问题暂时保留，接着往下看代码。

之后使用torch.nn.DataParallel()对模型包装时，没有如其他代码中的先对模型调用model.cuda()，而是在包装后，使用model.to(device)，将模型分发至指定的设备中。同时，输入的数据也是通过data.to(device)分发至指定的设备。
```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1: 
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

for data in rand_loader: 
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```
运行代码得到输出：

        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])


## 自己的实验

通过实际运行，确实可以得到文档中的结果。但是，从别处看到的许多pytorch多卡训练的代码和官方文档中的有稍许不同，主要是模型和输入数据的分发方式，因此我们修改代码进行实验，并且试图找到之前提出的那个问题的答案。

修改模型与数据的分发函数，文档中的是通过to(device)进行模型与数据的分发的。这里我们首先通过打印整个batch的输入及输出和每个gpu中模型的输入及输出验证一下to(device)是如何对数据进行分发的。通过观察运行结果，可以得出结论，`torch.nn.DataParallel()包装后的模型会将输入数据对半分为两部分，第一部分作为device='cuda:0'的输入，第二部分作为device='cuda:1'的输入，之后分别将两个device的输出按照相同顺序拼接成最终的输出，放置到device='cuda:0'上。`实际上to(device)并不会将数据或者模型进行指定设备以外的分发。

在官方文档中，可以查询到cuda()也可将所有模型参数和缓冲区移动到图形处理器。将to(device)修改为cuda()，再次运行程序，结果同to(device)。再次修改为cuda(0)，指定为0号设备，结果依然相同。

但是修改成cuda(1)之后，程序报错。
```
RuntimeError: module must have its parameters and buffers on device cuda:0 (device_ids[0]) but found one of them on device: cuda:1
```
说明在进行并行计算的时候，需要把所有数据统一发送至device_ids[0]，这里就需要阅读torch.nn.DataParallel()的代码。我们可以看到torch.nn.DataParallel()的构造器有如下的参数：
```
Args:
    module (Module): module to be parallelized
    device_ids (list of int or torch.device): CUDA devices (default: all devices)
    output_device (int or torch.device): device location of output (default: device_ids[0])
```
其中device_ids用于指定并行计算的设备，默认值为所有可用的cuda设备，output_device用于指定输出的设备，默认值为device_ids[0]。
报错是由以下代码抛出的：
```python
for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))
```
从中我们可以得到一些信息，在进行并行计算时，pytorch需要对数据进行校验，检查他们所在的位置，self.src_device_obj在之前进行了定义，为device_ids[0]。也就是说在进行并行计算前，pytorch要求将模型的参数、缓存都放在self.src_device_obj中，即device_ids[0]。

于是修改model.cuda(0)，但是仍然将输入data.cuda(1)，此时却能正常运行。说明对输入数据放置于哪张显卡中似乎并没有特别要求。`（进一步实验，似乎输入数据都不需要分发至gpu中）`

之后将代码修改为：
```python
model = nn.DataParallel(model.cuda(1),[1,0])
```
代码能够正常运行，这也验证了刚才我们的结论，需要`将模型分发至devices_ids[0]指代的那张显卡中`。同理，使用to(device)也需要遵循相同的原则。

然后在torch.nn.DataParallel中将输入平均划分到各个gpu上并执行。这也能回答刚才提出的问题，为什么`只需要将模型与数据分发至一个gpu上就能实现并行计算`。

## 关于to()
查询了一些资料，发现自从pytorch0.4.0发布之后，新增了to()方法，使Tensors和Modules可容易地将对象移动到不同的设备（代替以前的cpu()或cuda()方法）。因此我们可以得到结论：`所有的cuda()都能替换为to(device)`。

## 总结
1. 使用torch.nn.DataParallel()包装需要进行并行计算的模型，注意参数devices_id用于指定进行运算的gpu，默认值为检测到的所有可用gpu，output_device用于指定输出的gpu，默认值为device_ids[0]。
2. 输入数据进入模型进行计算时似乎会自动由device:cpu,type:torch.FloatTensor，转换为device:cuda:0,type:torch.cuda.FloatTensor(待验证)，保险起见还是将其同样分发给经过torch.nn.DataParallel()包装的模型的device_ids[0]。
3. 所有cuda()都可以被to(device)替换。
4. 加载预训练模型时，如果原来的权重是由单卡训练得到的，可能需要对OrderedDict的key进行修改。
5. 由于torch.nn.DataParallel()会将各个gpu的计算结果暴力地堆叠起来，所以如果你的模型的forward()中使用了permute()进行维度换位，需要特别注意。举个例子，在不使用torch.nn.DataParallel()的crnn中，cnn的输出为[32,512,65]，经过permute(2,0,1)之后，变为[65,32,512]输入到rnn中，最后得到输出[65,32,5530]。我们可以明显的看出，在这次的训练中，batchsize为32。但是当使用torch.nn.DataParallel()包装crnn之后，模型最后的输出却是[130,16,5530]。也就是说，在torch.nn.DataParallel()处理输出数据时，默认的将输出tensor的第一维度视为batchsize。因此将两张gpu上的输出数据进行合并时，会使输出tensor的第一维度数值翻倍。这与我们的预期不符。那么应该怎么做呢，只要保证被torch.nn.DataParallel()包装的整个模型的输出tensor的第一个维度为batchsize，之后再根据需要使用permute()进行维度换位就行了。

**ac酱**

**完成于2019-11-20 晚上**

> 参考资料：
* [可选: 数据并行处理](https://pytorch.apachecn.org/docs/1.2/beginner/blitz/data_parallel_tutorial.html)
* [Pytorch之Dataparallel源码解析](https://www.cnblogs.com/marsggbo/p/10962763.html)
* [Pytorch使用To方法编写代码在不同设备(CUDA/CPU)上兼容(device-agnostic)](https://ptorch.com/news/189.html)