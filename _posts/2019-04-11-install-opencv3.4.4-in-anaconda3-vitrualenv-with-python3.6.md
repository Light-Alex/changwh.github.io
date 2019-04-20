---
layout: post
title:  "在装有python3.6的Anaconda3虚拟环境中安装opencv3.4.4"
date:   2019-04-20 22:26:00 +0800
categories: 深度学习环境配置
tags: opencv anaconda python
author: ac酱
mathjax: true
---

* content
{:toc}
在python中使用opencv一直以来都是一件相对麻烦的事情，通过pip安装，经常有部分功能无法使用，通过conda安装，目前有一种方法能在我的环境中正常使用，但是最佳的方法，还是通过源代码编译，得到最适合自己环境的opencv包。现将在ubuntu18.04 + anaconda3 + python3.6环境下安装opencv3.4.4的过程做一个记录，以便将来的不时之需。



## 一、使用conda安装py-opencv
```bash
conda install py-opencv=3.4.2 # ubuntu18.04 python3.6
```
通过这种方法能够在conda中安装其他人编译好的opencv，是安装opencv最简单有效的方法。
## 二、使用源代码编译
### 1.确保本机环境为ubuntu18.04
目前只在ubuntu18.04中进行过测试，不能保证本安装过程适用于其他版本的ubuntu。
### 2.安装opencv的依赖项
```bash
$ sudo apt-get update

$ sudo apt-get install build-essential cmake unzip pkg-config   # developer tools

$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev       # image I/O packages

$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt-get install libxvidcore-dev libx264-dev              # video I/O packages

$ sudo apt-get install libgtk-3-dev                             # GTK library

$ sudo apt-get install libatlas-base-dev gfortran               # optimize various OpenCV functions 

$ sudo apt-get install python3-dev                              # Python 3 headers and libraries
```
### 3.下载opencv3.4.4源文件
```bash
$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.4.zip # opencv

$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.4.zip # opencv_contrib

$ unzip opencv.zip
$ unzip opencv_contrib.zip

$ mv opencv-3.4.4 opencv
$ mv opencv_contrib-3.4.4 opencv_contrib
```

### 4.准备anaconda3环境
```bash
$ conda create -n cv python=3.6
$ source activate cv

$ pip install numpy
```

### 5.配置及编译opencv
```bash
$ cd ~/opencv
$ mkdir build
$ cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
 -D CMAKE_INSTALL_PREFIX=/usr/local \
 -D INSTALL_PYTHON_EXAMPLES=ON \
 -D INSTALL_C_EXAMPLES=OFF \
 -D OPENCV_ENABLE_NONFREE=ON \
 -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
 -D BUILD_EXAMPLES=ON \
 -D BUILD_opencv_python3=ON\
 -D BUILD_opencv_python2=OFF\
 -D PYTHON3_EXCUTABLE=~/anaconda3/cv/chinese-ocr/bin/python3\
 -D PYTHON3_INCLUDE_DIR=~/anaconda3/cv/chinese-ocr/include/python3.6m\
 -D PYTHON3_LIBRARY=~/anaconda3/cv/chinese-ocr/lib/libpython3.6m.a\
 -D PYTHON_NUMPY_PATH=~/anaconda3/cv/chinese-ocr/lib/python3.6/site-packages ..
```
注意：涉及python位置的配置需要根据具体情况调整！这是整个安装过程中最关键的一步！！！

```bash
$ make -j8
```
make的参数由你的cpu核数决定：-j[num of cores + a few]

```bash	
$ sudo make install
$ sudo ldconfig

$ pkg-config --modversion opencv    # to verify the install
3.4.4
```
### 6.验证安装及安装后处理
经过上述步骤，编译好的opencv包应该已经存在于下面的文件夹中
```bash
$ ls /usr/local/python/cv2/python-3.6
cv2.cpython-36m-x86_64-linux-gnu.so
```
将其改名并链接到虚拟环境中python的site-packages文件夹中
```bash
$ cd /usr/local/python/cv2/python-3.6
$ sudo mv cv2.cpython-36m-x86_64-linux-gnu.so cv2.so

$ cd ~/anaconda3/cv/chinese-ocr/lib/python3.6/site-packages
$ ln -s /usr/local/python/cv2/python-3.6/cv2.so cv2.so
```
验证opencv是否安装成功
```bash
$ conda activate cv
$ python

>>> import cv2
>>> cv2.__version__
'3.4.4'
>>> quit()
```
注意：若此处出现ImportError: libfontconfig.so.1: undefined symbol: FT_Done_MM_Var，需要将/anaconda3/lib/下的libfontconfig*文件禁用或删除
```bash
$ cd ~/anaconda3/lib
$ delete libfontconfig*
```

清理安装过程文件
```bash
$ cd ~
$ rm opencv.zip opencv_contrib.zip
$ rm -rf opencv opencv_contrib
```


**ac酱**

**写于2019-04-20 晚上**

> 参考资料
* [ubuntu18.04LTS+Anaconda3+cmake编译安装opencv3.4.3](https://www.jianshu.com/p/6478b318cd8f)
* [Ubuntu 18.04: How to install OpenCV](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/)
*[make -j4 or -j8](https://stackoverflow.com/questions/15289250/make-j4-or-j8)