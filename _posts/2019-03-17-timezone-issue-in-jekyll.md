---
layout: post
title:  "使用Jekyll时遇到的时区问题"
date:   2019-03-17 16:55:00 +0800
categories: 建站相关
tags: Jekyll timezone
author: ac酱
mathjax: true
---

* content
{:toc}
在本地使用Jekyll预览刚写好的Blog就遇到了问题...



## 使用Jekyll时遇到的时区问题
使用Jekyll预览刚写好的Blog就遇到加载不到页面的问题，一开始还以为是路径不匹配的问题

输入
```bash
jekyll build --verbose 
```
可以看到输出：
```bash
Generating...

Reading: _posts/2019-03-17-first-article-of-my-blog.md
Skipping: _posts/2019-03-17-first-article-of-my-blog.md has a future date

Generating: Jekyll::Paginate::Pagination finished in 0.000929 seconds.
```
可知这个文件由于拥有一个**未来**的日期而被跳过了，因此推测是时间设定上出现了问题。在百度上查了些资料，发现在`_config.yml`中添加：
```xml
future: true
```
可以解决这个问题，实际测试后也确实如此。

但是心中总觉得似乎有点不对，这样做并没有从源头上解决问题，于是查找jekyll是否有与时区相关的设置，果不其然。找到的另一篇文章中写道：

> 这个问题的原因是，_config.yml中的timezone，没有被jekyll正确的识别出来。
因此，jekyll默认使用UTC时间。而UTC时间是比china的要慢的，那么，date中的时间对于UTC来说就是未来的时刻啦。
因此，这篇文章暂时不会被jekyll build出来。
* 错误的解决方法:
--future参数  意思是把未来所有的文章都一起build，造成的问题是，jekyll将该篇未来的文章的时间延后。
* 正确的解决方法：
修改`_config.yml` 中`timezone: +800`

之后进行尝试，竟然报错了：

    Dependency Error: Yikes! It looks like you don't have tzinfo or one of its dependencies installed. In order to use Jekyll as currently configured, you'll need to install this gem. The full error message from Ruby is: 'cannot load such file -- tzinfo' If you run into trouble, you can find helpful resources at https://jekyllrb.com/help/!

    jekyll 3.8.5 | Error:  tzinfo

继续查找资料，得知这是缺少了`tzinfo-data`这一文件引起的，于是在命令行中输入
```bash
gem install tzinfo-data
```
安装后再次运行`jekyll s`，结果又报了另一个错误：
```bash
jekyll 3.8.5 | Error:  Invalid identifier: 800
```
这就奇怪了，于是又翻了翻jekyll的issue，找到了与这个不同的时区写法：
```xml
timezone: America/New_York
```
于是依葫芦画瓢：
```xml
timezone: China/Beijing
```
然而依然报错，没有办法，只能找找jekyll的开发文档，果然发现了点线索，里面有个连接导向了[List of tz database time zones的wiki](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)，于是在页面中搜索`Beijing`，真相大白：并没有以北京作为时区数据库的名称，而是以`Asia/Shanghai`作为名称指代北京时间。因此这个问题真正的解决方法是在`_config.yml`中添加：
```xml
timezone: Asia/Shanghai
```

---

然而事情并没有这么简单，就在我以为这样就能解决问题时，再一次的运行又出现了**未来**日期的错误，在每篇文章开头部分将

    date:   2019-03-17 03:18:00

改为

    date:   2019-03-17 03:18:00 +0800

似乎有效，jekyll能够读取这两个页面，但是进入页面一看，上面的日期却是错误的，这到底是怎么回事呢？

经过尝试，将`_config.yml`中添加的`timezone: Asia/Shanghai`删除，保留markdown页面中date后添加的`+0800`似乎可行。在本地运行时没有出现错误，但是将页面push到github之后，时间又乱套了，因此应该是本地时间与github时间不一致的问题。于是现在陷入一种尴尬的境地：
* 在本地运行jekyll时，不对`_config.yml`进行修改，而是在每篇Blog的head中的date添加时区标志，如东八区，则将date写成`date:   2019-03-17 12:07:00 +0800`
* 将页面部署到github上时，则需要修改`_config.yml`，添加`timezone: Asia/Shanghai`，每篇Blog的head中的date不需要添加时区标志

又在jekyll的issues页面里翻了翻，发现一个有趣的issue:[Update dependency constraint to allow for tzinfo v2.0.0](https://github.com/jekyll/jekyll/issues/7565)，里面提到了jekyll对tzinfo的版本支持是`1.2.0`，对于最新的`2.0.0`是不支持的。因此联想到使用`jekyll build --verbose`进行debug的时候，显示：
```bash
Requiring: tzinfo
Timezone: Asia/Shanghai 00:00
```
十分异常，根据我的理解，输出的应为`Timezone: Asia/Shanghai +08:00`。尝试降低`tzinfo`版本至`1.2.0`：
```bash
gem uninstall tzinfo
gem install tzinfo -v 1.2.0
```
再次进行debug，输出：
```bash
Requiring: tzinfo
Timezone: Asia/Shanghai +08:00
```
似乎真相大白了。在md文件的头部的date字段修改为`date:   2019-03-17 03:18:00 +0800`后，无论是在本地或是github上，都能正确地显示页面了。但是由于不确定是否就是这个原因引起的这一系列的问题，因此还需要进行一段时间的测试。

## 暂时的解决方案

* 将`tzinfo`降级至`1.2.0`
* 对`_config.yml`进行修改，添加`timezone: Asia/Shanghai`
* 在每篇Blog的开头日期后添加时区标志，如东八区，则将date写成`date:   2019-03-17 12:07:00 +0800`


**ac酱**

**更新于2019-03-17 下午**


> 参考资料：
* [jekyll不编译_post目录里的md文件](https://ferrisyu.com/2018/03/28/jekyll_not_gen_html.html)
* [jekyll _config timezone](https://blog.csdn.net/think_ycx/article/details/77460567)