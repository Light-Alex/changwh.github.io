---
layout: post
title:  "更换评论控件为gitment"
date:   2019-04-05 15:20:00 +0800
categories: 建站相关
tags: gitment
author: ac酱
mathjax: true
---

* content
{:toc}
多说评论下线，disquo在国内使用不便，畅言需要网站备案，幸运的是还有gitment这样一种方便稳定的评论工具可供我们使用...



## 更换步骤
### 申请一个Github OAuth Application
Github头像下拉菜单 > Settings > 左边Developer settings下的OAuth Apps > New OAuth App，根据说明填写相关信息：[直接访问](https://github.com/settings/applications/new)

Application name, Homepage URL, Application description 都可以随意填写.

Authorization callback URL 一定要写自己Github Pages的URL.

填写完上述信息后按Register application按钮，得到`Client ID`和`Client Secret`.
### 修改代码进行调用
在每个需要加入gitment的页面下添加代码：
```html
<div id="gitmentContainer"></div> 
<script src="https://cdn.jsdelivr.net/gh/theme-next/theme-next-gitment@1/gitment.browser.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/theme-next/theme-next-gitment@1/default.css"/>
<script>
    var gitment = new Gitment({
        id: '{{page.date}}',
        owner: '{{site.github_username}}',
        repo: '{{site.comment_gitment_repo}}',
        oauth: {
            client_id: '{{site.comment_gitment_clientId}}',
            client_secret: '{{site.comment_gitment_clientSecret}}',
        },
    });
    gitment.render('gitmentContainer')
</script>
```

如果使用的blog框架支持批量添加评论栏，就在相应的文件中进行添加。如我使用的这个框架，只需要在`_includes/comments.html`中添加即可。

之后在`_config.yml`中添加相应的用户信息，根据上面的代码，我们可知需要添加的信息为：`github_username`,`comment_gitment_repo`,`comment_gitment_clientId`,`comment_gitment_clientSecret`。因此在`_config.yml`中添加：
```xml
#gitment
github_username: [your github user name]
comment_gitment_repo: [your github repo name]
comment_gitment_clientId: [your clientId]
comment_gitment_clientSecret: [your clienrSecret]
```

同时注意到，还有一个`page.date`属性，这个是根据你提交的blog页面中的header中的date获取的。
### 在每个调用评论控件的页面中初始化评论系统
当上述步骤都完成之后，我们需要在每个调用评论控件的页面中初始化评论系统。在引入评论控件的页面中login后找到`Initialize Comments`按钮，点击即可。
通过这个操作，gitment将在你的存放blog的repo中新建一个issue，并且打上`id`，`gitment`的label。按照之前添加的代码，这里的`id`将会是你填写的每篇blog的创建时间，如`2019-03-17 16:55:00 +0800`。由于为issue打上label的操作是需要repo的权限的，因此读者是无法初始化评论系统的，需要blog的拥有者手动初始化。事实上，有人编写了批量初始化的脚本，有兴趣的读者可以自行查找。
## 坑
### [object ProgressEvent]
前两个步骤完成之后，在调用评论控件的页面中初始化评论系统时，浏览器将会弹出[object ProgressEvent]提示框。之后初始化将会失败，加载图标不停转圈。

在gitment的issue中发现这样的问题是普遍存在的，原因是gitment作者提供给使用者的加CORS header的服务已经停了，即引入的gitment.js中的一个请求URL失效了。而作者也将加CORS header的js代码开源了，因此我们可以自行搭建服务器以提供相同的服务。此外还需修改引入的gitment.js中失效的URL，由于这个gitment.js托管在原作者的github page仓库中且不再维护，因此需要另外修改后再托管到其他位置。

这些繁琐的工作已经有其他人完成了，因此事实上我们只需要做一些简单的修改即可解决这个问题。将之前添加的代码中的js获取地址和css获取地址由：
```html
<link rel="stylesheet" href="https://imsun.github.io/gitment/style/default.css">
<script src="https://imsun.github.io/gitment/dist/gitment.browser.js"></script>
```

修改成：
```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/theme-next/theme-next-gitment@1/default.css"/>
<script src="https://cdn.jsdelivr.net/gh/theme-next/theme-next-gitment@1/gitment.browser.js"></script>
```

在之前的步骤中，我已提前修改好了，之后遇到相同的问题，只需要重新寻找可用的`gitment.browser.js`并进行替换修改即可。有条件也可自行搭建服务，这样更便于维护。
    
### [Error: Validation Failed]
由于之前添加代码中的id字段一直读取失败~~后来才发现是格式写错，真是太菜了~~，尝试使用post.url作为id时，浏览器弹出[Error: Validation Failed]提示框。又是在gitment的issue中，找到了引发这个问题的原因。gitment会在repo中创建issue，并且根据id作为该issue的label，正是label的长度限制引起了这个问题。事实上，gitment的原作者在使用说明中只提到了id是"An optional string to identify your page. Default: location.href"，因此，很多人选择将页面的URL作为id。也有人为了解决URL过长，提出了不少可行的办法。但更为巧妙的方法是使用每一篇blog的创建时间进行标识，因为一般人不会同时创建多篇blog，即blog的创建时间与blog是存在对应关系的。因此将时间作为id也不失为一种可行的方法。

**ac酱**

**写于2019-04-05 下午**

> 参考资料：
* [Jekyll博客添加Gitment评论系统](https://blog.csdn.net/zhangquan2015/article/details/80178794)
* [gitment issue #170](https://github.com/imsun/gitment/issues/170)
* [gitment issue #112](https://github.com/imsun/gitment/issues/112)