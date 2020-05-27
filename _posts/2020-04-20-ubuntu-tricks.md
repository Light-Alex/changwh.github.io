---
layout: post
title:  "Ubuntu的使用小技巧"
date:   2020-04-20 22:00:00 +0800
categories: Ubuntu
tags: Ubuntu Linux
author: ac酱
mathjax: true
---

* content
{:toc}
Ubuntu的使用小技巧




## 用户相关
### 查看本机所有用户
```bash
cat /etc/passwd
```
### 添加用户
```bash
# 新增用户，--home指定用户目录位置（若不存在则新建）
sudo adduser USER_NAME --home USER_HOME
# 将新增的用户添加至sudo用户组
sudo usermod -aG sudo USER_NAME
```
### 删除用户
```bash
# -r 表示将用户目录等一起删除
sudo userdel -r USER_NAME
```
## SCREEN相关
### 查看本机现有的Session
```bash
screen -ls
```
### 创建Session
```bash
screen -S YOUR_SESSION
```
### 进入Session
```bash
# -D 表示将Attach的Session Deattach
screen -D -r YOUR_SESSION
```
### 删除Session
```bash
screen -S YOUR_SESSION -X quit
```

## update-alternatives
用于处理 Linux 系统中软件版本的切换，使其多版本共存。
### 安装后添加切换选项
```bash
# sudo update-alternatives --install <链接> <名称> <路径> <优先级>
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 40
```
### 进行切换
```bash
sudo update-alternatives --config gcc
```
### 删除切换选项
```bash
sudo update-alternatives --remove gcc /usr/bin/gcc-4.9
```
