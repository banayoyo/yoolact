copy from https://github.com/JoshuaQYH/simple-yolact-pytorch

# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s

```
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 
```

# 前言

`YOLACT`是一个简单的用于实时实例分割的全卷积模型。本`repo`是我对这篇论文[YOLACT Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)的一个简单pytorch实现(力求达到精简有效），官方实现的全面（庞杂）版本请查看这个`repo`。[戳我](https://github.com/dbolya/yolact)

预想的主要成果：

- [ ] 打开电脑摄像头，进行实时实例分割，实现开箱即用(`linux`)；
- [ ] 输入一张图片，返回实例分割结果;
- [ ] 对论文进行深度解读，在代码中尽可能使用详细的中文来描述实时实例分割的过程，希望对实时分割学习者有一定的帮助；
- [ ] 增加对其他实例分割模型如Mask RCNN的对比见解，尝试对论文进行一定程度的改进和优化;

个人进展记录在[DailyNote](https://github.com/JoshuaQYH/yolact-pytorch/blob/master/DailyNote.md)文件中。

Well, start！
