﻿论文解析见[https://blog.csdn.net/iiiiiiimp/article/details/122328093](https://blog.csdn.net/iiiiiiimp/article/details/122328093)

（1）生成器

OASIS的生成器还是主要参考了SPADE，只不过就是去掉了编码器部分，换为了随机噪声

![在这里插入图片描述](https://img-blog.csdnimg.cn/bfeedff7b89c4f039aa9d136f6cc855b.png)

（2）鉴别器

鉴别器是Unet的结构

![在这里插入图片描述](https://img-blog.csdnimg.cn/b133106bd7c9433085d3f81f6ae4cc8e.png)

不同的是，这里并不是使用的全卷积网络，下采样用的`AvgPool2d`上采样用的是`Upsample`，且每层都有跳跃连接

![在这里插入图片描述](https://img-blog.csdnimg.cn/ace9a13f005e44cf916e2ed87aa8c8d6.png)

（3）没有使用VGGLoss

（4）对抗损失

首先获取类别权重，需要注意的是作者把第0类归为0代表不关心的类别，这也对应cityscapes数据集第0类为unlabel类别。

![在这里插入图片描述](https://img-blog.csdnimg.cn/0b2c057d7aba4c068b1cb1f38de3d6ff.png)

得到目标类别，即label到底是到底是哪一类，然后做交叉熵分类。训练鉴别器的时候因为要求生成器的为假，所以target的值全为0

![在这里插入图片描述](https://img-blog.csdnimg.cn/8b5838924ec84f3788d9d0990c643b67.png)

（5）标签混合的内容Loss

鉴别器处理判断真假的对抗Loss以外还有标签混合Loss，和论文说的一样，产生一个随机的二值图，再通过乘法把fake_image和real_image融合起来

![在这里插入图片描述](https://img-blog.csdnimg.cn/c769540758dd4c4789c5511dc03b62ab.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/64ebd70aa52e43d5b43f34f3f39f0477.png)