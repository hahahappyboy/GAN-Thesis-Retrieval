论文解析见[https://blog.csdn.net/iiiiiiimp/article/details/122328093](https://blog.csdn.net/iiiiiiimp/article/details/122328093)

参考了[作者的代码](https://github.com/NVIDIA/pix2pixHD)

时间原因只复现了模型和训练流程

作者默认只使用了全局生成器，下采样+残差块+上采样，注意残差块用的是相加，不是通道拼接

![在这里插入图片描述](https://img-blog.csdnimg.cn/888a4b34d55f4b91ae37420460b33f1a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

鉴别器主要就是要把各个模块分开定义，这是为了提取中间的特征图用于后面的loss计算，forward函数也是返回的一个特征图List

![在这里插入图片描述](https://img-blog.csdnimg.cn/d04badfd2f834ab1b00bc30230b4797b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

![在这里插入图片描述](https://img-blog.csdnimg.cn/ac1b8f6f5beb492da20c3aa7179c65d4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_13,color_FFFFFF,t_70,g_se,x_16)

多尺度鉴别器就是在模型中定义两个鉴别器，一个鉴别器原图尺寸输入，另一个做一个平局池化后输入，最后返回的也是两个鉴别器各模块特征图的list

![在这里插入图片描述](https://img-blog.csdnimg.cn/468b6f1282454abb9d76ebaa01cb8bac.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_16,color_FFFFFF,t_70,g_se,x_16)

VGG的定义也和鉴别器差不多，从torchvision中得到与训练的VGG模型把他差分层几个模块，forward也是返回的是特征图的list，用于后面计算感受野损失

![在这里插入图片描述](https://img-blog.csdnimg.cn/bbb26595e60e4f6b93b6cab988f2d118.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)
