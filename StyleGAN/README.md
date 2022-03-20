论文解析见[https://blog.csdn.net/iiiiiiimp/article/details/122328093](https://blog.csdn.net/iiiiiiimp/article/details/122328093)

参考了：[https://github.com/tomguluson92/StyleGAN_PyTorch](https://github.com/tomguluson92/StyleGAN_PyTorch)

（1）映射网络

就是一个PixelNorm加上8个MLP

![在这里插入图片描述](https://img-blog.csdnimg.cn/a5173236e3fe4899a19bf3ed46496c50.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_16,color_FFFFFF,t_70,g_se,x_16)

论文中的Style mixing样式混合其实就是z经过映射网络得到w过后将w复制18份前8份乘上0.7，后10份乘上1.0

![在这里插入图片描述](https://img-blog.csdnimg.cn/75b883ed49144f2d823a66119abd0402.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

（2）主干网络

输入的是[1，512，4，4]的固定的高斯噪声，然后加上一个可学习的偏执bias

之后就是论文所讲的先up_sample->noise->adaIN->conv->noise->adaIN为一个block

![在这里插入图片描述](https://img-blog.csdnimg.cn/7b68aff67fb6425c947e54e88bea2d14.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

（3）各层加上Noise

noise的维度在模型初始化的时候就已经计算好了的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/73b61b21e8e34984a1c412447f394400.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

之后每个阶段把上采样或者卷积过后的x直接加上对应尺寸大小和维度的noise就行了

![在这里插入图片描述](https://img-blog.csdnimg.cn/8018ab9aacf1450994ca9f6b55ac17fc.png)

做到可学习是因为把noise和可学习参数相乘之后再加上x的

![在这里插入图片描述](https://img-blog.csdnimg.cn/a275b92ca8904ee3ada76503e0344823.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

（4）鉴别器结构

鉴别器结构就非常的常规了![在这里插入图片描述](https://img-blog.csdnimg.cn/69304ff23a36437e9621f4b1a72b84b0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)