论文解析见[https://blog.csdn.net/iiiiiiimp/article/details/122328093](https://blog.csdn.net/iiiiiiimp/article/details/122328093)

（1）编码器

编码器就是把输入的风格图像resize为256 * 256大小，然后经过几个Conv2d+spectral_norm+InstanceNorm2d+LeakyReLU后变为[1,512,4,4]大小然后展平送个两个全连接得到两个[1,256]的向量作为均值和标准差得到高斯分布z[1,256]

![在这里插入图片描述](https://img-blog.csdnimg.cn/fec6e3e83c04487ba5b7fad8260e89c8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_18,color_FFFFFF,t_70,g_se,x_16)

（2）生成器

生成器就是把Encoder生成的z[1,256]通过fc和reshape后变为[1,1024,4,8]，之后就是连续的几个上采样+SPADE，得到图片[1,3,256,512]。

![在这里插入图片描述](https://img-blog.csdnimg.cn/1f3a8efe85234857a9e7942017686224.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_12,color_FFFFFF,t_70,g_se,x_16)

SPADE

感觉没什么好说的，就和论文的结构是一样，把mask裁剪为和上一层的输出x一样的大小然后去运算。

![在这里插入图片描述](https://img-blog.csdnimg.cn/e107381a619248be99c771fb7d715a81.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

（3）鉴别器

用的多尺度鉴别器和以前模型唯一的区别去就是Conv后加了spectral_norm才InstanceNorm

![在这里插入图片描述](https://img-blog.csdnimg.cn/1ed5eae4f61e4483a8b04ff0cf90f459.png)

（4）训练

需要注意的是作者是把伪造图片和真实图片，说是为了避免真假图片的统计差异。

![在这里插入图片描述](https://img-blog.csdnimg.cn/ed70f2a0260448afbe1652b7c86dc4d6.png)