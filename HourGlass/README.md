论文解析见[https://blog.csdn.net/iiiiiiimp/article/details/122328093](https://blog.csdn.net/iiiiiiimp/article/details/122328093)

[参考了photo2cartoon代码](https://github.com/minivision-ai/photo2cartoon)

残差模块其实用的是Inception结构

![在这里插入图片描述](https://img-blog.csdnimg.cn/77b556d3af8e463d9c1a53a41d3463c8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

HourGlass模块的特征一个经过残差模型用于之后的跳跃连接，另一个经过池化后继续下采样编码。

![在这里插入图片描述](https://img-blog.csdnimg.cn/82910b97dd4d43c68f0e20c709a5340c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)