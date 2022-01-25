# (Pix2Pix)Image-to-Image Translation with Conditional Adversarial Networks


参考了[官方代码](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
时间原因只复现了模型和训练流程
作者生成器下采样用的是LeakyReLU，而上采样用的是ReLU，并且上采样还加了Dropout，归一化用的是BN，最后输出为Tanh
![在这里插入图片描述](https://img-blog.csdnimg.cn/2752977de5a7463d93ec4f00875035ac.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

跳跃连接前的通道拼接是按NCHW的C维度进行拼接

![在这里插入图片描述](https://img-blog.csdnimg.cn/4a8c7464135641008a45a5704255fac2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_16,color_FFFFFF,t_70,g_se,x_16)

鉴别器，输入是生成器的输入和真实/伪造图片，所以为6维，输出为1维的patch

![在这里插入图片描述](https://img-blog.csdnimg.cn/479af770f03e443da5eb774d75f13eb4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

训练，由于鉴别器输入用了detach，所以训练生成器时还要重新forward一次鉴别器

![在这里插入图片描述](https://img-blog.csdnimg.cn/54f8a32196af4c87b05db8dcaa0bbe9b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_18,color_FFFFFF,t_70,g_se,x_16)