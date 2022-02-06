
[参考了官方代码](https://github.com/znxlwm/UGATIT-pytorch)

CAM是把经过下采样和Res模块后得到的特征r3[1,64,64,64]\(NCHW)，进行全局平局池化和全局最大池化得到2个通道维度数量的向量[1,64,1,1]，再将这个向量拉平后送入64->1的FC层，从而得到两个1*1的值gap_logit和gmp_logit，然后将FC的权重取出来乘到r3上得到注意力图gap和gmp，经过通道合并后送入1*1卷积将通道还原。

![在这里插入图片描述](https://img-blog.csdnimg.cn/ea4364e0794f44e390d7f28b73493fe4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

AdaLIN是将CAM得到的注意力特征图直接拉平x_送入全连接self.gamma和self.beta得到AdaLIN参数gamma和beta，再送给上采样模块。

![在这里插入图片描述](https://img-blog.csdnimg.cn/86dd306fc1cd4eb4a40288cc0881c994.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

AdaLIN中可学习参数为rho[1,64,1,1]，计算注意力特征图的IN[1,64,64,64]和LN[1,64,64,64]，然后用rho控制其占比，最终用之前全连接得到的参数参数gamma和beta再标准化一次

![在这里插入图片描述](https://img-blog.csdnimg.cn/d849828897164f53b72485f80ef37c13.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

CAMLoss中注意，鉴别器的希望真实图片在辅助分类器中的得分越接近于1越好，希望伪造图片在辅助分类器的得分越接近于0越好，并且用的是MSELoss。

![在这里插入图片描述](https://img-blog.csdnimg.cn/a819b3b219b3418db1a1879afd92acde.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

生成器希望生成的伪造图在判别器的辅助分类器得分越接近1越好，并且还希望生成器输入为源域时A2A或B2B\<IdentityLoss需要>其辅助分类器接近于0，输入为目标域时A2B或B2A接近于1，并且用的是BCELoss。![在这里插入图片描述](https://img-blog.csdnimg.cn/f29701f4c0ec41efab948196b7d05f48.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)
