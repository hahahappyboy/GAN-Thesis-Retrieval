论文解析见[https://blog.csdn.net/iiiiiiimp/article/details/122328093](https://blog.csdn.net/iiiiiiimp/article/details/122328093)

（1）注意，pSp网络是没有鉴别器的，只有生成器。生成器是由编码器encoder和解码器decoder组成，decoder就是StyleGAN2的代码，完全没改，可以见我复现的StyleGAN2的代码。唯一变的就是StyleGAN2的latent code不再是来自18层全连接后的。而是由encoder提供。所以我们重点讲encoder，这也是作责的创新部分。

（2）encoder网络结构

输入是mask语义图

![在这里插入图片描述](https://img-blog.csdnimg.cn/aa0246ef9ef644089203b7489a580b52.png)

然后经过这么多个bottleneck_IR_SE，bottleneck_IR_SE就是resnet结构。所有bottleneck_IR_SE一共分为低中高3个阶段分界线为第6、20、23个bottleneck_IR_SE。可以认为是FPN结构中coarse、mid、fine的分界点，但是这个FPN结构只有下采样，没有上采样。这个过程使得输入从[1,11,256,256]下采样到[1,512,16,16]

![在这里插入图片描述](https://img-blog.csdnimg.cn/67bcdd03148a4f3daff1ee1597ceceef.png)

取出第6、20、23个bottleneck_IR_SE的feature map用于生成最终的w。这一部分对应论文图2中map2style的左边部分。

![在这里插入图片描述](https://img-blog.csdnimg.cn/3720161a7c4247088114969a7aabd9f3.png)

把这3组feature map送人map2style得到w。同理map2style也是分了3组的，总共有16个map2style，正好对应StyleGAN2的16层。

![在这里插入图片描述](https://img-blog.csdnimg.cn/92850d32a1cb4f85974267032a8b9f5c.png)

map2style就是几个卷积后再经过全连接，得到一个[1,512]大小的w

![在这里插入图片描述](https://img-blog.csdnimg.cn/9244dbd2178c4b7ebdb89a5df81811cb.png)

（3）不关心背景

作者所谓的让网络不关心背景，就是把生成图的中间部分截取出来再多计算一次损失。

![在这里插入图片描述](https://img-blog.csdnimg.cn/8c400bb508434543955e56d79f915093.png)