论文解析见[https://blog.csdn.net/iiiiiiimp/article/details/122328093](https://blog.csdn.net/iiiiiiimp/article/details/122328093)

（1）生成器的权重预测网络有点类似与之前说的HourGlass结构，就是跳跃连接之前其实是把下采样的图多经过了一个卷积层(入下图中的labellat1)，这样做的原因是想过滤调一些下采样特征图中concat后对上采样特征图来说没有必要补充的信息。

![在这里插入图片描述](https://img-blog.csdnimg.cn/903d23e2a36f450da3cf5aa1996748f9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_15,color_FFFFFF,t_70,g_se,x_16)

之后就是把concat后的特征图后经过一层decoder之后得到的预测权重再与mask(reszie成与特征图一样大小)caoncat后送入到主干网络的CC-Block中去了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/59c8434d86b545e186062ce6eb940a90.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_14,color_FFFFFF,t_70,g_se,x_16)

CC-Block中把seg(就是权重预测网络得到的特征图)经过gen_weights(其实就是两个卷积)得到的两个卷积权重分别作为通道卷积权重和条件注意力机制权重。可以看到，条件注意力卷积权重其实就是点乘。

![在这里插入图片描述](https://img-blog.csdnimg.cn/198da41c727a48609e0a57377c6da7aa.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

通道卷积的预测权重其实也是点乘

![在这里插入图片描述](https://img-blog.csdnimg.cn/9bc8dfacc2454e83b83eb2e6ea047509.png)

（2）鉴别器

先把拼接在一起的真实图像和伪造图像下采样得到的特征图作为特征匹配损失用到输入

![在这里插入图片描述](https://img-blog.csdnimg.cn/53d47608aa2e4ba9b3f1ce51b6a59cb0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

之后把上采样concat后的特征图与经过经过一层卷积后变为通道为1的patch(图中的patch_pred2、3、4)作为真假判断的patch

又上采样concat后的特征图经过一层卷积得到的特征图与缩放成相同大小的mask图相乘得到语义对齐度的patch

最后把真假判断的patch和语义对齐度的patch相加作为鉴别器的输出

![在这里插入图片描述](https://img-blog.csdnimg.cn/134ce36b3ad84dcb839adf575d430cf0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)