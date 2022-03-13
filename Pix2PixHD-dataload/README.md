# 写在前面

项目需要，研究了一下Pix2PixHD的运行方法，这里对一些个人感觉难以理解的代码进行讲解，如果有写的不对的地方欢迎指正~

## 数据格式

这里以citycapes数据集为例，参考了[这篇博客](https://blog.csdn.net/MVandCV/article/details/115331719?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-6.pc_relevant_paycolumn_v3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-6.pc_relevant_paycolumn_v3&utm_relevant_index=10)

**语义图像**：每个类别一种标签颜色，例如图片中的所有车的像素值都是26。citycapes的语义图像大小为[2048, 1024, 1]。

![在这里插入图片描述](https://img-blog.csdnimg.cn/1b3c3a7121324588852ff9a55ce0e4fe.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

注意Pix2PixHD并没有直接使用这幅语义图像，而是使用的上面那副。

![在这里插入图片描述](https://img-blog.csdnimg.cn/e039c2f299454c67a789adf838f2fec5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

**实例图像**：同一类别的不同实例个体用的标签颜色也不一样，例如下面四辆车，用肉眼看是一个颜色，但是用opencv工具显示出来发现像素值还是有差别的，四辆车的像素值分别为26010、26005、26008、26003。这里之所以像素值大小为26010应该是用的16位存的，而不是0~255那种8位来存。

citycapes的实例图像大小为[2048, 1024, 1]。

注意需通过cv2.imread(path, cv2.IMREA_UNCHANGED)才能无损的读取到实例的uint图像数据。

![在这里插入图片描述](https://img-blog.csdnimg.cn/8f0fea38aa824b79a948ae4f861a9beb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

## 数据处理

**one-hot编码**：citycapes一共有35类(34个类别+1个背景)，作者将语义图片NCHW[1, 1, 2048, 1024]编码成one-hot格式[1, 35, 2048 ,1024]，其中每一个通道代表一个类别，这里我们取出第26个通道[1, 1, 2048 ,1024]出来，发现这一个通道全是车子

![在这里插入图片描述](https://img-blog.csdnimg.cn/9783a6fff82245419a07edff59e10f9d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

**边缘映射**：作者将实例图像[1, 1, 2048 ,1024]转换为边缘图像[[1, 1, 2048 ,1024]]，如果转换的论文里面有。

![在这里插入图片描述](https://img-blog.csdnimg.cn/7b187119c8b6430cb299ef2a5fb75fee.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

**生成器输入**：将one-hot编码[1, 35, 2048 ,1024]与边缘映射[1, 1, 2048 ,1024按通道维度拼接变为[1, 36, 2048 ,1024]送入生成器进行训练。

## 如何运行

**如果你只有语义图像**

python train.py --name 训练名字 --no_instance --label_nc 0  --resize_or_crop none --dataroot 数据集名字

**如果你有语义图像+实例图片**

python train.py --name 训练名字  --label_nc 类别数  --resize_or_crop none --dataroot 数据集名字

**如果你想联合训练**（先训练全局生成器再训练局部增强器）

首先执行：

python train.py --name 训练名字  --label_nc 类别数 --netG global --resize_or_crop none --dataroot 数据集名字

训练完后再执行：

python train.py --name 训练名字  --label_nc 类别数 --netG local --resize_or_crop none --dataroot 数据集名字 --load_pretrain 训练全局生成器网络保存路径

## 可视化代码

下面对各个文件简单介绍一下

**get_edges.py**：边缘映射，将实例图变为边缘图。![在这里插入图片描述](https://img-blog.csdnimg.cn/fd04c57589ae4e2699e094ea680767e5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)

**get_onehot.py**：one-hot编码，把语义图进行one-hot编码，每一通道是一个类别，一个35个类别，所以有35个通道。

![在这里插入图片描述](https://img-blog.csdnimg.cn/9f764b4cfa5546d59f159a32e5741647.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_20,color_FFFFFF,t_70,g_se,x_16)**get_imgvalue.py**：鼠标指在哪里现实哪里的像素值，把鼠标移到车子上可以看到车子的像素值大小为26000多。

# 写在后面 

看论文固然重要，但是要更加深入的理解作者作者的思想，阅读和复现代码是少不了的。希望本文对大家学习Pix2PixHD有所帮助。

 <img src="https://img-blog.csdnimg.cn/9b03fb4cbf76498180bfeb48d08fe95e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_10,color_FFFFFF,t_70,g_se,x_16" width="10%">
