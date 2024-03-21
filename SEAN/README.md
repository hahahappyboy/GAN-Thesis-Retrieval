论文解析见[https://blog.csdn.net/iiiiiiimp/article/details/122328093](https://blog.csdn.net/iiiiiiimp/article/details/122328093)

SEAN主要就是生成器的结构很复杂
（1）提取风格矩阵ST
输入的是rgb真实图像和segmap

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a720979291534a1f86c839a1b6ae37db.png)

论文中的ConvLayers和T-ConvLayers就是一个encoder和decoder结构

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/78a598a874024814b9316e56e128f12c.png)

将真实图像[1，3，512，512]编码为[1,512,256,256]的风格无关特征图

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/cd2b24b195714119815fab5cd4251ad2.png)

之后就是将风格无关特征图codes[1，512，256，256]和缩放后的语义图Segmap[1，512，256，256]编码为风格矩阵ST的步骤为
1）遍历语义图每个通道j，这里假设one-hot编码后的语义图有12个通道。统计通道j中值为1的像素个数
2）如果像素个数>0说明该通道有语义信息，所以要编码该区域
3）设在语义图Segmap的通道j中为1值的位置（x,y），然后取出codes中（x,y）的值。例如假设语义图Segmap的通道j中为1值有35783，那么取出codes中（x,y）的值就有[512, 35783]，从而得到codes_component_feature [512, 35783]
4）然后对codes_component_feature 按维度1平均池化得到[512]。
5）这个[512]的向量就是真实图像在语义图第j个语义通道的风格编码向量
6）语义图有12个通道，所以会得到12个风格编码向量，大小就为[1,12,512]，这就是风格矩阵ST

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/1406d42c68004852aac37bdd501231f2.png)

（2）SEAN归一化
在得到风格矩阵ST后，论文叫style_codes
就将其style_codes和语义图seg，上一层特性图x送入到SEAN模块中。假设上一层特征图为[1，1024，16，16]，seg缩放后为[[1，12，16，16]
SEAN模块整体看就是残差结构

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/98ce765177314faa97466cf7b7c67e0a.png)

SEAN归一化在论文中叫ACE模块
首先是给上一层特性图x加上StyleGAN的Noise

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/54cd2acb31f4413eb75ea610b5d13eea.png)

假设语义图seg的j通道有104个值为1的像素
SEAN归一化就是先将第j个语义的风格向量[512]经过一个全连接j（这个全连接j只用于第j个语义向量），论文中应该叫风格卷积。
然后复制104份（论文中叫广播）得到component_mu [512,104]
最后将语义图seg中j通道值为1的104个值用风格图的值替换。从而得到风格图（Style Map）middle_avg [1，512，16，16]

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/44bbb838ea3c46a49c71b947ccd6cc3e.png)

之后便是SPADE的归一化了
将刚刚得到的风格图（Style Map）middle_avg做SEAN归一化得到参数gamma_avg，beta_avg
SPADE对segmap归一化得到gamma_spade、beta_spade
然后用一个可学习参数gamma_alpha、beta_alpha将SEAN归一化和SPADE归一化融合起来

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/cdb9698785d2417cb467ea44499c2aa4.png)