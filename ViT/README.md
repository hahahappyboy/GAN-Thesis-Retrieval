论文解析见[https://blog.csdn.net/iiiiiiimp/article/details/122328093](https://blog.csdn.net/iiiiiiimp/article/details/122328093)

[参考了的代码](https://github.com/lucidrains/vit-pytorch)

（1）把[1,3,256,256]的图片拆分成64个patch得到[1,8 * 8, 32 *  32 * 3]，然后经过 Liner Projection of Flattened Patches把32 * 32 * 映射为1024得到[1,64,1024]

![在这里插入图片描述](https://img-blog.csdnimg.cn/09aed7dc07074f95a9f4aa21abf37d36.png)

（2）用`self.cls_token = nn.Parameter(torch.randn(1, 1, dim))`生成类别patch[1,1,1024]与（1）的[1,64,1024]按通道拼接得到[1,65,1024]的Patch Embedding

![在这里插入图片描述](https://img-blog.csdnimg.cn/d8223c0702574c8f822187f78b718e18.png)

（3）用`self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))`生成[1,65,1024]大小的Position Embedding直接加到Patch Embedding上得到[1,65,1024]的Embeded Patches，因为Liner Projection of Flattened Patches是MLP所以又加了个dropout层

![在这里插入图片描述](https://img-blog.csdnimg.cn/936b8255672641299d92c54d9e1d67c5.png)

（4）在Transformer中先用`nn.LayerNorm(dim)`对[1,65,1024]的Embeded Patches进行归一化然后送入全连接` self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)`中得到q、k、v [1，65，512]再用rearrange将其reshape为矩阵形状[1,8,65,64]，其中8为head的个数。![](https://img-blog.csdnimg.cn/afb5879762014491b183242414f468c8.png)

（5）按照如下公式进行点积得到

![在这里插入图片描述](https://img-blog.csdnimg.cn/75dc3f852f254665a7f2eaa600a59cd5.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/7e5a61b5dcee40a9b07e3f13e8b8a3d0.png)

（6）将点积后的结果(多个heads的)concat到一起在经过一个全连接，这是为了使得多个head的信息进行交互并且把维度转为最初的1024维，所以最终输出为[1，65，1024]。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2673706e76c34ce58595ef18e55ed8c2.png)