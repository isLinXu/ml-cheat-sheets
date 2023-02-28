# 什么是转置卷积层？

## 通过动画 gif 和[python 代码进行解释](https://github.com/aqeelanwar/conv_layers_animation)

转置卷积层也（错误地）称为反卷积层。反卷积层反转标准卷积层的操作，即如果通过标准卷积层生成的输出被反卷积，您将得到原始输入。转置卷积层类似于反卷积层，因为两者生成的空间维度相同。转置卷积不会按值反转标准卷积，而是仅按维度反转。

![img](https://miro.medium.com/max/1050/1*gjDHvfY6XELWPZ50rqFs1A.png)

转置卷积层与标准卷积层完全相同，但在修改后的输入特征图上。在解释相似性之前，让我们先看看标准的卷积层是如何工作的。

## 标准卷积层：

***大小为ixi\***的输入的标准卷积层由以下两个参数定义。

- **填充** ***(p)：\***在原始输入周围填充零的数量，将大小增加到***(i+2\*p)x(i+2\*p)\***
- **Stride** ***(s)：\***内核在输入图像上滑动时移动的量。

下图显示了卷积层如何作为两步过程工作。

![img](https://miro.medium.com/max/1050/1*gYAQUBj741P5-gbuboXfMA.png)

在第一步中，输入图像用零填充，而在第二步中，内核被放置在填充的输入上并滑动生成输出像素作为内核和重叠输入区域的点积。内核通过采用由步幅定义的大小的跳跃来滑过填充的输入。卷积层通常进行下采样，即输出的空间维度小于输入的空间维度。

下面的动画解释了不同步幅和填充值的卷积层的工作原理。

![img](https://miro.medium.com/max/1500/1*YvlCSNzDEBGEWkZWNffPvw.gif)

![img](https://miro.medium.com/max/1500/1*gXAcHnbTxmPb8KjSryki-g.gif)

![img](https://miro.medium.com/max/1500/1*34_365CJB5seboQDUrbI5A.gif)

![img](https://miro.medium.com/max/1500/1*WpOcRWlofm0Z0EDUTKefzg.gif)

对于给定大小的输入***(i)\***、内核***(k)\***、填充***(p)\***和步幅***(s) ，生成的输出特征图\******(o)\***的大小由下式给出

![img](https://miro.medium.com/max/1050/1*6OdZgz15qxMQKjp70Z30Kw.png)

## 转置卷积层：

另一方面，转置卷积层通常用于上采样，即生成空间维度大于输入特征图的输出特征图。就像标准卷积层一样，转置卷积层也由填充和步幅定义。padding 和 stride 的这些值是假设在输出上执行以生成输入的值。也就是说，如果您获取输出，并执行定义了步长和填充的标准卷积，它将生成与输入相同的空间维度。

实施转置卷积层可以更好地解释为 4 步过程

- **第 1 步：**计算新参数 z 和 p'
- **第 2 步：**在输入的每一行和每一列之间，插入 z 个零。这会将输入的大小增加到***(2\*i-1)x(2\*i-1)\***
- **第 3 步：**用 p' 个零填充修改后的输入图像
- **第四步：**对第三步生成的图像进行标准卷积步幅为 1

完整的步骤如下图所示。

![img](https://miro.medium.com/max/1050/1*54-7typHLLXhdvAhlku9SQ.png)

图片作者

下面的动画解释了不同步幅和填充值的卷积层的工作原理。

![img](https://miro.medium.com/max/1500/1*SpxCUPzNfb9C8TiAcrRr5A.gif)

![img](https://miro.medium.com/max/1500/1*gff0oa2iPygyCEjj7Fb3yg.gif)

![img](https://miro.medium.com/max/1500/1*WaBzh5OkmD-9EBLy5aXiug.gif)

![img](https://miro.medium.com/max/1500/1*L_hJRnywTpeTFJAaVZTRfQ.gif)

对于给定大小的输入***(i)\***、内核***(k)\***、填充***(p)\***和步幅***(s) ，生成的输出特征图\******(o)\***的大小由下式给出

![img](https://miro.medium.com/max/1050/1*2Bh5FuM2B6UAM9DC_-j95Q.png)

## 概括：

下表总结了两种卷积，标准的和转置的。

![img](https://miro.medium.com/max/1050/1*zbVS6lHvo9J4aRZeE-77lA.png)

- 转置卷积背后的思想是进行可训练的上采样
- 转置卷积是标准卷积，但具有修改的输入特征图。
- 步幅和填充**不**对应于图像周围添加的零的数量以及在将图像滑过输入时内核中的移位量，就像它们在标准卷积运算中那样。

## Python代码：

这些 gif 是使用 python 生成的。[完整的代码可以在https://github.com/aqeelanwar/conv_layers_animation](https://github.com/aqeelanwar/conv_layers_animation)找到