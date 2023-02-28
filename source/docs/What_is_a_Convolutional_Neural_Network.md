# 基本 CNN 模板：

一个基本的 CNN 由三种层组成。输入、隐藏、输出如下图。数据通过输入层进入CNN，在到达输出层之前经过各个隐藏层。输出层是网络的预测。网络的输出在损失或错误方面与实际标签进行比较。对于要学习的网络，计算此损失对可训练权重的偏导数，并通过使用反向传播的各种方法之一更新权重。

基本 CNN 的完整视觉模板如下所示。

![img](https://miro.medium.com/max/1050/1*nWwSb30Q0CiEDPVPblS05g.png)

基本 CNN 的模板

# CNN 的隐藏层

网络中的隐藏层提供了一个基本的构建块来转换数据（输入层或先前隐藏层的输出）。大多数常用的隐藏层（不是全部）都遵循一种模式。它从将函数应用于其输入开始，然后进行池化、归一化，最后在将其作为输入馈送到下一层之前应用激活。因此，每一层可以分解为以下4个子功能

1. **层函数：**基本的变换函数，例如卷积层或全连接层。
2. **池化：**用于改变特征图的空间大小，增加（上采样）或减少（最常见）。例如最大池化、平均池化和反池化。
3. **归一化：**此子函数将数据归一化为零均值和单位方差。这有助于解决梯度消失、内部协变量偏移等问题[（更多信息）](https://towardsdatascience.com/difference-between-local-response-normalization-and-batch-normalization-272308c034ac)。最常用的两种归一化技术是局部响应归一化和批量归一化。
4. **激活：**应用非线性并限制输出过高或过低。

我们将通过每个子功能来解释它们最常见的示例。

> 那里有更复杂的 CNN 架构，它们具有各种其他层和相当复杂的架构。并非所有 CNN 架构都遵循此模板。

# 1.图层功能

最常用的层函数是全连接层、卷积层和转置卷积层（错误地称为反卷积层）。

![img](https://miro.medium.com/max/1050/1*Kg5cA0WNLjDnS3F6gbwFYQ.gif)

## A。全连接层：

这些层由输入和输出之间的线性函数组成。对于*i 个*输入节点和*j*个输出节点，可训练的权重是 wij 和 bj。左图说明了 3 个输入节点和 2 个输出节点之间的全连接层是如何工作的。

## b. 卷积层：

这些层应用于 2D（和 3D）输入特征图。可训练权重是一个 2D（或 3D）内核/过滤器，它在输入特征图上移动，生成与输入特征图重叠区域的点积。以下是用于定义卷积层的 3 个参数

- **Kernel Size K：**滑动内核或过滤器的大小。
- **Stride Length S：**定义在进行点积生成输出像素之前内核滑动了多少
- **Padding P：**在输入特征图周围插入零的帧大小。

下面的 4 张图直观地解释了输入大小为 ( ***i\*** ) 5x5 的卷积层，内核大小 ( ***k\*** ) 为 3 x 3 和不同的步长 ( ***s\*** ) 和填充 ( ***p\*** )

![img](https://miro.medium.com/max/1500/1*YvlCSNzDEBGEWkZWNffPvw.gif)

![img](https://miro.medium.com/max/1500/1*gXAcHnbTxmPb8KjSryki-g.gif)

![img](https://miro.medium.com/max/1500/1*34_365CJB5seboQDUrbI5A.gif)

![img](https://miro.medium.com/max/1500/1*WpOcRWlofm0Z0EDUTKefzg.gif)

动画卷积层（来源：Aqeel Anwar）

步幅和填充以及输入特征图控制输出特征图的大小。输出大小由下式给出

## C。转置卷积（DeConvolutional）层：

通常用于增加输出特征图的大小（上采样）。转置卷积层背后的想法是撤销（不完全）卷积层。正如卷积层一样，它也由步长和填充定义。如果我们在输出上应用提供的步幅和填充，并应用提供大小的卷积核，它将生成输入。

![img](https://miro.medium.com/max/1050/1*aQoJO4cxEhJLxyGNVeqq_g.png)

转置卷积层（来源：Aqeel Anwar）

要生成输出，需要执行两件事

- 零插入 ( ***z\*** )：在原始输入的行和列之间插入的零的数量
- 填充 ( ***p'\*** )：在输入特征图周围插入的零的帧大小。

***下面的 4 张图直观地解释了输入不同大小 ( i\*** )的转置卷积层，内核大小 ( ***k\*** ) 为 3x3，步幅 ( ***s\*** ) 和填充 ( ***p\*** ) 不同，而输出***(o)\***固定为 5x5

![img](https://miro.medium.com/max/1500/1*SpxCUPzNfb9C8TiAcrRr5A.gif)

![img](https://miro.medium.com/max/1500/1*gff0oa2iPygyCEjj7Fb3yg.gif)

![img](https://miro.medium.com/max/1500/1*WaBzh5OkmD-9EBLy5aXiug.gif)

![img](https://miro.medium.com/max/1500/1*L_hJRnywTpeTFJAaVZTRfQ.gif)

动画转置卷积层（来源：Aqeel Anwar）

可以在下面找到有关转置卷积层的详细信息

[什么是转置卷积层？通过动画 gif 和 python 代码进行解释。towardsdatascience.com](https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11)

# 2. 池化

最常用的池化有 Max、average pooling 和 max average unpooling。

## 最大/平均池化：

不可训练层用于根据选择内核定义的接受域中的最大值/平均值来**减小输入层的空间大小。**内核以给定的步幅滑过输入特征图。对于每个位置，输入特征图与内核重叠的部分的最大值/平均值是对应的输出像素。

![img](https://miro.medium.com/max/1500/1*kW4HcS4zFxoKv6R4xtqFlg.gif)

![img](https://miro.medium.com/max/1500/1*LjXV6eQKTQcg-PJnBRE0VA.gif)

动画最大池化层（来源：Aqeel Anwar）

## 取消池化：

不可训练层用于根据将输入像素放置在由内核定义的输出的感受野中的某个索引处来**增加**输入层的空间大小。对于反池化层，网络中需要有一个对应的池化层。来自相应池化层的最大值/平均值的索引被保存并在反池化层中使用。在反池化层中，每个输入像素都放置在输出中池化层中出现最大值/平均值的索引处，而其他像素设置为零

# 3.归一化

归一化通常在激活函数之前使用，以限制无界激活将输出层值增加得太高。通常使用两种归一化技术

## A。局部响应归一化 LRN：

LRN 是一个**不可训练的层**，它对局部邻域内特征图中的像素值进行平方归一化。基于邻域定义的LRN有Inter-channel和Intra-channel两种类型，如下图所示。

![img](https://miro.medium.com/max/1050/1*MFl0tPjwvc49HirAJZPhEA.png)

![img](https://miro.medium.com/max/1797/1*VoBDhNBaJPCxgcITvrQaXA.png)

![img](https://miro.medium.com/max/2400/1*hSM8Prmr58B7GvnkjUj42w.png)

**左：**通道内 LRN …**右**：通道间 LRN

## b. 批量归一化 BN：

另一方面，BN 是一种可训练的数据标准化方法。在批量归一化中，隐藏神经元的输出在被馈送到激活函数之前按以下方式处理。

1. 将整个批次*B*归一化为零均值和单位方差

- 计算整个小批量输出的均值：*u_B*
- 计算整个mini-batch输出的方差：s *igma_B*
- 通过减去均值并除以方差来归一化小批量

\2. 引入两个可训练参数（*Gamma:* scale_variable 和*Beta:* shift_variable）来缩放和移动归一化的小批量输出

\3. 将这个经过缩放和移位的归一化小批量输入到激活函数中。

两种归一化技术的总结如下所示

![img](https://miro.medium.com/max/1050/1*J7rxGz1f_2YWjdcsvqNCNA.png)

[**可以在此处**](https://towardsdatascience.com/difference-between-local-response-normalization-and-batch-normalization-272308c034ac)找到有关这些规范化技术的详细文章

# 4.激活

激活函数的主要目的是引入非线性，因此 CNN 可以有效地映射输入和输出之间的非线性复杂映射。可以根据底层需求使用多个激活函数。

- **非参数/静态函数：**线性、ReLU
- **参数函数：** ELU、tanh、sigmoid、Leaky ReLU
- **有界函数：** tanh、sigmoid

下面的 gif 动图直观地解释了最常用激活函数的性质。

![img](https://miro.medium.com/max/1350/1*EmTYifwsrA6YNPI2vYRf7g.gif)

![img](https://miro.medium.com/max/900/1*HBvDu4Rl56AEz_jvF3BYBQ.gif)

动画激活函数（来源：Aqeel Anwar）

最常用的激活函数是 ReLU。有界激活函数（如 tanh 和 sigmoid）在涉及更深的神经网络时会遇到梯度消失的问题，通常会避免使用。

# 5. 损失计算：

定义 CNN 后，需要选择一个损失函数来量化 CNN 预测与实际标签的差距。然后在梯度下降法中使用这种损失来训练网络变量。与激活函数一样，损失函数有多个候选对象。

## 回归损失函数

- Mean Absolute Error：估计值和标签都是实数
- 均方误差：估计值和标签都是实数
- Huber Loss：估计值和标签都是实数

## 分类损失函数

- 交叉熵：估计值和标签是概率（0,1）
- Hinge Loss：估计值和标签都是实数

这些损失函数的详细信息可以在下图中看到

![img](https://miro.medium.com/max/1050/1*ddQRcK9U-1Epmlfn6FT7ZQ.gif)

动画 ML 损失函数（来源：Aqeel Anwar）

# 6.反向传播

反向传播不是 CNN 的结构元素，而是我们通过在梯度变化（梯度下降）的相反方向上更新权重来了解潜在问题的方法。[可以在此处](https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms)找到有关不同梯度下降算法的详细信息。

# 概括：

在本文中，呈现了基本 CNN 不同元素的动画可视化，这将有助于更好地理解它们的功能。