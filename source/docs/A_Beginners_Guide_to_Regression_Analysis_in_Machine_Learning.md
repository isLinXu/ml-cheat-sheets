# 机器学习回归分析初学者指南

## 通过示例、插图、动画和备忘单解释回归分析。

# 语境：

为了理解回归背后的动机，让我们考虑以下简单示例。下面的散点图显示了 2001 年至 2012 年美国大学毕业生的数量。

![img](https://miro.medium.com/max/1050/1*x2cMaV5t46OKp56aoOt7tg.jpeg)

图片作者

现在根据现有数据，如果有人问你2018年有多少大学硕士毕业生呢？可以看出，具有硕士学位的大学毕业生人数几乎与年份呈线性增长。所以通过简单的视觉分析，我们可以粗略估计这个数字在2.0到210万之间。让我们看看实际数字。下图绘制了从 2001 年到 2018 年的同一变量。可以看出，我们的预测数字与实际值大致相符。

![img](https://miro.medium.com/max/1050/1*g0TgDvCUAyNDMsmW7bRNfQ.jpeg)

图片作者

由于这是一个更简单的问题（将一条线拟合到数据），我们的头脑很容易做到这一点。这种将函数拟合到一组数据点的过程称为回归分析。

# 什么是回归分析？

回归分析是估计因变量和自变量之间关系的过程. 简单来说，就是在某个误差函数下，将选定函数族中的一个函数拟合到采样数据中。回归分析是用于预测的机器学习领域中最基本的工具之一。使用回归，您可以在可用数据上拟合一个函数，并尝试预测未来或保留数据点的结果。这种函数匹配有两个目的。

1. 您可以估计数据范围内的缺失数据（插值）
2. 您可以估计数据范围之外的未来数据（外推）

回归分析的一些现实示例包括根据房屋特征预测房屋价格、预测 SAT/GRE 分数对大学录取的影响、根据输入参数预测销售额、预测天气等。

让我们考虑前面大学毕业生的例子。

1. **插值法：**假设我们可以访问有些稀疏的数据，我们知道每 4 年大学毕业生的数量，如下方的散点图所示。

![img](https://miro.medium.com/max/1050/1*4eVV0rx9n81pMOnEDW9Lqg.jpeg)

图片作者

我们想估计中间所有缺失年份的大学毕业生人数。我们可以通过将一条线拟合到有限的可用数据点来做到这一点。这个过程称为插值。

![img](https://miro.medium.com/max/1050/1*4XIWpLQXdZgiUXsWtHuibg.jpeg)

**图 4**：作者图片

**推断**：假设我们可以访问 2001 年至 2012 年的有限数据，并且我们想要预测 2013 年至 2018 年的大学毕业生人数。

![img](https://miro.medium.com/max/1050/1*x2cMaV5t46OKp56aoOt7tg.jpeg)

图片作者

可以看出，具有硕士学位的大学毕业生人数几乎与年份呈线性增长。因此，将一条线拟合到数据集是有意义的。用这12个点拟合出一条线，然后在未来的6个点上测试这条线的预测，可以看出预测很接近。

![img](https://miro.medium.com/max/1050/1*ee_S3W1O36MfwXBYO_uH5Q.png)

外推——预测未来看不见的价值——图片作者：Author

## 从数学上讲

![img](https://miro.medium.com/max/1050/1*tW8bZzZNt17N0lD0P-z7Vg.png)

# 回归分析的类型

现在让我们谈谈我们可以进行回归的不同方法。基于函数族 (f_beta) 和使用的损失函数 (l)，我们可以将回归分为以下几类。

## 1. 线性回归

在线性回归中，目标是通过最小化每个数据点的均方误差之和来拟合超平面（二维数据点的一条线）。

从数学上讲，线性回归解决了以下问题

![img](https://miro.medium.com/max/1050/1*j3LRR2Z-g-r1vZgTqnmPAw.png)

因此，我们需要找到 2 个用 beta 表示的变量来参数化线性函数 f(.)。线性回归的一个例子可以在上面的图 4 中看到，其中 P=5。该图还显示了 beta_0 = -90.798 和 beta_1 = 0.046 的拟合线性函数

## 2. 多项式回归

线性回归假设因变量 (y) 和自变量 (x) 之间的关系是线性的。当数据点之间的关系不是线性时，它无法拟合数据点。多项式回归通过将 m 次多项式拟合到数据点来扩展线性回归的拟合能力。所考虑的功能越丰富，（通常）其拟合能力就越好。从数学上讲，多项式回归解决了以下问题。

![img](https://miro.medium.com/max/1050/1*rkiqZeZnIoVIifam5y-FEQ.png)

多项式回归的数学公式 — 图片来自 Author

因此，我们需要找到 (m+1) 个由 beta_0, …,beta_m 表示的变量。可以看出，线性回归是2次多项式回归的特例。

考虑以下绘制为散点图的数据点集。如果我们使用线性回归，我们得到的拟合显然无法估计数据点。但是如果我们使用 6 次多项式回归，我们会得到更好的拟合，如下所示

![img](https://miro.medium.com/max/3000/1*yL67Ufrbs1mWkAcrPFMzCw.jpeg)

![img](https://miro.medium.com/max/3000/1*6v5ae_cuXa_7-tDcbo1W4Q.jpeg)

![img](https://miro.medium.com/max/3000/1*ghD5_zDVZIfe-NMUwXJLGQ.jpeg)

**[左]**数据散点图 - **[中]**数据线性回归 - **[右]** 6 次多项式回归

由于数据点在因变量和自变量之间不存在线性关系，因此线性回归无法估计出良好的拟合函数。另一方面，多项式回归能够捕获非线性关系。

## 3.岭回归

岭回归解决了回归分析中的过度拟合问题。要理解这一点，请考虑与上述相同的示例。当一个25次的多项式拟合到10个训练点的数据上时，可以看出它完美地拟合了红色数据点（下图中心）。但这样做会损害中间的其他点（最后两个数据点之间的尖峰）。这可以在下图中看到。岭回归试图解决这个问题。它试图通过折中训练点的拟合来最小化泛化误差。

![img](https://miro.medium.com/max/3000/1*yL67Ufrbs1mWkAcrPFMzCw.jpeg)

![img](https://miro.medium.com/max/3000/1*KUXhQQg7x-2AZn3-GCeqOA.jpeg)

![img](https://miro.medium.com/max/3000/1*L7vEo8hAEdjF1VhEWfjJuQ.jpeg)

**[左]**数据散点图 - **[中]** 25 次多项式回归 - **[右]** 25 次多项式岭回归

从数学上讲，岭回归通过修改损失函数来解决以下问题。

![img](https://miro.medium.com/max/1050/1*H22xduqYQEnfuC1IJvMJyg.png)

函数 f(x) 可以是线性函数或多项式函数。在没有岭回归的情况下，当函数过度拟合数据点时，学习到的权重往往会很高。岭回归通过在损失函数中引入权重的缩放 L2 范数 (beta) 来限制正在学习的权重范数，从而避免过度拟合。因此，经过训练的模型会在完美拟合数据点（学习权重的大范数）和限制权重范数之间进行权衡。比例常数 alpha>0 用于控制这种权衡。较小的 alpha 值将导致更高的范数权重和过度拟合训练数据点。另一方面，较大的 alpha 值将导致函数与训练数据点的拟合较差，但权重范数非常小。

## 4. LASSO 回归

LASSO 回归类似于 Ridge 回归，因为它们都用作正则化器以防止训练数据点上的过度拟合。但是 LASSO 有一个额外的好处。它对学习到的权重强制执行稀疏性。

岭回归强制学习权重的范数较小，从而产生一组总范数减少的权重。大多数权重（如果不是全部）将是非零的。另一方面，LASSO 试图通过使大多数权重真正接近于零来找到一组权重。这会产生一个稀疏权重矩阵，其实现比非稀疏权重矩阵更节能，同时在数据点拟合方面保持相似的精度。

下图试图在与上图相同的示例中形象化这个想法。使用 Ridge 和 Lasso 回归拟合数据点，并且它们相应的拟合和权重按升序绘制。可以看出，LASSO回归中的大部分权重确实接近于零。

![img](https://miro.medium.com/max/1050/1*3tDfPkR_U1VSPUCYYV9QmA.jpeg)

![img](https://miro.medium.com/max/1050/1*OKyl-HOTMJd4pMOSkEFkWQ.jpeg)

图片作者

从数学上讲，LASSO 回归通过修改损失函数来解决以下问题。

![img](https://miro.medium.com/max/1050/1*3uLgJx3x26o9GlgqK-Z7Ow.png)

LASSO 和 Ridge 回归的区别在于 LASSO 使用权重的 L1 范数而不是 L2 范数。损失函数中的 L1 范数倾向于增加学习权重的稀疏性。有关它如何强制执行稀疏性的更多详细信息，请参阅下面文章的***L1 正则化部分。\***

[机器学习中的正则化类型机器学习正则化初学者指南。towardsdatascience.com](https://towardsdatascience.com/types-of-regularization-in-machine-learning-eb5ce5f9bf50)

常量 alpha>0 用于控制学习权重中拟合和稀疏性之间的权衡。较大的 alpha 值会导致拟合不佳，但学习的权重集会更稀疏。另一方面，较小的 alpha 值会导致训练数据点紧密拟合（可能导致过度拟合），但权重集较少。

## 5. 弹性网络回归

ElasticNet 回归是 Ridge 和 LASSO 回归的组合。损失项包括权重的 L1 和 L2 范数及其各自的比例常数。它通常用于解决 LASSO 回归的局限性，例如非凸性。ElasticNet 添加了权重的二次惩罚，使其主要是凸的。

从数学上讲，ElasticNet 回归通过修改损失函数来解决以下问题。

![img](https://miro.medium.com/max/1050/1*c2xXFTNme6aksG7JKz4RGw.png)

## 6. 贝叶斯回归

对于上面讨论的回归（频率论者方法），目标是找到一组解释数据的确定性权重值 (beta)。在贝叶斯回归中，我们不是为每个权重找到一个值，而是尝试在假设先验的情况下找到这些权重的分布。

因此，我们从权重的初始分布开始，并根据数据利用贝叶斯定理将先验分布与基于可能性和证据的后验分布相关联，从而将分布推向正确的方向。

![img](https://miro.medium.com/max/1050/1*3HNstMp2IXo9TyQOmkOmwQ.png)

当我们有无限的数据点时，权重的后验分布在普通最小二乘解的解中成为一个脉冲，即方差趋近于零。

找到权重的分布而不是一组确定性值有两个目的

1. 它自然地防止过度拟合的问题，因此充当正则化器
2. 它提供了权重的置信度和范围，这比仅返回一个值更符合逻辑。

让我们以数学方式表述问题并陈述其解决方案。

![img](https://miro.medium.com/max/1050/1*3opwYaIL7S7XcB4W7ehwdw.png)

**让我们对具有均值μ**和协方差**Σ 的**权重进行高斯先验，即

![img](https://miro.medium.com/max/1050/1*QQ40kokCVZDsDjI3lpHTVw.png)

根据可用数据 D，我们更新此分布。对于手头的问题，后验将是具有以下参数的高斯分布

![img](https://miro.medium.com/max/1050/1*NOFWBSBv1pJUXxkOtORAXQ.png)

[可以在此处](https://cedar.buffalo.edu/~srihari/CSE574/Chap3/3.4-BayesianRegression.pdf)找到详细的数学解释

让我们通过一次更新一个数据点的权重分布来查看顺序贝叶斯线性回归，从而从视觉上尝试理解它。下图

![img](https://miro.medium.com/max/1050/1*Ii-Uan8neJUpgH_-pYWQ3w.png)

贝叶斯回归根据输入数据 (x, y) 将后验分布推向正确的方向 — 图片作者

通过包含每个数据点，权重的分布更接近实际的基础分布。

下面的动画绘制了考虑单个新数据点时的原始数据、预测的四分位间距、权重的边际后验分布以及每个时间步的权重联合分布。可以看出，随着我们包含更多的点，四分位数范围变窄（绿色阴影区域），边缘分布围绕两个方差接近零的权重参数分布，并且联合分布收敛于实际权重。

![img](https://miro.medium.com/max/1050/1*dbNBFq6FLyMqRxLqosBaSQ.gif)

动画贝叶斯回归——来源：[www.MLinGIFS.aqeel-anwar.com](http://www.mlingifs.aqeel-anwar.com/)

## 7.逻辑回归

逻辑回归在分类任务中派上用场，其中输出需要是给定输入的输出的条件概率。从数学上讲，逻辑回归解决了以下问题

![img](https://miro.medium.com/max/1050/1*TaH5Z6JnOxloz9B6wvPsmA.png)

考虑以下示例，其中数据点属于两个类别之一：{0（红色），1（黄色）}，如下面的散点图所示。

![img](https://miro.medium.com/max/3000/1*uUVC_q34eKpv6TBnF8Cf-w.jpeg)

![img](https://miro.medium.com/max/3000/1*YzyOiG3VG-lMJYXoGpJzPA.jpeg)

[左] 数据点的散点图 - [右] 在以蓝色绘制的数据点上训练的逻辑回归

逻辑回归在线性或多项式函数的输出端使用 sigmoid 函数将输出从 (-♾️, ♾️) 映射到 (0, 1)。然后使用阈值（通常为 0.5）将测试数据分类为两个类别之一。

> 这可能看起来像逻辑回归不是回归而是一种分类算法。但事实并非如此。您可以在 Adrian 的帖子中[找到](https://www.linkedin.com/posts/adrianolszewski_medium-where-good-ideas-find-you-activity-6699301848865656832-_tsI)更多相关信息。

# 概括

在本文中，我们研究了回归分析中的各种方法、它们的动机是什么以及如何使用它们。下面的表格和备忘单总结了上面讨论的不同方法。

![img](https://miro.medium.com/max/1050/1*9t0WcZEzrL5DPG8WJsKYZA.png)

回归分析总结 — 图片来自 Author