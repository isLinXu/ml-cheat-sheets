# 机器学习面试主题备忘单

> [从http://cheatsheets.aqeel-anwar.com/](http://cheatsheets.aqeel-anwar.com/)下载更新版本的备忘单

本文的其余部分基于这些备忘单。对于每个主题，提供

- 备忘单形式的概述
- 面试问题示例
- 为详细了解该主题而推荐的文章。

> **注 1：**这些备忘单旨在刷新概念，并不意味着为初学者提供对主题的深入理解。
>
> **注2：**文章不断更新以获取更多备忘单。
>
> **资料来源：**所有这些备忘单（以及更多）都可以从[**www.cheatsheets.aqeel-anwar.com**](http://www.cheatsheets.aqeel-anwar.com/)**下载 pdf 格式。**

# 机器学习模型中的偏差和方差

## 一）概述：

![img](https://miro.medium.com/max/1050/1*gt38bg0Um9qwc4fi0_n9_Q.png)

偏差方差权衡备忘单 — 图片作者

## b) 示例问题：

1. ML 模型中的偏差是什么？
2. ML 模型中的方差是什么？
3. 偏差和方差之间的权衡是什么？
4. 高偏差/高方差 ML 模型的缺点是什么？
5. 您如何根据训练数据大小选择模型（高偏差或高方差）？

## **c) 详细文章：**

- [了解偏差方差权衡并使用示例和 Python 代码将其可视化](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-and-visualizing-it-with-example-and-python-code-7af2681a10a7)

# 机器学习中的不平衡数据

## 一）概述：

![img](https://miro.medium.com/max/1050/1*wEniP5HewaUSHeIF1_bEiw.png)

分类中的不平衡数据 — 图片来自 Author

## b) 示例问题：

1. 什么是分类中的不平衡数据？
2. 准确性是一个好的性能指标吗？它什么时候无法捕获 ML 系统的性能？
3. 什么是精确率和召回率？举个例子
4. 如何解决数据不平衡的问题？

## c) 详细文章：

- [通过视觉备忘单了解机器学习中的不平衡类](https://towardsdatascience.com/a-walk-through-imbalanced-classes-in-machine-learning-through-a-visual-cheat-sheet-974740b19094)

# 贝叶斯定理

## 一）概述：

![img](https://miro.medium.com/max/1050/1*YS0HjCoJw3kByQxqUJORHA.png)

贝叶斯定理和分类器 — 图片来自 Author

## b) 示例问题：

1. 什么是贝叶斯定理？
2. 实施贝叶斯定理的玩具示例
3. MLE 和 MAP 有什么区别？
4. MAP 和 MLE 什么时候相等？

## c) 详细文章：

- [机器学习贝叶斯定理简介](https://machinelearningmastery.com/bayes-theorem-for-machine-learning/)

# 主成分分析与降维

## 一）概述：

![img](https://miro.medium.com/max/1050/1*LTP0rpo9DZEcx0N_oapizQ.png)

PCA 和降维 — 图片来自 Author

## b) 示例问题：

1. 什么是主成分分析？
2. 我们如何使用 PCA 来降低维度？
3. 在 PCA 的上下文中，特征值表示什么？*（特征值的大小越大，如果我们将相应的特征向量作为数据的特征向量，则保留的信息越多）*

## c) 详细文章：

- [逐步了解主成分分析（PCA）。](https://medium.com/analytics-vidhya/understanding-principle-component-analysis-pca-step-by-step-e7a4bb4031d9)

# 机器学习中的回归

## 一）概述：

![img](https://miro.medium.com/max/1050/1*k9H8UBrmECwKy-jqAhaiyg.jpeg)

回归分析 — 图片由作者提供

## b) 示例问题：

1. ML 中的回归是什么？
2. 我们如何在回归中引入正则化？*（套索和山脊）*
3. LASSO 和 Ridge 回归对模型的权重有什么影响？*（Ridge 试图减小所学权重的大小，而 LASSO 试图将它们强制为零，从而创建一组更稀疏的权重）*
4. 贝叶斯线性回归的预测何时接近线性回归的预测？*（当数据点数量足够多时）*
5. 逻辑回归是用词不当吗？*（是的，因为不是回归，而是基于回归的分类）*

## c) 详细文章：

- [5 种回归类型及其性质](https://towardsdatascience.com/5-types-of-regression-and-their-properties-c5e1fa12d55e)
- [贝叶斯线性回归简介](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7)

# 机器学习中的正则化

## 一）概述：

![img](https://miro.medium.com/max/1050/1*KBU0hPg94w2TLa1SUgrAEg.png)

ML 中的正则化 — 图片由作者提供

## b) 示例问题：

1. ML 中的正则化是什么？
2. 我们如何解决过度拟合？
3. 什么是 K 折交叉验证？
4. L1 和 L2 正则化有什么区别？
5. 我们为什么要使用 dropout？

## c) 详细文章：

- [正则化](https://ml-cheatsheet.readthedocs.io/en/latest/regularization.html)
- [机器学习中的正则化](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

# 卷积神经网络基础

## 一）概述：

![img](https://miro.medium.com/max/1050/1*shs86xjDskPSMHGneie5pA.png)

卷积神经网络 — 图片由作者提供

## b) 示例问题：

1. 什么是美国有线电视新闻网？
2. 解释卷积层和转置卷积层的区别。
3. 用于分类的损失函数有哪些？

## c) 详细文章：

- [什么是卷积神经网络？](https://towardsdatascience.com/a-visualization-of-the-basic-elements-of-a-convolutional-neural-network-75fea30cd78d?source=friends_link&sk=680f483949434299ba538a3e0674a40a)

# 机器学习中著名的 DNN

## 一）概述：

![img](https://miro.medium.com/max/1050/1*SsxkxMrjQm2lknRUvEQ6RA.png)

著名的 CNN — 图片来自 Author

## b) 示例问题：

1. ResNet 网络如何解决梯度消失的问题？
2. Inception Network 的主要关键特征之一是什么？
3. ResNet 网络中的快捷连接是什么？

## c) 详细文章：

- [AlexNet、VGGNet、ResNet 和 Inception 之间的区别](https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96)

# 机器学习中的集成方法

## 一）概述：

![img](https://miro.medium.com/max/1050/1*TkXkL0mpzWB1PJcHrK3FOg.png)

## b) 示例问题：

1. 什么是集成学习？
2. 什么是 ML 中的 bagging、boosting 和 stacking？
3. bagging 和 boosting 有什么区别？
4. 举几个boosting方法

## c) 详细文章：

- [集成方法：bagging、boosting 和 stacking](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205)
- [机器学习中的集成方法是什么？](https://towardsdatascience.com/what-are-ensemble-methods-in-machine-learning-cac1d17ed349)

# 自编码器和变分自编码器

## 一）概述：

![img](https://miro.medium.com/max/1050/1*eyYkfUdklK1EincTjzKf1Q.jpeg)

## b) 示例问题：

1. 什么是自动编码器？
2. 自动编码器的潜在空间是否正则化？
3. 变分自动编码器的损失函数是什么？
4. 自动编码器和变分自动编码器有什么区别？

## c) 详细文章：

- [自动编码器 (AE) 和变分自动编码器 (VAE) 之间的区别](https://towardsdatascience.com/difference-between-autoencoder-ae-and-variational-autoencoder-vae-ed7be1c038f2)

# 概括

本文提供了一份备忘单列表，涵盖了机器学习面试的重要主题，并附有一些示例问题。文章中不断添加主题列表和备忘单数量。