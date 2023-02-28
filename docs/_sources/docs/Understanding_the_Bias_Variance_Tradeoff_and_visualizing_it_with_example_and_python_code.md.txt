# 了解偏差方差权衡并使用示例和 Python 代码将其可视化

机器学习中最重要和最基本的主题之一是偏差方差权衡。在本文中，我们将详细介绍什么是偏差方差权衡，它源于何处，为什么需要考虑它，它如何影响我们的底层系统，以及如何改进我们的 ML模型。

# 背景：它源于哪里？

ML 系统背后的总体思想是根据收集的样本对数据集的隐藏分布进行建模。如果您从分布中采样足够多，您可以获得相当准确的分布重新创建，如下所示。

![img](https://miro.medium.com/max/1050/1*DfRL_SLfgLVwIpzHw71LtQ.png)

给定 (x,y) 对，估计 f。

![img](https://miro.medium.com/max/1050/1*hymytuOcW8zjS-glWC5XbQ.png)

但问题是在现实生活中采集的样本通常有噪声。这种噪声的来源可能有很多因素，例如量化、感官限制等。因此我们没有得到分布的确切输出，而是在其中添加了噪声因子。所以现在的问题是在给定输入 x 及其对应的噪声输出 y 的情况下估计函数 f。

![img](https://miro.medium.com/max/1050/1*eSiVJ_aZwVfo_fXE1xJUng.png)

![img](https://miro.medium.com/max/1050/1*MW0JR7IZtjq5LCOweAoL0A.png)

目标是有效地估计函数 f 并滤除噪声。现在，由于您不太了解添加到样本输出中的噪声，如果处理不当，ML 系统最终会将输入 x 映射到噪声输出 y（称为过度拟合）。此映射不是函数 f 的准确表示，并且会为看不见的测试数据产生高误差。

偏差方差权衡告诉我们，在存在噪声的情况下，我们的基础系统 (f`(x)) 应该有多复杂才能相当准确地表示我们数据集的分布。

# 术语：什么是 ML 系统中的偏差和方差：

偏差和方差都可以作为我们 ML 系统中的错误来源来解决。假设我们有一个训练数据集***D\***，由从隐藏分布 ( ***y=f(x)+e\*** ) 中采样的 (x,y) 对组成。我们从数据集***D\***构建模型***f`\***，使训练标签和预测值之间的误差最小化 ( ***error = yf`(x)\*** )。

# **偏见：**

***偏差被称为平均模型预测f`(x)\***和基本事实***f(x)\***之间的误差

![img](https://miro.medium.com/max/1050/1*tge8zF6kZXBbR2g_10V0_w.png)

**对从不同数据子集D_i**预测的函数进行期望。简而言之，您从隐藏分布中采样**n 个**不同的数据集**D_i**（每个数据集由多个 (x,y) 对组成），并估计**n 个**不同的函数***f_i。\***然后

![img](https://miro.medium.com/max/1050/1*qv-QpT5o8u2HiAkmPzXKeA.png)

估计函数的偏差告诉我们基础模型预测值的能力。通常，更简单的模型无法捕捉高维数据的复杂性，因此它们具有更高的偏差。例如，您有一个从正弦曲线采样的数据集，您正尝试使用 1 次多项式（即通过函数***y = ax+b）对其进行估计。\***无论您采样多少个数据点 (x,y)，直线永远无法捕捉到正弦曲线的趋势。因此，对于正弦曲线，线模型具有非常高的偏差。另一方面，假设您将多项式的次数增加到 3，即现在用***y = ax² +bx+c 来估计它。\***该多项式的性能将比前一个多项式好得多，因此在估计正弦曲线时，3 次多项式的偏差比 1 次多项式的偏差小得多。

![img](https://miro.medium.com/max/1050/1*cp_O8__Gm663JLw-I0S1TA.png)

模型的高偏差与以下相关

1. 欠拟合——无法捕捉数据趋势
2. 更强调泛化
3. 训练和测试数据集的高误差
4. 过于简化的模型

# 方差：

方差是指给定数据集的模型预测的平均变异性。

![img](https://miro.medium.com/max/1050/1*bpHcbO7ajSVsl2Chqeqh2Q.png)

估计函数的方差告诉您该函数能够在多大程度上适应数据集中的变化。方差越大，函数对变化的数据集越稳健。例如，假设您训练两个不同的多项式来拟合从正弦曲线采样的数据，即 1 次多项式和 3 次多项式。您在三个不同的数据集（D1、D2 和 D3）上训练这两个多项式。下图显示了多项式次数和数据集的估计函数。

![img](https://miro.medium.com/max/1050/1*eW3iQHXDQ3uvM3O3Rw0LdA.png)

可以看出，1 次多项式在三个不同的数据集之间变化不大（因此方差较低），而 3 次多项式的估计彼此差异很大（因此方差较高）。

模型的高方差与以下相关

1. 过度拟合——最终对数据集中的噪声进行建模
2. 更加强调尽可能接近地拟合每个数据点
3. 训练数据的低误差，但测试数据的高误差
4. 过于复杂的模型和稀疏的训练数据。

> **通常，如果增加底层系统的复杂性，系统的偏差会降低，而方差会增加。它们彼此成反比。你不能同时减少它们。这一点将是偏差方差权衡的基础。**

# 权衡：

ML模型的预期测试误差可以通过以下公式分解为其偏差和方差

> 测试误差 = bias² + 方差 + 不可约误差

[可以在此处](https://en.wikipedia.org/wiki/Bias–variance_tradeoff)找到对此的完整推导。不可约误差是由于数据中的噪声而产生的误差，与模型的选择无关。

因此，为了减少估计误差，您需要同时减少偏差和方差。您必须选择一个模型（在我们的示例中是多项式的次数），该模型以最小化误差的方式权衡偏差和方差。在过度拟合和欠拟合的情况下，测试误差都可能很高。因此，我们需要偏差和方差的最佳平衡，以便我们的模型既不会过度拟合也不会欠拟合我们的数据。

考虑之前的以下示例。我们必须使用多项式从噪声采样数据***D估计正弦曲线。\***我们可以从各种不同次数的多项式中进行选择。在理想情况下，我们可以访问足够多的无噪声数据（噪声 = 0），泰勒级数展开告诉我们，如果我们继续增加多项式。但在嘈杂且有限的数据集的情况下，增加多项式的次数也会开始拟合数据中的噪声，并且在测试数据集上表现不佳。因此，我们需要找到在我们的数据集上最有效（即测试误差最小化）的多项式的最佳次数。

下图绘制了不同多项式的正弦曲线估计。1 次多项式太简单，无法捕捉正弦曲线，而 11 次多项式足够复杂，甚至可以封装噪声（因此偏离我们试图估计的实际正弦曲线）。

![img](https://miro.medium.com/max/1050/1*9WZsMZ5ZIeRUcsdjSwG_Bg.png)

正弦曲线的多项式回归

为了找到估计正弦曲线的最佳多项式，我们找到测试数据集的多项式（1、3、5、7、9 和 11 次）的偏差和方差。该图绘制如下。

![img](https://miro.medium.com/max/1050/1*DMv9-Wbt6pyLlFYrmhUh0w.png)

可以看出，在存在噪声的情况下，7 次多项式最适合我们的正弦曲线。该多项式的选择取决于

1. 样本数据中噪声的数量和性质
2. 训练数据量

让我们假设采样的数据集没有噪声。对这个干净的数据集运行相同的实验给出了下图

![img](https://miro.medium.com/max/1050/1*dL7ALyUlApU27b_I0UYxCg.png)

上图符合正弦曲线的泰勒级数展开，表明多项式的次数越高，逼近效果越好。完整的代码附在下面。

# Python代码：

正弦曲线估计问题的 python 代码可以在下面找到。模块 gen_data() 中的变量 b 用于控制采样数据中的噪声量。估计函数 f`(x) 的期望值是通过估计 num_data=2000 个不同（但重叠）数据集的 f`(x) 并将其取平均值来找到的。

```python
# Author: aqeelanwar 
# Created: 16 February,2020, 7:12 PM
# Email: aqeel.anwar@gatech.edu

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "cmr10"
import random
np.random.seed(54)
# random.seed(4959)

def dist_function(x):
    f_x = np.sin(x)
    return f_x

def gen_data(x):
    f_x = dist_function(x)
    b = 0.4
    y = f_x + np.random.uniform(-b, b, len(x))
    sigma = 1/12*(2*b)**2
    return y, f_x, sigma


def get_rand_data(len_ratio, x, y, f_x):
    m = len(x)
    m_new = int(np.round(len_ratio*m))
    ind = random.sample(range(m), m_new)
    ind = np.sort(ind)

    x1 = x[ind]
    y1 = y[ind]
    f_x1 = f_x[ind]

    return x1, y1, f_x1


x_orig = np.arange(0, 4 * np.pi, .2)

y_orig, f_x_orig, sigma = gen_data(x_orig)
p_order = [1, 3, 5, 7, 9, 11]
num_data = 2000

f, ax = plt.subplots(1,len(p_order), figsize=(15, 1.5), dpi=120, facecolor='w', edgecolor='k')
t = np.arange(0.1, 4*np.pi, .2)
P=[]
bb=[]
vv=[]
for i, p_val in enumerate(p_order):
    exp_f_x = np.zeros(len(t))
    exp_f_x_plot = np.zeros(len(x_orig))
    var = np.zeros(len(t))
    for j in range(num_data):
        x, y, f_x = get_rand_data(0.7, x_orig, y_orig, f_x_orig)
        p = np.poly1d(np.polyfit(x, y, p_val))
        # ax[j, i].plot(x, y, 'o', t, p(x_orig), '-')
        P.append(p)
        exp_f_x = exp_f_x + p(t)
        exp_f_x_plot = exp_f_x_plot + p(x_orig)

    exp_f_x = exp_f_x/num_data
    exp_f_x_plot = exp_f_x_plot / num_data
    ax[i].plot(x, y, 'o', x_orig, exp_f_x_plot, '-')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    ax[i].set_xlabel('P='+str(p_order[i]))
    bias = np.linalg.norm(exp_f_x-dist_function(t))
    bb.append(bias)

    for j in range(num_data):
        p_t = P.pop(0)
        var = var + np.square(p_t(t)-exp_f_x)
    var = var/num_data
    variance = np.linalg.norm(var)
    vv.append(variance)


print("bias: ", bb)
print("var: ",vv)

fig, ax1 = plt.subplots(dpi=200)

color = 'tab:orange'
ax1.set_xlabel('Polynomial order')
ax1.set_ylabel('Variance', color=color)
ax1.plot(p_order, vv, color=color, label = 'Variance')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Bias', color=color)
ax2.plot(p_order, bb, color=color,  label = 'Bias')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()

error = np.square(bb)+vv+sigma*len(t)
min_ind = np.argmin(error)
plt.plot(p_order, error, 'k--', label = 'error')
plt.plot(p_order[min_ind], error[min_ind], 'ro', label= 'Minimum Error')
leg = ax2.legend(loc='upper right')
leg = ax1.legend(loc='upper left')
plt.show()
```



# 概括：

## **对于干净且足够大的数据集：**

1. 模型复杂度越大，估计误差越低，近似越好。
2. 模型越复杂，偏差越小
3. 模型复杂度越大，方差**越小**

## 对于嘈杂的数据集：

1. 更高的模型复杂性并不意味着更低的估计误差
2. 模型越复杂，偏差越小
3. 模型复杂度越大，方差**越大**

由于现实世界中的数据集几乎总是嘈杂和有限的，因此我们需要找到底层模型的最佳复杂性，使我们在数据集上的误差最小。更高的复杂性并不能保证最合适。