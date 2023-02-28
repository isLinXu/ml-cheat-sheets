# 为机器学习项目设置 Python 平台

有很多关于如何使用 Python 开始机器学习的深入教程。这些教程主要侧重于深度学习框架（例如 TensorFlow、PyTorch、Keras 等）的使用，例如如何设置基本的监督学习问题，或如何创建一个简单的神经网络并对其进行训练等。但即使在之前人们可以开始尝试使用此类教程，但主机上必须有一个可用的 Python 平台，才能进行这些实践操作。

几年前，当我开始使用 ML 时，我找不到一个关于如何在我的机器上设置工作平台的好教程。有很多选项/配置可用，我无法决定任何一个。在我决定配置之前，我必须浏览很多网页，并且花了大约 2 周的时间才最终拥有一个工作平台。几个月前，我的一位同事遇到了同样的问题，所以我帮助她搭建了平台，为她节省了不少时间。因此我决定写一篇关于它的详细文章。

我将这篇文章分为两部分，并将分别关注它们

- **第 1 部分：**选择和安装 Python 发行版和 IDE
- **第 2 部分：**创建新项目时所需的步骤

# 第 1 部分 — Python 发行版和 IDE

在本文中，我将使用以下 Python 发行版和 IDE。

- **Python 发行版：** Anaconda
- **Python 集成开发环境：** PyCharm

什么是**Python**？Anaconda 是 Python 和 R 语言的发行版，具有简化的包管理和部署。使用 anaconda，可以更轻松地拥有多个具有不同配置的 python 环境并在它们之间切换。anaconda 包管理器可以更轻松地解决不同包所需的包的多个版本之间的冲突。[可以在此处](https://www.quora.com/What-are-the-pros-and-cons-of-using-default-Python-and-using-Python-in-Anaconda)找到有关使用 Anaconda 的优缺点的详细说明

**下载和安装： Anaconda 发行版可以从**[这里](https://www.anaconda.com/distribution/)下载。安装说明非常简单。

**PyCharm**是什么？PyCharm 是可用于 python 的众多 IDE 之一。我更喜欢 PyCharm，因为与其他 IDE 相比，它更加用户友好、功能强大且可配置。它提供与 git 的集成，拥有自己的终端和 python 控制台，提供对各种方便的插件的支持，以及许多有用的键盘快捷键。

**下载和安装：**要下载 PyCharm，请转到此[链接](https://www.jetbrains.com/pycharm/download)并下载最新的社区（免费）版本并按照安装说明进行操作

# 第 2 部分 - 创建新项目时所需的步骤

您将从事的不同项目将需要具有不同版本要求的不同资源和包。因此，始终建议您为每个项目使用单独的虚拟 python 环境。这也确保您不会意外地用其他版本覆盖某些包的任何现有工作版本，从而使其对您当前的项目无用。

以下是创建新项目时应采取的步骤。

## 第一步：新建Anaconda虚拟环境：

打开 Anaconda Prompt 命令并键入

```
conda create -n myenv python==3.5
```

*这将创建一个名为myenv*的新虚拟环境，它将安装并加载 python 3.5 版

创建环境后，您可以通过使用激活环境来验证它

```
conda activate myenv #For Windows
source activate myenv #For MAC OS
```

您可以通过键入找到创建的环境 myenv 的位置

```
which python 
# 输出看起来像这样
# /Users/aqeelanwar/anaconda/envs/myenv/bin/python
```

我们将在第 2 步中使用它来定位我们的环境

## 第 2 步：安装必要的软件包：

激活环境后，您可以使用安装任何所需的包

```shell
#通用格式：
 conda install package_name#Example:
 conda install numpy #安装numpy
```

## 第 3 步：有用的 Conda 命令：

以下是有用的 conda 命令，它们将在管理 conda 环境时派上用场

```shell
# 激活环境
conda activate env_name
# 停用一个环境
deactivate #Windows 
source deactivate #Linux 和 macOS# 列出所有创建的环境
conda env list# 列出安装在环境中的包
conda list# 安装包
conda install package_name# 克隆一个 conda 环境
conda create --clone name_env_to_be_cloned --name name_cloned_env
```

## 第 4 步：在 PyCharm 上创建一个新项目

- 打开 PyCharm 并选择创建新项目。

![img](https://miro.medium.com/max/1050/1*rxXMDx8hA5L3teCTfoe5QA.png)

- 选择项目的位置和名称（在本例中为 medium_tutorial）。
- 展开 Project Interpreter 选项并选择 Existing interpreter
- 通过单击现有解释器下最右侧的三个点来找到您的环境

![img](https://miro.medium.com/max/1050/1*tOVqb35pTqUMoRO0d_5xnA.png)

- 此时，我们将使用在步骤 1 中从 which python 命令显示的位置定位我们的环境 myenv（如果它还没有在项目解释器列表中：稍后会详细介绍）。此外，我们希望这个环境是可用于我们将来创建的其他项目，因此我们将选中“*对所有项目可用”*复选框

![img](https://miro.medium.com/max/1050/1*bFRCnc5PD0Vdbd3NxkotSg.png)

- 点击确定，然后创建。

![img](https://miro.medium.com/max/1050/1*fBRoDY0nnerC8xTbDuPn5A.png)

- 该项目现在将使用环境 myenv。您可以使用 PyCharm 的内置终端将包安装到此环境中。
- 创建一个新的 .py 文件（比如 main.py）并使用

```
run >> run main.py
```

![img](https://miro.medium.com/max/1050/1*DQoctMVS2flFpKvyVIXh5Q.png)

## 第 5 步：在 conda 环境之间切换（可选）

如果以后你想为同一个项目在不同的 conda 环境之间切换，你可以按照下面的步骤进行

- PyCharm 只能选择已包含在其项目解释器列表中的环境
- 将新创建的（比如命名为 PyCaffe 的）环境添加到项目解释器列表

```
settings >> Project:project_name >> Project Interpreter
```

![img](https://miro.medium.com/max/1050/1*EQUFBwcopHgU4vZa-5vD9g.png)

- 点击右上角的齿轮图标并选择添加

![img](https://miro.medium.com/max/1050/1*O523j9KFnks5Zg8XJZ1Jcg.png)

- 选择

```
Conda Environment > Existing environment > <three dots>
```

并找到新创建的环境并点击确定

![img](https://miro.medium.com/max/1050/1*ur2FZV02yyuMBXVOPBQ60Q.png)

- 现在环境已经添加到Project Interpreter列表中，可以在下拉菜单中看到

![img](https://miro.medium.com/max/1050/1*GBbXaeYzmkYJ8PJmGEPqfw.png)

- 此列表显示现有环境以及您选择的任何环境将用于项目
- **注意：** PyCharm 终端不会自动激活当前选择的环境。如果你已经从项目解释器列表中选择了 PyCaffe env，现在想在其中安装一个新包，你必须首先在终端中激活环境，然后你可以使用 conda install package_name。否则会在之前激活的conda环境中安装包

![img](https://miro.medium.com/max/1050/1*2W3o1YE7SrW3-7lTjM6SJQ.png)

现在您已经完成了平台设置。此时，您可以安装所需的 ML 框架（TensorFlow、Keras、PyTorch）并开始尝试 ML 教程。

# 概括

在本教程中，我们了解了处理 ML 项目（或与此相关的任何 Python 项目）的先决条件。我们看到了如何使用 Anaconda 和 PyCharm 拥有多个 Python 环境并在它们之间切换。