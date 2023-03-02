# 图神经网络基础

## 深度学习基础

### 基础范式-监督学习

给定特征$x$，目标是预测标签$y$；

$x$可以是向量、文本序列、栅格图片、图；

本质可抽象为优化问题，最小化损失函数；

损失函数衡量模型的预测结果与真实的标注结果之间的区别。

例：回归问题常用的L2损失函数
$$
\mathcal{L}(\boldsymbol{y}, f(\boldsymbol{x}))=\|y-f(x)\|_{2}
$$
例：多分类问题的交叉熵损失函数
$$
f(\boldsymbol{x})=\operatorname{Softmax}(g(\boldsymbol{x}))
$$

$$
\mathrm{CE}(\boldsymbol{y}, f(\boldsymbol{x}))=-\sum_{i=1}^{C}\left(y_{i} \log f(x)_{i}\right)
$$

$$
\mathcal{L}=\sum_{(\boldsymbol{x}, \boldsymbol{y}) \in \mathcal{T}} \operatorname{CE}(\boldsymbol{y}, f(\boldsymbol{x}))
$$

### 优化目标函数

迭代的优化模型中的权重（参数），用求偏导数的方法，求得损失函数相对于每一个参数的偏导数。
$$
\nabla_{\Theta} \mathcal{L}=\left(\frac{\partial \mathcal{L}}{\partial \Theta_{1}}, \frac{\partial \mathcal{L}}{\partial \Theta_{2}}, \ldots\right)
$$

### 梯度下降

知道了偏导数，即知道了优化的方向，将其乘以学习率，按照梯度的方向去优化损失函数；
$$
\Theta \leftarrow \Theta-\eta \nabla_{\Theta} \mathcal{L}
$$
理想终止点是gradient = 0（碗底），但神经网络的拟合空间是一个非线性、非凸的高维空间。但可能没有一个真正的碗底，只有一个局部最优点。所以在训练过程中要时时刻刻关心验证集（测试集）上的表现。

#### 梯度下降面临的问题

如果把所有数据输入进去，算出损失函数的梯度，自然是一个全局的梯度，非常不错。但现代数据集常包含数十亿个样本，每迭代一次，需输入所有样本计算损失函数，太浪费时间和计算资源。

#### 解决：SGD

每迭代一次，只输入batch size个样本计算损失函数。

### Mini-batch SGD（随机梯度下降）

循环：

1. 采样生成Mini-batch；
2. 前向推断，求损失函数；
3. 反向传播，求每个权重的更新梯度；
4. 优化更新权重。

重要概念：

* iteration = step = 输入一个mini batch = 一次迭代 = 一步
* epoch = 一轮 = 完整遍历训练集的所有样本一遍

SGD是全局梯度的无偏估计！

### 反向传播

求一个mini-batch的损失函数，并且求得这个mini-batch上损失函数相对于每一个权重的梯度：
$$
\nabla_{W} \mathcal{L}=\left(\frac{\partial \mathcal{L}}{\partial w_{1}}, \frac{\partial \mathcal{L}}{\partial w_{2}}, \frac{\partial \mathcal{L}}{\partial w_{3}} \ldots\right)
$$
求梯度的过程，即是反向传播。是复合函数求偏导的过程，应用链式法则。

### 训练过程

前向预测，求损失函数：
$$
x \underset{\text { Multiply } W_{1}}{\longrightarrow} \longrightarrow \underset{\text { Multiply } W_{2}}{\longrightarrow} \longrightarrow \mathcal{L}
$$
反向传播，求梯度：
$$
\Theta=\left\{W_{1}, W_{2}\right\}
$$

$$
\frac{\partial \mathcal{L}}{\partial W_{2}}=\frac{\partial \mathcal{L}}{\partial f} \cdot \frac{\partial f}{\partial W_{2}}, \quad \frac{\partial \mathcal{L}}{\partial W_{1}}=\frac{\partial \mathcal{L}}{\partial f} \cdot \frac{\partial f}{\partial W_{2}} \cdot \frac{\partial W_{2}}{\partial W_{1}}
$$

注：可以复用外层函数求偏导的结果，节省内存和缓存。

### 优化策略

求得梯度之后去优化（下山）的时候，可以考虑不同的优化策略：

* SGD
* Momentum
* NAG
* Adam
* Adagrad
* Adadelta
* Rmsprop
* Nesterov

### 非线性激活函数

给每一个神经元的输出，接入一个激活函数，使得神经网络具备表达非线性数据分布的能力。

例：Rectified linear unit $\operatorname{ReLU}(x)=\max (x, 0)$

例：Sigmoid $\sigma(x)=\frac{1}{1+e^{-x}}$

### 多层感知机

最简单的神经网络，
$$
\boldsymbol{x}^{(l+1)}=\sigma\left(W_{l} \boldsymbol{x}^{(l)}+b^{l}\right)
$$
其中，$W$是权重，$b^{l}$是偏置项，$\sigma$是非线性激活函数

是全连接神经网络，即每一层的神经元，都和上一层的所有神经元相连。

### 一些神经网络技巧

Batch Normlization、Dropout、Attention / Gating

#### Suggested GNN Layer

$$
\begin{array}{c}\downarrow \\ \hline \text { Linear }  \\ \hline \downarrow \\ \hline \text { BatchNorm } \\ \hline \downarrow \\ \hline \text { Dropout } \\ \hline \downarrow \\ \hline \text { Activation } \\ \hline \downarrow \\ \hline \text { Attention } \\ \hline \downarrow \\ \hline \text { Aggregation } \\ \hline \downarrow\end{array}
$$

## 图神经网络难点

1. 网络是复杂的，可能是任意尺寸的输入；
2. 没有图固定的节点顺序和参考锚点
3. 经常动态变化，并有多模态特征