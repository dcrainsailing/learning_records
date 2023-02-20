# 图嵌入表示学习

如何把节点映射成D维向量？

* 人工特征工程：节点重要度、集群系数、Graphlet
* 图表示学习：通过随机游走构造自监督学习任务。DeepWalk、Node2Vec
* 矩阵分解
* 深度学习：图神经网络

## 图嵌入

### 表示学习

1. 把各模态输入转为向量，向量保留原始的信息，以及便于下游机器学习预测的信息
2. 不需要人工设计特征，自动学习特征

### 图表示学习

$$
f: u \rightarrow \mathbb{R}^{d}
$$

把节点映射为d维度向量，并且与下游任务无关（无监督方法），特点：

* 低维：向量维度远小于节点数
* 连续：每个元素都是实数
* 稠密：每个元素都不为0

### 嵌入

把一个节点嵌入到d维空间变成一个点

d维空间中向量的相似度反映原图中节点的相似度

嵌入向量包含网络连接信息

能够用于下游的各种任务

### 基本框架（编码器-解码器）

$$
节点集 \ V:\{1,2,3,4\} \\
无权图 \ 
A=\left(\begin{array}{llll}0 & 1 & 0 & 1 \\ 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 1 \\ 1 & 1 & 1 & 0\end{array}\right)
$$

#### 编码器

输入一个节点，输出节点对应的d维向量
$$
ENC(u) = \mathbf{z}_{v}
$$

##### 最简单的编码器：查表

提取Z矩阵的某一列（一个节点的嵌入向量）
$$
\begin{aligned} \mathrm{ENC}(v) & =\mathbf{z}_{v}=\mathbf{Z} \cdot v \\ \mathbf{Z} & \in \mathbb{R}^{d \times|\mathcal{V}|} \\ v & \in \mathbb{I}^{|\mathcal{V}|}\end{aligned}
$$

#### 解码器

向量点乘值（余弦相似度），反映节点在原图的相似度（需人为定义）
$$
\operatorname{similarity}(u, v) \approx \mathbf{z}_{v}^{\mathrm{T}} \mathbf{z}_{u}
$$

* 如果两个节点完全不相似，对应两个向量就是正交的，向量点乘值为0
* 如果两个节点直接向量，相似度（向量点乘值）是1，对应两个向量就是共线的

编码器和解码器的结构可以替换，例如用深度学习做编码器，用两个向量L2距离做解码器；

可以定义其他的节点相似度，例如间接相连，相同功能角色

#### 优化目标

迭代优化每个节点的d维向量，

使得图中相似节点向量数量积大，不相似节点向量数量积小

## 基于随机游走的方法

在图中开始节点，随机地选择一个相邻节点移动，不断游走。最终得到随机游走序列，很多的随机游走序列，不同长度，不同随机游走策略。

* 图机器学习和自然语言处理是能够一一对应的
  * 图——文章
  * 随机游走序列——句子
  * 节点——单词
  * DeepWalk——Skip-Cram
  * Node Embedding——Word Embeding

### 定义

从u节点出发的随机游走序列经过v节点的条件概率
$$
P\left(v \mid \mathbf{z}_{u}\right)
$$
非线性Softmax函数用于计算概率
$$
\sigma(\mathbf{z})[i]=\frac{e^{z[i]}}{\sum_{j=1}^{K} e^{z[j]}}
$$
节点“相似”定义：共同出现在同一个随机游走序列。

### 步骤

1. 采样得到若干随机游走序列，计算条件概率
2. 迭代优化每个节点的d维向量，使得序列中共现节点向量数量积大，不共线节点向量数量积小 

### 优点

表示能力：能够捕捉局部和更高阶邻域大信息，通过大量随机游走序列的生成，能够很好的表述起始节点的结构信息，能够捕捉多跳的信息

计算便捷：不需要大量计算资源，只需要暴力的方法去采样即可

无监督/自监督学习问题

### 优化目标

使用极大似然估计，优化目标函数
$$
\max _{f} \sum_{u \in V} \log \mathrm{P}\left(N_{\mathrm{R}}(u) \mid \mathbf{z}_{u}\right)
$$

$$
N_{\mathrm{R}}(u) 是从u节点出发的随机游走序列所有邻域节点
$$

遍历所有节点，在每个节点u下，遍历从u节点出发的随机游走序列所有邻域节点
$$
\mathcal{L}=\sum_{u \in V} \sum_{v \in N_{R}(u)}-\log \left(P\left(v \mid \mathbf{z}_{u}\right)\right)
$$
表示节点u和节点v在该随机游走序列中共现
$$
P\left(v \mid \mathbf{z}_{u}\right)=\frac{\exp \left(\mathbf{z}_{u}^{\mathrm{T}} \mathbf{z}_{v}\right)}{\sum_{n \in V} \exp \left(\mathbf{z}_{u}^{\mathrm{T}} \mathbf{z}_{n}\right)}
$$
整体目标使得概率最大化，即上述损失函数L最小化。

#### 工程优化

$$
\mathcal{L}=\sum_{u \in V} \sum_{v \in N_{R}(u)}-\log \left(P\left(v \mid \mathbf{z}_{u}\right)\right)
$$

式子需要遍历所有节点，每个节点下还要遍历所有节点对，复杂度非常高。

##### 负采样

$$
\begin{array}{c}\log \left(\frac{\exp \left(\mathbf{z}_{u}^{\mathrm{T}} \mathbf{z}_{v}\right)}{\sum_{n \in V} \exp \left(\mathbf{z}_{u}^{\mathrm{T}} \mathbf{z}_{n}\right)}\right) \\ \approx \log \left(\sigma\left(\mathbf{z}_{u}^{\mathrm{T}} \mathbf{z}_{v}\right)\right)-\sum_{i=1}^{k} \log \left(\sigma\left(\mathbf{z}_{u}^{\mathrm{T}} \mathbf{z}_{n_{i}}\right)\right), n_{i} \sim P_{V}\end{array}
$$

只算u节点和k个负样本的数量积，k=5～20.

理论上，同一个随机游走序列中的节点，不应该被采样为“负样本”。但是大样本中很难采到，所以工程上可以随机采。

##### 随机梯度下降

全局梯度下降：

​	随机初始化，

​	求所有节点u总梯度，
$$
\frac{\partial \mathcal{L}}{\partial z_{u}}
$$
​	迭代更新。
$$
z_{u} \leftarrow z_{u}-\eta \frac{\partial \mathcal{L}}{\partial z_{u}}
$$
随机梯度下降：

​	每个训练样本优化一次，即每次随机游走优化一次，迭代更新。

## Node2Vec

步骤和DeepWalk是一模一样的，区别在于是有偏二阶随机游走。

通过设置p和q两个超参数，取控制随机游走是探索邻域还是探索远方。一个参数（1/p概率）控制是否回上一个节点，另一个参数（1/q概率）控制是否走向更远的节点。1的权重表示走向上一个节点距离相等的节点。
$$
\mathrm{W} \rightarrow\left|\begin{array}{c} Target t \\ \mathrm{S}_{1} \\ \mathrm{~S}_{2} \\ \mathrm{~S}_{3} \\ \mathrm{~S}_{4}\end{array}\right|\left|\begin{array}{c} Prob. \\ 1 / p \\ 1 \\ 1 / q \\ 1 / q\end{array}\right|\left|\begin{array}{c} Dist.(s_{1},t) \\ 0 \\ 1 \\ 2 \\ 2\end{array}\right|
$$
按照概率，采样出下一个随机游走的节点。

通过不同的p、q组合达到不同的效果，

p大q小：DFS深度优先（探索远方），探索出同质社群属性

p小q大：BFS宽度优先（探索近邻），反映节点的功能角色特征（中枢、桥接、biany）

### node算法

1. 计算每条边的权重和概率
2. 以u节点为出发点，长度为l，生成r个随机游走序列
3. 用随机梯度下降优化目标函数

## 其他基于随机游走的图嵌入方法

* 基于节点属性特征（Dong et al., 2017）
* 基于学习到的连接权重（Abu-El-Haija et al., 2017）
* 基于1跳和2跳随机游走概率（as in LINE from Tang et al. 2015）
* 在原图基础上改了一个新图，在新图上做随机游走（Ribeiro et al. 2017's struct2vec, Chen et al. 2016's HARP）

不管如何，都是希望得到的向量相似度，能够反映节点相似度。

## 矩阵分解角度理解图嵌入和随机游走

矩阵分解和随机游走，本质上在数学上是一回事。
$$
A=\boldsymbol{Z}^{T} \mathbf{Z}
$$
对邻接矩阵做矩阵分解，邻接矩阵元素为1，即表示两个节点相连；元素为0，两个节点正交（不直接相连）

A是半正定矩阵，而Z是方阵，不太好做矩阵分解。得到的解不唯一，在数学上不好做解析解。

从数值计算估计，目标
$$
\min _{\mathbf{Z}}\left\|A-\boldsymbol{Z}^{T} \boldsymbol{Z}\right\|_{2}
$$
DeepWalk也可以写成矩阵分解形式，

可以视为对下式做矩阵分解
$$
\begin{array}{c}\log \left(\operatorname{vol}(G)\left(\frac{1}{T} \sum_{r=1}^{T}\left(D^{-1} A\right)^{r}\right) D^{-1}\right)-\log b \\ \quad \operatorname{vol}(G)=\sum_{i} \sum_{j} A_{i, j} \\ D_{u, u}=\operatorname{deg}(u)\end{array}
$$
其中，vol是邻接矩阵所有元素求和，即连接个数*2；T是上下文滑窗宽度；D是对角矩阵，是节点连接数；r是幂；b是负采样的样本数。

## 随机游走的图嵌入的讨论

### 缺点

* 无法立刻泛化到新加入的节点
* 只是探索相邻局部信息，只能采样出地理上相近作为节点相似的指标
* 仅利用图本身的连接信息，并没有使用属性信息

### DeepWalk

* 首个将深度学习和自然语言处理的思想用于图机器学习
* 在稀疏标注节点分类场景下，嵌入性能卓越
* 均匀随机游走，没有偏向的游走方向
* 需要大量随机游走序列训练
* 基于随机游走，管中窥豹，距离较远的两个节点无法相互影响。看不到全图信息。
* 无监督，仅编码图的连接信息，没有利用节点的属性特征。
* 没有真正用到神经网络和深度学习

### Node2Vec

- 解决图嵌入问题，将图中的每个节点映射为一个向量（嵌入）。
- 向量（嵌入）包含了节点的语义信息（相邻社群和功能角色）。
- 语义相似的节点，向量（嵌入）的距离也近。
- 向量（嵌入）用于后续的分类、聚类、Link Prediction、推荐等任务。
- 在DeepWalk完全随机游走的基础上，Node2Vec增加参数p和q，实现有偏随机游走。不同的p、q组合，对应了不同的探索范围和节点语义。
- DFS深度优先探索，相邻的节点，向量（嵌入）距离相近。
- BFS广度优先探索，相同功能角色的节点，向量（嵌入）距离相近。
- DeepWalk是Node2Vec在p=1，q=1的特例。



## 嵌入整张图

### 直接对所有节点嵌入求和

$$
\mathbf{z}_{\boldsymbol{G}}=\sum_{v \in G} \mathbf{z}_{v}
$$

### 引入虚拟节点

和全图的所有节点都相连。求出该虚拟节点的嵌入，即作为全图的嵌入。

### 匿名随机游走嵌入

每次见到不同节点，就发一个新编号。”认号不认人“

PS：3个节点的匿名随机游走有5种可能

#### Bag of "Anonymous Walks"

采样出不同匿名随机游走序列的个数，构成的向量作为全图的向量。

匿名随机游走长度固定时，欲使误差大于*epsilon*的概率小于*delta*，需采样m次
$$
m=\left\lceil\frac{2}{\varepsilon^{2}}\left(\log \left(2^{\eta}-2\right)-\log (\delta)\right)\right\rceil
$$

#### New idea：Learn Walk Embeddings

1. 给每种匿名随机游走序列单独嵌入编码，再一个全图的嵌入编码。

2. 将随机游走的编码平均一下，再和全图的编码堆叠，得到一个2d维向量。
   $$
   \operatorname{cat}\left(\frac{1}{\Delta} \sum_{i=1}^{\Delta} \mathbf{z}_{i}, \mathbf{z}_{\boldsymbol{G}}\right)
   $$

3. 再接一个线性分类层和softmax。
   $$
   y\left(w_{t}\right)=b+U \cdot\left(\operatorname{cat}\left(\frac{1}{\Delta} \sum_{i=1}^{\Delta} \mathbf{z}_{i}, \mathbf{z}_{\boldsymbol{G}}\right)\right)
   $$

   $$
   P\left(w_{t} \mid\left\{w_{t-\Delta}, \ldots, w_{t-1}, z_{G}\right\}\right)=\frac{\exp \left(y\left(w_{t}\right)\right)}{\sum_{i=1}^{\eta} \exp \left(y\left(w_{i}\right)\right)}
   $$

4. 让其预测出额外一个匿名随机游走概率，越大越好，构建出一个自监督学习的问题。
   $$
   \max _{z_{i}, z_{G}} \frac{1}{T} \sum_{t=\Delta}^{T} \log P\left(w_{t} \mid\left\{w_{t-\Delta}, \ldots, w_{t-1}, z_{G}\right\}\right)
   $$



