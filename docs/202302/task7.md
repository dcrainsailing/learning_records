# 半监督节点分类

## 基本概念

有一半节点有标签，有一半节点无标签；

目标是由已知标签节点，预测出未知节点标签。

是直推式学习（Transductive），即不需要对新节点泛化学习；

与之相对的是归纳式学习（Inductive），即要泛化到新的节点。

### 求解思路

节点特征工程

节点表示学习（图嵌入）

标签传播（消息传递）

图神经网络

### 数学模式

给出一个$n \times n$的邻接矩阵A（可以是无权重的，可以是有权重的，有权重的邻接矩阵中的值就是连接的权重）；

给出一个类别标签的向量$Y=\{0,1\}^{n}$，其中$Y_{v}=1$属于类别1，$Y_{v}=0$属于类别0；

很多节点是unlabeled，目标就是由已知0或1的节点去预测这些unlabeled的节点。

每个节点有属性特征向量$f_{v}$，还有在图中和其他节点的连接方式。

### 从图的视角审视各种模态数据

可以把很多其他数据挖掘问题，抽象成半监督节点分类问题

* 文件分类
* 词性标注
* 连接预测
* OCR光学字符识别问题
* 图像分割
* 实体统一（例如：一家公司有两个名字）
* 垃圾邮件/欺诈检测

### 求解方法对比

| 方法               | 图嵌入 | 表示学习 | 使用属性特征 | 使用标注 | 直推式 | 归纳式 |
| ------------------ | ------ | -------- | ------------ | -------- | ------ | ------ |
| 人工特征工程       | 是     | 否       | 否           | 否       | /      |        |
| 基于随机游走的方法 | 是     | 是       | 否           | 否       | 是     | 否     |
| 基于矩阵分解的方法 | 是     | 是       | 否           | 否       | 是     | 否     |
| 标签传播           | 否     | 否       | 是/否        | 是       | 是     | 否     |
| 图神经网络         | 是     | 是       | 是           | 是       | 是     | 是     |

- 人工特征工程：节点重要度、集群系数、Graphlet等

- 基于随机游走的方法，构建自监督表示学习任务实现图嵌入，无法泛化到新节点。

  例如DeepWalk、Node2Vec、LINE、SDNE等

- 标签传播：假设“物以类聚、人以群分”，利用领域节点类别预测当前节点类别，无法泛化到新节点。

  例如Label Propagation、Interative Classification、Belief Propagation、Correct & Smooth等

- 图神经网络：利用深度学习和神经网络，构建领域节点信息聚合计算图，实现节点嵌入和类别预测，可泛化到新节点。

  例如GCN、GraphSAGE、GAT、GIN等

## 标签传播

基本假设：物以类聚，人以群分。

利用领域节点类别猜测当前节点类别。

标签传播和集体分类：

- Label Propagation（Relational Classification）
- Iterative Classification
- Correct & Smooth
- Belief Propagation
- Masked Label Prediction

### 大自然对图的基本假设

homophily在图中是广泛存在的。

homophily：具有相似属性特征的节点，更可能相连且具有相同类别。

influence：社交会影响节点类别。

如何充分利用网络中的correlation相关的信息，去进行半监督节点分类？

最简单的方法：KNN最近邻分类；

利用节点自身属性特征，更要利用邻域节点类别和属性特征。

## Label Propagation

算法步骤：

1. 初始化

   对已知标注的节点 ，打为1和0，$Y_{v}=\{0,1\}$；

   其他未知类别的，打为0.5，$Y_{v}=0.5$ 。

2. 迭代

   节点的类别，迭代更新为该节点周围的所有节点标注求平均值（加权平均）
   $$
   P\left(Y_{v}=C\right)=\frac{1}{\sum_{(v, u) \in E} A_{v, u}} \sum_{(v, u) \in E} P\left(Y_{u}=c\right)
   $$

3. 节点收敛

   当节点都收敛之后，可以设定阈值，进行类别划分，例如大于0.5设置为类别1，小于0.5设置为类别0

问题：

* 不保证收敛
* 仅用到网络连接信息，没有用到节点属性特征。



## Iterative Classification

定义节点属性特征$f_{v}$，连接特征$z_{v}$；

训练两个分离器：

1. base classifier $\phi_{1}(f_{v})$：仅使用节点属性特征；
2. relational classifier $\phi_{2}(f_{v},z_{v})$：使用节点属性特征和网络连接特征$z_{v}$（邻域节点信息）。

可以自定义$z_{v}$的算法，以包含邻域节点类别信息的向量

* 节点周围不同类别节点的直方图
* 周围最多的类别
* 周围不同类别的个数

算法步骤：

1. 使用已标注的数据训练两个分类器：$\phi_{1}(f_{v})$、$\phi_{2}(f_{v},z_{v})$
2. 迭代直至收敛：用$\phi_{1}$预测未知类别的节点$Y_{v}$，用$Y_{v}$计算$z_{v}$，然后再用$\phi_{2}$预测所有节点类别，更新领域节点$z_{v}$，用新的$\phi_{2}$更新$Y_{v}$。

 问题：

* 不保证收敛

这种模式可以抽象为collective classification，

它的基本假设是马尔科夫假设，我是什么类别仅取决于与我相连的节点是什么类别。
$$
P\left(Y_{v}\right)=P\left(Y_{v} \mid N_{v}\right)
$$

## Correct & Smooth

属于一种后处理的方法。

算法步骤：

1. 在有类别标注的节点上训练一个基础的分离器

2. 用分离器去预测所有节点（包含有类别标注的节点）的预测结果（是soft -labels，即对于二分类，两种类别的概率都不是非0即1的，加和为1）

3. 后处理

   1. correct step：计算（仅有标注的节点）training error $\boldsymbol{E}^{(0)}$；

      假设error在图中也有homophily，让error更均匀分布在每一点。
      $$
      \boldsymbol{E}^{(t+1)} \leftarrow(1-\alpha) \cdot \boldsymbol{E}^{(t)}+\alpha \cdot \widetilde{\boldsymbol{A}} \boldsymbol{E}^{(t)} 
      $$
      $A$为邻接矩阵，将邻接矩阵$A$进行归一化，得到扩散矩阵$\tilde{A}=D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$；

      $D$是对角矩阵，对角线上元素即节点的度，$\boldsymbol{D} \equiv \operatorname{Diag}\left(d_{1}, \ldots, d_{N}\right)$；

      把$\tilde{A}$的元素当作某种权重，作为error扩散的权重。

      $\tilde{A}$性质：

      * $\tilde{A}$的特征值$\lambda \in[-1,1]$，特征值为1的时候，特征向量为$D^{\frac{1}{2}}$

      * $\widetilde{A} D^{1 / 2} 1=D^{-1 / 2} A D^{-1 / 2} D^{1 / 2} 1=D^{-1 / 2} A 1$

        $A 1 = D 1$

        $\widetilde{A} D^{1 / 2} 1 = \boldsymbol{D}^{-1 / 2} \boldsymbol{D} \mathbf{1}=1 \cdot \boldsymbol{D}^{1 / 2} \mathbf{1}$

      * 幂运算之后，依然保证收敛（特征值在[-1,1]之间）。

   2. smooth step：对最终的预测结果进行label propagation

      通过图得到$\boldsymbol{Z}^{(0)}$

      把置信度$Z$进行传播扩散，$Z^{(t+1)} \leftarrow(1-\alpha) \cdot Z^{(t)}+\alpha \cdot \widetilde{A} Z^{(t)}$

      最终的$Z$矩阵并不是求和为1的，但是大小关系仍然是有意义的，仍然可以把大小关系作为类别预测的结果。



## Belief Propagation

类似消息传递，基于动态规划，即下一时刻的状态仅取决于上一时刻，当所有节点达到共识时，可得最终结果。

算法思路：

1. 定义一个节点序列
2. 按照边的有向顺序排列
3. 从节点$i$到节点$i+1$计数（类似报数）

定义：

* Label-label potential matrix $\psi$ ：当邻居节点 $i$ 为类别 $Y_{i}$ 时，节点 $j$ 为类别$Y_{j}$的概率（标量），反映了节点和其邻居节点之间的依赖关系；
* Prior belief $\phi: \phi\left(Y_{i}\right)$ 表示节点 $i$ 为类别$Y_{i}$ 的概率；
* $m_{i \rightarrow j}\left(Y_{j}\right)$ : 表示节点 $i$ 认为节点 $j$ 是类别$Y_{j}$的概率；
* $\mathcal{L}$ ：表示节点的所有标签。

算法步骤：

1. 初始化所有节点信息都为1；

2. 对每个节点迭代运算：
   $$
   m_{i \rightarrow j}\left(Y_{j}\right)=\sum_{Y_{i} \in \mathcal{L}} \psi\left(Y_{i}, Y_{j}\right) \phi_{i}\left(Y_{i}\right) \prod_{k \in N_{j} \backslash j} m_{k \rightarrow i}\left(Y_{i}\right), \forall Y_{j} \in \mathcal{L}
   $$

3. 收敛之后，
   $$
   b_{i}\left(Y_{i}\right)=\phi_{i}\left(Y_{i}\right) \prod_{j \in N_{i}} m_{j \rightarrow i}\left(Y_{i}\right), \forall Y_{j} \in \mathcal{L}
   $$

优点：

* 非常容易编程和并行
* 可以把potential matrix高阶的去设置，例如：$\psi\left(Y_{i}, Y_{j}, Y_{k}, Y_{v} \ldots\right)$

缺点：

* 不保证收敛（尤其是图中有环的时候）
* Label-label potential matrix里的参数需要训练才优化得到的



## Masked Label Prediction

灵感来自于语言模型BERT，自监督学习。

算法思路：

1. 随机将节点的标签设置为0，用$[X, \tilde{Y}]$预测已标记的节点标签；
2. 构造自监督的学习场景，构造损失函数，迭代地去优化。



