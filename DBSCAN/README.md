### 带噪声的基于密度的空间聚类应用(Density-Based Spatial Clustering of Applications with Noise)原理

#### 主要概念

- 核心对象:若某个点的密度达到算法设定的阈值则其为核心点.(即 r 邻域内点的数量不小于 minPts)
- $\epsilon$-邻域的距离阈值:设定的半径r
- 直接密度可达:若某点q在中心点p的r邻域内,则称 $p-q$ 直接密度可达
- 密度可达:若有一个点的序列 $p_0,p_1,p_2\dots$ ,其中对任意 $p_i-p_{i-1}$ 是直接密度可达,那么可以称 $p_0-p_i$ 密度可达
- 密度相连:若从点 $p$ 出发,点 $j,k$ 都是密度可达的,那么可以称点 $j,k$ 是密度相连的
- 边界点:属于某一个簇的非核心点,不能找到新的直接密度可达的点了,就是边界点
- 噪声点:不属于任何一个簇类,从任何一个核心点出发都不是密度可达的

#### 工作流程

> 标记所有对线是unvisited
> 	
> Do
> 	
> &nbsp;&nbsp;随机选择一个unvisited对象为$p$
> 	
> &nbsp;&nbsp;标记p为visited
> 	
> &nbsp;&nbsp;if $p$的$\epsilon$-邻域内至少有minPts个对象
> 	
> &nbsp;&nbsp;&nbsp;&nbsp;创建一个新簇$C$,并把$p$添加到$C$内
> 	
> &nbsp;&nbsp;&nbsp;&nbsp;令$N$为 $p$的$\epsilon$-邻域中的对象的集合
> 	
> &nbsp;&nbsp;&nbsp;&nbsp;For $N$中的每一个点$p\prime$
> 	
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if $p\prime$是unvisited
> 	
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;标记$p\prime$为visited
> 	
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if $p\prime$的$\epsilon$-邻域内至少有minPts个对象
> 	
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;把这些对象加到$C$里面
> 	
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if $p\prime$还不是任何簇的成员
>	
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;把$p\prime$添加到$C$里面
> 	
> &nbsp;&nbsp;输出$C$
> 	
> &nbsp;&nbsp;else 标记$p$为噪声点
> 	
> while 没有标记为unvisited的对象

#### 参数选择

- 半径 $\epsilon$ :可以根据 $K$ 距离来设定,找 $K$ 距离的突变点
  - K距离:给定数据集为 $P=\{p(i);i=0,1,\dots,n\}$ ,计算点 $p(i)$ 到集合 $D$ 的子集 $S$ 间所有点的距离,距离按照从小到大排序, $d(k)$ 就称为 $k-$ 距离
- MinPts: $k-$ 距离中 $k$ 的值,一般取小点,可以多试几次

#### 优缺点

- 优点:
  - 不用指定簇的个数
  - 可以发现任意形状的簇(因为是通过密度去发现的)
  - 擅长找出离群点(做检测任务)
  - 要的参数不多,就两个

- 缺点:
  - 高维度数据有些困难(容易出现内存不足的问题,可以做降维来解决)
  - 参数不好选(参数对结果的影响非常大)
  - Sklearn中效率很慢(可以做数据削减)
