### 支持向量机原理

支持向量机(SVM)也是用来解决经典的二分问题的一种算法.SVM的目的就是找出能够一个能把样本二分类的面(在二维空间里就是一条线),并且使这两类样本点到这个面最近的的那个距离最大,通俗的说就是在两个样本最中间的那个面.那么SVM需要解决的问题就是找到那个面,假设这个面的方程为:

$$
W^TX+b=0
$$


那么问题就变成了到这个平面最近的样本点 $X(x,y)$ 到这个平面的距离 $h$ 最大.由于点到面的距离没那么好求,那么可以把这个距离做个转换.把这个距离转换成从平面上一点 $X^\prime$ 到样本的一个向量 $X-X^\prime$ ,在平面的单位法向量 $e \over ||W||$ 上的投影.那么距离公式就变成了:

$$
distance(X,b,W)=|{e \over ||W||}(X-X^\prime)|
$$

并且有由前面的面方程可以将式子转换成:

$$
distance(X,b,W)={1 \over ||W||}|W^T{X}+b| \tag{决策方程}
$$

现在假设样本集是 $(X_1,Y_1),(X_2,Y_2)\dots(X_n,Y_n)$ ,样本的类可以设成,当 $X$ 为正例的时候 $Y=1$ ,此时点到面的距离为正,当 $X$ 为负例的时候 $Y=-1$ ,此时点到面的距离为负数.并且设决策方程是 $y(x)$ .那么这样决策方程就有:

$$
\begin{cases}
distance(x_i)>0 \Leftrightarrow y_i=1 \\
distance(x_i)<0 \Leftrightarrow y_i=-1
\end{cases}
\Rightarrow y_iy(x_i)>0
$$

那么由于y的绝对值是1,并且将维度扩展到高维度的时候可以用核函数 $\phi$ 进行转换,那么整体的距离公式最后就变成:

$$
distance(x_i)={{y_i\cdot(w^T \cdot \phi(x_i)+b)} \over ||w||}
$$

由于我们需要的是决策方程的极值点,而不是极值,那么就可以通过一些放缩的变换使得 $|Distance|>1$ ,即有:

$$
y_i\cdot(w^T \cdot \phi(x_i)+b)\geq1 \tag{相对于前面的条件更加严格点}
$$

那么最后可以得到优化目标:

$$
\operatorname{arg\,max}_{w,b}\{{1\over||w||}\operatorname{min}[y_i\cdot(w^T \cdot \phi(x_i)+b)]\} \tag{方程可以理解为,离得最近的样本,找最远距离}
$$

由于$y_i\cdot(w^T \cdot \phi(x_i)+b)\geq1$,那么只需要考虑$\operatorname{arg\,max}_{w,b}{1\over||w||}$就行了,于是就得到了目标函数:

$$
当前目标:\operatorname{arg\,max}_{w,b}{1\over||w||} \\
约束条件:y_i\cdot(w^T \cdot \phi(x_i)+b)\geq1
$$

按照机器学习的传统套路就可以将求极大值的任务转换成求极小值的任务:

$$
\operatorname{arg\,min}_{w,b}{{1\over2} w^2}
$$

那么对于给定条件,求极值的问题,可以用拉格朗日乘子法

$$
带约束的优化问题:\operatorname{min}_{x}f_0(x)　subject　to　f_i(x)\le0,i=1,\dots,m　h_i(x)=0,i=1,\dots,q　\\
原式转换:\operatorname{min}L(x,\lambda,v)=f_0(x)+\sum_{i=1}^{m}\lambda_if_i(x)+\sum_{i=1}^{q}v_ih_i(x)
$$

来进行求解,那么就可以把我们的式子变成:

$$
L(w,b,\alpha)={{1\over2} ||w||^2}-\sum_{i=1}^{n}\alpha_i(y_i\cdot(w^T \cdot \phi(x_i)+b)-1) 
\\ subject　to　y_i\cdot(w^T \cdot \phi(x_i)+b)\geq1
$$

并且由于对偶性质(KKT条件),我们可以将 $\operatorname{min}_{w,b}\operatorname{max}_\alpha L(w,b,\alpha)$ 的问题转换成  $\operatorname{max}_{\alpha}\operatorname{mim}_{w,b}L(w,b,\alpha)$ 的问题,然后就分别对 $w$ , $b$ 求偏导,然后带入原式就得到一个与 $\alpha$ 有关的式子,然后对 $\alpha$ 求极大值,此时可以将这个式子取负转换成求 $\alpha$ 的极小值,求得了 $\alpha$ 之后就能反推得到 $w,b$ 了

#### 软间隔

为了处理数据中的噪声点,那么我们可以把条件放松一些,那么就可以将限制条件变成下面的式子

$$
y_i\cdot(w^T \cdot \phi(x_i)+b)\geq1-\xi_i
$$

那么原来的目标函数就变成了

$$
\operatorname{min}{{1\over2} ||w||^2}+C\sum_{i=1}^{n}\xi_i
$$

可以看到 $C$ 越大那么为了使目标函数小, $\xi$ 就会越小,即限制条件就越严格,反过来 $C$ 越小,那么限制条件就会越松散

#### 核函数的说明

有时候样本在低维度很难区分的时候,如果把它扩展到高维度可能就更好区分.那么这个时候就可以用核函数将低维数据扩展到高维.而在实际使用过程中,并不是对参数本身做核函数的运算,而是对结果做核函数的运算
