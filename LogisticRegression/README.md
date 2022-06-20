### 原理分析

逻辑回归实际上是在线性回归的基础上完成,由于通过线性回归已经得到了一个预测值,那么逻辑回归可以通过一个sigmoid函数:

$$g(z)=\frac{1}{1+e^x}$$

来将结果映射到$[0,1]$中间,那么此时就将值映射到一个概率空间里面,来完成一个分类任务.

整合之后的预测函数就变成了:

$$h_\theta(x)=g({\theta^T}x)=\frac{1}{1+e^{{\theta^T}x}}$$

那么对于一个二分类的任务来说就有

$$P(y=1|x;\theta)=h_\theta(x)$$

$$P(y=0|x;\theta)=1-h_\theta(x)$$

将上面两个式子整合起来就有:

$$P(y|x;\theta)={h_\theta(x)}^y(1-h_\theta(x))^{1-y}$$

即对于一个二分类任务来说,对于y取0或1的时候上面的式子就变成前面两个式子

那么就可以知道逻辑回归的似然函数就是:

$$L(\theta)=\prod_{i=1}^{m}{h_\theta(x_i)}^{y_i}(1-h_\theta(x_i))^{1-y_i}$$

对应的对数似然就是:

$$l(\theta)=\log{L(\theta)}=\sum_{i=1}^{m}(y_i\log(h_\theta(x_i))+(1-y_i)\log(1-h_\theta(x_i)))$$

此时是梯度上升求最大值,那么可以引入:

$$J(\theta)=-\frac{1}{m}l(\theta)$$

转换成梯度下降的任务,那么对于特征j来说就有:

$$\frac{\partial}{\partial{\theta_j}}J(\theta)=-\frac{1}{m}\sum_{i=1}^m(h_\theta(x_i)-y_i)x_i^j$$

那么最终的一个二分类的逻辑回归的参数更新就是:

$$\theta=\theta-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x_i)-y_i)x_i^j$$

并且将一个二分类做三次可以完成一个三分类任务,由此可以引申出多分类任务.
