### 贝叶斯算法原理

贝叶斯解决的问题实际是一个逆向概率的问题.

例如,已知一个学校男女比例,并且男生一定穿长裤,女生有一定概率长裤.那么正向概率就是选一个学生,问他穿长裤的概率,这个概率很直观.但是如果现在是有个学生穿了长裤,那这个学生是女生的概率是多少就不是很直观了.

首先假设总的人数是$U$.

然后我们可以知道穿长裤的男生的人数是:$U*P(boy)*P(pans|boy)$,其中$P(boy)$是已知的,$P(pans|boy)$由条件可知是为1.

接着求穿长裤的女生的人数是:$U*P(girl)*P(pans|girl)$,其中各个概率是可知的

最后$P(girl|pans)$的值就是穿长裤的女生除以总穿长裤的人数,即:

$$
P(girl|pans)={{U*P(girl)*P(pans|girl)} \over U*P(girl)*P(pans|girl)+U*P(boy)*P(pans|boy)}
$$

最后发现可以将$U$约掉,并且约掉后,分子就是$P(pans|girl)*P(girl)$,分母就是$P(pans)$,最后就有:

$$
P(girl|pans)={P(pans|girl)*P(girl) \over P(pans)}
$$

总结出的贝叶斯公式就是:

$$
P(A|B)={P(B|A)P(A) \over P(B)}
$$



