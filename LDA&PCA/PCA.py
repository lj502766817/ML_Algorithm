"""
主成分分析原理
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取数据集
df = pd.read_csv('../data/iris.data')
# 原始数据没有列名
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
print(df.head())

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

# 画图展示数据特征
# 展示我们标签用的
# label_dict = {1: 'Iris-Setosa',
#               2: 'Iris-Versicolor',
#               3: 'Iris-Virgnica'}
#
# # 展示特征用的
# feature_dict = {0: 'sepal length [cm]',
#                 1: 'sepal width [cm]',
#                 2: 'petal length [cm]',
#                 3: 'petal width [cm]'}
#
# # 指定绘图区域大小
# plt.figure(figsize=(8, 6))
# for cnt in range(4):
#     # 这里用子图来呈现4个特征
#     plt.subplot(2, 2, cnt + 1)
#     for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
#         plt.hist(X[y == lab, cnt],
#                  label=lab,
#                  bins=10,
#                  alpha=0.3, )
#     plt.xlabel(feature_dict[cnt])
#     plt.legend(loc='upper right', fancybox=True, fontsize=8)
#
# plt.tight_layout()
# plt.show()

# 先对数据做标准化
X_std = StandardScaler().fit_transform(X)
# 计算协方差矩阵
# 平均值
mean_vec = np.mean(X_std, axis=0)
# 协方差矩阵
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
print('协方差矩阵 \n%s' % cov_mat)
# 或者直接用np来做
print('numpy协方差矩阵: \n%s' % np.cov(X_std.T))

# 计算特征值和特征向量
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('特征向量 \n%s' % eig_vecs)
print('\n特征值 \n%s\n' % eig_vals)
# 把特征值和特征向量对应起来
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
print(eig_pairs)
print('----------')
# 把它们按照特征值大小进行排序
eig_pairs.sort(key=lambda x: x[0], reverse=True)
# 打印排序结果
print('特征值从大到小排序结果:')
for i in eig_pairs:
    print(i[0])

# 计算累加结果,看降维的维度对应的特征信息占比
tot = sum(eig_vals)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
print(var_exp)
cum_var_exp = np.cumsum(var_exp)
# 例如只用两个维度就能做到95%的效果
print(cum_var_exp)

# 画图展示
# plt.figure(figsize=(6, 4))
#
# plt.bar(range(4), var_exp, alpha=0.5, align='center',
#         label='individual explained variance')
# plt.step(range(4), cum_var_exp, where='mid',
#          label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

# 选前两个维度作为一组基
matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),
                      eig_pairs[1][1].reshape(4, 1)))

print('Matrix W:\n', matrix_w)

# 降维后的数据
Y = X_std.dot(matrix_w)

# 降维前后的对比图,并且降维后的数据不具备可解释性
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                    ('blue', 'red', 'green')):
    plt.scatter(X[y == lab, 0],
                X[y == lab, 1],
                label=lab,
                c=col)
plt.xlabel('sepal_len')
plt.ylabel('sepal_wid')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                    ('blue', 'red', 'green')):
    plt.scatter(Y[y == lab, 0],
                Y[y == lab, 1],
                label=lab,
                c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()
