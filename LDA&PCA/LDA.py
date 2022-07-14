"""
线性判别分析原理
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 使用鸢尾花数据集
df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
)
# print(df.head())
# 下载的数据没列名,自定义列名
feature_dict = {i: label for i, label in
                zip(range(4),
                    ('sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm',)
                    )}

df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
# print(df.head())

X = df[['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']].values
y = df['class label'].values
# 处理标签列,{1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}
enc = LabelEncoder()
label_encoder = enc.fit(y)
# transform函数是从0开始的
y = label_encoder.transform(y) + 1

# 首先算各个类别的均值
# 设置小数点的位数
np.set_printoptions(precision=4)
# 这里会保存所有的均值
mean_vectors = []
# 要计算3个类别
for cl in range(1, 4):
    # 求当前类别各个特征均值
    mean_vectors.append(np.mean(X[y == cl], axis=0))
    print('类别:%s,均值:%s\n' % (cl, mean_vectors[cl - 1]))

# 然后计算类内散布矩阵
# 原始数据中有4个特征,那么散布矩阵就是4*4的
S_W = np.zeros((4, 4))
for cl, mv in zip(range(1, 4), mean_vectors):
    # 每个类别自身的类内散布矩阵
    class_sc_mat = np.zeros((4, 4))
    # 选中属于当前类别的数据
    for row in X[y == cl]:
        # 这里相当于对各个特征分别进行计算，用矩阵的形式
        row, mv = row.reshape(4, 1), mv.reshape(4, 1)
        # 跟公式一样
        class_sc_mat += (row - mv).dot((row - mv).T)
    # 最终的散布矩阵就是各个类别的散布矩阵之和
    S_W += class_sc_mat
print('类内散布矩阵:\n', S_W)

# 然后算类间散布矩阵
# 全局均值
overall_mean = np.mean(X, axis=0)
# 构建类间散布矩阵
S_B = np.zeros((4, 4))
for i, mean_vec in enumerate(mean_vectors):
    # 当前类别的样本数
    n = X[y == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(4, 1)
    overall_mean = overall_mean.reshape(4, 1)
    # 采用改进的计算方式,每类样本均值减去全局的均值然后乘以转置矩阵,然后乘以每类样本的数量,最后按类做累加
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
print('类间散布矩阵:\n', S_B)

# 最后,求解矩阵特征值，特征向量
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# 拿到每一个特征值和其所对应的特征向量
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:, i].reshape(4, 1)
    print('\n特征向量 {}: \n{}'.format(i + 1, eigvec_sc.real))
    print('特征值 {:}: {:.2e}'.format(i + 1, eig_vals[i].real))

# 特征值和特征向量配对
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
# 按特征值大小进行排序
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print('特征值排序结果:\n')
for i in eig_pairs:
    print(i[0])

# 选择前两维的特征向量构成w矩阵
W = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))
print('矩阵W:\n', W.real)
# 然后根据w做降维操作
X_lda = X.dot(W)

# 可视化展示
label_dict = {i: label for i, label in zip(range(1, 4), ('Setosa', 'Versicolor', 'Virginica'))}


# 原始数据
def plot_step_lda():
    ax = plt.subplot(111)
    for label, marker, color in zip(
            range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=X[:, 0].real[y == label],
                    y=X[:, 1].real[y == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )

    plt.xlabel('X[0]')
    plt.ylabel('X[1]')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('Original data')

    # 把边边角角隐藏起来
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    # 为了看的清晰些，尽量简洁
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.show()


plot_step_lda()


# 降维后的数据
def plot_step_lda():
    ax = plt.subplot(111)
    for label, marker, color in zip(
            range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=X_lda[:, 0].real[y == label],
                    y=X_lda[:, 1].real[y == label],
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
                    )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA on iris')

    # 把边边角角隐藏起来
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    # 为了看的清晰些，尽量简洁
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.show()


plot_step_lda()

# 用sklearn的组件来做
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)


def plot_scikit_lda(X, title):
    ax = plt.subplot(111)
    for label, marker, color in zip(
            range(1, 4), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(x=X[:, 0][y == label],
                    y=X[:, 1][y == label] * -1,  # flip the figure
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.show()


plot_scikit_lda(X_lda_sklearn, title='Default LDA via scikit-learn')
