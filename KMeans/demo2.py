import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 图像分隔,分隔前景和背景
# 读取图像
image = imread('../img/ladybug.png')
print(image.shape)
# 用kmeans做分隔,就把数据做成二维的
X = image.reshape(-1, 3)
print(X.shape)

# 做聚类
kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
# 聚类后的中心点位置
print(kmeans.cluster_centers_)
# 将各个簇的样本替换成质心,并重新转换成原来图像的维度
segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(533, 800, 3)

# 做不同的颜色类目来对比
segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_cluster in n_colors:
    kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))
plt.figure(figsize=(10, 5))
plt.subplot(231)
plt.imshow(image)
plt.title('Original image')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title('{}colors'.format(n_clusters))

plt.show()

# 半监督学习
# 做点数据
X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

# 假设现在样本里只有50个数据有标签,那么就只能拿50个样本做线性回归
n_labeled = 50
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
print(log_reg.score(X_test, y_test))
# 可以看到学习的效果不是很好,因为有可能这50个样本量太小,而且这些样本本身不够好
# 那么先用聚类找出50个特征比较好的样本试试
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
# 获得每个样本到簇中心的距离
X_digits_dist = kmeans.fit_transform(X_train)
# 找到每个簇里最靠近中心点的样本,这些样本就是50个类别里特征最好的样本
representative_digits_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digits_idx]
# 手动做标记
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')
plt.show()
y_representative_digits = np.array([
    4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
    5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
    1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
    6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
    4, 2, 9, 4, 7, 6, 2, 3, 1, 1])

# 现在测试集的样本就是每个类别里最好的样本了,不是随机选择的样本,用这些样本来训练看效果
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
print(log_reg.score(X_test, y_test))
# 可以看到用特征好的样本来做,效果会好很多

# 然后将特征好的样本做下扩展,把标签扩展到簇里面的每个样本
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    # 每次迭代都把对应簇的样本的标签设置成对应的标签
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train_propagated)
print(log_reg.score(X_test, y_test))
# 可以看到效果加的不多

# 然后选择每个簇里面最靠近中心点的20个样本来做训练
percentile_closest = 20
# 这里是取出每个样本到他最近中心点的距离
X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    # 选择属于当前簇的所有样本
    cluster_dist = X_cluster_dist[in_cluster]
    # 找到当前簇前20个样本的区分距离,即小于这个值的样本在前20个里面
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    # 距离大于区分值的样本的索引,即不需要的那些样本
    above_cutoff = (X_cluster_dist > cutoff_distance)
    # 将属于当前簇并且距离不属于前20的样本标识成-1
    X_cluster_dist[in_cluster & above_cutoff] = -1
# 距离属于前20的样本
partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
print(log_reg.score(X_test, y_test))
# 可以看到样本数量足够并且样本特征都比较好的情况下,做出的效果不错
