import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.cluster import KMeans

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
