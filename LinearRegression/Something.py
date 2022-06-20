"""
各种综合性的对比
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
np.random.seed(42)

# 做些数据
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # 加上一点高斯抖动
# plt.plot(X, y, 'b.')
# plt.xlabel('X_1')
# plt.ylabel('y')
# plt.axis([0, 2, 0, 15])
# plt.show()

# 最优解的情况
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# 预测结果与原始结果进行比较
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
# plt.plot(X_new,y_predict,'r--')
# plt.plot(X,y,'b.')
# plt.axis([0,2,0,15])
# plt.show()

# 梯度下降的方式,用到了sklearn,https://scikit-learn.org/stable/modules/classes.html
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.coef_)  # 参数项,即θ1,θ2
print(lin_reg.intercept_)  # 偏置项,即θ0

# 不使用框架的批量梯度下降
eta = 0.1  # 学习率
n_iterations = 1000  # 迭代1000次
m = 100  # 前面弄了100个样本
theta = np.random.randn(2, 1)  # 初始化theta值
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)  # 计算梯度
    theta = theta - eta * gradients  # 沿着梯度下降更新theta
print(theta)

# 对比不同学习率对结果的影响
theta_path_bgd = []


def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)  # 采用批量梯度下降
    plt.plot(X, y, 'b.')
    n_iterations = 1000
    for _ in range(n_iterations):
        y_predict = X_new_b.dot(theta)
        plt.plot(X_new, y_predict, 'b-')  # 画出每一次的预测值
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)  # 如果传了theta的保存位置就保存历史theta
    plt.xlabel('X_1')
    plt.axis([0, 2, 0, 15])
    plt.title('eta = {}'.format(eta))


# 三个不同的学习率
# plt.figure(figsize=(10, 4))
# plt.subplot(131)
# theta = np.random.randn(2, 1)
# plot_gradient_descent(theta, eta=0.02)
# plt.subplot(132)
# theta = np.random.randn(2, 1)
# plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
# plt.subplot(133)
# theta = np.random.randn(2, 1)
# plot_gradient_descent(theta, eta=0.5)
# plt.show()

# 多项式回归

# 做些多项式数据
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + np.random.randn(m, 1)
# plt.plot(X, y, 'b.')
# plt.xlabel('X_1')
# plt.ylabel('y')
# plt.axis([-3, 3, -5, 10])
# plt.show()

# 对数据进行幂为2的多项式处理
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[0], X_poly[0])
# 用多项式处理后的结果进行训练
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.coef_)
print(lin_reg.intercept_)
# 对比查看
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, 'b.')
plt.plot(X_new, y_new, 'r--', label='prediction')
plt.axis([-3, 3, -5, 10])
plt.legend()
plt.show()

# 幂的次数对结果的影响,用sklearn.pipeline的流程模块去做
plt.figure(figsize=(12, 6))
for style, width, degree in (('g-', 1, 100), ('b--', 1, 2), ('r-+', 1, 1)):
    # 自定义多项式的次幂
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    # 用sklearn自带的标准化组件
    std = StandardScaler()
    lin_reg = LinearRegression()
    # 创建流程
    polynomial_reg = Pipeline([('poly_features', poly_features),
                               ('StandardScaler', std),
                               ('lin_reg', lin_reg)])
    # 通过流程进行训练
    polynomial_reg.fit(X, y)
    # 预测结果
    y_new_2 = polynomial_reg.predict(X_new)
    plt.plot(X_new, y_new_2, style, label='degree   ' + str(degree), linewidth=width)
plt.plot(X, y, 'b.')
plt.axis([-3, 3, -5, 10])
plt.legend()
plt.show()
# 可以看到把特征做的越复杂化,过拟合风险越高

# 正则化:对权重参数进行惩罚,让权重参数的分布较为平滑,有两种方式岭回归和lasso回归

# 岭回归:||y - Xw||^2_2 + alpha * ||w||^2_2,就是在损失值后面加上一个参数的平方,通过alpha来调节两边的权重

# 做些数据
np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 0.5 * X + np.random.randn(m, 1) / 1.5 + 1
X_new = np.linspace(0, 3, 100).reshape(100, 1)


def plot_model(model_class, polynomial, alphas, **model_kwargs):
    for alpha, line_style in zip(alphas, ('b-', 'g--', 'r:')):
        # 线性回归的模型采用传入的模型,岭回归或者lasso回归等等
        model = model_class(alpha, **model_kwargs)
        if polynomial:
            model = Pipeline([('poly_features', PolynomialFeatures(degree=10, include_bias=False)),
                              ('StandardScaler', StandardScaler()),
                              ('lin_reg', model)])
        model.fit(X, y)
        y_new_prediction = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_prediction, line_style, linewidth=lw, label='alpha = {}'.format(alpha))
    plt.plot(X, y, 'b.', linewidth=3)
    plt.legend()


plt.figure(figsize=(14, 6))
plt.subplot(121)
# 不做多项式处理的情况,alpha实际就是没有惩罚项
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100))
plt.subplot(122)
# 做多项式处理的情况
plot_model(Ridge, polynomial=True, alphas=(0, 10 ** -5, 1))
plt.show()
# 可以看到惩罚力度越大，alpha值越大的时候，得到的决策方程越平稳。

# lasso回归:(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1,后面加的是是参数的绝对值
plt.figure(figsize=(14, 6))
plt.subplot(121)
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1))
plt.subplot(122)
plot_model(Lasso, polynomial=True, alphas=(0, 10 ** -1, 1))
plt.show()
