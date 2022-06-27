# -*- coding: UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator


# 做些数据
def create_data_set():
    data_set = [[0, 0, 0, 0, 'no'],
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return data_set, labels


def create_tree(dataset, labels, feature_labels):
    # 获得样本的所有标签值
    class_list = [example[-1] for example in dataset]
    # 如果只有一种标签值,那么就返回这个标签值
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果特征值全部被使用完了(每把一个特征用来做决策,就把这个特征从样本里移除),返回数量最多的那个标签
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    # 选出最合适的用来切分的特征的索引
    best_feature = choose_best_feature_to_split(dataset)
    # 具体的特征的名称
    best_feat_label = labels[best_feature]
    # 把选出来的最佳特征放到特征集合里
    feature_labels.append(best_feat_label)
    # 构建当前特征的决策树
    my_tree = {best_feat_label: {}}
    # 从可选的特征中去掉已经选出来的特征
    del labels[best_feature]
    # 得到最佳特征下的各种值(相当于树的分叉),然后继续对每个分支迭代构建树
    feature_values = [example[best_feature] for example in dataset]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(
            # 分割样本集,把最佳特征下,值等于这个值的样本分离出来
            split_data_set(dataset, best_feature, value)
            , sub_labels, feature_labels)
    return my_tree


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            # 先初始化数量为0
            class_count[vote] = 0
        # 对每种标签值做累加
        class_count[vote] += 1
    # 按大小排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回最大的那个
    return sorted_class_count[0][0]


def choose_best_feature_to_split(dataset):
    # 特征的数量
    num_features = len(dataset[0]) - 1
    # 计算样本集基础的熵值
    base_entropy = calc_shannon_ent(dataset)
    # 最好的信息增益值
    best_info_gain = 0
    # 最好的信息增益值对应的特征值索引
    best_feature = -1
    # 针对每个特征迭代
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)
        new_entropy = 0
        # 对每个特征下的每种值做迭代
        for val in unique_vals:
            # 根据特征和值将样本做切割,得到的是某个特征下,特征是某个值的样本
            sub_data_set = split_data_set(dataset, i, val)
            # 这个特征下根据每类值区分出来的样本的占比
            prob = len(sub_data_set) / float(len(dataset))
            # 得到这个特征下不同各类值的加权集合熵
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        # 这个特征下的信息增益
        info_gain = base_entropy - new_entropy
        # 如果是最大增益,就更新相关值
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def split_data_set(dataset, axis, val):
    ret_data_set = []
    for featVec in dataset:
        if featVec[axis] == val:
            reduced_feat_vec = featVec[:axis]
            reduced_feat_vec.extend(featVec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def calc_shannon_ent(dataset):
    # 总样本数量
    numexamples = len(dataset)
    label_counts = {}
    # 每类标签的样本舒朗
    for featVec in dataset:
        currentlabel = featVec[-1]
        if currentlabel not in label_counts.keys():
            label_counts[currentlabel] = 0
        label_counts[currentlabel] += 1

    shannon_ent = 0
    for key in label_counts:
        prop = float(label_counts[key]) / numexamples
        # 总的熵值 = (- 概率 * log(概率))做累加
        shannon_ent -= prop * log(prop, 2)
    return shannon_ent


def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args, FontProperties=font)


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(my_tree, parent_pt, node_txt):
    decision_node = dict(boxstyle="sawtooth", fc="0.8")
    leaf_node = dict(boxstyle="round4", fc="0.8")
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = next(iter(my_tree))
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d


def create_plot(in_tree):
    # 创建fig
    fig = plt.figure(1, facecolor='white')
    # 清空fig
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 去掉x、y轴
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 获取决策树叶结点数目
    plot_tree.total_w = float(get_num_leafs(in_tree))
    # 获取决策树层数
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    # x偏移
    plot_tree.y_off = 1.0
    # 绘制决策树
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    dataset, labels = create_data_set()
    featLabels = []
    myTree = create_tree(dataset, labels, featLabels)
    create_plot(myTree)
