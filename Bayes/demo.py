"""
贝叶斯实现垃圾邮件过滤
"""
import numpy as np
import re
import random


def text_parse(input_string):
    """
    切分出邮件中的单词,并转换成小写
    """
    list_of_tokens = re.split(r'\W+', input_string)
    return [tok.lower() for tok in list_of_tokens if len(list_of_tokens) > 2]


def create_vocab_list(doc_list):
    """
    创建去重后的词汇集合
    """
    vocab_set = set([])
    for document in doc_list:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_word_2_vec(vocab_list, input_set):
    """
    将邮件中的单词转换成词库的词向量,如果词库中的单词出现在邮件中就标1,否者标0
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
    return return_vec


def train_nb(train_mat, train_class):
    """
    进行词向量的模型训练 todo:没懂
    """
    # 样本数
    num_train_docs = len(train_mat)
    # 特征数
    num_words = len(train_mat[0])
    # 标志成垃圾邮件的样本比例
    p1 = sum(train_class) / float(num_train_docs)
    # 做了一个平滑处理
    p0_num = np.ones(num_words)
    # 拉普拉斯平滑
    p1_num = np.ones(num_words)
    p0_denom = 2
    # 通常情况下都是设置成类别个数
    p1_denom = 2

    for i in range(num_train_docs):
        if train_class[i] == 1:
            # 垃圾邮件
            p1_num += train_mat[i]
            p1_denom += sum(train_mat[i])
        else:
            p0_num += train_mat[i]
            p0_denom += sum(train_mat[i])
    p1_vec = np.log(p1_num / p1_denom)
    p0_vec = np.log(p0_num / p0_denom)
    return p0_vec, p1_vec, p1


def classify_nb(word_vec, p0_vec, p1_vec, p1_class):
    p1 = np.log(p1_class) + sum(word_vec * p1_vec)
    p0 = np.log(1.0 - p1_class) + sum(word_vec * p0_vec)
    if p0 > p1:
        return 0
    else:
        return 1


def spam():
    doc_list = []
    class_list = []
    for i in range(1, 26):
        word_list = text_parse(open('../data/email/spam/%d.txt' % i, 'r').read())
        # 将邮件中出现过的单词放到集合中,并把这些单词标记成垃圾邮件单词
        doc_list.append(word_list)
        # 1表示垃圾邮件
        class_list.append(1)

        word_list = text_parse(open('../data/email/ham/%d.txt' % i, 'r').read())
        # 将邮件中出现过的单词放到集合中,并把这些单词标记成正常邮件单词
        doc_list.append(word_list)
        # 0表示正常邮件
        class_list.append(0)

    vocab_list = create_vocab_list(doc_list)
    train_set = list(range(50))
    test_set = []
    for i in range(10):
        # 从训练集中随机选取10个作为测试集
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_index])
        del (train_set[rand_index])
    train_mat = []
    train_class = []
    for doc_index in train_set:
        # 相当于一行记录的特征值
        train_mat.append(set_of_word_2_vec(vocab_list, doc_list[doc_index]))
        # 相当于这行记录的标签
        train_class.append(class_list[doc_index])
    p0_vec, p1_vec, p1 = train_nb(np.array(train_mat), np.array(train_class))
    error_count = 0
    for doc_index in test_set:
        word_vec = set_of_word_2_vec(vocab_list, doc_list[doc_index])
        if classify_nb(np.array(word_vec), p0_vec, p1_vec, p1) != class_list[doc_index]:
            error_count += 1
    print('当前10个测试样本，错了：', error_count)


if __name__ == '__main__':
    spam()
