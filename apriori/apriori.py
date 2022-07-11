"""
apriori算法实现
"""


def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(data_set):
    """
    创建只含单个项的不可变项集
    """
    C1 = []
    for transaction in data_set:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scan_D(D, CK, min_support):
    ss_cnt = {}
    # 遍历每个事物
    for tid in D:
        # 遍历项集
        for can in CK:
            # 如果项集是实物的子集,就把这个项放到map里计数
            if can.issubset(tid):
                if not can in ss_cnt:
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    # 事物数量
    num_items = float(len(list(D)))
    ret_list = []
    support_data = {}
    # 遍历每个项集
    for key in ss_cnt:
        # 计算支持度
        support = ss_cnt[key] / num_items
        # 支持度达标放入返回的项集里,并把对应支持度放到支持度映射表里
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data


def apriori_gen(LK, k):
    """
    根据初始项集LK,生成每个项集都有k项的超集,
    生成规则是将前k-1项元素相同的项集合并
    """
    ret_list = []
    len_LK = len(LK)
    for i in range(len_LK):
        for j in range(i + 1, len_LK):
            L1 = list(LK[i])[:k - 2]
            L2 = list(LK[j])[:k - 2]
            if L1 == L2:
                ret_list.append(LK[i] | LK[j])
    return ret_list


def apriori(data_set, min_support=0.5):
    """
    根据最小支持度得到全部的项集以及对应支持度
    """
    # 找到全部单独项的项集
    C1 = create_c1(data_set)
    # 找出支持度符合条件的单独项集以及对应支持度
    L1, support_data = scan_D(dataSet, C1, min_support)
    L = [L1]
    k = 2
    # 循环找出2~n项的项集,并计算对应的支持率,第一轮是根据单元素项集生成两个元素的项集,
    # 第二轮是根据两个元素的项集生成三个元素的项集,以此类推
    while len(L[k - 2]) > 0:
        CK = apriori_gen(L[k - 2], k)
        LK, supk = scan_D(dataSet, CK, min_support)
        support_data.update(supk)
        L.append(LK)
        k += 1
    return L, support_data


def generate_rules(L, support_data, min_conf=0.6):
    """
    计算各个多元素项集的置信度
    """
    rule_list = []
    # index从1开始,因为index为0的项集都是单元素的项集,没有置信度
    for i in range(1, len(L)):
        for freq_set in L[i]:
            H1 = [frozenset([item]) for item in freq_set]
            rules_from_conseq(freq_set, H1, support_data, rule_list, min_conf)
    return rule_list


def rules_from_conseq(freq_set, H, support_data, rule_list, min_conf=0.6):
    """
    计算freq_set项集拆开后的项H的置信度
    """
    m = len(H[0])
    while len(freq_set) > m:
        H = cal_conf(freq_set, H, support_data, rule_list, min_conf)
        if len(H) > 1:
            apriori_gen(H, m + 1)
            m += 1
        else:
            break


def cal_conf(freq_set, H, support_data, rule_list, min_conf=0.6):
    prunedh = []
    for conseq in H:
        conf = support_data[freq_set] / support_data[freq_set - conseq]
        if conf >= min_conf:
            print(freq_set - conseq, '-->', conseq, 'conf:', conf)
            rule_list.append((freq_set - conseq, conseq, conf))
            prunedh.append(conseq)
    return prunedh


if __name__ == '__main__':
    dataSet = load_data_set()
    L, support = apriori(dataSet)
    i = 0
    for freq in L:
        print('项数', i + 1, ':', freq)
        i += 1
    rules = generate_rules(L, support, min_conf=0.5)
    print(rules)
