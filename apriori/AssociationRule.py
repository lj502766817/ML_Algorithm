"""
关联规则demo
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 做点数据
data = {'ID': [1, 2, 3, 4, 5, 6],
        'Onion': [1, 0, 0, 1, 1, 1],
        'Potato': [1, 1, 0, 1, 1, 1],
        'Burger': [1, 1, 0, 0, 1, 1],
        'Milk': [0, 1, 1, 1, 0, 1],
        'Beer': [0, 0, 1, 0, 1, 0]}
df = pd.DataFrame(data)
# 根据api的要求转成bool格式
df_bool = df[['ID']].join(df[['Onion', 'Potato', 'Burger', 'Milk', 'Beer']].astype(bool))
print(df_bool)
# 找出支持度大于50%的频繁项集
frequent_item_sets = apriori(df_bool[['Onion', 'Potato', 'Burger', 'Milk', 'Beer']], min_support=0.50,
                             use_colnames=True)
print(frequent_item_sets)
# 计算关联规则,阈值取1
rules = association_rules(frequent_item_sets, metric='lift', min_threshold=1)
# 取项集和置信度,提升度看看
print(rules[['antecedents', 'consequents', 'confidence', 'lift']])
# 取些有意义的数据
print(rules[(rules['lift'] > 1.125) & (rules['confidence'] > 0.8)])

# 实际场景中,数据可能不是这种0|1的格式,那就需要用one-hot编码来转换
# 做点数据
retail_shopping_basket = {'ID': [1, 2, 3, 4, 5, 6],
                          'Basket': [['Beer', 'Diaper', 'Pretzels', 'Chips', 'Aspirin'],
                                     ['Diaper', 'Beer', 'Chips', 'Lotion', 'Juice', 'BabyFood', 'Milk'],
                                     ['Soda', 'Chips', 'Milk'],
                                     ['Soup', 'Beer', 'Diaper', 'Milk', 'IceCream'],
                                     ['Soda', 'Coffee', 'Milk', 'Bread'],
                                     ['Beer', 'Chips']
                                     ]
                          }
retail = pd.DataFrame(retail_shopping_basket)
retail = retail[['ID', 'Basket']]
pd.options.display.max_colwidth = 100
# 取出id列
retail_id = retail.drop('Basket', 1)
# 用','来连接数组
retail_Basket = retail.Basket.str.join(',')
# 根据','来做one-hot编码,并转成bool格式
retail_Basket = retail_Basket.str.get_dummies(',').astype(bool)
retail = retail_id.join(retail_Basket)
print(retail)
# 查看规则
frequent_item_sets_2 = apriori(retail.drop('ID', 1), use_colnames=True)
print(frequent_item_sets_2)
rules2 = association_rules(frequent_item_sets_2, metric='lift')
print(rules2)

# 拿个真实的数据集玩玩
# https://grouplens.org/datasets/movielens/
movies = pd.read_csv('../data/movies.csv')
movies_ohe = movies.drop('genres', 1).join(movies.genres.str.get_dummies().astype(bool))
pd.options.display.max_columns = 100
print(movies_ohe)
# 把movieId和title设置成df的index
movies_ohe.set_index(['movieId', 'title'], inplace=True)
# 实际情况下的支持度是个绝对的概念,通常是很小的
frequent_item_sets_movies = apriori(movies_ohe, use_colnames=True, min_support=0.025)
# 由于提升度是一个相对的概念,所以实际值是偏高的
rules_movies = association_rules(frequent_item_sets_movies, metric='lift', min_threshold=1.25)
# 查看提升度高的项集
print(rules_movies[(rules_movies.lift > 4)].sort_values(by=['lift'], ascending=False))
