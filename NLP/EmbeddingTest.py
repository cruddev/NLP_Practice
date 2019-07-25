# -*- coding: utf-8 -*-

# @Time    : 2019-07-24 15:06

# @Author  : yuxi

# @Project : NLP

# @FileName: EmbeddingTest.py

# @Software: PyCharm

from bert_serving.client import BertClient
import numpy as np
from termcolor import colored
bc = BertClient(ip='localhost')
topk = 2
questions_list_storage = ['忘记密码怎么办', '明天天气怎么样', '明天会下雨吗', '有什么好看的电影', '今天会下雨吗', '昨天有下雨吗',
                          '附近有哪些电影院', '有什么有趣的游戏', '附近最近的游泳池在哪里', '最近有什么好看的小说', '如何注册账号',
                          '附近哪里有网球场', '如何开通VPN专线']
doc_vecs = bc.encode(questions_list_storage)
def give_answer(input_string, datasets, datasets_vec):
    questions_test = input_string
    query_vec = bc.encode([questions_test])[0]
    score = np.sum(query_vec * datasets_vec, axis=1) / np.linalg.norm(datasets_vec, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    print('top %d questions similar to "%s"' % (topk, colored(questions_test, 'green')))
    for idx in topk_idx:
        print('> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(datasets[idx], 'yellow')))
# 基于匹配的方法无法得到这样的回答
give_answer('明天出门需要带伞吗', questions_list_storage, doc_vecs)
give_answer('玩什么', questions_list_storage, doc_vecs)
give_answer('我无法登录了', questions_list_storage, doc_vecs)
give_answer('怎么开通VPN', questions_list_storage, doc_vecs)

knowledge_graph_storage = ['蔡徐坤_粉丝', '蔡徐坤_出道时间', '蔡徐坤_生日', '蔡徐坤_作品', '周杰伦_出道时间']
doc_vecs = bc.encode(knowledge_graph_storage)
give_answer('蔡徐坤唱过哪些歌', knowledge_graph_storage, doc_vecs)
give_answer('蔡徐坤什么时候成名的', knowledge_graph_storage, doc_vecs)
give_answer('谁喜欢蔡徐坤', knowledge_graph_storage, doc_vecs)

# top 2 questions similar to "明天出门需要带伞吗"
# > 16.8	明天会下雨吗
# > 16.8	今天会下雨吗
# top 2 questions similar to "玩什么"
# > 16.7	有什么有趣的游戏
# > 15.6	有什么好看的电影
# top 2 questions similar to "我无法登录了"
# > 16.1	忘记密码怎么办
# > 15.7	如何注册账号


# top 2 questions similar to "蔡徐坤唱过哪些歌"
# > 17.5	蔡徐坤_出道时间
# > 17.1	蔡徐坤_粉丝
# top 2 questions similar to "蔡徐坤什么时候成名的"
# > 17.4	蔡徐坤_出道时间
# > 17.0	周杰伦_出道时间
# top 2 questions similar to "谁喜欢蔡徐坤"
# > 16.9	蔡徐坤_粉丝
# > 16.7	蔡徐坤_生日