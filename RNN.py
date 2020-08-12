#!/usr/bin/env python
# encoding: utf-8
'''
# Author:        Guo Qi
# File:          RNN.py
# Date:          2020/7/16
# Description:   🍣RNN
'''

import numpy as np
from utils import rnn_utils
from utils import cllm_utils
import time


def cell_forward(xt, a_prev, params):
    '''
    神经元上的单步前向传播, tanh + softmax
    :param xt:       shape = n_a, m
    :param a_prev:   shape = n_a, m
    :param params:   dict of params { Waa, Wax, Way, ba, by }
    :return:
        a_next - shape=n_a, m
        y -  shape = n_y, m
        cache - 反向工具 (a_next, a_prev, xt, params)
    '''
    Wax, Waa, Way, ba, by = params['Wax'], params['Waa'], params['Wya'], params['ba'], params['by']

    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    y = rnn_utils.softmax(np.dot(Way, a_next) + by)

    cache = (a_next, a_prev, xt, params)
    return a_next, y, cache


def cell_backward(da_next, cache):
    '''
    单神经元上的单层后向传播
    '''
    a_next, a_prev, xt, param = cache
    Waa, Wax, Wya, ba, by = param['Waa'], param['Wax'], param['Wya'], param['ba'], param['by']

    dtanh = (1 - np.square(a_next)) * da_next

    dWaa = np.dot(dtanh, a_prev.T)
    dWax = np.dot(dtanh, xt.T)
    dba = np.sum(dtanh, axis=1, keepdims=True)
    dxt = np.dot(Wax.T, dtanh)
    da_prev = np.dot(Waa.T, dtanh)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients


def rnn_forward(x, a0, params):
    '''
    RNN网络的前向传播
    '''
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = params['Wya'].shape

    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0

    for t in range(T_x):
        a_next, yt, cache = cell_forward(x[:, :, t], a_prev=a_next, params=params)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt
        caches.append(cache)

    caches = (caches, x)
    return a, y_pred, caches


def rnn_backward(da, caches):
    '''
    rnn反向传播
    :param da:
    :param cache:
    :return:
    '''
    caches, x = caches
    a1, a0, xt, param = caches[0]

    n_a, m, T_x = da.shape
    n_x, m = xt.shape

    dx = np.zeros([n_x, m, T_x])
    dWax = np.zeros([n_a, n_x])
    dWaa = np.zeros([n_a, n_a])
    dba = np.zeros([n_a, 1])
    da_prev = np.zeros([n_a, m])

    for t in reversed(range(T_x)):
        grads = cell_backward(da_next=da[:, :, t] + da_prev, cache=caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = grads["dxt"], grads["da_prev"], grads["dWax"], grads["dWaa"], grads["dba"]
        dx[:, :, t] = dxt

        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    da0 = da_prevt

    grads = {
        'dWaa': dWaa,
        'dWax': dWax,
        'dba': dba,
        'dx':  dx,
        'da0': da0
    }
    return grads


def clip(grads, maxValue):
    '''
    梯度修剪, 避免梯度爆炸
    '''
    dWaa, dWax, dWya, db, dby = grads['dWaa'], grads['dWax'], grads['dWya'], grads['db'], grads['dby']
    for grad in [dWaa, dWax, dWya, db, dby]:
        np.clip(grad, -maxValue, maxValue, out=grad)
    grads['dWaa'], grads['dWax'], grads['dWya'], grads['db'], grads['dby'] = dWaa, dWax, dWya, db, dby
    return grads


def sample(parameters, char_to_is, seed):
    """
    根据RNN输出的概率分布序列对字符序列进行采样
    参数：
        parameters -- 包含了Waa, Wax, Wya, by, b的字典
        char_to_ix -- 字符映射到索引的字典
        seed -- 随机种子
    返回：
        indices -- 包含采样字符索引的长度为n的列表。
    """

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    indices = []

    idx = -1

    # 循环遍历时间步骤t。在每个时间步中，从概率分布中抽取一个字符，并将其索引附加到“indices”
    counter = 0
    newline_character = char_to_ix["\n"]

    while (idx != newline_character and counter < 50):
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = cllm_utils.softmax(z)

        np.random.seed(counter + seed)

        # 步骤3：从概率分布y中抽取词汇表中字符的索引
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        # 添加到索引中
        indices.append(idx)

        # 步骤4:将输入字符重写为与采样索引对应的字符。
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a

        seed += 1
        counter += 1

    if (counter == 50):
        indices.append(char_to_ix["\n"])

    return indices


def optimize(X, Y, a_prev, params, learning_rate=0.01):
    '''
    模型单步优化
    '''
    loss, cache = cllm_utils.rnn_forward(X, Y, a_prev, params)
    grads, a = cllm_utils.rnn_backward(X, Y, params, cache)

    grads = clip(grads, 5)
    params = cllm_utils.update_parameters(params, grads, learning_rate)
    return loss, grads, a[len(X)-1]


def model(data, ix_to_char, char_to_ix, num_iterations=3500, n_a=50, dino_names=7, vocab_size=27):
    n_x, n_y = vocab_size, vocab_size
    parameters = cllm_utils.initialize_parameters(n_a, n_x, n_y)
    loss = cllm_utils.get_initial_loss(vocab_size, dino_names)

    with open("./data/dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # 打乱全部的恐龙名称
    np.random.seed(0)
    np.random.shuffle(examples)

    a_prev = np.zeros((n_a, 1))
    for j in range(num_iterations):
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix['\n']]

        cur_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        loss = cllm_utils.smooth(loss, cur_loss)  # 平滑损失来加速训练 | 梯度变化相对更小，训练时不容易跑飞

        # 每2000次迭代，通过sample()生成“\n”字符，检查模型是否学习正确
        if j % 200 == 0:
            print("第" + str(j + 1) + "次迭代，损失值为：" + str(loss))

            seed = 0
            for name in range(dino_names):
                # 采样
                sampled_indices = sample(parameters, char_to_ix, seed)
                # cllm_utils.print_sample(sampled_indices, ix_to_char)

                seed += 1
    return parameters


def get_data():
    data = open('./data/dinos.txt', 'r').read()
    data = data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print("Load data done, {} chars and {} unique chars".format(data_size, vocab_size))
    return data, chars


data, chars = get_data()
char_to_ix = {ch:i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i:ch for i, ch in enumerate(sorted(chars))}


start_time = time.perf_counter()
parameters = model(data, ix_to_char, char_to_ix, num_iterations=5000)
end_time = time.perf_counter()
minium = end_time - start_time

print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")
