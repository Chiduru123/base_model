#!/usr/bin/env python
# encoding: utf-8
'''
# Author:        Guo Qi
# File:          NN.py
# Date:          2020/7/14
# Description:   手写logistics模型(input-> relu -> mlp -> sigmoid)， logistics激活函数+交叉熵损失函数
'''

import numpy as np
from utils import testCases
from utils.dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from utils import lr_utils
from pprint import pprint
np.random.seed(1)


def initial_params_deep(layer_dims):
    '''
    :desc 初始化多层神经元参数
    :param layer_dims: [l0, l1, l2, l3, ……, ln ]
    :return:
        params {
            Wl - (nl, nl-1)
            bl - (nl, 1)
        }
    '''
    np.random.seed(3)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        # tips: np.random.randn 是按标准正态分布生成shape=(m.n)的随机数矩阵，不能与random.random((n, x))生成n个0-x范围内的随机数功能混淆
        params["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layers_dims[l-1])
        params['b'+str(l)] = np.random.randn(layer_dims[l], 1)
    return params


def linear_forward(A, W, b):
    '''
    前向传播的线性部分
    :param A: pre-layer output or cur-layer input, shape=(nl-1, 1)
    :param W: shape=(nl, nl-1)
    :param b: shape=(nl, 1)
    :return:
        Z - Wx+b
        cache - {
                'W': W,
                'b': b
                }
    '''
    Z = np.dot(W, A) + b
    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)
    return Z, cache


def linear_active_forward(A_prev, W, b, activation):
    '''
    前向传播 linear + active
    :param A_prev: 前一层的输出
    :param W: 当前层的权重矩阵, shape=nl, nl-1
    :param b: 当前层的权重偏置, shape=nl, 1
    :param activation: 激活函数 sigmoid/ relu
    :return:
        A - 当前层的输出；
        cache - 参数字典  linear_cache + sigmoid_cache
    '''
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)    # cache 是Z
    elif activation == 'relu':
        A, activation_cache = relu(Z)

    return A, (linear_cache, activation_cache)


def L_model_forward(X, params):
    '''
    mlp 前向传播， relu*(L-1) + sigmoid
    :param X:       shape= l0, m
    :param params:  参数字典，key是每一层的W， b
    :return:
        AL - shap= nl, 1
        caches - 每层的linear, active cache
    '''
    caches = []
    A = X
    L = len(params)//2

    for l in range(1, L):
        print('当前层：{}, A.shape={}'.format(l, A.shape))
        A, cache = linear_active_forward(A, params['W'+str(l)], params['b'+str(l)], activation='relu')
        caches.append(cache)   # Z

    AL, cache = linear_active_forward(A, params["W"+str(L)], params['b'+str(L)], activation='sigmoid')
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])
    return AL, caches


def compute_loss(AL, Y):
    '''
    交叉熵计算损失函数
    :param AL:  与标签预测相对应的概率向量， shape=1, m
    :param Y:   0-1 标签， shape=1, m
    :return:
        cost - 交叉熵成本
    '''
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    cost = np.squeeze(cost)

    assert cost.shape == ()
    return cost


def linear_backward(dZ, cache):
    '''
    线性后向传播
    :param dZ:      当前第l层的成本梯度
    :param cache:   前向传播时的(A_prev, W, b)
    :return:
        dA_prev - 当前层的inputA的梯度
        dW      - 当前层的W梯度
        db      - 当前层的b梯度
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1]              # 样本数
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert dW.shape[0] == db.shape[0]
    assert dW.shape==W.shape
    assert db.shape==b.shape

    return dA_prev, dW, db


def linear_active_backward(dA, cache, active='relu'):
    '''
    两层神经网络的linear+activation 后向传播
    :param dA:      当前层的A梯度
    :param cache:   前向传播的linear(A, w, b)和 sigmoid_cache(Z)
    :param active:
    :return:
    '''
    linear_cache, active_cache = cache

    if active == 'relu':
        dZ = relu_backward(dA, cache=active_cache)
    elif active == 'sigmoid':
        dZ = sigmoid_backward(dA, cache=active_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    '''
    多层神经网络的 linear+activation 一次后向传播
    :param AL:      正向传播预测结果
    :param Y:       真实结果
    :param caches:   [linear+active1, 2, ……]
    :return:
        grads - 每层的梯度字典, dWi, dbi, dA
    '''
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)      # AL对应交叉熵损失函数的求导
    grads = {}

    cur_cache = caches[L-1]
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db' + str(L)] = linear_active_backward(dA=dAL, cache=cur_cache, active='sigmoid')

    for l in reversed(range(L-1)):      # l-2, l-3, ……, 1, 0
        cur_cache = caches[l]
        grads['dA' + str(l+1)], grads['dW'+str(l+1)], grads['db' + str(l+1)] = \
                linear_active_backward(dA=grads['dA'+str(l+2)], cache=cur_cache, active='relu')

    return grads


def update_params(params, grads, learning_rate):
    '''
    梯度更新， w = w - learning_rate * dw
    :param params:          {w, b}
    :param grads:           dw, db
    :param learning_rate:   学习率/ 步长
    :return:
        updated params
    '''

    L = len(grads) // 3
    for l in range(L):
        params['W' + str(l+1)] -= learning_rate * grads['dW'+str(l+1)]
        params['b' + str(l+1)] -= learning_rate * grads['db'+str(l+1)]

    return params


def plot(x):
    import matplotlib.pyplot as plt
    plt.plot(np.squeeze(x))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.show()


def mlp(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    '''
    多层全连接神经网络，(L-1)relu + sigmoid
    :param X:               输入数据,shape=n_x, m
    :param Y:               输出数据,shape=1,n_y
    :param layer_dims:      [n_1, n_2, ……, n_y]
    :param learning_rate:   学习率
    :param num_interations: 迭代次数
    :return:
        params -  参数字典
    '''
    np.random.seed(1)
    costs = []
    parameters = initial_params_deep(layer_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)     # 一次前向
        cost = compute_loss(AL, Y)
        grads = L_model_backward(AL, Y, caches=caches)       # 梯度
        parameters = update_params(parameters, grads, learning_rate)
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print('第{}次迭代损失为: {}'.format(i, cost))

    # if isPlot:
    #     plot(costs)

    return parameters


def predict(X, y, parameters):
    """
    该函数用于预测L层神经网络的结果，当然也包含两层
    参数：
         X - 测试集
         y - 标签
         parameters - 训练模型的参数
    返回：
        p - 给定数据集X的预测
    """

    m = X.shape[1]
    p = np.zeros((1, m))

    # 根据参数前向传播
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("准确度为: " + str(float(np.sum((p == y)) / m)))
    return p


train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

layers_dims = [12288, 20, 7, 5, 1]      # 5-layer model
parameters = mlp(train_x, train_y, layers_dims, learning_rate=0.0075, num_iterations=2500, print_cost=True, isPlot=True)

print('训练集-- ')
pred_train = predict(train_x, train_y, parameters)      # 训练集
print('测试集-- ')
pred_test = predict(test_x, test_y, parameters)         # 测试集
