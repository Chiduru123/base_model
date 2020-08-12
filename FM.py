#!/usr/bin/env python
# encoding: utf-8
'''
# Author:        Guo Qi
# File:          FM.py
# Date:          2020/7/16
# Description:   FM(因子分解机)模型算法：稀疏数据下的特征二阶组合问题（个性化特征）
#               1、应用矩阵分解思想，引入隐向量构造FM模型方程
#               2、目标函数（损失函数复合FM模型方程）的最优问题：链式求偏导
#               3、SGD优化目标函数
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def sigmoid(x):
    '''二分类输出非线性映射'''
    return 1 / (1 + np.exp(-x))


def mse(y, y_hat):
    return ((y_hat - y) ** 2).mean(axis=1)


def df_mse(y, y_hat):
    '''均方差损失函数的偏导'''
    return np.sum(y_hat - y, axis=1, keepdims=True)


def logit(y, y_hat):
    '''
    计算logit损失函数
    '''
    return np.log(1 + np.exp(-y * y_hat))


def df_logit(y, y_hat):
    '''计算logit损失函数的外层偏导(不含y_hat的一阶偏导)'''
    return sigmoid(-y * y_hat) * (-y)


def FM(Xi, w0, W, V):
    '''FM的模型方程：LR线性组合 + 交叉项组合 = 1阶特征组合 + 2阶特征组合'''
    # 样本Xi的特征分量xi和xj的2阶交叉项组合系数wij = xi和xj对应的隐向量Vi和Vj的内积
    interaction = np.sum((Xi.dot(V)) ** 2 - (Xi ** 2).dot(V ** 2))
    y_hat = w0 + Xi.dot(W) + interaction / 2
    return y_hat[0]


def FM_SGD(X, y, k=2, alpha=0.01, iter=50):
    '''SGD更新FM模型的参数列表：[w0, W, V]'''
    m, n = np.shape(X)
    w0, W = 0, np.zeros((n, 1))
    V = np.random.normal(loc=0, scale=1, size=(n, k))  # 按标准正态分布随机初始化隐向量矩阵V=(n, k)~N(0, 1)，其中Vj是第j维特征的隐向量
    all_FM_params = []  # [w0, W, V]

    for it in range(iter):
        total_loss = 0
        for i in range(m):
            y_hat = FM(Xi=X[i], w0=w0, W=W, V=V)
            total_loss += logit(y=y[i], y_hat=y_hat)

            dloss = df_logit(y=y[i], y_hat=y_hat)  # logit损失函数的外层偏导

            dloss_w0 = dloss * 1
            w0 = w0 - alpha * dloss_w0

            for j in range(n):  # 遍历n维向量X[i]
                if X[i, j] != 0:
                    dloss_Wj = dloss * X[i, j]  # l(y, y_hat)中y_hat展开y_hat，求关于W[j]的内层偏导

                    W[j] = W[j] - alpha * dloss_Wj  # 梯度下降更新W[j]

                    for f in range(k):  # 遍历k维隐向量Vj
                        # l(y, y_hat)中y_hat展开V[j, f]，求关于V[j, f]的内层偏导
                        dloss_Vjf = dloss * (X[i, j] * (X[i].dot(V[:, f])) - V[j, f] * X[i, j] ** 2)

                        V[j, f] = V[j, f] - alpha * dloss_Vjf  # 梯度下降更新V[j, f]
        if it % 5 == 0:
            print('FM第{}次迭代，当前损失值为：{:.4f}'.format(it + 1, total_loss / m))

        all_FM_params.append([w0, W, V])  # 保存当前迭代下FM的参数列表:[w0, W, V]
    return all_FM_params


def FM_predict(X, w0, W, V):
    '''FM模型预测测试集分类结果'''
    predicts, threshold = [], 0.5  # sigmoid阈值
    for i in range(X.shape[0]):
        y_hat = FM(Xi=X[i], w0=w0, W=W, V=V)
        predicts.append(-1 if sigmoid(y_hat) < threshold else 1)  # 分类结果非线性映射
    return np.array(predicts)


def draw_research(all_FM_params, X_train, y_train, X_test, y_test):
    '''FM在不同迭代次数下的参数列表中，训练集的损失值和测试集的准确率变化'''
    all_total_loss, all_total_accuracy = [], []
    for w0, W, V in all_FM_params:
        total_loss = 0
        for i in range(X_train.shape[0]):
            total_loss += logit(y=y_train[i], y_hat=FM(Xi=X_train[i], w0=w0, W=W, V=V))
        all_total_loss.append(total_loss / X_train.shape[0])
        all_total_accuracy.append(accuracy_score(y_test, FM_predict(X=X_test, w0=w0, W=W, V=V)))
    plt.plot(np.arange(len(all_FM_params)), all_total_loss, color='#FF4040', label='loss on train_data')
    plt.plot(np.arange(len(all_FM_params)), all_total_accuracy, color='#4876FF', label='acc on test_data')
    plt.xlabel('SGD iterations')
    plt.title('FM')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(123)
    df = pd.read_csv('data/fm_xg.csv')

    df['Class'] = df['Class'].map({0: -1, 1: 1})  # 标签列从[0, 1]离散到[-1, 1]
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1].values, df.iloc[:, -1].values, test_size=0.3, random_state=123)
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)
    print('Data preprocessing done. \nTrain data shape: {}, and test data shape: {}'.format(X_train.shape, X_test.shape))

    '''*****************FM预测模型*****************'''
    all_FM_params = FM_SGD(X=X_train, y=y_train, k=2, alpha=0.008, iter=300)

    w0, W, V = all_FM_params[-1]
    predicts = FM_predict(X=X_test, w0=w0, W=W, V=V)
    print('FM在测试集的分类准确率为: {:.2%}'.format(accuracy_score(y_test, predicts)))
    draw_research(all_FM_params=all_FM_params, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
