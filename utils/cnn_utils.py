#!/usr/bin/env python
# encoding: utf-8
'''
# Author:        小.今天也没有被富婆爱护.锅
# File:          cnn_utils.py
# Date:          2020/7/26
# Description:   一个CNN的utils(废话
'''
import sys
import numpy as np
from struct import unpack


def read_img(file):
    with open(file, 'rb') as fr:
        magic, nums, rows, cols = unpack('>4I', fr.read(16))
        img = np.fromfile(fr, dtype=np.uint8).reshape(nums, rows, cols, 1)
    return img


def read_label(file):
    with open(file, 'rb') as fr:
        magic, num = unpack('>2I', fr.read(8))
        label = np.fromfile(fr, dtype=np.uint8)
    return label


def nomalize_image(image):
    img =image.astype(np.float32)/255
    return img


def one_hot_label(label, label_size=10):    # mnist手写数字的label有10种
    lab = np.zeros((label.size, label_size))
    for i, row in enumerate(lab):
        lab[label[i]] = i
    return lab


def add_bias(conv, bias):
    assert conv.shape[-1] == bias.shape[0]
    for i in range(bias.shape[0]):
        conv[:, :, i] += bias[i, 0]
    return conv


def rot180(conv_filters):
    rot180_filters = np.zeros((conv_filters.shape))
    for filter_num in range(conv_filters.shape[0]):
        for img_ch in range(conv_filters.shape[-1]):
            rot180_filters[filter_num, :, :, img_ch] = np.flipud(np.fliplr(conv_filters[filter_num, :, :, img_ch]))
    return rot180_filters


def relu(feature):
    return feature*(feature>0)


def d_relu(feature):
    return 1* (feature>0)


def softmax(z):
    tmp = np.max(z)
    z -= tmp    # trick：元素值缩放，避免溢出
    return np.exp(z) / np.sum(z)


def padding(image, p):
    '''
    图片零填充
    '''
    if len(image.shape) == 4:
        image_padding = np.zeros((image.shape[0], image.shape[1]+2*p, image.shape[2]+2*p, image.shape[3]))
        image_padding[:, p:image.shape[1]+p, p:image.shape[2]+p, :] = image
    elif len(image.shape) == 3:
        image_padding = np.zeros((image[0]+2*p, image.shape[1]+2*p, image.shape[2]))
        image_padding[p:image.shape[0]+p, p:image.shape[1]+p, :] = image
    else:
        print("WARNING | wrong image dimensions!")
        sys.exit()
    return image_padding
