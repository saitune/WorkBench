# -*- encoding: shift_jis -*-
# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "ryo"
__date__ = "$2016/05/28 16:20:48$"


import numpy as np

#sigmoid関数
def sigmoid(v): return 1.0 / (1.0 + np.exp(-v))

#sigmmoid関数の微分
def dsigmoid(y): return (1.0 - y) * y

#1次元配列
#入力配列に対応し、確率を要素とした配列を返す
def softmax(v):
    e = np.exp(v - np.max(v))
    return e / np.sum(e)

#softmaxの微分
def dsoftmax(y): return (1.0 - y) * y

#rectified linear unit関数
def relu(v): return v * (v > 0)

#rectified linear unit関数の微分
def drelu(y): return 1.0 * (0 < y)

'<EOF>'