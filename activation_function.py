# -*- encoding: shift_jis -*-
# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "ryo"
__date__ = "$2016/05/28 16:20:48$"


import numpy as np

#sigmoid�֐�
def sigmoid(v): return 1.0 / (1.0 + np.exp(-v))

#sigmmoid�֐��̔���
def dsigmoid(y): return (1.0 - y) * y

#1�����z��
#���͔z��ɑΉ����A�m����v�f�Ƃ����z���Ԃ�
def softmax(v):
    e = np.exp(v - np.max(v))
    return e / np.sum(e)

#softmax�̔���
def dsoftmax(y): return (1.0 - y) * y

#rectified linear unit�֐�
def relu(v): return v * (v > 0)

#rectified linear unit�֐��̔���
def drelu(y): return 1.0 * (0 < y)

'<EOF>'