#-*- encoding: shift_jis -*-
# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "ryo"
__date__ = "$2016/05/29 6:05:24$"

import activation_function as af
import numpy as np
import sqlite3
from PIL import Image

#MNIST�̉摜�f�[�^�̃T�C�Y
MNIST_SIZE = 28

#MNIST��train�f�[�^�̐�
NUM_OF_TRAIN_DATA = [5922,6741,5957,6130,5841,5420,5917,6264,5850,5948]

#��ݍ��݃j���[�����l�b�g���[�N�N���X
class CNN:
    
    def __init__(self, num_of_conv_filter, num_of_hidden_node, num_of_output):
        #convlayer0�̃t�B���^����
        #conv0_filter[3][2][1] -> 0���琔����3���ڂ̃t�B���^��y = 2,x = 1�̏d��
        self.conv0_filter = self.make_filter(num_of_conv_filter[0:1])
        
        #convlayer0�̃o�C�A�X�d�ݐ���
        #conv0_bweight[2] -> 0���琔����2���ڂ̃t�B���^�ɑΉ�����o�C�A�X�d��
        self.conv0_bweight = np.zeros([num_of_conv_filter[0]])
        
        #convlayer1�̃t�B���^����
        #conv1_filter[4][3][2][1] -> 0���琔����4�Ԗڂ̃t�B���^�Q��3���ڂ̃t�B���^��y = 2,x = 1�̏d��
        self.conv1_filter = self.make_filter(num_of_conv_filter[0:2])
        
        #convlayer1�̃o�C�A�X�d�ݐ���
        #conv1_bweight[3] -> 0���琔����3�Ԗڂ̃t�B���^�Q�ɑΉ�����o�C�A�X�d��
        self.conv1_bweight = np.zeros([num_of_conv_filter[1]])
        
        
        len_of_one_side = MNIST_SIZE + 2
        
        #��ݍ��݁A�v�[�����O�̌�̈�ӂ̒��������߂�
        for i in range(num_of_conv_filter.size): len_of_one_side = (len_of_one_side - 2) / 2
        
        #NN�w�̓��͐�
        num_of_nn_input = (len_of_one_side ** 2) * num_of_conv_filter[-1]
        
        #NN�w�̓��o�͑w���܂߂��e�w�̃m�[�h���̃��X�g�𐶐�
        self.num_of_nn_node = [int(num_of_nn_input)] + list(num_of_hidden_node) + [num_of_output]
        
        # NN�w�̏d�݂̐���
        # nn_weight[2][3][4] -> NN�w��0���琔����2�w�ڂ�3�Ԗڂ̃m�[�h���玟�̑w��4�Ԗڂ̃m�[�h���Ȃ��d��
        self.nn_weight = self.make_weight()
        
        #�������֐��Z�b�g
        self.hidden_act_func = af.sigmoid
        self.output_act_func = af.softmax
        self.hidden_differential_func = af.dsigmoid
        self.output_differential_func = af.dsoftmax
        
    #��ݍ��ݑw�̃t�B���^�𐶐��A�ċA�𗘗p
    def make_filter(self, num_of_conv_filter, lv = 0):
        if lv == len(num_of_conv_filter) - 1: return [np.zeros([3,3]) for i in range(num_of_conv_filter[0])]
        else: return [self.make_filter(num_of_conv_filter, lv + 1) for i in range(num_of_conv_filter[len(num_of_conv_filter) - (lv + 1)])]
    
    #�S�����w�̏d�݂𐶐�
    def make_weight(self):
        return [np.zeros([self.num_of_nn_node[i] + 1,self.num_of_nn_node[i + 1]]) for i in range(len(self.num_of_nn_node) - 1)]
    
    #�l�b�g���[�N�̃p�����[�^�������_��(-0.5����0.5)�ɐݒ�
    def random_params(self):
        #conv0_filter
        for i in range(len(self.conv0_filter)): self.conv0_filter[i] = np.random.rand(3,3) - 0.5
        
        #conv0_bweight
        self.conv0_bweight = np.random.rand(self.conv0_bweight.size) - 0.5
        
        #conv1_filter
        for i in range(len(self.conv1_filter)):
            for j in range(len(self.conv1_filter[i])): self.conv1_filter[i][j] = np.random.rand(3,3) - 0.5 
        
        #conv1_bweight
        self.conv1_bweight = np.random.rand(self.conv1_bweight.size) - 0.5
        
        #nn_weight
        for i in range(len(self.nn_weight)): self.nn_weight[i] = np.random.rand(self.nn_weight[i].shape[0],self.nn_weight[i].shape[1]) - 0.5
        
    #�f�[�^�x�[�X�ɐڑ��A�p�����[�^���Z�b�g
    def set_params(self):
        #�f�[�^�x�[�X�ڑ�
        conn = sqlite3.connect('cnn_params.db')
        
        #�J�[�\������
        c = conn.cursor()
        
        #��0��ݍ��ݑw�̊e�t�B���^�̏d�݂��Z�b�g
        for i in range(len(self.conv0_filter)):
            for y in range(self.conv0_filter[i].shape[0]):
                for x in range(self.conv0_filter[i].shape[1]):
                    sql = "select value from conv0_filter where channel=%d and y=%d and x=%d" %(i,y,x)
                    value = c.execute(sql).fetchone()
                    self.conv0_filter[i][y][x] = value[0]
        
        #��1��ݍ��ݑw�̊e�t�B���^�̏d�݂��Z�b�g
        for i in range(len(self.conv1_filter)):
            for j in range(len(self.conv1_filter[i])):
                for y in range(self.conv1_filter[i][j].shape[0]):
                    for x in range(self.conv1_filter[i][j].shape[1]):
                        sql = "select value from conv1_filter where channel=%d and before_channel=%d and y=%d and x=%d" %(i,j,y,x)
                        value = c.execute(sql).fetchone()
                        self.conv1_filter[i][j][y][x] = value[0]
        
        #��0��ݍ��ݑw�̃o�C�A�X���Z�b�g
        for i in range(self.conv0_bweight.size):
            sql = "select value from conv_bweight where layer=%d and channel=%d" %(0,i)
            value = c.execute(sql).fetchone()
            self.conv0_bweight[i] = value[0]
        
        #��1��ݍ��ݑw�̃o�C�A�X���Z�b�g
        for i in range(self.conv1_bweight.size):
            sql = "select value from conv_bweight where layer=%d and channel=%d" %(1,i)
            value = c.execute(sql).fetchone()
            self.conv1_bweight[i] = value[0]
        
        #�S�����w�̊e�w�̏d�݂��Z�b�g
        for i in range(len(self.nn_weight)):
            for j in range(self.nn_weight[i].shape[0]):
                for k in range(self.nn_weight[i].shape[1]):
                    sql = "select value from nn_weight where layer=%d and fromid=%d and toid=%d" %(i,j,k)
                    value = c.execute(sql).fetchone()
                    self.nn_weight[i][j][k] = value[0]
        
        #�f�[�^�x�[�X�ڑ���ؒf
        conn.close()
        
        
    
    #�p�����[�^�̕ۑ�
    def save_params(self):
        #�f�[�^�x�[�X�ڑ�
        conn = sqlite3.connect('cnn_params.db')
        #�J�[�\���̐���
        c = conn.cursor()
        
        
        #�f�[�^�x�[�X��̊e�p�����[�^�l���X�V
        #��0��ݍ��ݑw�̊e�t�B���^�̏d�݂��X�V
        for i in range(len(self.conv0_filter)):
            for y in range(self.conv0_filter[i].shape[0]):
                for x in range(self.conv0_filter[i].shape[1]):
                    sql = "update conv0_filter set value=%.8f where channel=%d and y=%d and x=%d" %(self.conv0_filter[i][y][x],i,y,x)
                    c.execute(sql)
        print('conv0 saved')
        
        #��1��ݍ��ݑw�̊e�t�B���^�̏d�݂��X�V
        for i in range(len(self.conv1_filter)):
            for j in range(len(self.conv1_filter[i])):
                for y in range(self.conv1_filter[i][j].shape[0]):
                    for x in range(self.conv1_filter[i][j].shape[1]):
                        sql = "update conv1_filter set value=%.8f where channel=%d and before_channel=%d and y=%d and x=%d" %(self.conv1_filter[i][j][y][x],i,j,y,x)
                        c.execute(sql)
        print('conv1 saved')
        
        #��0��ݍ��ݑw�̃o�C�A�X���X�V
        for i in range(self.conv0_bweight.size):
            sql = "update conv_bweight set value=%.8f where layer=%d and channel=%d" %(self.conv0_bweight[i],0,i)
            c.execute(sql)
        print('convbweight0 saved')
        
        #��1��ݍ��ݑw�̃o�C�A�X���X�V
        for i in range(self.conv1_bweight.size):
            sql = "update conv_bweight set value=%.8f where layer=%d and channel=%d" %(self.conv1_bweight[i],1,i)
            c.execute(sql)
        print('convbweight1 saved')
        
        #�S�����w�̊e�d�݂��X�V
        for i in range(len(self.nn_weight)):
            for j in range(self.nn_weight[i].shape[0]):
                for k in range(self.nn_weight[i].shape[1]):
                    sql = "update nn_weight set value=%.8f where layer=%d and fromid=%d and toid=%d" %(self.nn_weight[i][j][k],i,j,k)
                    c.execute(sql)
        print('nn_weight saved')
        
        #�f�[�^�x�[�X�̕ύX���X�V
        conn.commit()
        #�f�[�^�x�[�X�ڑ���ؒf
        conn.close()
    
    #��ݍ����feature_map��Ԃ�
    def convolution_layer0(self):
        for i in range(len(self.conv0_filter)):
            feature_map = np.zeros([self.input.shape[0] - 2,self.input.shape[1] - 2])
            for y in range(feature_map.shape[0]):
                for x in range(feature_map.shape[1]):
                    sum = np.sum(self.input[y:y + 3,x:x + 3] * self.conv0_filter[i])
                    feature_map[y][x] = af.relu(sum + self.conv0_bweight[i])
            self.conv0_feature_map.append(feature_map)
    
    #feature_map���v�[�����O�������̂�Ԃ�
    def maxpool(self,feature_map):
        new_feature_map = []
        for i in range(len(feature_map)):
            pool_out = np.zeros([int(feature_map[i].shape[0] / 2),int(feature_map[i].shape[1] / 2)])
            for y in range(pool_out.shape[0]):
                for x in range(pool_out.shape[1]):
                    v = feature_map[i][2 * y:2 * (y + 1),2 * x:2 * (x + 1)]
                    pool_out[y][x] = np.max(v)
            new_feature_map.append(pool_out)
        return new_feature_map
    
    def convolution_layer1(self):
        for i in range(len(self.conv1_filter)):
            v = []
            for j in range(len(self.conv1_filter[i])):
                v.append(np.zeros([self.pool0_out[j].shape[0] - 2,self.pool0_out[j].shape[1] - 2]))
                for y in range(v[j].shape[0]):
                    for x in range(v[j].shape[1]):
                        v[j][y][x] = np.sum(self.pool0_out[j][y:y+3,x:x+3] * self.conv1_filter[i][j])
            feature_map = af.relu(sum(v) + self.conv1_bweight[i])
            self.conv1_feature_map.append(feature_map)
    
    #pool1layer���featuremap��1�����z��ɕϊ�
    #1�����z���Ԃ�
    def transform_map(self):
        dist = np.zeros([len(self.pool1_out) * self.pool1_out[0].shape[0] * self.pool1_out[0].shape[1]])
        for i in range(len(self.pool1_out)):
            for y in range(self.pool1_out[i].shape[0]):
                for x in range(self.pool1_out[i].shape[1]):
                    dist[self.pool1_out[i].shape[0] * self.pool1_out[i].shape[1] * i + self.pool1_out[i].shape[0] * y + x] = self.pool1_out[i][y][x]
        return dist
    
    #�w�̔���
    #input 1�����z��
    #weight [input][output]�̓񎟌��z�� input��output���Ȃ��d��
    def fire(self,input,num_of_layer,act_func):
        #�o�͂̌��̗v�f��������1�����z��
        dist = np.zeros([self.nn_weight[num_of_layer].shape[1]])
        for i in range(dist.size): dist[i] = sum([input[j] * self.nn_weight[num_of_layer][j][i] for j in range(input.size)])
        return act_func(dist)
        
    def feedforward(self,input):
        #input�Ƀp�f�B���O��������
        self.input = np.zeros([MNIST_SIZE + 2,MNIST_SIZE + 2])
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                self.input[1 + i][1 + j] = input[i][j]
        
        #conv0layer
        #conv0_feature_map�����
        self.conv0_feature_map = []
        self.convolution_layer0()
        
        #pooling0layer
        #pool0_out�Ƀv�[�����O���feature_map���i�[
        self.pool0_out = self.maxpool(self.conv0_feature_map)
        
        #conv1layer
        self.conv1_feature_map = []
        self.convolution_layer1()
        
        #pooling1layer
        self.pool1_out = self.maxpool(self.conv1_feature_map)
        
        #�S�����w
        
        #pool1_out����nn�w��input(1�����z��)�ɕϊ�
        self.transformed_pool1_out = self.transform_map()
        bias = np.array([1])
        self.nn_input = np.r_[bias,self.transformed_pool1_out]
        
        #�e�w�̏o�͂̃��X�g
        self.nn_out = []
        
        #�e�w�̔���
        for i in range(len(self.nn_weight)):
            if i == 0: nn_layer_out = self.fire(self.nn_input,i,self.hidden_act_func)
            else:
                layer_input = np.r_[bias,self.nn_out[i-1]]
                if i == len(self.nn_weight) - 1: nn_layer_out = self.fire(layer_input,i,self.output_act_func)
                else: nn_layer_out = self.fire(layer_input,i,self.hidden_act_func)
            self.nn_out.append(nn_layer_out)
        return self.nn_out[-1]
    
    def backprop(self,teacher_signal,larning_rate = 0.5):
        #�S�����w�̌덷���i�[����z��̃��X�g
        nn_delta = []
        for i in range(len(self.nn_out)):
            nn_delta.append(np.zeros([self.nn_out[i].size]))
        
        #�o�͂Ƌ��t�M���̌덷�����߂�
        output_error = self.nn_out[-1] - teacher_signal
        
        #�S�����w�̏o�͑w
        #�덷���v�Z����nn_delta�Ɋi�[
        for i in range(nn_delta[-1].size):
            nn_delta[-1][i] = output_error[i] * self.output_differential_func(self.nn_out[-1][i])
        
        
        #�S�����w�̉B��w�̌덷�����߂�
        for i in range(len(nn_delta) - 1)[::-1]:
            for j in range(nn_delta[i].size):
                delta = 0
                for k in range(nn_delta[i + 1].size):
                    delta += self.nn_weight[i + 1][j + 1][k] * nn_delta[i + 1][k]
                nn_delta[i][j] = self.hidden_differential_func(self.nn_out[i][j]) * delta
        
        #�S�����w�̓��͑w�̌덷�����߂�
        nn_input_delta = np.zeros([self.transformed_pool1_out.size])
        for i in range(self.transformed_pool1_out.size):
            delta = 0
            for j in range(nn_delta[0].size):
                delta += self.nn_weight[0][i+1][j] * nn_delta[0][j]
            nn_input_delta[i] = delta
        
        
        #�S�����w�̏d�݂��X�V
        for i in range(len(self.nn_weight))[::-1]:
            if i == 0:
                for j in range(self.nn_weight[0].shape[1]):
                    for k in range(self.nn_weight[0].shape[0]):
                        x = 1 if j == 0 else self.transformed_pool1_out[j - 1]
                        self.nn_weight[i][k][j] -= larning_rate * x * nn_delta[i][j]
            else:
                for j in range(self.nn_weight[i].shape[1]):
                    for k in range(self.nn_weight[i].shape[0]):
                        x = 1 if j == 0 else self.nn_out[i - 1][j - 1]
                        self.nn_weight[i][k][j] -= larning_rate * x * nn_delta[i][j]
        
        
        #�S�����w�̓��͑w�̌덷(�ꎟ���z��)��pool1_out�ɑΉ������񎟌��z��̃��X�g�ɒu��������
        pool1_delta = []
        for i in range(len(self.pool1_out)): pool1_delta.append(np.zeros([self.pool1_out[i].shape[0],self.pool1_out[i].shape[1]]))
        for i in range(len(pool1_delta)):
            for y in range(pool1_delta[i].shape[0]):
                for x in range(pool1_delta[i].shape[1]):
                    pool1_delta[i][y][x] = nn_input_delta[pool1_delta[i].size * i + pool1_delta[i].shape[0] * y + x]
        
        #conv1�w�̌덷�����߂�
        conv1_delta = []
        for i in range(len(self.conv1_feature_map)):
            conv1_delta.append(np.zeros([self.conv1_feature_map[i].shape[0],self.conv1_feature_map[i].shape[1]]))
        
        for i in range(len(self.pool1_out)):
            for y in range(self.pool1_out[i].shape[0]):
                for x in range(self.pool1_out[i].shape[1]):
                    for yy in range(2):
                        for xx in range(2):
                            m = 2 * y + yy
                            n = 2 * x + xx
                            conv1_delta[i][m][n] = af.drelu(self.conv1_feature_map[i][m][n]) * (self.pool1_out[i][y][x] == self.conv1_feature_map[i][m][n]) * pool1_delta[i][y][x]
        
        #pool0�w�̌덷�����߂�
        pool0_delta = []
        for i in range(len(self.pool0_out)):
            pool0_delta.append(np.zeros([self.pool0_out[i].shape[0],self.pool0_out[i].shape[1]]))
        
        for i in range(len(self.conv1_feature_map)):
            for y in range(self.conv1_feature_map[i].shape[0]):
                for x in range(self.conv1_feature_map[i].shape[1]):
                    for j in range(len(self.pool0_out)):
                        pool0_delta[j][y:y+3,x:x+3] += self.conv1_filter[i][j] * conv1_delta[i][y][x]
        
        #conv0�w�̌덷�����߂�
        conv0_delta = []
        for i in range(len(self.conv0_feature_map)):
            conv0_delta.append(np.zeros([self.conv0_feature_map[i].shape[0],self.conv0_feature_map[i].shape[1]]))
        for i in range(len(self.pool0_out)):
            for y in range(self.pool0_out[i].shape[0]):
                for x in range(self.pool0_out[i].shape[1]):
                    for yy in range(2):
                        for xx in range(2):
                            m = 2 * y + yy
                            n = 2 * x + xx
                            conv0_delta[i][m][n] = af.drelu(self.conv0_feature_map[i][m][n]) * (self.pool0_out[i][y][x] == self.conv0_feature_map[i][m][n]) * pool0_delta[i][y][x]
        
        #conv1�w�̃t�B���^�d�݂��X�V
        for i in range(len(self.conv1_filter)):
            for j in range(len(self.conv1_filter[i])):
                for y in range(self.conv1_filter[i][j].shape[0]):
                    for x in range(self.conv1_filter[i][j].shape[1]):
                        change = self.pool0_out[j][y:y + self.conv1_feature_map[i].shape[0], x:x + self.conv1_feature_map[i].shape[1]] \
                             * conv1_delta[i] * larning_rate
                        self.conv1_filter[i][j][y][x] -= np.average(change)
        
        #conv1�w�̃o�C�A�X�d�݂��X�V
        for i in range(self.conv1_bweight.size):
            change = 1 * conv1_delta[i] * larning_rate
            self.conv1_bweight[i] -= np.average(change)
        
        
        #conv0�w�̃t�B���^�d�݂��X�V
        for i in range(len(self.conv0_filter)):
            for y in range(self.conv0_filter[i].shape[0]):
                for x in range(self.conv0_filter[i].shape[1]):
                    change = self.input[y:y + self.conv0_feature_map[i].shape[0],x:x + self.conv0_feature_map[i].shape[1]]\
                        * conv0_delta[i] * larning_rate
                    self.conv0_filter[i][y][x] -= np.average(change)
        
        #conv0�w�̃o�C�A�X�d�݂��X�V
        for i in range(self.conv0_bweight.size):
            change = 1 * conv0_delta[i] * larning_rate
            self.conv0_bweight[i] -= np.average(change)

#�摜�f�[�^��2�l�����ē񎟌��z��ɕϊ�
def transform_image(image):
    size = image.size
    data = np.zeros([size[1],size[0]])
    
    for y in range(size[1]):
        for x in range(size[0]):
            r,g,b = image.getpixel((x,y))
            data[y][x] = (255 - (r + g + b) / 3)
    
    return data

#�l�b�g���[�N�̐���
def build_cnn():
    #�Œ�p�����[�^
    #��ݍ��ݑw��2�w�A�t�B���^����8,16�Ƃ���
    num_of_conv_filter = np.array([8,16])
    
    #�S�����w�̉B��w��3�w�A�m�[�h����96,64,32�Ƃ���
    num_of_hidden_node = np.array([96,64,32])
    
    #�o�͂̃N���X��
    num_of_output = 10
    
    #CNN�N���X�̃C���X�^���X�𐶐����ĕԂ�
    cnn = CNN(num_of_conv_filter, num_of_hidden_node, num_of_output)
    return cnn

def train():
    cnn = build_cnn()
    print('set_params')
    cnn.set_params()
    for i in range(10):
        print(i)
        teacher_signal = np.zeros([10])
        teacher_signal[i] = 1.0
        for j in range(NUM_OF_TRAIN_DATA[i]):
            if j % 1000 == 0: print(str(j)+'trained')
            try: im = Image.open("c:/users/ryo/pictures/train%d/%d.jpg" %(i,j))
            except:
                print('error')
                return
            input = transform_image(im)
            cnn.feedforward(input)
            cnn.backprop(teacher_signal)
            #while 1:
                #ans = cnn.feedforward(input)
                #if np.argmax(ans) == i: break
                #else: cnn.backprop(teacher_signal)
    print('save_params')
    cnn.save_params()

#MNIST��TEST�f�[�^�ɑ΂��ăl�b�g���[�N�ŕ��ށA��������Ԃ�
def classify():
    
    #�듚��
    error = 0
    
    #�T���v����
    num_of_sample = 0
    
    #�l�b�g���[�N�̐���
    cnn = build_cnn()
    
    #�l�b�g���[�N�Ƀp�����[�^�l���Z�b�g
    cnn.set_params()
    
    #0����9�܂�
    for i in range(10):
        print(i)
        n = 0
        while 1:
            
            #�f�[�^�ǂݍ���
            try: im = Image.open("c:/users/ryo/pictures/test%d/%d.jpg" %(i,n))
            except:
                num_of_sample += n
                break
            input = transform_image(im)
            result = cnn.feedforward(input)
            if np.argmax(result) != i: error += 1
            n += 1
    print('correct answers')
    correct_percent = 1.0 - error / num_of_sample
    print(correct_percent)
    print('error')
    print(error)
    return correct_percent

#�f�[�^�x�[�X�쐬
def create_db(cnn):
    #�f�[�^�x�[�X�ڑ�
    conn = sqlite3.connect('cnn_params.db')
    
    #�J�[�\���𐶐�
    c = conn.cursor()
    
    #�e�e�[�u���̍쐬
    c.execute('''create table conv0_filter (channel integer,y integer,x integer,value real)''')
    c.execute('''create table conv1_filter (channel integer,before_channel integer,y integer,x integer,value real)''')
    c.execute('''create table conv_bweight (layer integer,channel integer,value real)''')
    c.execute('''create table nn_weight (layer integer,fromid integer,toid integer,value real)''')
    
    #�S�����w�̏d�݂̂ݒT�����x���グ�邽�߃C���f�b�N�X���쐬
    c.execute('''create index nn_weight_index on nn_weight(layer,fromid,toid)''')
    
    #�l�̑}��
    #����cnn�̃p�����[�^��}�����Ă���
    
    #��0��ݍ��ݑw�̊e�t�B���^�̏d�݂̑}��
    sql = "insert into conv0_filter values (?,?,?,?)"
    for i in range(len(cnn.conv0_filter)):
        for y in range(cnn.conv0_filter[i].shape[0]):
            for x in range(cnn.conv0_filter[i].shape[1]): c.execute(sql,(i,y,x,cnn.conv0_filter[i][y][x]))
    
    #��1��ݍ��ݑw�̊e�t�B���^�̏d�݂̑}��
    sql = "insert into conv1_filter values (?,?,?,?,?)"
    for i in range(len(cnn.conv1_filter)):
        for j in range(len(cnn.conv1_filter[i])):
            for y in range(cnn.conv1_filter[i][j].shape[0]):
                for x in range(cnn.conv1_filter[i][j].shape[1]): c.execute(sql,(i,j,y,x,cnn.conv1_filter[i][j][y][x]))
    
    #�e��ݍ��ݑw�̃o�C�A�X��}��
    sql = "insert into conv_bweight values (?,?,?)"
    #��0��ݍ��ݑw�̃o�C�A�X��}��
    for i in range(cnn.conv0_bweight.size): c.execute(sql,(0,i,cnn.conv0_bweight[i]))
    #��1��ݍ��ݑw�̃o�C�A�X��}��
    for i in range(cnn.conv1_bweight.size): c.execute(sql,(1,i,cnn.conv1_bweight[i]))
    
    #�S�����w�̏d��(�o�C�A�X��)��}��
    sql = "insert into nn_weight values (?,?,?,?)"
    for i in range(len(cnn.nn_weight)):
        for j in range(cnn.nn_weight[i].shape[0]):
            for k in range(cnn.nn_weight[i].shape[1]): c.execute(sql,(i,j,k,cnn.nn_weight[i][j][k]))
    
    #�f�[�^�x�[�X�̕ύX��ۑ�
    conn.commit()
    
    #�f�[�^�x�[�X�Ƃ̐ڑ���؂�
    conn.close()

#�p�����[�^�������l�̏�ݍ��݃j���[�����l�b�g���[�N�𐶐�
#�l�b�g���[�N��Ԃ�
def initialize():
    cnn = build_cnn()
    cnn.random_params()
    create_db(cnn)
    return cnn

if __name__ == "__main__":
    #cnn�̏�����(�f�[�^�x�[�X�t�@�C�����폜���Ă���)
    #cnn = initialize()
    
    #���N���X����
    #classify()
    
    #�w�K
    #train()
        
    '<EOF>'