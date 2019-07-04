import os
import time

import numpy as np
import pandas as pd
import loc_record
import feature_extra
import math
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, TimeDistributed, Bidirectional
from keras.layers import Dropout
from keras.layers import LSTM
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models import Word2Vec
from keras import backend as K
import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from matplotlib import pyplot


''''
    # integer encode direction
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    print(reframed.head())
    '''
start_traindata = time.time()
#构造训练测试集
#path1 = r'F:\GPS\Feature_Data\Effica_Feature_Data'
path1 = r'F:\GPS\Feature_Data\Effica_Feature_Data'
user_list = loc_record.curdir_file(path1)
seq_length = 4
total_x = []
train_x = []
train_y = []
test_x = []
test_y = []

for user in user_list:
    path = path1 + os.path.sep + user
    data = pd.read_csv(open(path))
    values = data.values
    day_interval_index = []

    for i in range(0, len(data)-1):
        if data.iloc[i]['date_string'] != data.iloc[i + 1]['date_string']:
            day_interval_index.append(i)

    day_interval_index.append(len(data))

    #构造滑动窗口
    x = []
    y = []
    begin = 0


    #隔天的标签号
    for m in range(0,len(day_interval_index)):
        #同天划分
        for n in range(begin,day_interval_index[m]-seq_length+1 ):

            #到末尾了
            if n + seq_length == len(data)-1:

                # 对应向量
                given = values[n:n + seq_length, 17:31]
                predict = values[n + seq_length, 31:41]
                #given = values[n:n + seq_length, :]
                #predict = values[n + seq_length, :]
                x.append(given)
                y.append(predict)
                break
            #对应向量
            given = values[n:n + seq_length,17:31]
            predict = values[n + seq_length, 31:41]
            #全部信息
            #given = values[n:n + seq_length,:]
            #predict = values[n + seq_length,:]
            x.append(given)
            y.append(predict)


        begin = day_interval_index[m] + 1


    #构建训练、测试集
    train_count = math.ceil(len(x)*0.7)
    train_x_list = x[0:train_count]
    train_y_list = y[0:train_count]
    test_x_list = x[train_count::]
    test_y_list = y[train_count::]
    #print('数据总条数:',len(x))
    #print('训练集条数:',len(train_x_list))
    #print('测试集条数:',len(test_x_list))
    #print('###############################')

    total_x.extend(x)
    train_x.extend(train_x_list)
    train_y.extend(train_y_list)
    test_x.extend(test_x_list)
    test_y.extend(test_y_list)


#ANN需要的数组格式：(sample,nx*ny)
train_x = np.reshape(train_x,(-1,seq_length*14))
test_x = np.reshape(test_x, (-1,seq_length*14))
train_y = np.reshape(train_y,(-1,10))
test_y = np.reshape(test_y, (-1, 10))
print(train_x.shape)
print(train_x[0:2])
print(type(train_x))
print(train_y.shape)
print(train_y[0:2])

print('训练集、测试集构建完成')
print('数据总条数:',len(total_x))
print('训练集条数:',len(train_x))
print('测试集条数:',len(test_x))
end_tarindata = time.time()
print('构造训练集时间:',end_tarindata - start_traindata)

'''
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


# 神经元个数
units = [32, 64, 128]
for unit in units:
    # 激活函数：relu, logistic, tanh
    # 优化算法：lbfgs, sgd, adam。adam适用于较大的数据集，lbfgs适用于较小的数据集。
    #初始化模型
    print('创建模型...')
    ann_model = MLPRegressor(hidden_layer_sizes=[unit], activation='logistic', solver='adam', random_state=0)
    #训练模型
    print('训练模型...')
    ann_model.fit(train_x, train_y)
    print('评估模型...')
    print('神经元个数={}，准确率：{:.4f}'.format(unit, ann_model.score(test_x, test_y)))
'''
'''
pyplot.plot(hist_1.history['acc'], label='ANN')
#pyplot.plot(hist.history['val_acc'], label='test')
pyplot.legend(loc='lower right')
#设置坐标轴范围
pyplot.xlim((0, 110))
#pyplot.ylim((0.75, 0.85))
#设置坐标轴名称
pyplot.xlabel('epoch')
pyplot.ylabel('accuracy')
#设置坐标轴刻度
my_x_ticks = np.arange(0, 110, 10)
#my_y_ticks = np.arange(0.75, 0.85, 0.01)
pyplot.xticks(my_x_ticks)
#pyplot.yticks(my_y_ticks)

pyplot.show()

pyplot.plot(hist_1.history['loss'], label='ANN')
#pyplot.plot(hist.history['val_acc'], label='test')
pyplot.legend(loc='lower right')
#设置坐标轴范围
pyplot.xlim((0, 110))
#pyplot.ylim((0.75, 0.85))
#设置坐标轴名称
pyplot.xlabel('epoch')
pyplot.ylabel('loss')
#设置坐标轴刻度
my_x_ticks = np.arange(0, 110, 10)
#my_y_ticks = np.arange(0.75, 0.85, 0.01)
pyplot.xticks(my_x_ticks)
#pyplot.yticks(my_y_ticks)

pyplot.show()

#pyplot.plot(hist.history['acc'], label='train')
pyplot.plot(hist_1.history['val_acc'], label='ANN')
pyplot.legend()
#设置坐标轴范围
pyplot.xlim((0, 110))
#pyplot.ylim((0.75, 0.85))
#设置坐标轴名称
pyplot.xlabel('epoch')
pyplot.ylabel('accuracy')
#设置坐标轴刻度
my_x_ticks = np.arange(0, 110, 10)
#my_y_ticks = np.arange(0.75, 0.85, 0.01)
pyplot.xticks(my_x_ticks)
#pyplot.yticks(my_y_ticks)
pyplot.show()

pyplot.plot(hist_1.history['val_loss'], label='ANN')
#pyplot.plot(hist.history['val_acc'], label='test')
pyplot.legend()
#设置坐标轴范围
pyplot.xlim((0, 110))
#pyplot.ylim((0.75, 0.85))
#设置坐标轴名称
pyplot.xlabel('epoch')
pyplot.ylabel('loss')
#设置坐标轴刻度
my_x_ticks = np.arange(0, 110, 10)
#my_y_ticks = np.arange(0.75, 0.85, 0.01)
pyplot.xticks(my_x_ticks)
#pyplot.yticks(my_y_ticks)

pyplot.show()
'''
ANN_loss = []
ANN_acc = []
ANN_train_time = []

Epoch = 100
batch_size_num = 64
num_units = 32

for i in range(0,10):
    #搭建神经网络

    # 创建模型
    start_1 = time.time()
    print('创建模型...')
    model_1 = Sequential()
    model_1.add(Dense(num_units, input_shape=(seq_length*14,)))#input_shape=(seq_length, 14)))
        #防止过拟合，在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接
    model_1.add(Dropout(0.2))
        #全连接层
    model_1.add(Dense(10, activation="softmax"))#输出维度，激活函数


    model_1.summary()

    # 编译模型
    print('编译模型...')
    #model_1.compile(loss="mse", optimizer="RMSProp",metrics = ['acc'])#RMSProp,adam,#metrics = ['accuracy']),metrics = ['acc',recall,fmeasure]
    model_1.compile(loss="categorical_crossentropy", optimizer="RMSProp",metrics = ['acc'])#RMSProp,adam,#metrics = ['accuracy']),metrics = ['acc',recall,fmeasure]

    # 训练模型
    print('训练模型...')                                                #验证集取训练集的10%
    hist_1 = model_1.fit(train_x, train_y,nb_epoch=Epoch, batch_size=batch_size_num,shuffle=True,validation_split=0.1)
          #validation_split=0.1,validation_data=(test_x, test_y)

    end_1 = time.time()



    print('\n')
    train_time_ANN = end_1 - start_1
    print('ANN训练时间:',train_time_ANN)
    print('\n')
    print('ANN训练效果')
    print(hist_1.history)
    print('\n')




    #评估模型
    print('评估模型...')
    print('ANN')

    #print(model.evaluate(test_x, test_y, batch_size=32))
    score, acc = model_1.evaluate(test_x, test_y, batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)
    ANN_loss.append(score)
    ANN_acc.append(acc)
    ANN_train_time.append(train_time_ANN)

print('\n')
print('Epoch:', Epoch)
print('batch_size:', batch_size_num)
print('num_units:', num_units)

print('\n')
print('ANN_loss',ANN_loss)
print('ANN_acc',ANN_acc)
print('ANN_train_time',ANN_train_time)


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)

y = x * 2

plt.title("一元一次函数")
plt.plot(x, y)

plt.show()



