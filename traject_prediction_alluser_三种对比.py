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

#LSTM需要的数组格式： [样本数，时间步伐，特征]
train_x = np.reshape(train_x,(-1,seq_length,14))
test_x = np.reshape(test_x, (-1, seq_length, 14))
train_y = np.reshape(train_y,(-1,10))
test_y = np.reshape(test_y, (-1, 10))
train_x_a = np.reshape(train_x,(-1,seq_length*14))
test_x_a = np.reshape(test_x, (-1, seq_length*14))

print('训练集、测试集构建完成')
print('数据总条数:',len(total_x))
print('训练集条数:',len(train_x))
print('测试集条数:',len(test_x))
end_tarindata = time.time()
print('构造训练集时间:',end_tarindata - start_traindata)
#print(train_x.shape)
#print(train_y.shape)
#print(test_x.shape)
#print(test_y.shape)


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
#         val_targ = self.validation_data[1]
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        print(' — val_f1:' ,_val_f1)
        return

# 其他metrics可自行添加
metrics = Metrics()

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)


from keras import backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


class Attention_layer(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(Attention_layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        a = K.exp(uit)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

start_1 = time.time()

#搭建LSTM

# 创建模型
Epoch = 200
batch_size_num = 256
num_units = 32

print('创建模型...')
model_1 = Sequential()
model_1.add(LSTM(num_units, dropout=0.2,input_shape=(seq_length, 14)))
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
hist_1 = model_1.fit(train_x, train_y,nb_epoch=Epoch, batch_size=batch_size_num,shuffle=True,validation_split=0.1,callbacks=[metrics])
          #validation_split=0.1,validation_data=(test_x, test_y)

end_1 = time.time()

start = time.time()
#搭建双向LSTM

# 创建模型
print('创建模型...')
model_2 = Sequential()
input_shape = (seq_length, 14)
model_2.add(Bidirectional(LSTM(num_units, return_sequences=False),input_shape=input_shape))
    #防止过拟合，在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接
model_2.add(Dropout(0.2))
model_2.add(BatchNormalization())
#model.add(TimeDistributed(Dense(10, activation='softmax')))
    #全连接层
model_2.add(Dense(10, activation="softmax"))#输出维度，激活函数


model_2.summary()

# 编译模型
print('编译模型...')
#model_2.compile(loss="mse", optimizer="RMSProp",metrics = ['acc'])#RMSProp,adam,#metrics = ['accuracy']),metrics = ['acc',recall,fmeasure]
model_2.compile(loss="categorical_crossentropy", optimizer="RMSProp",metrics = ['acc'])#RMSProp,adam,#metrics = ['accuracy']),metrics = ['acc',recall,fmeasure]

# 训练模型
print('训练模型...')                                                #验证集取训练集的10%
hist_2 = model_2.fit(train_x, train_y,nb_epoch=Epoch, batch_size=batch_size_num,shuffle=True,validation_split=0.1,callbacks=[metrics])
          #validation_split=0.1,validation_data=(test_x, test_y)


end = time.time()



#搭建ANN
start_2 = time.time()
# 创建模型
print('创建模型...')
model_3 = Sequential()
model_3.add(Dense(num_units, input_shape=(seq_length*14,)))#input_shape=(seq_length, 14)))
        #防止过拟合，在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接
model_3.add(Dropout(0.2))
        #全连接层
model_3.add(Dense(10, activation="softmax"))#输出维度，激活函数


model_3.summary()

# 编译模型
print('编译模型...')
#model_3.compile(loss="mse", optimizer="RMSProp",metrics = ['acc'])#RMSProp,adam,#metrics = ['accuracy']),metrics = ['acc',recall,fmeasure]
model_3.compile(loss="categorical_crossentropy", optimizer="RMSProp",metrics = ['acc'])#RMSProp,adam,#metrics = ['accuracy']),metrics = ['acc',recall,fmeasure]

# 训练模型
print('训练模型...')                                                #验证集取训练集的10%
hist_3 = model_3.fit(train_x_a, train_y,nb_epoch=Epoch, batch_size=batch_size_num,shuffle=True,validation_split=0.1)
          #validation_split=0.1,validation_data=(test_x, test_y)

end_2 = time.time()


print('\n')
print('Epoch:', Epoch)
print('batch_size:', batch_size_num)
print('num_units:', num_units)
print('\n')
print('LSTM训练时间:',end_1 - start_1)
print('Bi-LSTM训练时间:',end - start)
print('\n')
print('LSTM训练效果')
print(hist_1.history)
print('\n')
print('Bi-LSTM训练效果')
print(hist_2.history)

pyplot.plot(hist_1.history['acc'], label='LSTM')
pyplot.plot(hist_2.history['acc'], label='Bi-LSTM')
pyplot.plot(hist_3.history['acc'], label='ANN')
#pyplot.plot(hist.history['val_acc'], label='test')
pyplot.legend(loc='lower right')
#设置坐标轴范围
pyplot.xlim((0, 220))
#pyplot.ylim((0.75, 0.85))
#设置坐标轴名称
pyplot.xlabel('epoch')
pyplot.ylabel('accuracy')
#设置坐标轴刻度
my_x_ticks = np.arange(0, 220, 20)
#my_y_ticks = np.arange(0.75, 0.85, 0.01)
pyplot.xticks(my_x_ticks)
#pyplot.yticks(my_y_ticks)

pyplot.show()

pyplot.plot(hist_1.history['loss'], label='LSTM')
pyplot.plot(hist_2.history['loss'], label='Bi-LSTM')
pyplot.plot(hist_3.history['loss'], label='ANN')
#pyplot.plot(hist.history['val_acc'], label='test')
pyplot.legend(loc='lower right')
#设置坐标轴范围
pyplot.xlim((0, 220))
#pyplot.ylim((0.75, 0.85))
#设置坐标轴名称
pyplot.xlabel('epoch')
pyplot.ylabel('loss')
#设置坐标轴刻度
my_x_ticks = np.arange(0, 220, 20)
#my_y_ticks = np.arange(0.75, 0.85, 0.01)
pyplot.xticks(my_x_ticks)
#pyplot.yticks(my_y_ticks)

pyplot.show()

#pyplot.plot(hist.history['acc'], label='train')
pyplot.plot(hist_1.history['val_acc'], label='LSTM')
pyplot.plot(hist_2.history['val_acc'], label='Bi-LSTM')
pyplot.plot(hist_3.history['val_acc'], label='ANN')
pyplot.legend()
#设置坐标轴范围
pyplot.xlim((0, 220))
#pyplot.ylim((0.75, 0.85))
#设置坐标轴名称
pyplot.xlabel('epoch')
pyplot.ylabel('accuracy')
#设置坐标轴刻度
my_x_ticks = np.arange(0, 220, 20)
#my_y_ticks = np.arange(0.75, 0.85, 0.01)
pyplot.xticks(my_x_ticks)
#pyplot.yticks(my_y_ticks)
pyplot.show()

pyplot.plot(hist_1.history['val_loss'], label='LSTM')
pyplot.plot(hist_2.history['val_loss'], label='Bi-LSTM')
pyplot.plot(hist_3.history['val_loss'], label='ANN')
#pyplot.plot(hist.history['val_acc'], label='test')
pyplot.legend()
#设置坐标轴范围
pyplot.xlim((0, 220))
#pyplot.ylim((0.75, 0.85))
#设置坐标轴名称
pyplot.xlabel('epoch')
pyplot.ylabel('loss')
#设置坐标轴刻度
my_x_ticks = np.arange(0, 220, 20)
#my_y_ticks = np.arange(0.75, 0.85, 0.01)
pyplot.xticks(my_x_ticks)
#pyplot.yticks(my_y_ticks)

pyplot.show()



#评估模型
print('评估模型...')
print('LSTM')

#print(model.evaluate(test_x, test_y, batch_size=32))
score, acc = model_1.evaluate(test_x, test_y, batch_size=32)
#score, acc = model_1.evaluate(test_x, test_y, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
#print('Test precision:', precision)
#print('Test recall:', recall)
#print('Test fmeasure:', fmeasure)

#评估模型
print('评估模型...')
print('Bi-LSTM')
#print(model.evaluate(test_x, test_y, batch_size=32))
score, acc = model_2.evaluate(test_x, test_y, batch_size=32)
#score, acc = model_2.evaluate(test_x, test_y, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
#print('Test precision:', precision)
#print('Test recall:', recall)
#print('Test fmeasure:', fmeasure)

#评估模型
print('评估模型...')
print('ANN')
#print(model.evaluate(test_x, test_y, batch_size=32))
score, acc = model_3.evaluate(test_x_a, test_y, batch_size=32)
#score, acc = model_2.evaluate(test_x, test_y, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
#print('Test precision:', precision)
#print('Test recall:', recall)
#print('Test fmeasure:', fmeasure)




