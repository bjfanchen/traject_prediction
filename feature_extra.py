import os
import pandas as pd
import numpy as np
import math
import loc_record
import feature_generation
import datetime
import time
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from sklearn.preprocessing import MinMaxScaler


# 方位角特征提取
def add_degree_feature(path1, path2):
    user_list = loc_record.curdir_file(path1)
    for user in user_list:
        path = path1 + os.path.sep + user
        data1 = pd.read_csv(open(path))
        data2 = pd.read_csv(open(path))

        a = np.zeros(len(data2))
        degree_col = pd.DataFrame(a, columns=['degree_fea'])

        for j in range(0, len(data2)):
            degree = data2.iloc[j]['degree']
            if degree == 0:
                degree_col.iloc[j] = 0
            elif (0 < degree and degree < 22.5) or (337.5 < degree):
                degree_col.iloc[j] = 1
            else:
                degree_col.iloc[j] = math.ceil((degree - 22.5) / 45)

        data1['degree_fea'] = degree_col


        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user
        data1.to_csv(new_path, index=False, encoding='gbk')
        print(user + ' Degree_fea done')

# 时间特征提取
def add_time_feature(path1, path2,time_rage):

    user_list = loc_record.curdir_file(path1)

    for user in user_list:
        path = path1 + os.path.sep + user
        data1 = pd.read_csv(open(path))
        data2 = pd.read_csv(open(path))
        data2['date_string'] = pd.to_datetime(data2['date_string'], format='%Y-%m-%d')
        data2['arrive_time'] = pd.to_datetime(data2['arrive_time'], format='%H:%M:%S')
        #data2['arrive_time'] = datetime.datetime.strptime(data2.iloc[1]['arrive_time'], "%H:%M:%S")
        data2['leave_time'] = pd.to_datetime(data2['leave_time'], format='%H:%M:%S')

        a = np.zeros(len(data2))
        ari_time_col = pd.DataFrame(a, columns=['arrive_time_fea'])
        b = np.zeros(len(data2))
        lea_time_col = pd.DataFrame(b, columns=['leave_time_fea'])

        t0 = pd.to_datetime('0:0:0', format='%H:%M:%S')

        for j in range(0, len(data2)):

            arrive_time = data2.iloc[j]['arrive_time']
            leave_time = data2.iloc[j]['leave_time']

            #计算时间差
            ta = (arrive_time - t0 ).seconds
            tl = (leave_time - t0 ).seconds

            #时间划分，向上取整
            ari_time_col.iloc[j] = math.ceil(ta/(time_rage*60))
            lea_time_col.iloc[j] = math.ceil(tl/(time_rage*60))

        #print(degree_col)

        data1['arrive_time_fea'] = ari_time_col
        data1['leave_time_fea'] = lea_time_col


        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user
        data1.to_csv(new_path, index=False, encoding='gbk')
        print(user + ' Time_fea_Data done')

# 地点整合
def integrate_poi(path1, path2):

    user_list = loc_record.curdir_file(path1)

    for user in user_list:
        path = path1 + os.path.sep + user
        data = pd.read_csv(open(path))

        a = np.zeros(len(data))
        poi_col = pd.DataFrame(a, columns=['poi'])
        not_city_index = []


        for i in range(0,len(data)):

            city = data.iloc[i]['city']

            #如果不在北京市
            if city != '北京市':
                not_city_index.append(i)

        for j in range(0,len(data)):

            if j in not_city_index :
                continue
            else:
                street = data.iloc[j]['street']
                poiReg_name = data.iloc[j]['poiReg_name']

                #如果区域范围不存在，则用街道表示
                if poiReg_name is np.nan:
                    poi_col.iloc[j] = street

                #如果区域范围名存在，则表示为区域范围名
                else:
                    poi_col.iloc[j] = poiReg_name

        data['poi'] = poi_col

        new_data = data.drop(not_city_index)

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user
        new_data.to_csv(new_path, index=False, encoding='gbk')
        print(user + ' poi_Data done')

'''
# 读取原始csv文件，将同天轨迹放在一行，poi放入txt中
def poi_transform(path1,path2):

    user_list = loc_record.curdir_file(path1)

    for user in user_list:

        path = path1 + os.path.sep + user
        data = pd.read_csv(open(path))

        all_traj = []
        oneday_traj = ''

        #遍历提取poi
        for j in range(0,len(data)-1):
            #如果不是最后一条记录
            if j != len(data) :
                #如果是同一天的，放进同一条行
                if data.iloc[j]['date_string'] == data.iloc[j+1]['date_string'] :
                    if data.iloc[j]['poi'] is np.nan:
                        #print(data.iloc[j])
                        continue
                    else:
                    #oneday_traj.append(data.iloc[j]['poi'])
                        poi =  data.iloc[j]['poi'] + ' '
                        oneday_traj = oneday_traj + poi
                else:
                    all_traj.append(oneday_traj)
                    oneday_traj =''
            else:
                poi = data.iloc[j]['poi'] + ' '
                oneday_traj = oneday_traj + poi
                all_traj.append(oneday_traj)

        #print(all_traj)

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user.strip('.csv') + '.txt'
        with open(new_path, 'a',encoding='utf-8') as f:
            for lines in all_traj:
                f.write(lines)
                f.write('\n')

        print(user.strip('.csv') + ' poi_transform done')
'''
# 读取原始csv文件，将同天轨迹放在一行，poi放入txt中（修改版，每条轨迹最后一个点保留）
def poi_transform(path1,path2):

    user_list = loc_record.curdir_file(path1)

    for user in user_list:

        path = path1 + os.path.sep + user
        data = pd.read_csv(open(path))

        all_traj = []
        j = 0

        #遍历提取poi
        while j < len(data):

            '''
            #如果是最后一条记录
            if j == len(data)-1 :

                if data.iloc[j]['date_string'] == data.iloc[j-1]['date_string'] :
                    poi = data.iloc[j]['poi'] + ' '
                    oneday_traj = oneday_traj + poi
                    all_traj.append(oneday_traj)
                    break
                else:
                    oneday_traj = data.iloc[j]['date_string']
                    all_traj.append(oneday_traj)
                    break
            
            # 如果不是最后一条记录
            else:
            '''
            # 如果当前记录poi为空
            if data.iloc[j]['poi'] is np.nan:
                j = j+1
                continue
            # 如果当前记录poi不为空
            else:
                oneday_traj = data.iloc[j]['poi']

                #与后续数据对比
                for i in range(1, len(data)+1):

                    #如果超过数据长度
                    if j+i >= len(data):
                        all_traj.append(oneday_traj)
                        j = j + i
                        break
                    #如果在数据长度内
                    else:
                        # 如果是同一天的，放进同一条行
                        if data.iloc[j]['date_string'] == data.iloc[j + i]['date_string']:
                            #判断是否为空
                            if data.iloc[j + i]['poi'] is np.nan:
                                continue
                            #不为空与当前记录 字符串 合并
                            else:
                                # oneday_traj.append(data.iloc[j]['poi'])
                                poi = ' ' + data.iloc[j + i]['poi']
                                oneday_traj = oneday_traj + poi
                        # 如果不是同一天的，跳出i的循环重新开始比较
                        else:
                            all_traj.append(oneday_traj)
                            j = j + i
                            break


        #print(all_traj)

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user.strip('.csv') + '.txt'
        with open(new_path, 'a',encoding='utf-8') as f:
            for lines in all_traj:
                f.write(lines)
                f.write('\n')

        print(user.strip('.csv') + ' poi_transform done')
'''
# 读取原始csv文件，将同天轨迹放在一行，poi放入txt中
def poi_transform(path1,path2):

    user_list = loc_record.curdir_file(path1)

    for user in user_list:

        path = path1 + os.path.sep + user
        data = pd.read_csv(open(path))

        all_traj = []
        oneday_traj = ''

        #遍历提取poi
        for j in range(0,len(data)-1):
            #如果不是最后一条记录
            if j != len(data) :

                # 如果当前记录poi为空
                if data.iloc[j]['poi'] is np.nan:
                    continue
                # 如果当前记录poi不为空
                else:
                    if oneday_traj == '':
                        oneday_traj = data.iloc[j]['poi']

                #如果是同一天的，放进同一条行
                if data.iloc[j]['date_string'] == data.iloc[j+1]['date_string'] :
                    if data.iloc[j+1]['poi'] is np.nan:
                        #print(data.iloc[j])
                        continue
                    else:
                    #oneday_traj.append(data.iloc[j]['poi'])
                        poi =' ' + data.iloc[j + 1]['poi']
                        oneday_traj = oneday_traj + poi
                else:
                    all_traj.append(oneday_traj)
                    oneday_traj =''
                    print('同一天：' + oneday_traj)

            # 如果是最后一条记录
            else:
                if data.iloc[j]['date_string'] == data.iloc[j - 1]['date_string']:
                    poi = data.iloc[j]['poi'] + ' '
                    oneday_traj = oneday_traj + poi
                    all_traj.append(oneday_traj)
                else:
                    oneday_traj = data.iloc[j]['date_string']
                    all_traj.append(oneday_traj)


        #print(all_traj)

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user.strip('.csv') + '.txt'
        with open(new_path, 'a',encoding='utf-8') as f:
            for lines in all_traj:
                f.write(lines)
                f.write('\n')

        print(user.strip('.csv') + ' poi_transform done')
'''

# word2vec的位置特征向量提取
def get_poi_model(path1,path2,model_name):

    sentences = PathLineSentences(path1)
    #model = Word2Vec(sentences, size=100, iter=10, min_count=20)
    model = Word2Vec(sentences, sg=1, size=10, window=5, min_count=0, hs=1, negative=3, sample=0.001)
    ''''
    · sentences：可以是一个list，对于大语料集，建议使用BrownCorpus,Text8Corpus或LineSentence构建。
    · sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    · size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    · window：表示当前词与预测词在一个句子中的最大距离是多少，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）。
    · alpha: 是学习速率
    · seed：用于随机数发生器。与初始化词向量有关。
    · min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    · max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
    · sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
    · workers参数控制训练的并行数。此参数只有在安装了Cpython后才有效，否则只能使用单核。
    · hs: 如果为1则会采用hierarchical softmax技巧。如果设置为0（defaut），则negative sampling负采样会被使用。
    · negative: 如果>0,则会采用negativesamping，用于设置多少个noise words
    · cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defaut）则采用均值。只有使用CBOW的时候才起作用。
    · hashfxn： hash函数来初始化权重。默认使用python的hash函数
    · iter： 迭代次数，默认为5
    · trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
    · sorted_vocab： 如果为1（defaut），则在分配word index 的时候会先对单词基于频率降序排序。
    · batch_words：每一批的传递给线程的单词的数量，默认为10000

    '''
    if not os.path.exists(path2):  # 检验给出的路径是否存在
        os.makedirs(path2)
    save_model_name = path2 + os.path.sep + model_name + '.model'
    model.save(save_model_name)
    model.wv.save_word2vec_format(save_model_name + ".bin", binary=True)
    print('model done')

    #with open(r'F:\Data\cutWords_list\cutWords_list.txt') as file:
        #cutWords_list = [k.split() for k in file.readlines()]
        #print(cutWords_list)
        #word2vec_model = Word2Vec(cutWords_list, size=100, iter=10, min_count=20)



def use_model(path1,model_name,word):
    model_path = path1 + os.path.sep + model_name + '.model'
    model = Word2Vec.load(model_path)

    #y1 = model.wv.most_similar(u'清华大学',topn=10)
    try:
        y2 = model[word]
        print(word,':',y2)
    except KeyError:
        print(word,"not in vocabulary")
        y2 = 0

    #y3 = model.most_similar(positive=[u'女人', u'先生'], negative=[u'男人'], topn=1)
    #y3 = model.similarity("清华大学", "北京大学")

    return y2

    #print(y1)
    #print(y2)
    #print(y3)

#各种特征向量整合
def feature_represent(path1, path2,poi_model_path):
    # load dataset
    user_list = loc_record.curdir_file(path1)

    for user in user_list:
        path = path1 + os.path.sep + user
        data = pd.read_csv(open(path))

        # a = np.zeros(len(data))
        # poi_col = pd.DataFrame(a, columns=['poi(t)'])
        # b = np.zeros(len(data))
        # poi_next_col = pd.DataFrame(b, columns=['poi(t+1)'])
        poi_col = []
        poi_next_col = []

        not_poi = []

        model = Word2Vec.load(poi_model_path)

        # 获取poi的词向量
        for i in range(0, len(data)):
            # 如果不是末尾
            if i != len(data) - 1:
                # 如果是同一天
                if data.iloc[i]['date_string'] == data.iloc[i + 1]['date_string']:

                    # 如果poi和下一个poi存在，用模型得到向量
                    if (data.iloc[i]['poi'] in model.wv.vocab) and (data.iloc[i + 1]['poi'] in model.wv.vocab):
                        poi_fea = model[data.iloc[i]['poi']]
                        poi_next_fea = model[data.iloc[i + 1]['poi']]
                        # poi_col.iloc[i] = poi_fea
                        # poi_next_col.iloc[i+1] = poi_next_fea
                        poi_col.append(poi_fea)
                        poi_next_col.append(poi_next_fea)

                    # 不存在
                    else:
                        not_poi.append(i)

                # 如果不是同一天
                else:
                    if data.iloc[i]['poi'] in model.wv.vocab:
                        poi_fea = model[data.iloc[i]['poi']]
                        # poi_col.iloc[i] = poi_fea
                        # poi_next_col.iloc[i + 1] = poi_fea
                        poi_col.append(poi_fea)
                        poi_next_col.append(poi_fea)
                    else:
                        not_poi.append(i)
            # 如果是末尾
            else:
                if data.iloc[i]['poi'] in model.wv.vocab:
                    poi_fea = model[data.iloc[i]['poi']]
                    # poi_col.iloc[i] = poi_fea
                    # poi_next_col.iloc[i + 1] = poi_fea
                    poi_col.append(poi_fea)
                    poi_next_col.append(poi_fea)
                else:
                    not_poi.append(i)

        # 去除无效的poi特征的行
        data_drop = data.drop(not_poi)
        new_data_drop = pd.DataFrame(
            columns=['lat', 'lon', 'date_string', 'arrive_time', 'leave_time',
                    'time_diff', 'degree', 'lat_baidu', 'lon_baidu', 'city', 'street',
                    'poiReg_name', 'sematic_descrip', 'arrive_time_fea', 'leave_time_fea',
                    'degree_fea', 'poi'])

        new_data_drop = new_data_drop.append(data_drop, ignore_index=True)
        poi_fea_data = pd.DataFrame(poi_col)

        # 合并有效数据及t时刻poi特征
        new_data1 = pd.concat([new_data_drop, poi_fea_data], axis=1)

        # 添加其他维度特征
        # arrive_time_values = data[:,12:15].values
        arrive_time_values = new_data_drop['arrive_time_fea'].values
        leave_time_values = new_data_drop['leave_time_fea'].values
        time_diff_values = new_data_drop['time_diff'].values
        degree_fea_values = new_data_drop['degree_fea'].values

        #归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_arrive_time = scaler.fit_transform(arrive_time_values)
        scaled_leave_time = scaler.fit_transform(leave_time_values)
        scaled_time_diff = scaler.fit_transform(time_diff_values)
        scaled_degree_fea = scaler.fit_transform(degree_fea_values)

        arrive_time_col = pd.DataFrame(scaled_arrive_time, columns=['arrive_time_fea(t)'])
        leave_time_col = pd.DataFrame(scaled_leave_time, columns=['leave_time_fea(t)'])
        time_diff_col = pd.DataFrame(scaled_time_diff, columns=['time_diff(t)'])
        degree_fea_col = pd.DataFrame(scaled_degree_fea, columns=['degree_fea(t)'])

        new_data1['arrive_time_fea(t)'] = arrive_time_col
        new_data1['leave_time_fea(t)'] = leave_time_col
        new_data1['time_diff(t)'] = time_diff_col
        new_data1['degree_fea(t)'] = degree_fea_col

        # t+1时刻的poi特征向量
        poi_next_fea_data = pd.DataFrame(poi_next_col)
        new_data = pd.concat([new_data1, poi_next_fea_data], axis=1)

        # print(new_data)
        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user
        new_data.to_csv(new_path, index=False, encoding='gbk')
        print(user + ' Feature_Data done')

# 时间特征提取
#add_time_feature(r'F:\GPS\Clusters_Data\Semantic_Data1',r'F:\GPS\Feature_Data\Time_fea_Data1',10)#时间划分

# 方位角特征提取
#add_degree_feature(r'F:\GPS\Feature_Data\Time_fea_Data',r'F:\GPS\Feature_Data\Degree_fea_Data')

# poi地点整合，如果区域范围名存在，则表示为区域范围名，不存在，则用街道表示
#integrate_poi(r'F:\GPS\Feature_Data\Degree_fea_Data',r'F:\GPS\Feature_Data\poi_Data')

#feature_generation.screen_user(r'F:\GPS\Feature_Data\poi_Data1', r'F:\GPS\Feature_Data\Effica_Data1_v1', 5, 20)

# 读取原始csv文件，将同天轨迹放在一行，poi放入txt中，为word2vec做准备
#poi_transform(r'F:\GPS\Feature_Data\Effica_Data_v1',r'F:\GPS\Feature_Data\poi_trans_Data')#txt是往下继续写，不是重写

# word2vec的位置特征向量提取
#get_poi_model(r'F:\GPS\Feature_Data\poi_trans_Data',r'F:\GPS\Feature_Data\poi_model_Data','alluser_model_10')#model_name
#use_model(r'F:\GPS\Feature_Data\poi_model_Data','alluser_model_10','八达岭野生动物世界')

# 各种特征向量整合
#feature_represent(r'F:\GPS\Feature_Data\Effica_Data_v1',r'F:\GPS\Feature_Data\Feature_Data',r'F:\GPS\Feature_Data\poi_model_Data\alluser_model_10.model')

#feature_generation.screen_user(r'F:\GPS\Feature_Data\Feature_Data', r'F:\GPS\Feature_Data\Effica_Feature_Data', 5, 20)