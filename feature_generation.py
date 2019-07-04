import os
import pandas as pd
import loc_record
import numpy as np
import trajectory_extraction
from math import *
import datetime
from geographiclib.geodesic import Geodesic
from geopy.distance import geodesic

#获取方位角
def get_degree(lat1,lon1,lat2,lon2):
    geodict = Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)
    new_geodict = geodict['azi1']
    if new_geodict < 0 and new_geodict > -180 :
        new_geodict = (new_geodict + 360)%360
    if new_geodict == -90 :
        new_geodict = 270
    return new_geodict

#获取方位角差值
def get_degree_diff(degree1,degree2):
    degree_diff = abs(degree1-degree2)
    if degree_diff > 180 :
        if degree1 > degree2:
            degree_diff = 360-degree1+degree2
        else :
            degree_diff = 360 - degree2 + degree1
    return degree_diff

#筛选有效用户,traje_range为每条轨迹拥有多少个及以下点数要求,date_range为用户拥有多少条及以下，则剔除
def screen_user(path1,path2,traje_range,date_range):
    user_list = loc_record.curdir_file(path1)
    for i in user_list:
        path = path1 + os.path.sep + i
        data1 = pd.read_csv(open(path))
        #data1.rename(columns={'data_string':'date_string'},inplace=True)
        #data2 = pd.read_csv(open(path))
        #data2['data_string'] = pd.to_datetime(data2['data_string'], format='%Y-%m-%d')
        #data2['time_string'] = pd.to_datetime(data2['time_string'], format='%H:%M:%S')
        date_value = []

        distinct_values = data1['date_string'].unique()
        total_count = data1['date_string'].nunique()
        for j in distinct_values :
            repet_count = list(data1['date_string']).count(j)
            if repet_count <= traje_range :
                date_value.append(j)

        valid_date =  total_count - len(date_value)
        if valid_date <= date_range :
            continue
        else:
            new_data = data1[~data1['date_string'].isin(date_value)]
            #print(new_data)

            if not os.path.exists(path2):  # 检验给出的路径是否存在
                os.makedirs(path2)
            new_path = path2 + os.path.sep + i
            new_data.to_csv(new_path, index=False, encoding='gbk')
            print(i + ' Effica_Data done') # 有效数据


#添加方位角特征
def add_degree(path1,path2):
    user_list = loc_record.curdir_file(path1)
    for i in user_list:
        path = path1 + os.path.sep + i
        data1 = pd.read_csv(open(path))
        data2 = pd.read_csv(open(path))
        data2['date_string'] = pd.to_datetime(data2['date_string'], format='%Y-%m-%d')

        a = np.zeros(len(data2))
        degree_col = pd.DataFrame(a,columns=['degree'])
        degree_col.iloc[0] = 0

        for j in range(0,len(data2) - 1):
            day_interval = trajectory_extraction.get_day_interval(data2.iloc[j]['date_string'], data2.iloc[j+1]['date_string'])
            if day_interval == 0:
                degree = get_degree(data2.iloc[j]['lat'], data2.iloc[j]['lon'], data2.iloc[j+1]['lat'],data2.iloc[j+1]['lon'])
                degree_col.iloc[j+1] = degree
            else:
                degree_col.iloc[j + 1] = 0
        data1.insert(6,'degree',degree_col)


        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + i
        data1.to_csv(new_path, index=False, encoding='gbk')
        print(i + ' Degree_Data done')

#添加方位角变化差
def add_degree_diff(path1,path2):
    user_list = loc_record.curdir_file(path1)
    for i in user_list:
        path = path1 + os.path.sep + i
        data1 = pd.read_csv(open(path))
        data2 = pd.read_csv(open(path))
        data2['date_string'] = pd.to_datetime(data2['date_string'], format='%Y-%m-%d')
        data2['time_string'] = pd.to_datetime(data2['time_string'], format='%H:%M:%S')
        #data2 = pd.DataFrame(data1,columns=['lat', 'lon','alt','date_string','time_string'])
        #data2 = data1.copy()
        a = np.zeros(len(data2))
        degree_diff_col = pd.DataFrame(a,columns=['degree_diff'])
        degree_diff_col.iloc[0] = nan
        degree_diff_col.iloc[1] = nan

        for j in range(1,len(data2) - 1):
            day_interval = trajectory_extraction.get_day_interval(data2.iloc[j]['date_string'], data2.iloc[j+1]['date_string'])
            if day_interval == 0:
                degree_diff = get_degree_diff(data2.iloc[j]['degree'],data2.iloc[j+1]['degree'])
            #print(degree)
                degree_diff_col.iloc[j+1] = degree_diff
            #data2.iloc[j + 1,'degree'] = degree
            #print(degree_col)
            else:
                degree_diff_col.iloc[j + 1] = nan
        data1.insert(5,'degree_diff',degree_diff_col)
        #print(data2)

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + i
        data1.to_csv(new_path, index=False, encoding='gbk')
        print(i + ' Degree_diff_Data done')

# 添加轨迹点停留时间
def add_time_interval(path1, path2):

    user_list = loc_record.curdir_file(path1)
    for user in user_list:
        path = path1 + os.path.sep + user
        data1 = pd.read_csv(open(path))
        data2 = pd.read_csv(open(path))
        data2['date_string'] = pd.to_datetime(data2['date_string'], format='%Y-%m-%d')
        data2['arrive_time'] = pd.to_datetime(data2['arrive_time'], format='%H:%M:%S')
        data2['leave_time'] = pd.to_datetime(data2['leave_time'], format='%H:%M:%S')

        a = np.zeros(len(data2))
        time_interval_col = pd.DataFrame(a, columns=['time_interval'])


        for j in range(0, len(data2) - 1):

            if data2.iloc[j]['leave_time'] != 'None':
                time_interval = trajectory_extraction.get_time_interval(data2.iloc[j]['arrive_time'], data2.iloc[j]['leave_time'])
                time_interval_col.iloc[j] = time_interval

            else:
                time_interval_col.iloc[j] = nan

        data1.insert(5, 'time_diff', time_interval_col)


        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user
        data1.to_csv(new_path, index=False, encoding='gbk')
        print(user + ' Time_interval_Data done')


#获取轨迹及轨迹点数目(修改日期字段后)
def get_trace_count_and_point(path1):
    user_list = loc_record.curdir_file(path1)
    trace_count = 0 #轨迹数
    trace_point = 0 #轨迹点数
    for i in user_list:
        path = path1 + os.path.sep + i
        data = pd.read_csv(open(path))
        trace_count = trace_count + data['date_string'].nunique()
        trace_point = trace_point + len(data)
    print("轨迹数：",trace_count,"轨迹点数：",trace_point)
    return trace_count,trace_point

#获取轨迹及驻足点数目(修改日期字段后)
def get_trace_count_and_cluster(path1,id):
    user_list = loc_record.curdir_file(path1)
    trace_count = 0 #轨迹数
    trace_cluster = 0 #驻足点数
    for i in user_list:
        path = path1 + os.path.sep + i
        data = pd.read_csv(open(path))
        trace_count = trace_count + data['date_string'].nunique()
        trace_cluster = trace_count + data[id].nunique()
    print("轨迹数：",trace_count,"驻足点数：",trace_cluster)
    return trace_count,trace_cluster

#提取停留点 带时间窗的区域一致性
import datetime
from geopy.distance import geodesic

def get_stay_points(path1,path2,scale_factor,consistency_range,density_range,window_size):
    user_list = loc_record.curdir_file(path1)

    for user in user_list:
        path = path1 + os.path.sep + user
        userid = user.split('.')[0]
        data1 = pd.read_csv(open(path))
        data2 = pd.read_csv(open(path))
        data2['date_string'] = pd.to_datetime(data2['date_string'], format='%Y-%m-%d')
        data2['time_string'] = pd.to_datetime(data2['time_string'], format='%H:%M:%S')

        clusters_list = []
        a = np.zeros(len(data2))
        classified = pd.DataFrame(a, columns=['classified'])
        cluster_id = 1
        new_stay_point_data = pd.DataFrame(columns=['cluster_id', 'index_id', 'lat', 'lon', 'date_string', 'time_string'])


        for i in range(0,len(data2) - 1):
            if classified.iloc[i]['classified'] == 0 :
                classified.iloc[i]['classified'] = 1

                #区域一致性扩展算法
                clusters = []
                seeds = []
                points = []
                clusters.append(i)
                seeds.append(i)
                while seeds != [] :
                    seed = seeds[0]
                    begin = seed
                    time_window = 0
                    for j in range(begin + 1, len(data2) - 1):
                        day_interval = trajectory_extraction.get_day_interval(data2.iloc[begin]['date_string'],
                                                                              data2.iloc[j]['date_string'])
                        if day_interval == 0:

                            if classified.iloc[j]['classified'] == 0:

                                distance_interval = trajectory_extraction.get_distance(data2.iloc[begin]['lat'],
                                                                                   data2.iloc[begin]['lon'],
                                                                                   data2.iloc[j]['lat'],
                                                                                   data2.iloc[j]['lon'])

                                speed = trajectory_extraction.get_speed(data2.iloc[begin]['lat'], data2.iloc[begin]['lon'],
                                                                        data2.iloc[j]['lat'],
                                                                        data2.iloc[j]['lon'],
                                                                        data2.iloc[begin]['time_string'],
                                                                        data2.iloc[j]['time_string'])

                                consistency_value = exp(-(distance_interval / scale_factor) - speed)

                                if consistency_value >= consistency_range :

                                    seeds.append(j)
                                    clusters.append(j)
                                    classified.iloc[j]['classified'] = 1
                                    time_window = 0
                                else:
                                    time_window = time_window + 1
                            else:
                                time_window = 0

                            if time_window == window_size:
                                break

                        if day_interval != 0:
                            break

                    del(seeds[0])

                if len(clusters) >= density_range:
                    for index in clusters:
                        # new_data.append({'a':10,'b':11,'c':12,'d':13},ignore_index=True)
                        new_stay_point_data = new_stay_point_data.append({'cluster_id':cluster_id,'index_id':index,'lat':data1.iloc[index]['lat'],
                                          'lon':data1.iloc[index]['lon'],'date_string':data1.iloc[index]['date_string'],'time_string':data1.iloc[index]['time_string']}, ignore_index=True)
                    cluster_id = cluster_id + 1

        #print(new_stay_point_data)

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user
        new_stay_point_data.to_csv(new_path, index=False, encoding='gbk')
        print(user + ' Stay_point_Data done')
''''
                if len(clusters)>= density_range :
                    clusters_list.append(clusters)
                    #print(clusters)

        clusters_dict[userid] = clusters_list

    #print (clusters_dict)
    return clusters_dict
'''''

# 提取重要位置
def get_important_point(path1,path2,distance_range):
    user_list = loc_record.curdir_file(path1)

    for user in user_list:
        path = path1 + os.path.sep + user
        data1 = pd.read_csv(open(path))
        data2 = pd.read_csv(open(path))
        data2['date_string'] = pd.to_datetime(data2['date_string'], format='%Y-%m-%d')
        data2['time_string'] = pd.to_datetime(data2['time_string'], format='%H:%M:%S')

        impoint_id = 1
        new_impoint_data = pd.DataFrame(columns=['impoint_id','cluster_id','index_id','lat','lon','date_string','time_string'])
        cluster_id = data2['cluster_id'].unique()

        ''''
        lat_avge = data2.groupby(['cluster_id'])[['lat']].mean()
        leave_time = data2.groupby(['cluster_id'])[['time_string']].max()
        date = data2.groupby(['cluster_id'])[['date_string']].max()
        print(date)
        date = data2[data2['cluster_id'] == 1]['lat'].mean()
        print(date)
        '''''
        impoint_id_dict = {1:1}

        #按簇读取
        for i in cluster_id:

            #如果是同一天
            if data2[data2['cluster_id'] == i]['date_string'].max() == data2[data2['cluster_id'] == (i+1)]['date_string'].max() :

                #分别求两点到达和离开时间
                arrive_time1 = data2[data2['cluster_id'] == i]['time_string'].min()
                arrive_time2 = data2[data2['cluster_id'] == (i+1)]['time_string'].min()
                leave_time1 = data2[data2['cluster_id'] == i]['time_string'].max()
                leave_time2 = data2[data2['cluster_id'] == (i + 1)]['time_string'].max()

                #如果两点是时间包含关系
                if (arrive_time1 < arrive_time2 and leave_time1 > leave_time2) or (arrive_time1 > arrive_time2 and leave_time1 < leave_time2) :
                    #和前一个簇合并
                    impoint_id_dict[i+1] = impoint_id

                #如果两个点时间不包含,计算距离
                else:
                    lat1 = data2[data2['cluster_id'] == i]['lat'].mean()
                    lat2 = data2[data2['cluster_id'] == (i+1)]['lat'].mean()
                    lon1 = data2[data2['cluster_id'] == i]['lon'].mean()
                    lon2 = data2[data2['cluster_id'] == (i + 1)]['lon'].mean()

                    distance_interval = trajectory_extraction.get_distance(lat1,lon1,lat2,lon2)
                    #如果满足距离阈值，和前一个簇合并，否则新开一个
                    if distance_interval <= distance_range :
                        impoint_id_dict[i + 1] = impoint_id

                    else:
                        impoint_id = impoint_id + 1
                        impoint_id_dict[i + 1] = impoint_id

            else :
                impoint_id = impoint_id + 1
                impoint_id_dict[i + 1] = impoint_id

        #print(impoint_id_dict)

        #结果合并
        for i in cluster_id :
            #print(impoint_id_dict[i])
            stay_point_data = data1[data1['cluster_id'] == i]
            new_data = stay_point_data.reindex(
            columns=['impoint_id', 'cluster_id', 'index_id', 'lat', 'lon', 'date_string', 'time_string'], fill_value=impoint_id_dict[i])
            new_impoint_data = new_impoint_data.append(new_data,ignore_index=True)

        #结果输出至csv
        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user
        new_impoint_data.to_csv(new_path, index=False, encoding='gbk')
        print(user + ' Important_point_Data done')

# 聚类结果
def get_clusters(path1, path2):

    user_list = loc_record.curdir_file(path1)

    for user in user_list:

        path = path1 + os.path.sep + user
        data1 = pd.read_csv(open(path))
        data2 = pd.read_csv(open(path))
        data2['date_string'] = pd.to_datetime(data2['date_string'], format='%Y-%m-%d')
        data2['time_string'] = pd.to_datetime(data2['time_string'], format='%H:%M:%S')


        new_clusters_data = pd.DataFrame(
            columns=['impoint_id', 'lat', 'lon', 'date_string', 'arrive_time','leave_time','arrive_index','leave_index'])
        important_point_id = data1['impoint_id'].unique()

        ''''
        lat_avge = data1.groupby(['cluster_id'])[['lat']].mean()
        leave_time = data1.groupby(['cluster_id'])[['time_string']].max()
        date = data1.groupby(['cluster_id'])[['date_string']].max()
        print(pd.concat(lat_avge,leave_time,date),axis=1,ignore_index=True)
        '''
        # 结果合并
        for i in important_point_id:

            lat_aver = data1[data1['impoint_id'] == i]['lat'].mean()
            lon_aver = data1[data1['impoint_id'] == i]['lon'].mean()
            date = data1[data1['impoint_id'] == i]['date_string'].max()
            arrive_time = data1[data1['impoint_id'] == i]['time_string'].min()
            leave_time = data1[data1['impoint_id'] == i]['time_string'].max()
            arrive_index = data1[data1['impoint_id'] == i]['index_id'].min()
            leave_index = data1[data1['impoint_id'] == i]['index_id'].max()

            #new_impoint_data = new_impoint_data.append({'cluster_id': cluster_id, 'index_id': index, 'lat': data1.iloc[index]['lat'],
                                        #'lon': data1.iloc[index]['lon'],
                                        #'date_string': data1.iloc[index]['date_string'],
                                        #'time_string': data1.iloc[index]['time_string']}, ignore_index=True)
            new_clusters_data = new_clusters_data.append(
                {'impoint_id': i, 'lat':lat_aver, 'lon':lon_aver, 'date_string':date, 'arrive_time':arrive_time,'leave_time':leave_time,'arrive_index':arrive_index,'leave_index':leave_index}, ignore_index=True)
        #print(new_clusters_data)
        # 结果输出至csv
        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + user
        new_clusters_data.to_csv(new_path, index=False, encoding='gbk')
        print(user + ' Clusters_Data done')

#重要位置与移动点结合
def get_imp_movie_point(path_effica,path_clusters,out_path):
    user_list = loc_record.curdir_file(path_effica)

    for user in user_list:

        path_effica1 = path_effica + os.path.sep + user
        path_clusters1 = path_clusters + os.path.sep + user

        effica_data = pd.read_csv(open(path_effica1))
        clusters_data = pd.read_csv(open(path_clusters1))

        a = np.zeros(len(clusters_data))
        new_space_feature_data = pd.DataFrame(
            columns=['lat', 'lon', 'date_string', 'arrive_time', 'leave_time'])
        impoint_id = 1
        i = 0

        #for i in range(begin,len(effica_data)-1):
        while i < len(effica_data)-1 :

            ''''
            impoint_id_index_list = impoint_data[impoint_data.index_id == i].index.tolist()
            impoint_index_id = impoint_id_index_list[0]
            impoint_id = impoint_data.iloc[impoint_index_id]['impoint_id']
            print(type(impoint_id))
            '''
            #在兴趣点队列中
            if impoint_id <= len(clusters_data):

                #兴趣区域，这是用索引的，以后要修改

                arrive_index = int(clusters_data.iloc[impoint_id-1]['arrive_index'])
                leave_index = int(clusters_data.iloc[impoint_id-1]['leave_index'])


                # 如果轨迹点在兴趣区域
                if arrive_index <= i and i <= leave_index:

                    new_data = clusters_data[clusters_data['impoint_id'] == impoint_id][['lat', 'lon', 'date_string', 'arrive_time','leave_time']]
                    new_space_feature_data = new_space_feature_data.append(new_data, ignore_index=True)
                    impoint_id = impoint_id + 1
                    i = leave_index + 1


                #如果轨迹不在兴趣区域
                else:
                    lat = effica_data.iloc[i]['lat']
                    lon = effica_data.iloc[i]['lon']
                    date_string = effica_data.iloc[i]['date_string']
                    arrive_time = effica_data.iloc[i]['time_string']

                    #如果是同一天
                    if effica_data.iloc[i]['date_string'] == effica_data.iloc[i+1]['date_string'] :
                        leave_time =  effica_data.iloc[i+1]['time_string']

                    #如果是轨迹末尾，不是同一天
                    else:
                        leave_time = effica_data.iloc[i]['time_string']

                    #添加移动点
                    new_space_feature_data = new_space_feature_data.append(
                        {'lat': lat, 'lon': lon, 'date_string': date_string, 'arrive_time': arrive_time,
                        'leave_time': leave_time},
                        ignore_index=True)
                    i = i + 1

            #兴趣点末尾，以后要修改优化
            else:
                lat = effica_data.iloc[i]['lat']
                lon = effica_data.iloc[i]['lon']
                date_string = effica_data.iloc[i]['date_string']
                arrive_time = effica_data.iloc[i]['time_string']

                # 如果是同一天
                if effica_data.iloc[i]['date_string'] == effica_data.iloc[i + 1]['date_string']:
                    leave_time = effica_data.iloc[i + 1]['time_string']

                # 如果是轨迹末尾，不是同一天
                else:
                    leave_time = effica_data.iloc[i]['time_string']

                # 添加移动点
                new_space_feature_data = new_space_feature_data.append(
                    {'lat': lat, 'lon': lon, 'date_string': date_string, 'arrive_time': arrive_time,
                     'leave_time': leave_time},
                    ignore_index=True)
                i = i + 1

        if not os.path.exists(out_path):  # 检验给出的路径是否存在
            os.makedirs(out_path)
        new_path = out_path + os.path.sep + user
        new_space_feature_data.to_csv(new_path, index=False, encoding='gbk')
        print(user + ' Imp_movie_point_Data done')

# 提取停留点 带时间窗的区域一致性，计算经纬度平均值，到达离开时间
#,缩放因子1000米，一致性权值0.2，密度为3，时间窗为10
#get_stay_points(r'F:\GPS\Feature_Data\Effica_Data',r'F:\GPS\Feature_Data\Stay_point_Data', 1000, 0.2, 3,10)  # 先划分，后聚类最准

# 提取重要位置，如果满足距离阈值，和前一个簇合并，否则新开一个
#get_important_point(r'F:\GPS\Feature_Data\Stay_point_Data',r'F:\GPS\Feature_Data\Important_point_Data',1000)

# 重要位置聚类结果，经纬度平均值，到达离开时间
#get_clusters(r'F:\GPS\Feature_Data\Important_point_Data',r'F:\GPS\Feature_Data\Clusters_Data')

# 重要位置与移动点结合，移动点添加到达离开时间，离开时间为下一个点
#get_imp_movie_point(r'F:\GPS\Feature_Data\Effica_Data',r'F:\GPS\Feature_Data\Clusters_Data',r'F:\GPS\Feature_Data\Imp_movie_point_Data')

# 添加轨迹点停留时间
#add_time_interval(r'F:\GPS\Feature_Data\Space_feature_Data',r'F:\GPS\Feature_Data\Time_interval_Data')

#添加方位角特征
#add_degree(r'F:\GPS\Feature_Data\Time_interval_Data',r'F:\GPS\Feature_Data\Degree_Data')


#screen_user(r'F:\GPS\Feature_Data\poi_Data',r'F:\GPS\Feature_Data\Effica_Data',3,20)
#get_trace_count_and_point(r'F:\GPS\Feature_Data\Important_point_Data')
#get_trace_count_and_cluster(r'F:\GPS\Feature_Data\Stay_point_Data','cluster_id')
#get_trace_count_and_cluster(r'F:\GPS\Feature_Data\Important_point_Data','impoint_id')

#添加方位角变化差
#add_degree_diff(r'F:\GPS\Feature_Data\Degree_Data_oneday_degree',r'F:\GPS\Feature_Data\Degree_Data_oneday_degree_diff')
#get_trace_count_and_point(r'F:\GPS\Feature_Data\Effica_Data')

