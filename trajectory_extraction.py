import os
import pandas as pd
import loc_record
import csv
from math import *
import datetime
from geopy.distance import geodesic

#提取城市数据
def city_data(path1,path2,lat_range1,lat_range2,lon_range1,lon_range2):
    file_list = loc_record.curdir_file(path1)
    for i in file_list:
        #new_data = []
        path = path1 + os.path.sep + i
        data = pd.read_csv(open(path))
        new_data = data[(data['lat']>lat_range1 ) & (data['lat']<lat_range2 ) & (data['lon']>lon_range1) & (data['lon']<lon_range2)]
        print(data['data_string'].dtype)

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + i
        new_data.to_csv(new_path, index=False, encoding='gbk')
        print(i + 'Beijing_Data save csv done')
        ''''
        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + i
        new_file = open(new_path, 'w', newline='')
        myWriter = csv.writer(new_file, dialect='excel');
        myWriter.writerow(['lat', 'lon', '0', 'alt', 'timestamp', 'data', 'time']);
        myWriter.writerows(file_list);
        new_file.close()
        print(i + ' filtrate done')
        '''''

def rad(d):
    pi = 3.1415926535898
    return d*pi/180

# haversine公式
def get_distance(lat1,lon1,lat2,lon2): #date_type : float,米
    #lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])  # 经纬度转换成弧度
    rlat1 = rad(lat1)
    rlon1 = rad(lon1)
    rlat2 = rad(lat2)
    rlon2 = rad(lon2)

    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1

    R = 6378.137 # 地球平均半径,km
    a = sin(dlat/2)**2 + cos(rlat1) * cos(rlat2) * sin(dlon/2)**2
    distance = 2 * R * asin(sqrt( a )) * 1000
    #print(distance)
    return distance
#print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).m) #计算两个坐标直线距离
#print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).km) #计算两个坐标直线距离

def get_time_interval(start_time,end_time): #date_type : datetime，秒
    time_interval = (end_time - start_time).seconds
    return time_interval

def get_day_interval(start_time,end_time): #date_type : datetime，天
    time_interval = (end_time - start_time).days
    return time_interval

def get_speed(lat1,lon1,lat2,lon2,start_time,end_time):#date_type : ，米/秒
    distance = get_distance(lat1,lon1,lat2,lon2)
    time = get_time_interval(start_time,end_time)
    if time == 0 :
        return 0
    else:
        speed = distance/time
        return speed
#预处理：去除速度大于300km/h的后继点
def preprocessing(path1,path2,speed_range):
    file_list = loc_record.curdir_file(path1)
    for i in file_list:
        path = path1 + os.path.sep + i
        data1 = pd.read_csv(open(path))
        data2 = pd.read_csv(open(path))
        data2['data_string'] = pd.to_datetime(data2['data_string'],format = '%Y-%m-%d')
        data2['time_string'] = pd.to_datetime(data2['time_string'],format = '%H:%M:%S')

        new_index = []
        for j in range(0,len(data2)-1):
            day_interval = get_day_interval(data2.iloc[j]['data_string'],data2.iloc[j+1]['data_string'])
            if day_interval == 0 :
                speed = get_speed(data2.iloc[j]['lat'],data2.iloc[j]['lon'],data2.iloc[j+1]['lat'],data2.iloc[j+1]['lon'],data2.iloc[j]['time_string'],data2.iloc[j+1]['time_string'])
                if speed == 0:
                    continue
                #print(speed)
                #print(j)
                if speed > ((speed_range*1000) / (1*60*60)) :
                    #print(j)
                    new_index.append(j)
        new_data = data1.drop(new_index)

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + i
        new_data.to_csv(new_path, index=False, encoding='gbk')
        print(i + ' Prepro_Data save csv done')

        ''''
        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + i
        new_file = open(new_path, 'w', newline='')
        myWriter = csv.writer(new_file, dialect='excel');
        myWriter.writerow(['lat', 'lon', '0', 'alt', 'timestamp', 'data', 'time']);
        myWriter.writerows(file_list);
        new_file.close()
        print(i + ' filtrate done')
        '''''

# 轨迹按时间和距离提取，10min，500m
def trajectory_extra(path1, path2,time_interval_range,distance_interval_range):
    user_list = loc_record.curdir_file(path1)
    for i in user_list:
        path = path1 + os.path.sep + i
        data1 = pd.read_csv(open(path))
        data2 = pd.read_csv(open(path))
        data2['data_string'] = pd.to_datetime(data2['data_string'], format='%Y-%m-%d')
        data2['time_string'] = pd.to_datetime(data2['time_string'], format='%H:%M:%S')

        new_index = [0]
        begin = 0
        j = 1
        while begin + j <= (len(data2) - 1) :
            day_interval = get_day_interval(data2.iloc[begin]['data_string'], data2.iloc[begin+j]['data_string'])
            if day_interval == 0:
                time_interval = get_time_interval(data2.iloc[begin]['time_string'], data2.iloc[begin+j]['time_string'])
                distance_interval = get_distance(data2.iloc[begin]['lat'], data2.iloc[begin]['lon'], data2.iloc[begin+j]['lat'],data2.iloc[begin+j]['lon'])
                if time_interval >= (time_interval_range * 60) :
                    new_index.append(begin + j)
                    begin = begin + j
                    j = 0
                if distance_interval >= distance_interval_range :
                    if (begin + j) not in new_index:
                        new_index.append(begin + j)
                        begin = begin + j
                        j = 0
            if day_interval != 0 :
                begin = begin + j
                j = 0
            j = j + 1
        if len(new_index) != 1 :
            new_data = data1.iloc[new_index, :]
        if len(new_index) == 1 :
            new_data = pd.DataFrame(columns=['lat', 'lon', '0','alt','date','data_string','time_string'])
        #print(new_index)

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + i
        new_data.to_csv(new_path, index=False, encoding='gbk')
        print(i + ' Extra_Data done')

#获取轨迹及轨迹点数目
def get_trace_count_and_point(path1):
    user_list = loc_record.curdir_file(path1)
    trace_count = 0 #轨迹数
    trace_point = 0 #轨迹点数
    for i in user_list:
        path = path1 + os.path.sep + i
        data = pd.read_csv(open(path))
        trace_count = trace_count + data['data_string'].nunique()
        trace_point = trace_point + len(data)
    print("轨迹数：", trace_count, "轨迹点数：", trace_point)
    return trace_count,trace_point
#get_distance(34.2676434736,108.920156846,34.2683182934,108.967080573)

#提取北京地区的数据
#city_data(r'F:\GPS\User_Data_csv_format',r'F:\GPS\Pre_Data\Beijing_Data',39.26,41.6,115.25,117.4))
#预处理：去除速度大于300km/h的后继点
#preprocessing(r'F:\GPS\Pre_Data\Beijing_Data',r'F:\GPS\Pre_Data\Prepro_Data,300)
# 轨迹按时间和距离提取，10min，500m
#trajectory_extra(r'F:\GPS\Pre_Data\Prepro_Data',r'F:\GPS\Pre_Data\Extra_Data',10,500)

# 获取轨迹及轨迹点数目
#get_trace_count_and_point(r'F:\GPS\Pre_Data\Prepro_Data')
#get_trace_count_and_point(r'F:\GPS\Pre_Data\Extra_Data')