''''
# -*- coding: utf-8 -*-
# 第一行必须有，否则报中文字符非ascii码错误
from urllib import parse
import hashlib

# 以get请求为例http://api.map.baidu.com/geocoder/v2/?address=百度大厦&output=json&ak=yourak
queryStr = '/geocoder/v2/?address=百度大厦&output=json&ak=bmyiBN8W1wwEdLZPNz9B5AabXCXyqKeP'

# 对queryStr进行转码，safe内的保留字符不转换
encodedStr = parse.quote(queryStr, safe="/:=&?#+!$,;'@()*[]")

# 在最后直接追加上yoursk
rawStr = encodedStr + 'e6uwIgUHovU0BO5B0GZzj6MKQCkYcMXQ'

# md5计算出的sn值7de5a22212ffaa9e326444c75a58f9a0
# 最终合法请求url是http://api.map.baidu.com/geocoder/v2/?address=百度大厦&output=json&ak=yourak&sn=7de5a22212ffaa9e326444c75a58f9a0
print (hashlib.md5(parse.quote_plus(rawStr).encode("utf8")).hexdigest())
'''
import os
import time
from urllib import parse
import hashlib
import urllib.request
import json


import loc_record

#原始例子
def get_urt():
    # 以get请求为例http://api.map.baidu.com/geocoder/v2/?address=百度大厦&output=json&ak=你的ak
    queryStr = '/geocoder/v2/?address=百度大厦&output=json&ak=bmyiBN8W1wwEdLZPNz9B5AabXCXyqKeP'

    # 对queryStr进行转码，safe内的保留字符不转换
    encodedStr = parse.quote(queryStr, safe="/:=&?#+!$,;'@()*[]")

    # 在最后直接追加上yoursk
    rawStr = encodedStr + 'e6uwIgUHovU0BO5B0GZzj6MKQCkYcMXQ'

    # 计算sn
    sn = (hashlib.md5(parse.quote_plus(rawStr).encode("utf8")).hexdigest())

    # 由于URL里面含有中文，所以需要用parse.quote进行处理，然后返回最终可调用的url
    url = parse.quote("http://api.map.baidu.com" + queryStr + "&sn=" + sn, safe="/:=&?#+!$,;'@()*[]")

    print(url)

    req = urllib.request.urlopen(url)  # json格式的返回数据
    res = req.read().decode("utf-8")  # 将其他编码的字符串解码成unicode

    print(json.loads(res))

    return json.loads(res)


#轨迹语义化
#encoding=utf8  #编码

import json
import urllib.request
from urllib import parse
import hashlib
import pandas as pd
import numpy as np

#基于百度地图API下的经纬度信息来解析地理位置信息
def getlocation(lat,lon):

    lat = str(lat)
    lon = str(lon)

    ak = 'bmyiBN8W1wwEdLZPNz9B5AabXCXyqKeP'
    radius = str(500)
    queryStr = '/geocoder/v2/?location=' + lat + ',' + lon + '&coordtype=wgs84ll&output=json&pois=1'+ '&radius='+ radius +'&ak=' + ak

    # 对queryStr进行转码，safe内的保留字符不转换
    encodedStr = parse.quote(queryStr, safe="/:=&?#+!$,;'@()*[]")
    # 在最后直接追加上yoursk
    sk = 'e6uwIgUHovU0BO5B0GZzj6MKQCkYcMXQ'
    rawStr = encodedStr + sk

    # 计算sn
    sn = (hashlib.md5(parse.quote_plus(rawStr).encode("utf8")).hexdigest())

    # 若URL里面含有中文，需要用parse.quote进行处理，然后返回最终可调用的url
    url = parse.quote("http://api.map.baidu.com" + queryStr + "&sn=" + sn, safe="/:=&?#+!$,;'@()*[]")

    req = urllib.request.urlopen(url)  # json格式的返回数据
    res = req.read().decode("utf-8")  # 将其他编码的字符串解码成unicode
    return json.loads(res)

#json序列化解析数据(lat:纬度，lng:经度)
def jsonFormat(lat,lon):

    json_str = getlocation(lat,lon)

    dictjson={}#声明一个字典
    #get()获取json里面的数据
    jsonResult = json_str.get('result')
    if jsonResult is not None:
        #转换成百度坐标的经纬度
        location = jsonResult.get('location')
        if location != []:
            lat_baidu = location.get('lat')
            lon_baidu = location.get('lng')
        else:
            lat_baidu = ''
            lon_baidu = ''

        #结构化地址信息
        formatted_address = jsonResult.get('formatted_address')

        address = jsonResult.get('addressComponent')
        if address != []:
            #城市
            city = address.get('city')
            #街道
            street = address.get('street')
        else:
            city = ''
            street = ''
        #归属区域面名称
        poiRegions = jsonResult.get('poiRegions')
        if poiRegions != []:
            poiReg_list = poiRegions[0]
            poiReg_name = poiReg_list.get('name')
        else:
            poiReg_name = ''

        #当前位置结合POI的语义化结果描述
        sematic_descrip = jsonResult.get('sematic_description')

        #把获取到的值，添加到字典里（添加）
        dictjson['lat_baidu'] = lat_baidu
        dictjson['lon_baidu'] = lon_baidu
        dictjson['city'] = city
        dictjson['street'] = street
        dictjson['poiReg_name'] = poiReg_name
        dictjson['sematic_descrip']=sematic_descrip

        #print(dictjson)
    return dictjson

def semantic_trasform(path_in,path_out):

    start = time.time()
    user_list = loc_record.curdir_file(path_in)

    for user in user_list:

        #数据读取
        path_in1 = path_in + os.path.sep + user
        gps_data = pd.read_csv(open(path_in1))

        #创建数据列
        a = np.zeros(len(gps_data))
        lat_baidu = pd.DataFrame(a, columns=['lat_baidu'])
        b = np.zeros(len(gps_data))
        lon_baidu = pd.DataFrame(b, columns=['lon_baidu'])
        city = pd.DataFrame(a, columns=['city'])
        street = pd.DataFrame(a, columns=['street'])
        poiReg_name = pd.DataFrame(a, columns=['poiReg_name'])
        sematic_descrip = pd.DataFrame(a, columns=['sematic_descrip'])

        #读取语义转换数据
        for i in range(0,len(gps_data)):

            lat = gps_data.iloc[i]['lat']
            lon = gps_data.iloc[i]['lon']

            sem_result = jsonFormat(lat,lon)
            if sem_result == {} :
                continue
            else:
                print(sem_result)
                lat_baidu.iloc[i] = sem_result['lat_baidu']
                lon_baidu.iloc[i] = sem_result['lon_baidu']
                city.iloc[i] = sem_result['city']
                street.iloc[i] = sem_result['street']
                poiReg_name.iloc[i] = sem_result['poiReg_name']
                sematic_descrip.iloc[i] = sem_result['sematic_descrip']

        #添加到原始数据后面
        gps_data['lat_baidu'] = lat_baidu
        gps_data['lon_baidu'] = lon_baidu
        gps_data['city'] = city
        gps_data['street'] = street
        gps_data['poiReg_name'] = poiReg_name
        gps_data['sematic_descrip'] = sematic_descrip
        #print(gps_data)

        if not os.path.exists(path_out):  # 检验给出的路径是否存在
            os.makedirs(path_out)
        new_path = path_out + os.path.sep + user
        gps_data.to_csv(new_path, index=False, encoding='gbk')
        print(user + ' Semantic_Data done')

        end = time.time()
        print(end - start)

#地点语义化
#semantic_trasform('F:\GPS\Feature_Data\Effica_again_Data_v5',r'F:\GPS\Feature_Data\Semantic_Data_v5')
#jsonFormat(39.90613,116.375697)