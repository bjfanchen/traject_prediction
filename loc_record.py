import os
import csv
import datetime

# 遍历指定目录，显示目录下的所有文件名
def curdir_file(dir_path):
    curdir_list = os.listdir(dir_path) #返回指定目录下的所有文件和目录名
    return curdir_list

#读取txt文件
def read_file(file_path):
    file = open(file_path,'r')
    content_list = file.readlines()
    file.close()
    return content_list

def user_list():
    user_list = curdir_file(r'F:\GPS\Data')
    return user_list

#去除开头文字
def remove_the_begin(path1,path2):
    user_list = curdir_file(path1)
    for i in user_list:
        old_file_path = path1 + os.path.sep + i + os.path.sep + 'Trajectory'
        old_file_list = curdir_file(old_file_path)
        for j in old_file_list :
            old_path = old_file_path + os.path.sep + j
            old_file = open(old_path, 'r')
            new_file_path = path2 + os.path.sep + i
            if not os.path.exists(new_file_path):#检验给出的路径是否存在
                os.makedirs(new_file_path)
            new_path = new_file_path + os.path.sep + j
            new_file = open(new_path, 'w')
            alllines = old_file.readlines()
            lines = alllines[6:]
            for line in lines:
                new_file.write(line)
            old_file.close()
            new_file.close()
        print(i + 'changes done')

#将单人轨迹合并
def merge_file(path1,path2):
    user_list = curdir_file(path1)
    for i in user_list:
        old_file_path = path1 + os.path.sep + i
        old_file_list = curdir_file(old_file_path)
        file_list = []
        for j in old_file_list :
            old_path = old_file_path + os.path.sep + j
            old_file = open(old_path, 'r')
            file_list.extend(old_file.readlines())
            old_file.close()

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + i + '.txt'
        new_file = open(new_path, 'w')
        for file in file_list:
            new_file.write(file)
        new_file.close()
        print( i + ' merge done')

#保存为csv格式
def save_csv(path1,path2):
    user_list = curdir_file(path1)
    for i in user_list:
        old_file_path = path1 + os.path.sep + i
        #in_txt = csv.reader(open(old_file_path, "r"), delimiter=',',, escapechar = '\n')
        file_list = []
        old_file = open(old_file_path, 'r')
        for line in old_file.readlines():
            file_list.append(line.replace('\n','').split(','))
        old_file.close()

        to_csv_list = []
        for j in file_list:
            format_conversion = []
            format_conversion.append(float(j[0]))
            format_conversion.append(float(j[1]))
            format_conversion.append(int(j[2]))
            format_conversion.append(float(j[3]))
            format_conversion.append(float(j[4]))
            format_conversion.append(datetime.datetime.strptime(j[5],'%Y-%m-%d').date())
            format_conversion.append(datetime.datetime.strptime(j[6],'%H:%M:%S').time())
            to_csv_list.append(format_conversion)

        if not os.path.exists(path2):  # 检验给出的路径是否存在
            os.makedirs(path2)
        new_path = path2 + os.path.sep + i.strip('.txt') + '.csv'
        new_file = open(new_path, 'w',newline='')
        myWriter = csv.writer(new_file,dialect='excel');
        myWriter.writerow(['lat', 'lon', '0','alt','date','data_string','time_string']);
        myWriter.writerows(to_csv_list);
        new_file.close()
        print(i.strip('.txt') + ' save csv done')

#remove_the_begin(r'F:\GPS\Data',r'F:\GPS\Only_Data')  #去除开头
#merge_file(r'F:\GPS\Only_Data',r'F:\GPS\User_Data')  #将单人轨迹合并
#save_csv(r'F:\GPS\User_Data',r'F:\GPS\User_Data_csv_format') #保存为csv格式
''''
old_file=open(r'F:\GPS\Data\000\Trajectory\20081023025304.txt',"r")
new_file=open(r'F:\GPS\Only_Data\000\20081023025304.txt',"w")
alllines = old_file.readlines()
lines=alllines[6:]
for line in lines:
   new_file.write(line)
old_file.close()
new_file.close()
'''

