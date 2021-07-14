#place this file inside the labels folder
#the results are saved in dataset_analysis_main.csv
#list of images is a space seperated string of text file names

import glob
import pandas as pd
l = glob.glob("*.txt")
#i = glob.glob("Images/*.jpg")

index_dict = {'0':'face','1':'mask-1','2':'mask-2','3':'mask-3','4':'eyewear','5':'person','6':'helmet'}

total_occurence_dict = {'face':[0,[]],'mask-1':[0,[]],'mask-2':[0,[]],'mask-3':[0,[]],'eyewear':[0,[]],'person':[0,[]],'helmet':[0,[]]}


label =[]
for la in l:
    label = label + [la.strip()]


if 'classes.txt' in label:
    label.remove('classes.txt')

if 'Classes.txt' in label:
    label.remove('Classes.txt')
    
for file in label:
    var = open(file, 'r')
    for line in var:
        yolo_class_num =(line.strip()[0])
        yolo_class = index_dict[yolo_class_num]
        occurance_dict_list = total_occurence_dict[yolo_class]
        occurance_dict_list[0] = occurance_dict_list[0] + 1
        if file not in occurance_dict_list[1]:
            occurance_dict_list[1] = occurance_dict_list[1] + [file]

names_list = ['face','mask-1','mask-2','mask-3','eyewear','person','helmet']
data = []

for i in names_list:
    ll = total_occurence_dict[i]
    x = ''
    for j in ll[1]:
        x = x + j + ' '
    temp_list =[i,ll[0],len(ll[1]),x]
    data =data +[temp_list]

df = pd.DataFrame(data, columns = ['Class', 'Total Occurences', 'Number of Images', 'List of Images'])

df.to_csv('dataset_analysis_main.csv')


    
