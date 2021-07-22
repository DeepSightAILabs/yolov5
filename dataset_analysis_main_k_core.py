#place this file inside the labels folder
#the results are saved in dataset_analysis_k_core.csv
#list of images is a space seperated string of text file names

import glob
import pandas as pd
l = glob.glob("*.txt")
#i = glob.glob("Images/*.jpg")
names_list = ['face','mask-1','mask-2','mask-3','eyewear','person','helmet']

index_dict = {'0':'face','1':'mask-1','2':'mask-2','3':'mask-3','4':'eyewear','5':'person','6':'helmet'}

total_occurence_dict = {'face':[],'mask-1':[],'mask-2':[],'mask-3':[],'eyewear':[],'person':[],'helmet':[]}
#----------------------------------
#change this variable for the limiting value
k_core = 5
#---------------------------------------
label =[]
for la in l:
    label = label + [la.strip()]

if 'classes.txt' in label:
    label.remove('classes.txt')

if 'Classes.txt' in label:
    label.remove('Classes.txt')

for file in label:
    var = open(file, 'r')
    occurence = {'face':0,'mask-1':0,'mask-2':0,'mask-3':0,'eyewear':0,'person':0,'helmet':0}
    for line in var:
        yolo_class_num =(line.strip()[0])
        yolo_class = index_dict[yolo_class_num]
        occurence[yolo_class] = occurence[yolo_class] +1

    for i in names_list:
        if occurence[i]>=k_core:
            total_occurence_dict[i] = total_occurence_dict[i] + [file]
            


data = []

for i in names_list:
    ll = total_occurence_dict[i]
    x = ''
    for j in ll:
        x = x + j + ' '
    temp_list =[i,k_core,len(ll),x]
    data =data +[temp_list]

df = pd.DataFrame(data, columns = ['Class', 'Minimum k Instances', 'Number of Images', 'List of Images'])

df.to_csv('dataset_analysis_k_core.csv')


    
