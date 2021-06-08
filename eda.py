import pandas as pd
import os
from os import walk
from pandas.core.frame import DataFrame
import regex as re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
from random import shuffle
import glob
from tabulate import tabulate

if __name__ == "__main__":
    folder_path = "..\images"
    images_list = glob.glob(os.path.join(folder_path,"*.jpg"))
    images_list += glob.glob(os.path.join(folder_path,"*.jpeg"))
    images_list += glob.glob(os.path.join(folder_path,"*.png"))
    images_list += glob.glob(os.path.join(folder_path,"*.JPG"))
    #run in cmd> 
    running = input('Are you using cmd to run the code? [y/n]: ')
    
    if running == 'y':    
        os.chdir('..')
        metadata=pd.read_csv('..\Covid19\metadata.csv')
    else:#run in vscode> 
        metadata = pd.read_csv('metadata.csv')

############################################################'removing' .gz files#######################################################################
    '''def create_and_compare_imagesFile():
        metadata_images = metadata.filename
        valid_img_list = list()
        for image in metadata_images:
            if not re.search('gz$',image):
                valid_img_list.append(image)

            else:
                print(f'{image} ---- "REMOVED"')
                pass
                
        print('-'*25)
        print('\n\n')
    
        valid_img = pd.Series(valid_img_list)
        if not os.path.exists('..\Covid19\metadata_Images.csv'):
            save_valid_img = valid_img.to_csv('..\Covid19\metadata_Images.csv',index=True,header=['Filename'])
            print('File metadata_Images.csv created')
        else:
            pass
            print(f'File metadata_Images.csv already exists')
            
        #testing if .csv has all images
        ignore1,ignore2,filenames = next(walk('..\Covid19\images'))
        #print('\n\n')
        
        for file in filenames:
            if file in valid_img_list:
                pass
            else:
                print(f'FILE: {file} NOT IN metadata_Images.csv')
    '''
#################################################################### CREATING DF OF VALID DATA #########################################################################

    def validating_data():    
        row = -1
        valid_data = pd.DataFrame(data = {'patientid':metadata['patientid'],'finding':metadata['finding'],'modality':metadata['modality'],'filename':metadata['filename']})
        for x in valid_data['filename']:       
            row+=1
            if re.search('gz',x):
                valid_data = valid_data.drop(valid_data.index[row])
                row -=1
        valid_data.to_csv('valid_data.csv',index=False)
        patients = pd.unique(valid_data['patientid']) #How Many patients in dataset
        diseases = pd.unique(valid_data['finding']) # How Many and which diseases in dataset
        return patients, diseases,valid_data
    patients, diseases,valid_data = validating_data()
    
    
    ################################################################FILTERING X-RAY###########################################
    
    
    def only_x_ray(valid_data):
        valid_data1 = valid_data[valid_data['modality'].isin(['X-ray'])]
        return valid_data1
    
    
    ####################################################VALIDATING ONLY XRAY####################################################
    valid_data = only_x_ray(valid_data)
    ################################################## IDENTIFY HOW MANY AND WICH DISEASES THE DATASET CONTAINS ###########################################

    def show_images():
        os.chdir('images')
        for img in valid_data['filename']:
            cond = input(f'Do you want to see {img}? [y/n] or stop: ')
            print('-'*60)
            if cond == 'y':
                cv.namedWindow('PITOIMAGE',0)
                image = cv.imread(img,1)
                cv.imshow('PITOIMAGE',image)
                cv.waitKey(0)
            elif cond == 'stop':
                break
            else:
                pass
        os.chdir('..')
        

    ###################################################################SHOWING DATA#######################################################################
    
    def ploting_imgs():
        data_to_plot = pd.DataFrame(data = {'filename':list(valid_data['filename'])},index= valid_data['finding'])
        plt.figure()
        plt.bar(data_to_plot.index.value_counts().index,data_to_plot.index.value_counts().values
                ,color='#113759',edgecolor='#cd4446',linewidth=2)

        for i,v in enumerate(data_to_plot.index.value_counts()):
            plt.text(data_to_plot.index.value_counts().index[i],v+3,str(v),color='#113759',fontweight='bold',
                     horizontalalignment='center',fontname='Segoe UI')
        plt.title('Photos per Disease',fontname='Segoe UI',fontweight='bold')
        locs, labels = plt.xticks()
        plt.setp(labels,rotation=25,fontsize=9)
        plt.show()
        
    ########################################################################################################################################    
    
    
    ################################################replacing COVID-19 for 1 and Not Covid for 0##############################################
    
    def replacing():
        filter = valid_data["finding"].str.contains("COVID")
        store_old_data = valid_data["finding"]
        valid_data["finding"] = filter
        valid_data["finding"][filter] = True
        valid_data["finding"][~filter] = False
        return valid_data['finding'],store_old_data
    
    ############################################################################################################################################
    
    valid_data = only_x_ray(valid_data)
    
    ############################################################################################################################################
    #GETTING UNIQUE IDs
    #pprint(pd.unique(valid_data['patientid']))
    print(f'The diseases are: {diseases}') #Which diseases 
    print(f'There are {len(diseases)} diseases') #How many diseases
    print(f'There are {len(patients)} patients') #Unique Patients
    
    
    finding_column_new,finding_column_old = replacing()
    
   
    def dataframe_to_train_test(dataframe):
        unique_patients = dataframe['patientid'].unique().tolist()
        shuffle(unique_patients)
        patient_id_train = unique_patients[:int(len(unique_patients)*0.95)]
        patient_id_test = unique_patients[int(len(unique_patients)*0.95):]
        patient_id_train.sort()
        patient_id_test.sort()
        dataframe = dataframe.set_index(dataframe['patientid'])
        dataframe_train_filter = dataframe['patientid'].loc[patient_id_train] 
        dataframe_test_filter = dataframe['patientid'].loc[patient_id_test]
        test_dataframe = dataframe[dataframe['patientid'].isin(dataframe_test_filter)]
        train_dataframe = dataframe[dataframe['patientid'].isin(dataframe_train_filter)]
        return test_dataframe,train_dataframe
    
  
  ############################################################################################################################################  
    #show_images()
    #ploting_imgs()
    dataframe_to_train_test(valid_data)
    test_dataframe,train_dataframe = dataframe_to_train_test(valid_data)
    print(test_dataframe.head(15).to_markdown())
 
    print('\n'*3)

    print(train_dataframe.head(15).to_markdown())
    print(len(list(train_dataframe.index)))

    #print(train_dataframe)
    
    '''valid_data = valid_data.drop_duplicates(subset = 'patientid')
    patient_id_train,patient_id_test = split_patientid()
    filter_patient_to_test = valid_data['patientid'].isin(patient_id_test)
    filter_patient_to_train = valid_data['patientid'].isin(patient_id_train)
    valid_data['patientid'][filter_patient_to_train] = patient_id_train
    pd.set_option('display.max_rows',valid_data.shape[0]+1)
    valid_data = valid_data.set_index(valid_data['patientid'])
    #valid_data = valid_data.sort_index()
    print(valid_data)
    print(valid_data['finding'].describe())'''