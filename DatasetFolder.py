# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:24:08 2020

@author: Šimon Bilík
"""

# This script prepares the dataset according to the set parameters and divides it into the required directory structure

from PIL import Image
from math import floor
import pandas as pd 
import os

# Set if the images should be cropped and resized
crop = 1
res = 1

# Set the image height and width
imWidth = 256
imHeight = 256

# Set the dataset parameters
nTrain_ok = 1000
nTrain_nok = 50
nValid_ok = 0
nValid_nok = 0
nTest_ok = 200
nTest_nok = 200

# Defect ratios
rNComplete = 0.4
rSObject = 0.3
rCDefect = 0.3

# Specify the image, annotations and destination path
imPath = "D:\\Programovani\\Datasets\\IndustryBiscuit\\Images"
anPath = "D:\\Programovani\\Datasets\\IndustryBiscuit\\AnotaceFinal.csv"
dsPath = "D:\\Programovani\\Datasets\\IndustryBiscuit_KerasApp"

# Counters initialization
cTrain_ok = 0
cValid_ok = 0
cTest_ok = 0

cTrainNC_nok = 0
cValidNC_nok = 0
cTestNC_nok = 0

cTrainSO_nok = 0
cValidSO_nok = 0
cTestSO_nok = 0

cTrainCD_nok = 0
cValidCD_nok = 0
cTestCD_nok = 0

# Defect limits
nTrNC = floor(nTrain_nok * rNComplete)
nVaNC = floor(nValid_nok * rNComplete)
nTeNC = floor(nTest_nok * rNComplete)

nTrSO = floor(nTrain_nok * rSObject)
nVaSO = floor(nValid_nok * rSObject)
nTeSO = floor(nTest_nok * rSObject)

nTrCD = floor(nTrain_nok * rCDefect)
nVaCD = floor(nValid_nok * rCDefect)
nTeCD = floor(nTest_nok * rCDefect)

# Create the directories for the image storage
if not os.path.exists(dsPath):
    
    # Create the folder structure
    os.mkdir(dsPath)
    os.mkdir(dsPath + '\\train')
    os.mkdir(dsPath + '\\train' + '\\ok')
    os.mkdir(dsPath + '\\train' + '\\nok')
    os.mkdir(dsPath + '\\valid')
    os.mkdir(dsPath + '\\valid' + '\\ok')
    os.mkdir(dsPath + '\\valid' + '\\nok')
    os.mkdir(dsPath + '\\test')
    os.mkdir(dsPath + '\\test' + '\\ok')
    os.mkdir(dsPath + '\\test' + '\\nok')

    # Load the filenames and the annotation from the .csv file
    data = pd.read_csv(anPath, usecols= ['file','licenceCode', 'upperLeftX', 'upperLeftY', 'lowerRightX', 'lowerRightY'])
    
    augm = 1226
    
    for key in range(1, 1226):
        
        for temp in range(0, 4):
            
            if temp == 0:
                index = key
            else:
                index = augm
                augm += 1
            
            value = data.iloc[index - 1, :]
            
            # Open the image file
            im = Image.open(os.path.join(imPath, value[0]))
            
            # Crop the images if set
            if crop == 1:
                im = im.crop((value[1], value[2], value[3], value[4]))
                
            # Resize the image if set
            if res == 1:
                im = im.resize((imWidth, imHeight))
            
            # Split the images to the categories
            if value[5] == "Bez_Vady":
                if (cTrain_ok < nTrain_ok):
                    im.save(os.path.join(dsPath + '\\train' + '\\ok', value[0]), format='jpeg')
                    cTrain_ok += 1
                elif (cValid_ok < nValid_ok):
                    im.save(os.path.join(dsPath + '\\valid' + '\\ok', value[0]), format='jpeg')
                    cValid_ok += 1
                elif (cTest_ok < nTest_ok):
                    im.save(os.path.join(dsPath + '\\test' + '\\ok', value[0]), format='jpeg')
                    cTest_ok += 1
                    
            elif value[5] == "Vada_Neuplnost":
                if (cTrainNC_nok < nTrNC):
                    im.save(os.path.join(dsPath + '\\train' + '\\nok', value[0]), format='jpeg')
                    cTrainNC_nok += 1
                elif (cValidNC_nok < nVaNC):
                    im.save(os.path.join(dsPath + '\\valid' + '\\nok', value[0]), format='jpeg')
                    cValidNC_nok += 1
                elif (cTestNC_nok < nTeNC):
                    im.save(os.path.join(dsPath + '\\test' + '\\nok', value[0]), format='jpeg')
                    cTestNC_nok += 1
                    
            elif value[5] == "Vada_CiziObjekt":
                if (cTrainSO_nok < nTrSO):
                    im.save(os.path.join(dsPath + '\\train' + '\\nok', value[0]), format='jpeg')
                    cTrainSO_nok += 1
                elif (cValidSO_nok < nVaSO):
                    im.save(os.path.join(dsPath + '\\valid' + '\\nok', value[0]), format='jpeg')
                    cValidSO_nok += 1
                elif (cTestSO_nok < nTeSO):
                    im.save(os.path.join(dsPath + '\\test' + '\\nok', value[0]), format='jpeg')
                    cTestSO_nok += 1
                    
            elif value[5] == "Vada_NestandardniBarva":
                if (cTrainCD_nok < nTrCD):
                    im.save(os.path.join(dsPath + '\\train' + '\\nok', value[0]), format='jpeg')
                    cTrainCD_nok += 1
                elif (cValidCD_nok < nVaCD):
                    im.save(os.path.join(dsPath + '\\valid' + '\\nok', value[0]), format='jpeg')
                    cValidCD_nok += 1
                elif (cTestCD_nok < nTeCD):
                    im.save(os.path.join(dsPath + '\\test' + '\\nok', value[0]), format='jpeg')
                    cTestCD_nok += 1
                
    # Print dataset statistics TODO
    
    print ("Dataset created successfully...")
    
else:
    print("Folder structure with the dataset already exists...")
