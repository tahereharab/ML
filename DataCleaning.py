#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:52:16 2019

@author: tahereh
"""

import re
import os
import pandas as pd


def toCSV(stppath, csvpath, filename):
    stppath = stppath + filename + '.stp'
    csvpath = csvpath + filename + '.csv'
    regex = '([A-Z]{1,2}\d{4,5}|999\d{1}|00009999) (.*?) \d{6}'

    with open(csvpath,'w') as file:         
        with open(stppath) as fp: 
            fp.readline()
            header = fp.readline()
            header = re.sub('Patter Block','Patter_Block',header)
            header = re.sub('Node ID','Node_ID',header)
            header = re.sub('WS1S2','W S1 S2',header)
            header = re.sub('TPofTP','TPo fTP',header)
            header = re.sub(' +',' ',header)
            header = re.sub(' ',',' , header)
            file.write(header)
            for line in fp:
                line = re.sub(' +',' ',line)
                matches = re.finditer(regex,line)
                for matchNum, match in enumerate(matches):
                    matchNum = matchNum + 1
                    match = match.group(2)
                    if(match.find('.')==-1):
                        rep = re.sub(' ', '_', match)
                        line = re.sub(match, rep , line)
    
                line = re.sub(' ',',',line) 
                line = re.sub('_',' ',line)
                file.write(line)  



def CleanInvalidChars(csvpath, csvpathclean, filename):
    
    path = csvpath + filename + '.csv'
    df = pd.read_csv(path, sep="," , index_col=False , dtype=str) 
    print(df.shape[0])

    df = df[['ID', 'Unique', 'Stop_Name', '_Date_', 'Arrive', 'Depart', 'Rout', 'on', 'off', 'load', 'Latitude',
             'Longitude', 'D', 'D.1', 'TRIPID__', 'Bus', 'Schd', 'Deviat', 'Dwell', 'Full', 'Over' , 'DOW', 'Service' ]]
    
    df = df.rename(columns={"Unique": "Uniqu", "D.1": "Direction", "TRIPID__": "TripID", "Bus": "BusID", 
                            "Schd": "ScheduleTime", "Deviat": "Deviation", "DOW": "DayofWeek"})
    #--------------------
    df['load'] = df['load'].str.replace('*','')
    df['load'] = df['load'].str.replace('B','')
    df = df[(df['DayofWeek'] != 2) & (df['DayofWeek'] != 3)]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[(df['Rout'] != 0) & (df['Uniqu'] != '00009999') & (df['Uniqu'] != '00009998') & (df['Uniqu'] != '00009997') & (df['Uniqu'] != '00009996') & (df['Uniqu'] != '00009995')]
    df['duplicate_flag'] = df.duplicated(subset=['BusID','_Date_','TripID', 'Uniqu'], keep=False) #we dont keep any of duplicates because load, on and off mught have changed and we dont know which one is correct!
    df = df.loc[df['duplicate_flag'] == False]
    #--------------------
    print(df.shape[0])
    path = csvpathclean + filename + '_weekdays'  + '.csv'
    df.to_csv(path, sep=',', index=False)  

def cleanInvalidDates(df):   
    df['_Date_'] = df['_Date_'].astype(str)
    df['_Date_'] = df['_Date_'].str.replace(r'.0$', '')
    print(df.shape[0])
    new_df = df[~(((df['_Date_'].str.len() >= 6) & (df['_Date_'].str[:2] > '12')) | (df['_Date_'].str.len() < 5))]
    print(new_df.shape[0])
    return new_df

def oneYearData(csvpathclean):
    
    df1 = pd.read_csv(csvpathclean + '1806_weekdays.csv', sep=",")
    df2 = pd.read_csv(csvpathclean + '1809_weekdays.csv', sep=",")
    df3 = pd.read_csv(csvpathclean + '1811_weekdays.csv', sep=",")
    df4 = pd.read_csv(csvpathclean + '1903_weekdays.csv', sep=",")
    
    df_union= pd.concat([df1, df2, df3, df4]).drop_duplicates() 
    df_union = cleanInvalidDates(df_union)
    df_union.to_csv(csvpathclean + 'oneYear18-19_clean.csv', sep=',' , index=False)  
    
if __name__ == '__main__':
    
    #specify folder paths for stp and csv files here
    stppath = 'ModelData/NewData/stp/'
    csvpath = 'ModelData/NewData/csv/'
    csvpathclean = 'ModelData/NewData/csv/clean/'
    
    for filee in os.listdir(csvpath): 
        path = os.path.join(csvpath, filee)
        if os.path.isdir(path):
            # skip directories
            continue
        filename = filee.split('.')[0]   
        CleanInvalidChars(csvpath, csvpathclean, filename)
    
    oneYearData(csvpathclean)   

        
        
        
        