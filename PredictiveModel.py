# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:01:24 2019

@author: Tahereh
"""

import pandas as pd
import numpy as np
import csv
import os
from functools import reduce
from pathlib import Path
import math
import statistics as st
import math
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf    
import statsmodels.api as sm 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K

#import multiprocessing  
pd.options.mode.chained_assignment = None 

def addWeather(df):
    
    df['Hour'] = 0
    df.loc[((df['Arrive'] >= 50000) & (df['Arrive'] < 60000)), 'Hour'] = 5
    df.loc[ (df.Arrive >= 60000) & (df.Arrive < 70000) , 'Hour'] = 6
    df.loc[ (df.Arrive >= 70000) & (df.Arrive < 80000) , 'Hour'] = 7
    df.loc[ (df.Arrive >= 80000) & (df.Arrive < 90000) , 'Hour'] = 8
    df.loc[ (df.Arrive >= 90000) & (df.Arrive < 100000) , 'Hour'] = 9
    df.loc[ (df.Arrive >= 100000) & (df.Arrive < 110000) , 'Hour'] = 10
    df.loc[ (df.Arrive >= 110000) & (df.Arrive < 120000) , 'Hour'] = 11
    df.loc[ (df.Arrive >= 120000) & (df.Arrive < 130000) , 'Hour'] = 12
    df.loc[ (df.Arrive >= 130000) & (df.Arrive < 140000) , 'Hour'] = 13
    df.loc[ (df.Arrive >= 140000) & (df.Arrive < 150000) , 'Hour'] = 14
    df.loc[ (df.Arrive >= 150000) & (df.Arrive < 160000) , 'Hour'] = 15
    df.loc[ (df.Arrive >= 160000) & (df.Arrive < 170000) , 'Hour'] = 16
    df.loc[ (df.Arrive >= 170000) & (df.Arrive < 180000) , 'Hour'] = 17
    df.loc[ (df.Arrive >= 180000) & (df.Arrive < 190000) , 'Hour'] = 18
    df.loc[ (df.Arrive >= 190000) & (df.Arrive < 200000) , 'Hour'] = 19
    df.loc[ (df.Arrive >= 200000) & (df.Arrive < 210000) , 'Hour'] = 20
    df.loc[ (df.Arrive >= 210000) & (df.Arrive < 220000) , 'Hour'] = 21
    df.loc[ (df.Arrive >= 220000) & (df.Arrive < 230000) , 'Hour'] = 22
    df.loc[ (df.Arrive >= 230000) & (df.Arrive < 240000) , 'Hour'] = 23
    df.loc[ (df.Arrive >= 240000) & (df.Arrive < 250000) , 'Hour'] = 0 
    df.loc[ (df.Arrive >= 250000) & (df.Arrive < 260000) , 'Hour'] = 1 
    df.loc[ (df.Arrive >= 20000) & (df.Arrive < 30000) , 'Hour'] = 2 
    df.loc[ (df.Arrive >= 30000) & (df.Arrive < 40000) , 'Hour'] = 3
    df.loc[ (df.Arrive >= 40000) & (df.Arrive < 50000) , 'Hour'] = 4
    
    df['Min'] = 0
    df['Arrive'] = df['Arrive'].astype(str)
    df = df[~df.Arrive.str.contains('n')]
    df['Min'] = np.where(((df['Hour'] >= 3) & (df['Hour'] <= 9)), df.Arrive.str[1:3] , df.Arrive.str[2:4])
    df['Min'] = df['Min'].astype(int)
    
    df['Minute1'] = np.where(((df['Min'] >= 0) & (df['Min'] < 15)),1,0)
    df['Minute2'] = np.where(((df['Min'] >= 15) & (df['Min'] < 30)),1,0)
    df['Minute3'] = np.where(((df['Min'] >= 30) & (df['Min'] < 45)),1,0)
    df['Minute4'] = np.where(((df['Min'] >= 45) & (df['Min'] <= 59)),1,0)
    
    weather_df = pd.read_csv('ModelData/weather/weather18-19_hourly2.csv', sep=",")
    weather_df['Snow_depth'] = weather_df['hourly_Snow_depth']
    weather_df['_Date_'] = pd.to_datetime(weather_df['Date'], format='%m/%d/%y %H:%M') #%m/%d/%Y %H:%M
    weather_df['Hour'] = weather_df._Date_.map(lambda x: x.hour)
    weather_df['_Date_'] = pd.DatetimeIndex(weather_df._Date_).normalize()
    mergedf = pd.merge(df , weather_df , on=['_Date_','Hour'], how='inner')
    #weather_hour_df = pd.read_csv('../2-CSV/ModelData/weather/weather17-18_hourly.csv', sep=",")
    weather_hour_df = pd.read_csv('ModelData/weather/weather17-18_hourly.csv', sep=",")
    #weather_hour_df = pd.read_csv('ModelData/weather/weather18-19_hourly.csv', sep=",")
    #weather_hour_df['_Date_'] = pd.to_datetime(weather_hour_df['Date'], format='%m/%d/%y %H:%M') #%m/%d/%Y %H:%M for 2018-19
    weather_hour_df['_Date_'] = pd.to_datetime(weather_hour_df['_Date_'], format='%m/%d/%Y %H:%M') #%m/%d/%Y %H:%M  for 2017-18
    weather_hour_df['Hour'] = weather_hour_df._Date_.map(lambda x: x.hour)
    weather_hour_df['_Date_'] = pd.DatetimeIndex(weather_hour_df._Date_).normalize()
    mergedfinal = pd.merge(mergedf , weather_hour_df, on=['_Date_','Hour'], how='inner')    
    return mergedfinal

def addIndependentVariables(df, isAllStops, stopid, isRegression):
    
    df['load1'] = np.where((df.ID == 0) , 0 , df['load'].shift(1))
    df['load2'] = np.where(((df.ID == 0) | (df.ID == 1)) , 0 , df['load'].shift(2))
    df['load3'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2)), 0 , df['load'].shift(3))
    df['load4'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3)), 0 , df['load'].shift(4))
    df['load5'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4)), 0 , df['load'].shift(5))
   
    df['load6'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4) | (df.ID == 5)), 0 , df['load'].shift(6))
    df['load7'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4) | (df.ID == 5)| (df.ID == 6)), 0 , df['load'].shift(7))
    df['load8'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4) | (df.ID == 5) | (df.ID == 6)| (df.ID == 7)), 0 , df['load'].shift(8))
    df['load9'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4) | (df.ID == 5) | (df.ID == 6) | (df.ID == 7) | (df.ID == 8)), 0 , df['load'].shift(9))
    df['load10'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4) | (df.ID == 5) | (df.ID == 6) | (df.ID == 7) | (df.ID == 8) | (df.ID == 9)), 0 , df['load'].shift(10))
        
    #---------------
     # add stops as variables : if isAllStops = true
    if (isAllStops == True) & (stopid == ""):
        df = pd.get_dummies(df, columns = ['Uniqu'] , drop_first = True)
        
    else:
        df = df[df['Uniqu'] == stopid]
    #---------------
    df['_Date_'] = df['_Date_'].astype(str)
    formats = ['%Y-%m-%d', '%m%d%y']
    df['_Date_'] = reduce(lambda l,r: l.combine_first(r), [pd.to_datetime(df['_Date_'], format=fmt, errors='coerce') for fmt in formats])
    df['Month'] = df._Date_.map(lambda x: x.month)
    
    #pandas.get_dummies could be also used!
    df['MOY1'] = np.where((df.Month == 1),1,0)
    df['MOY2'] = np.where((df.Month == 2),1,0)
    df['MOY3'] = np.where((df.Month == 3),1,0)
    df['MOY4'] = np.where((df.Month == 4),1,0)
    df['MOY5'] = np.where((df.Month == 5),1,0)
    df['MOY6'] = np.where((df.Month == 6),1,0)
    df['MOY7'] = np.where((df.Month == 7),1,0)
    df['MOY8'] = np.where((df.Month == 8),1,0)
    df['MOY9'] = np.where((df.Month == 9),1,0)
    df['MOY10'] = np.where((df.Month == 10),1,0)
    df['MOY11'] = np.where((df.Month == 11),1,0)
    df['MOY12'] = np.where((df.Month == 12),1,0)
    # -----------
    #DOW : Day of week : Mon:4, Tue: 5, Wed: 6, Thu: 7, Friday: 8
    df['DOW1'] , df['DOW2'] , df['DOW3'] , df['DOW4'], df['DOW5'] = 0 , 0, 0 , 0 , 0 
    df.loc[ (df.DayofWeek == 4) , 'DOW1'] = 1
    df.loc[ (df.DayofWeek == 5) , 'DOW2'] = 1
    df.loc[ (df.DayofWeek == 6) , 'DOW3'] = 1
    df.loc[ (df.DayofWeek == 7) , 'DOW4'] = 1
    df.loc[ (df.DayofWeek == 8) , 'DOW5'] = 1
    #df.loc[ (df.DayofWeek == 2) , 'DOW6'] = 1
    #df.loc[ (df.DayofWeek == 3) , 'DOW7'] = 1
    # -----------    
    # Time of the Day: 24 dummy variables    
    df['Arrive'] = df['Arrive'].astype(int)
    df['TOD1'] = np.where((df.Arrive >= 250000) & (df.Arrive < 260000),1,0)
    df['TOD2'] = np.where((df.Arrive >= 20000) & (df.Arrive < 30000),1,0)
    df['TOD3'] = np.where((df.Arrive >= 30000) & (df.Arrive < 40000),1,0)
    df['TOD4'] = np.where((df.Arrive >= 40000) & (df.Arrive < 50000),1,0)    
    df['TOD5'] = np.where((df.Arrive >= 50000) & (df.Arrive < 60000),1,0)
    df['TOD6'] = np.where((df.Arrive >= 60000) & (df.Arrive < 70000),1,0)
    df['TOD7'] = np.where((df.Arrive >= 70000) & (df.Arrive < 80000),1,0)
    df['TOD8'] = np.where((df.Arrive >= 80000) & (df.Arrive < 90000),1,0)
    df['TOD9'] = np.where((df.Arrive >= 90000) & (df.Arrive < 100000),1,0)
    df['TOD10'] = np.where((df.Arrive >= 100000) & (df.Arrive < 110000),1,0)
    df['TOD11'] = np.where((df.Arrive >= 110000) & (df.Arrive < 120000),1,0)
    df['TOD12'] = np.where((df.Arrive >= 120000) & (df.Arrive < 130000),1,0)
    df['TOD13'] = np.where((df.Arrive >= 130000) & (df.Arrive < 140000),1,0)
    df['TOD14'] = np.where((df.Arrive >= 140000) & (df.Arrive < 150000),1,0)
    df['TOD15'] = np.where((df.Arrive >= 150000) & (df.Arrive < 160000),1,0)
    df['TOD16'] = np.where((df.Arrive >= 160000) & (df.Arrive < 170000),1,0)
    df['TOD17'] = np.where((df.Arrive >= 170000) & (df.Arrive < 180000),1,0)
    df['TOD18'] = np.where((df.Arrive >= 180000) & (df.Arrive < 190000),1,0)
    df['TOD19'] = np.where((df.Arrive >= 190000) & (df.Arrive < 200000),1,0)
    df['TOD20'] = np.where((df.Arrive >= 200000) & (df.Arrive < 210000),1,0)
    df['TOD21'] = np.where((df.Arrive >= 210000) & (df.Arrive < 220000),1,0)
    df['TOD22'] = np.where((df.Arrive >= 220000) & (df.Arrive < 230000),1,0)
    df['TOD23'] = np.where((df.Arrive >= 230000) & (df.Arrive < 240000),1,0)
    df['TOD24'] = np.where((df.Arrive >= 240000) & (df.Arrive < 250000),1,0)
    # -----------
    # is  bus single or double
    df['BusID'] = df['BusID'].astype(int)
    df['IsDouble'] = np.where(((df.BusID >= 3000) & (df.BusID <= 3425)),1,0)    
    #------------
    # load factor: load/number_of_seats(Full)
    if isRegression == False:
        df['Full'] = df['Full'].astype(int)
        df['load'] = df['load'].astype(int)
        df['LoadFactor'] =  round(df.load/df.Full,2)
        df['LoadFactor'] = df['LoadFactor'].astype(float)
        #--------------
        # add isFull to be used in Logistic Regression
        df['isFull'] = np.where((df.LoadFactor > 0.75), 1, 0)   
        #----------------
        # for deep learning model
        df['load_category'] = 0
        df.loc[(df['LoadFactor'] < 0.5), 'load_category'] = 0
        df.loc[((df['LoadFactor'] < 0.8) & (df['LoadFactor'] >= 0.5)), 'load_category'] = 1
        df.loc[((df['LoadFactor'] < 1.1) & (df['LoadFactor'] >= 0.8)), 'load_category'] = 2
        df.loc[((df['LoadFactor'] < 1.4) & (df['LoadFactor'] >= 1.1)), 'load_category'] = 3
        df.loc[(df['LoadFactor'] >= 1.4), 'load_category'] = 4
    #--------------- 
    df = addWeather(df)
    #---------------        
    return df
    
def preapreDataSet(df, isAllStops, stopid, isRegression):
    
    new_df = addIndependentVariables(df, isAllStops, stopid, isRegression)
    return new_df


def trainANDtestSets(df):
    y = df.load
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20, random_state=200) 
    return X_train, X_test

def addFeatureInteractions(df_train, df_test):

    df_train.loc[:, 'TOD1*Minute1'] = df_train.loc[:, 'TOD1'] * df_train.loc[:, 'Minute1']
    df_train.loc[:, 'TOD2*Minute1'] = df_train.loc[:, 'TOD2'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD3*Minute1'] = df_train.loc[:, 'TOD3'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD4*Minute1'] = df_train.loc[:, 'TOD4'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD5*Minute1'] = df_train.loc[:, 'TOD5'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD6*Minute1'] = df_train.loc[:, 'TOD6'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD7*Minute1'] = df_train.loc[:, 'TOD7'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD8*Minute1'] = df_train.loc[:, 'TOD8'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD9*Minute1'] = df_train.loc[:, 'TOD9'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD10*Minute1'] = df_train.loc[:, 'TOD10'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD11*Minute1'] = df_train.loc[:, 'TOD11'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD12*Minute1'] = df_train.loc[:, 'TOD12'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD13*Minute1'] = df_train.loc[:, 'TOD13'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD14*Minute1'] = df_train.loc[:, 'TOD14'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD15*Minute1'] = df_train.loc[:, 'TOD15'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD16*Minute1'] = df_train.loc[:, 'TOD16'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD17*Minute1'] = df_train.loc[:, 'TOD17'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD18*Minute1'] = df_train.loc[:, 'TOD18'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD19*Minute1'] = df_train.loc[:, 'TOD19'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD20*Minute1'] = df_train.loc[:, 'TOD20'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD21*Minute1'] = df_train.loc[:, 'TOD21'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD22*Minute1'] = df_train.loc[:, 'TOD22'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD23*Minute1'] = df_train.loc[:, 'TOD23'] * df_train.loc[:, 'Minute1'] 
    df_train.loc[:, 'TOD24*Minute1'] = df_train.loc[:, 'TOD24'] * df_train.loc[:, 'Minute1'] 
    
    df_train.loc[:, 'TOD1*Minute2'] = df_train.loc[:, 'TOD1'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD2*Minute2'] = df_train.loc[:, 'TOD2'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD3*Minute2'] = df_train.loc[:, 'TOD3'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD4*Minute2'] = df_train.loc[:, 'TOD4'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD5*Minute2'] = df_train.loc[:, 'TOD5'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD6*Minute2'] = df_train.loc[:, 'TOD6'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD7*Minute2'] = df_train.loc[:, 'TOD7'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD8*Minute2'] = df_train.loc[:, 'TOD8'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD9*Minute2'] = df_train.loc[:, 'TOD9'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD10*Minute2'] = df_train.loc[:, 'TOD10'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD11*Minute2'] = df_train.loc[:, 'TOD11'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD12*Minute2'] = df_train.loc[:, 'TOD12'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD13*Minute2'] = df_train.loc[:, 'TOD13'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD14*Minute2'] = df_train.loc[:, 'TOD14'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD15*Minute2'] = df_train.loc[:, 'TOD15'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD16*Minute2'] = df_train.loc[:, 'TOD16'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD17*Minute2'] = df_train.loc[:, 'TOD17'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD18*Minute2'] = df_train.loc[:, 'TOD18'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD19*Minute2'] = df_train.loc[:, 'TOD19'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD20*Minute2'] = df_train.loc[:, 'TOD20'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD21*Minute2'] = df_train.loc[:, 'TOD21'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD22*Minute2'] = df_train.loc[:, 'TOD22'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD23*Minute2'] = df_train.loc[:, 'TOD23'] * df_train.loc[:, 'Minute2'] 
    df_train.loc[:, 'TOD24*Minute2'] = df_train.loc[:, 'TOD24'] * df_train.loc[:, 'Minute2'] 
    
    df_train.loc[:, 'TOD1*Minute3'] = df_train.loc[:, 'TOD1'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD2*Minute3'] = df_train.loc[:, 'TOD2'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD3*Minute3'] = df_train.loc[:, 'TOD3'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD4*Minute3'] = df_train.loc[:, 'TOD4'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD5*Minute3'] = df_train.loc[:, 'TOD5'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD6*Minute3'] = df_train.loc[:, 'TOD6'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD7*Minute3'] = df_train.loc[:, 'TOD7'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD8*Minute3'] = df_train.loc[:, 'TOD8'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD9*Minute3'] = df_train.loc[:, 'TOD9'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD10*Minute3'] = df_train.loc[:, 'TOD10'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD11*Minute3'] = df_train.loc[:, 'TOD11'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD12*Minute3'] = df_train.loc[:, 'TOD12'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD13*Minute3'] = df_train.loc[:, 'TOD13'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD14*Minute3'] = df_train.loc[:, 'TOD14'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD15*Minute3'] = df_train.loc[:, 'TOD15'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD16*Minute3'] = df_train.loc[:, 'TOD16'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD17*Minute3'] = df_train.loc[:, 'TOD17'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD18*Minute3'] = df_train.loc[:, 'TOD18'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD19*Minute3'] = df_train.loc[:, 'TOD19'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD20*Minute3'] = df_train.loc[:, 'TOD20'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD21*Minute3'] = df_train.loc[:, 'TOD21'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD22*Minute3'] = df_train.loc[:, 'TOD22'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD23*Minute3'] = df_train.loc[:, 'TOD23'] * df_train.loc[:, 'Minute3'] 
    df_train.loc[:, 'TOD24*Minute3'] = df_train.loc[:, 'TOD24'] * df_train.loc[:, 'Minute3'] 
    
    df_train.loc[:, 'TOD1*Minute4'] = df_train.loc[:, 'TOD1'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD2*Minute4'] = df_train.loc[:, 'TOD2'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD3*Minute4'] = df_train.loc[:, 'TOD3'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD4*Minute4'] = df_train.loc[:, 'TOD4'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD5*Minute4'] = df_train.loc[:, 'TOD5'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD6*Minute4'] = df_train.loc[:, 'TOD6'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD7*Minute4'] = df_train.loc[:, 'TOD7'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD8*Minute4'] = df_train.loc[:, 'TOD8'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD9*Minute4'] = df_train.loc[:, 'TOD9'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD10*Minute4'] = df_train.loc[:, 'TOD10'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD11*Minute4'] = df_train.loc[:, 'TOD11'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD12*Minute4'] = df_train.loc[:, 'TOD12'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD13*Minute4'] = df_train.loc[:, 'TOD13'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD14*Minute4'] = df_train.loc[:, 'TOD14'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD15*Minute4'] = df_train.loc[:, 'TOD15'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD16*Minute4'] = df_train.loc[:, 'TOD16'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD17*Minute4'] = df_train.loc[:, 'TOD17'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD18*Minute4'] = df_train.loc[:, 'TOD18'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD19*Minute4'] = df_train.loc[:, 'TOD19'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD20*Minute4'] = df_train.loc[:, 'TOD20'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD21*Minute4'] = df_train.loc[:, 'TOD21'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD22*Minute4'] = df_train.loc[:, 'TOD22'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD23*Minute4'] = df_train.loc[:, 'TOD23'] * df_train.loc[:, 'Minute4'] 
    df_train.loc[:, 'TOD24*Minute4'] = df_train.loc[:, 'TOD24'] * df_train.loc[:, 'Minute4'] 
    
    df_test.loc[:, 'TOD1*Minute1'] = df_test.loc[:, 'TOD1'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD2*Minute1'] = df_test.loc[:, 'TOD2'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD3*Minute1'] = df_test.loc[:, 'TOD3'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD4*Minute1'] = df_test.loc[:, 'TOD4'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD5*Minute1'] = df_test.loc[:, 'TOD5'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD6*Minute1'] = df_test.loc[:, 'TOD6'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD7*Minute1'] = df_test.loc[:, 'TOD7'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD8*Minute1'] = df_test.loc[:, 'TOD8'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD9*Minute1'] = df_test.loc[:, 'TOD9'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD10*Minute1'] = df_test.loc[:, 'TOD10'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD11*Minute1'] = df_test.loc[:, 'TOD11'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD12*Minute1'] = df_test.loc[:, 'TOD12'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD13*Minute1'] = df_test.loc[:, 'TOD13'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD14*Minute1'] = df_test.loc[:, 'TOD14'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD15*Minute1'] = df_test.loc[:, 'TOD15'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD16*Minute1'] = df_test.loc[:, 'TOD16'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD17*Minute1'] = df_test.loc[:, 'TOD17'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD18*Minute1'] = df_test.loc[:, 'TOD18'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD19*Minute1'] = df_test.loc[:, 'TOD19'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD20*Minute1'] = df_test.loc[:, 'TOD20'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD21*Minute1'] = df_test.loc[:, 'TOD21'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD22*Minute1'] = df_test.loc[:, 'TOD22'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD23*Minute1'] = df_test.loc[:, 'TOD23'] * df_test.loc[:, 'Minute1'] 
    df_test.loc[:, 'TOD24*Minute1'] = df_test.loc[:, 'TOD24'] * df_test.loc[:, 'Minute1'] 
    
    df_test.loc[:, 'TOD1*Minute2'] = df_test.loc[:, 'TOD1'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD2*Minute2'] = df_test.loc[:, 'TOD2'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD3*Minute2'] = df_test.loc[:, 'TOD3'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD4*Minute2'] = df_test.loc[:, 'TOD4'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD5*Minute2'] = df_test.loc[:, 'TOD5'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD6*Minute2'] = df_test.loc[:, 'TOD6'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD7*Minute2'] = df_test.loc[:, 'TOD7'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD8*Minute2'] = df_test.loc[:, 'TOD8'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD9*Minute2'] = df_test.loc[:, 'TOD9'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD10*Minute2'] = df_test.loc[:, 'TOD10'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD11*Minute2'] = df_test.loc[:, 'TOD11'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD12*Minute2'] = df_test.loc[:, 'TOD12'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD13*Minute2'] = df_test.loc[:, 'TOD13'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD14*Minute2'] = df_test.loc[:, 'TOD14'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD15*Minute2'] = df_test.loc[:, 'TOD15'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD16*Minute2'] = df_test.loc[:, 'TOD16'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD17*Minute2'] = df_test.loc[:, 'TOD17'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD18*Minute2'] = df_test.loc[:, 'TOD18'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD19*Minute2'] = df_test.loc[:, 'TOD19'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD20*Minute2'] = df_test.loc[:, 'TOD20'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD21*Minute2'] = df_test.loc[:, 'TOD21'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD22*Minute2'] = df_test.loc[:, 'TOD22'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD23*Minute2'] = df_test.loc[:, 'TOD23'] * df_test.loc[:, 'Minute2'] 
    df_test.loc[:, 'TOD24*Minute2'] = df_test.loc[:, 'TOD24'] * df_test.loc[:, 'Minute2'] 
    
    df_test.loc[:, 'TOD1*Minute3'] = df_test.loc[:, 'TOD1'] * df_test.loc[:, 'Minute3']
    df_test.loc[:, 'TOD2*Minute3'] = df_test.loc[:, 'TOD2'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD3*Minute3'] = df_test.loc[:, 'TOD3'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD4*Minute3'] = df_test.loc[:, 'TOD4'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD5*Minute3'] = df_test.loc[:, 'TOD5'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD6*Minute3'] = df_test.loc[:, 'TOD6'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD7*Minute3'] = df_test.loc[:, 'TOD7'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD8*Minute3'] = df_test.loc[:, 'TOD8'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD9*Minute3'] = df_test.loc[:, 'TOD9'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD10*Minute3'] = df_test.loc[:, 'TOD10'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD11*Minute3'] = df_test.loc[:, 'TOD11'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD12*Minute3'] = df_test.loc[:, 'TOD12'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD13*Minute3'] = df_test.loc[:, 'TOD13'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD14*Minute3'] = df_test.loc[:, 'TOD14'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD15*Minute3'] = df_test.loc[:, 'TOD15'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD16*Minute3'] = df_test.loc[:, 'TOD16'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD17*Minute3'] = df_test.loc[:, 'TOD17'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD18*Minute3'] = df_test.loc[:, 'TOD18'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD19*Minute3'] = df_test.loc[:, 'TOD19'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD20*Minute3'] = df_test.loc[:, 'TOD20'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD21*Minute3'] = df_test.loc[:, 'TOD21'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD22*Minute3'] = df_test.loc[:, 'TOD22'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD23*Minute3'] = df_test.loc[:, 'TOD23'] * df_test.loc[:, 'Minute3'] 
    df_test.loc[:, 'TOD24*Minute3'] = df_test.loc[:, 'TOD24'] * df_test.loc[:, 'Minute3'] 
    
    df_test.loc[:, 'TOD1*Minute4'] = df_test.loc[:, 'TOD1'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD2*Minute4'] = df_test.loc[:, 'TOD2'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD3*Minute4'] = df_test.loc[:, 'TOD3'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD4*Minute4'] = df_test.loc[:, 'TOD4'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD5*Minute4'] = df_test.loc[:, 'TOD5'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD6*Minute4'] = df_test.loc[:, 'TOD6'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD7*Minute4'] = df_test.loc[:, 'TOD7'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD8*Minute4'] = df_test.loc[:, 'TOD8'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD9*Minute4'] = df_test.loc[:, 'TOD9'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD10*Minute4'] = df_test.loc[:, 'TOD10'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD11*Minute4'] = df_test.loc[:, 'TOD11'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD12*Minute4'] = df_test.loc[:, 'TOD12'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD13*Minute4'] = df_test.loc[:, 'TOD13'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD14*Minute4'] = df_test.loc[:, 'TOD14'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD15*Minute4'] = df_test.loc[:, 'TOD15'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD16*Minute4'] = df_test.loc[:, 'TOD16'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD17*Minute4'] = df_test.loc[:, 'TOD17'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD18*Minute4'] = df_test.loc[:, 'TOD18'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD19*Minute4'] = df_test.loc[:, 'TOD19'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD20*Minute4'] = df_test.loc[:, 'TOD20'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD21*Minute4'] = df_test.loc[:, 'TOD21'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD22*Minute4'] = df_test.loc[:, 'TOD22'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD23*Minute4'] = df_test.loc[:, 'TOD23'] * df_test.loc[:, 'Minute4'] 
    df_test.loc[:, 'TOD24*Minute4'] = df_test.loc[:, 'TOD24'] * df_test.loc[:, 'Minute4']
    
    return df_train, df_test

def crossValidation(clf, trainX, trainy):  
    print('before cross validation..')
    scores1 = cross_val_score(estimator=clf, X=trainX, y=trainy, cv=5, scoring='f1_micro')
    print("F1 score: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
    
    try:
        scores2 = cross_val_score(estimator=clf, X=trainX, y=trainy, cv=5, scoring='neg_log_loss') 
        print("log loss: %0.2f (+/- %0.2f)" % (-scores2.mean(), scores2.std() * 2))
        return -scores2.mean(), scores1.mean()
    
    except:
        print("log loss: ", 0.0)
        return 0.0 , scores1.mean()
        pass
            
def gridSearchParamOptimization(RF, trainX, trainy):
    
    parameters = {'n_estimators': [100,500,1000],'max_depth' : [None,2,4,5]}
    #'max_features': ['sqrt', 'log2', None]} 
    #'n_estimators': [100, 500],
    #'max_depth' : [4,5,6,7,8],
    #'criterion' :['gini', 'entropy']
    grid = GridSearchCV(RF, parameters , scoring = 'f1_micro', cv=5)
    grid.fit(trainX, trainy)
    print('best parameters: ', grid.best_params_, 'best f1 score: ' ,  grid.best_score_ , grid.best_estimator_)
    return grid

def KerasGridSearchCVParamOptimization(trainX, trainy):
    model_keras = KerasClassifier(build_fn = build_keras_base)
    dropout_rate  = [0.0, 0.5]
    hidden_layers_opts = [[100]]
    optimizer = ['Adam']
    parameters =  {
        'hidden_layers': hidden_layers_opts,
        'epochs': [50,100],
        'batch_size': [32,64],
        'validation_split': [0.2],
        'optimizer' : optimizer,
        'dropout_rate': dropout_rate
    }
    rs_keras = GridSearchCV( estimator=model_keras , param_grid=parameters , scoring = 'f1_micro', cv=5)      
    rs_keras.fit(trainX, trainy)
          
    print('Best score obtained: {0}'.format(rs_keras.best_score_))
    print('Parameters:')
    for param, value in rs_keras.best_params_.items():
        print('\t{}: {}'.format(param, value))
    
    return rs_keras


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def build_keras_base(hidden_layers = [64, 64, 64], dropout_rate = 0, optimizer = 'adam', n_input = 183, n_class = 5):
    
    model = Sequential()
    model.add(Dense(100, input_dim=n_input, activation='relu')) 
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model
    
def trainDeepClassificationModel(df_train, df_test):   
    
    train , test = addFeatureInteractions(df_train, df_test)
    
    #select diffrent subsets of columns in order to build models with different feature sets
    cols = ['TOD2*Minute1','TOD3*Minute1', 'TOD4*Minute1','TOD5*Minute1', 'TOD6*Minute1' , 'TOD7*Minute1' , 'TOD8*Minute1' , 'TOD9*Minute1' , 'TOD10*Minute1' , 'TOD11*Minute1', 'TOD12*Minute1' ,
                           'TOD13*Minute1' , 'TOD14*Minute1' , 'TOD15*Minute1' , 'TOD16*Minute1' , 'TOD17*Minute1' , 'TOD18*Minute1' , 'TOD19*Minute1' ,  'TOD20*Minute1' , 'TOD21*Minute1' , 
                            'TOD22*Minute1' , 'TOD23*Minute1' , 'TOD24*Minute1' ,
            'TOD1*Minute2','TOD2*Minute2','TOD3*Minute2', 'TOD4*Minute2','TOD5*Minute2', 'TOD6*Minute2' , 'TOD7*Minute2' , 'TOD8*Minute2' , 'TOD9*Minute2' , 'TOD10*Minute2' , 'TOD11*Minute2', 'TOD12*Minute2' ,
                           'TOD13*Minute2' , 'TOD14*Minute2' , 'TOD15*Minute2' , 'TOD16*Minute2' , 'TOD17*Minute2' , 'TOD18*Minute2' , 'TOD19*Minute2' ,  'TOD20*Minute2' , 'TOD21*Minute2' , 
                            'TOD22*Minute2' , 'TOD23*Minute2' , 'TOD24*Minute2' , 
            'TOD1*Minute3','TOD2*Minute3' , 'TOD3*Minute3',  'TOD4*Minute3' , 'TOD5*Minute3' , 'TOD6*Minute3' ,  'TOD7*Minute3' ,  'TOD8*Minute3' , 'TOD9*Minute3' , 'TOD10*Minute3','TOD11*Minute3' , 'TOD12*Minute3',
                            'TOD13*Minute3' , 'TOD14*Minute3' , 'TOD15*Minute3' , 'TOD16*Minute3' , 'TOD17*Minute3' , 'TOD18*Minute3' , 'TOD19*Minute3' , 'TOD20*Minute3' , 'TOD21*Minute3' ,
                             'TOD22*Minute3' , 'TOD23*Minute3' , 'TOD24*Minute3' , 
            'TOD1*Minute4','TOD2*Minute4' , 'TOD3*Minute4' , 'TOD4*Minute4' , 'TOD5*Minute4' , 'TOD6*Minute4' , 'TOD7*Minute4' , 'TOD8*Minute4' , 'TOD9*Minute4' , 'TOD10*Minute4' , 'TOD11*Minute4' , 'TOD12*Minute4' ,
                            'TOD13*Minute4' , 'TOD14*Minute4' , 'TOD15*Minute4' , 'TOD16*Minute4' , 'TOD17*Minute4' , 'TOD18*Minute4' , 'TOD19*Minute4' , 'TOD20*Minute4' , 'TOD21*Minute4' ,
                            'TOD22*Minute4' , 'TOD23*Minute4' , 'TOD24*Minute4', 
                            'MOY2', 'MOY3', 'MOY4' , 'MOY5' , 'MOY6' , 'MOY7',  'MOY8', 'MOY9' , 'MOY10' , 'MOY11' , 'MOY12' 
                            , 'DOW2' , 'DOW3' , 'DOW4' , 'DOW5' 
                            , 'IsDouble' , 'Snow_depth' , 'Avg_temp' , 'Precipitation', 
                             'load1', 'load2' , 'load3' , 'load4' , 'load5']
    
    newcolumns = [col for col in train if col.startswith('Uniqu_')]
    cols.extend(newcolumns) 
    #---------------------------    
    trainX=train[cols]
    trainy=train['load_category']
    testX = test[cols]
    testy = test['load_category']
    #--------------------------
    inputcol = len(cols)
    #----------------------
    #### tune hyperparameters 
    """
    rs_keras = KerasGridSearchCVParamOptimization(trainX, trainy)
    best_model = rs_keras.best_estimator_.model
    metric_names = best_model.metrics_names
    metric_values = best_model.evaluate(trainX, trainy)
    for metric, value in zip(metric_names, metric_values):
        print(metric, ': ', value)
        
    acc = 100
    los = 0   
    """
    #----------------------
    neurons = int(len(trainX) / (2 * (inputcol + 5)))  
    print("neurons = ", neurons)
    #run with best parameters from tuning
    model = Sequential()
    model.add(Dense(5, input_dim=inputcol, activation='relu')) 
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ###model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
    model.fit(trainX, trainy, validation_split=0.2, epochs=100, batch_size=64, verbose=0)
    scores = model.evaluate(testX, testy, verbose=0) 
    acc = scores[1]
    los = scores[0]
    print("accuracy: ", round(acc,2))
    print("logloss: ", round(los,2))
    return round(los,2) , round(acc,2) 
        
def trainClassificationModel(df_train, df_test, isMulticlass, isAllStops, stopid, isCrossValidation, isRF):   
    
    train , test = addFeatureInteractions(df_train, df_test)
    #select diffrent subsets of columns in order to build models with different feature sets
    cols = ['TOD2*Minute1','TOD3*Minute1', 'TOD4*Minute1','TOD5*Minute1', 'TOD6*Minute1' , 'TOD7*Minute1' , 'TOD8*Minute1' , 'TOD9*Minute1' , 'TOD10*Minute1' , 'TOD11*Minute1', 'TOD12*Minute1' ,
                           'TOD13*Minute1' , 'TOD14*Minute1' , 'TOD15*Minute1' , 'TOD16*Minute1' , 'TOD17*Minute1' , 'TOD18*Minute1' , 'TOD19*Minute1' ,  'TOD20*Minute1' , 'TOD21*Minute1' , 
                            'TOD22*Minute1' , 'TOD23*Minute1' , 'TOD24*Minute1' ,
            'TOD1*Minute2','TOD2*Minute2','TOD3*Minute2', 'TOD4*Minute2','TOD5*Minute2', 'TOD6*Minute2' , 'TOD7*Minute2' , 'TOD8*Minute2' , 'TOD9*Minute2' , 'TOD10*Minute2' , 'TOD11*Minute2', 'TOD12*Minute2' ,
                           'TOD13*Minute2' , 'TOD14*Minute2' , 'TOD15*Minute2' , 'TOD16*Minute2' , 'TOD17*Minute2' , 'TOD18*Minute2' , 'TOD19*Minute2' ,  'TOD20*Minute2' , 'TOD21*Minute2' , 
                            'TOD22*Minute2' , 'TOD23*Minute2' , 'TOD24*Minute2' , 
            'TOD1*Minute3','TOD2*Minute3' , 'TOD3*Minute3',  'TOD4*Minute3' , 'TOD5*Minute3' , 'TOD6*Minute3' ,  'TOD7*Minute3' ,  'TOD8*Minute3' , 'TOD9*Minute3' , 'TOD10*Minute3','TOD11*Minute3' , 'TOD12*Minute3',
                            'TOD13*Minute3' , 'TOD14*Minute3' , 'TOD15*Minute3' , 'TOD16*Minute3' , 'TOD17*Minute3' , 'TOD18*Minute3' , 'TOD19*Minute3' , 'TOD20*Minute3' , 'TOD21*Minute3' ,
                             'TOD22*Minute3' , 'TOD23*Minute3' , 'TOD24*Minute3' , 
            'TOD1*Minute4','TOD2*Minute4' , 'TOD3*Minute4' , 'TOD4*Minute4' , 'TOD5*Minute4' , 'TOD6*Minute4' , 'TOD7*Minute4' , 'TOD8*Minute4' , 'TOD9*Minute4' , 'TOD10*Minute4' , 'TOD11*Minute4' , 'TOD12*Minute4' ,
                            'TOD13*Minute4' , 'TOD14*Minute4' , 'TOD15*Minute4' , 'TOD16*Minute4' , 'TOD17*Minute4' , 'TOD18*Minute4' , 'TOD19*Minute4' , 'TOD20*Minute4' , 'TOD21*Minute4' ,
                            'TOD22*Minute4' , 'TOD23*Minute4' , 'TOD24*Minute4', 
                            'MOY2', 'MOY3', 'MOY4' , 'MOY5' , 'MOY6' , 'MOY7',  'MOY8', 'MOY9' , 'MOY10' , 'MOY11' , 'MOY12' 
                            , 'DOW2' , 'DOW3' , 'DOW4' , 'DOW5' 
                            , 'IsDouble' , 'Snow_depth' , 'Avg_temp' , 'Precipitation', 
                             'load1', 'load2' , 'load3' , 'load4' , 'load5','load6' , 'load7' , 'load8' , 'load9' , 'load10']
    
    if (isAllStops == True) & (stopid == ""):
        newcolumns = [col for col in train if col.startswith('Uniqu_')]
        cols.extend(newcolumns) 
        
    trainX=train[cols]
    trainy=train['load_category']
    testX = test[cols]
    testy = test['load_category']
    trainX.to_csv('../2-CSV/ModelData/tmp/12-inbound_train.csv', sep=',')
    
    if isRF == True:
        clf = RandomForestClassifier(n_estimators=500, criterion = 'entropy', random_state = 200) #max_features= auto
    else:    
        clf = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
    #----------------
    #tune parameters for RF 
    #grid = gridSearchParamOptimization(clf, trainX, trainy) 
    #clf = RandomForestClassifier(**grid.best_params_)
    #----------------

    if isCrossValidation == True:
        ll, f1 = crossValidation(clf, trainX, trainy)
        return ll , f1
           
    else:        
        
        model = clf.fit(trainX, trainy)  
        #----------------------------------------              
        y_predict = model.predict_proba(testX)
        
        try:
            ll = log_loss(testy, y_predict, labels=[1,2,3,4,5]) #
        except:
            ll = 0
            pass
        
        ll = round(ll,2)
        print("log_loss = " , ll)
        #----------------------------------------
        y_pred = model.predict(testX)
        F1 = round(f1_score(testy, y_pred, labels=[1,2,3,4,5], average = 'micro'),2)
        print('F1-micro Score = ', F1) #F1 micro score = accuracy
        print('\n')
        return ll, F1, model


def trainRegressionModel(df_train, df_test, stopid, isAllStops):
    train , test = addFeatureInteractions(df_train, df_test)
    
    cols = ['TOD2*Minute1','TOD3*Minute1', 'TOD4*Minute1','TOD5*Minute1', 'TOD6*Minute1' , 'TOD7*Minute1' , 'TOD8*Minute1' , 'TOD9*Minute1' , 'TOD10*Minute1' , 'TOD11*Minute1', 'TOD12*Minute1' ,
                           'TOD13*Minute1' , 'TOD14*Minute1' , 'TOD15*Minute1' , 'TOD16*Minute1' , 'TOD17*Minute1' , 'TOD18*Minute1' , 'TOD19*Minute1' ,  'TOD20*Minute1' , 'TOD21*Minute1' , 
                            'TOD22*Minute1' , 'TOD23*Minute1' , 'TOD24*Minute1' ,
            'TOD1*Minute2','TOD2*Minute2','TOD3*Minute2', 'TOD4*Minute2','TOD5*Minute2', 'TOD6*Minute2' , 'TOD7*Minute2' , 'TOD8*Minute2' , 'TOD9*Minute2' , 'TOD10*Minute2' , 'TOD11*Minute2', 'TOD12*Minute2' ,
                           'TOD13*Minute2' , 'TOD14*Minute2' , 'TOD15*Minute2' , 'TOD16*Minute2' , 'TOD17*Minute2' , 'TOD18*Minute2' , 'TOD19*Minute2' ,  'TOD20*Minute2' , 'TOD21*Minute2' , 
                            'TOD22*Minute2' , 'TOD23*Minute2' , 'TOD24*Minute2' , 
            'TOD1*Minute3','TOD2*Minute3' , 'TOD3*Minute3',  'TOD4*Minute3' , 'TOD5*Minute3' , 'TOD6*Minute3' ,  'TOD7*Minute3' ,  'TOD8*Minute3' , 'TOD9*Minute3' , 'TOD10*Minute3','TOD11*Minute3' , 'TOD12*Minute3',
                            'TOD13*Minute3' , 'TOD14*Minute3' , 'TOD15*Minute3' , 'TOD16*Minute3' , 'TOD17*Minute3' , 'TOD18*Minute3' , 'TOD19*Minute3' , 'TOD20*Minute3' , 'TOD21*Minute3' ,
                             'TOD22*Minute3' , 'TOD23*Minute3' , 'TOD24*Minute3' , 
            'TOD1*Minute4','TOD2*Minute4' , 'TOD3*Minute4' , 'TOD4*Minute4' , 'TOD5*Minute4' , 'TOD6*Minute4' , 'TOD7*Minute4' , 'TOD8*Minute4' , 'TOD9*Minute4' , 'TOD10*Minute4' , 'TOD11*Minute4' , 'TOD12*Minute4' ,
                            'TOD13*Minute4' , 'TOD14*Minute4' , 'TOD15*Minute4' , 'TOD16*Minute4' , 'TOD17*Minute4' , 'TOD18*Minute4' , 'TOD19*Minute4' , 'TOD20*Minute4' , 'TOD21*Minute4' ,
                            'TOD22*Minute4' , 'TOD23*Minute4' , 'TOD24*Minute4', 
                            'MOY2', 'MOY3', 'MOY4' , 'MOY5' , 'MOY6' , 'MOY7',  'MOY8', 'MOY9' , 'MOY10' , 'MOY11' , 'MOY12' 
                            , 'DOW2' , 'DOW3' , 'DOW4' , 'DOW5' 
                            , 'IsDouble' , 'Snow_depth' , 'Avg_temp' , 'Precipitation', 
                             'load1', 'load2' , 'load3' , 'load4' , 'load5', 'load6' , 'load7' , 'load8' , 'load9' , 'load10']

    if (isAllStops == True) & (stopid == ""):
        newcolumns = [col for col in train if col.startswith('Uniqu_')]
        cols.extend(newcolumns) 
    """
    trainX=train[cols]
    trainy=train['load']
    testX = test[cols]
    testy = test['load']
    model = linear_model.LinearRegression().fit(trainX, trainy)
    """
    X = train[cols]
    y = train['load'].values
    testX = test[cols]
    testy = test['load']
    X_sm = sm.add_constant(X[cols].copy())
    model = sm.NegativeBinomial(y, X_sm).fit()
    predictedValues = model.predict(testX)
    diff = testy - predictedValues    
    rmse = math.sqrt(st.mean((diff)**2))
    return round(rmse,2), model   
    
def trainNBModel(df_train, df_test, stopid, isAllStops):
    
    train , test = addFeatureInteractions(df_train, df_test)
    
    formula_array = ['TOD2*Minute1','TOD3*Minute1', 'TOD4*Minute1','TOD5*Minute1', 'TOD6*Minute1' , 'TOD7*Minute1' , 'TOD8*Minute1' , 'TOD9*Minute1' , 'TOD10*Minute1' , 'TOD11*Minute1', 'TOD12*Minute1' ,
                           'TOD13*Minute1' , 'TOD14*Minute1' , 'TOD15*Minute1' , 'TOD16*Minute1' , 'TOD17*Minute1' , 'TOD18*Minute1' , 'TOD19*Minute1' ,  'TOD20*Minute1' , 'TOD21*Minute1' , 
                            'TOD22*Minute1' , 'TOD23*Minute1' , 'TOD24*Minute1' ,
            'TOD1*Minute2','TOD2*Minute2','TOD3*Minute2', 'TOD4*Minute2','TOD5*Minute2', 'TOD6*Minute2' , 'TOD7*Minute2' , 'TOD8*Minute2' , 'TOD9*Minute2' , 'TOD10*Minute2' , 'TOD11*Minute2', 'TOD12*Minute2' ,
                           'TOD13*Minute2' , 'TOD14*Minute2' , 'TOD15*Minute2' , 'TOD16*Minute2' , 'TOD17*Minute2' , 'TOD18*Minute2' , 'TOD19*Minute2' ,  'TOD20*Minute2' , 'TOD21*Minute2' , 
                            'TOD22*Minute2' , 'TOD23*Minute2' , 'TOD24*Minute2' , 
            'TOD1*Minute3','TOD2*Minute3' , 'TOD3*Minute3',  'TOD4*Minute3' , 'TOD5*Minute3' , 'TOD6*Minute3' ,  'TOD7*Minute3' ,  'TOD8*Minute3' , 'TOD9*Minute3' , 'TOD10*Minute3','TOD11*Minute3' , 'TOD12*Minute3',
                            'TOD13*Minute3' , 'TOD14*Minute3' , 'TOD15*Minute3' , 'TOD16*Minute3' , 'TOD17*Minute3' , 'TOD18*Minute3' , 'TOD19*Minute3' , 'TOD20*Minute3' , 'TOD21*Minute3' ,
                             'TOD22*Minute3' , 'TOD23*Minute3' , 'TOD24*Minute3' , 
            'TOD1*Minute4','TOD2*Minute4' , 'TOD3*Minute4' , 'TOD4*Minute4' , 'TOD5*Minute4' , 'TOD6*Minute4' , 'TOD7*Minute4' , 'TOD8*Minute4' , 'TOD9*Minute4' , 'TOD10*Minute4' , 'TOD11*Minute4' , 'TOD12*Minute4' ,
                            'TOD13*Minute4' , 'TOD14*Minute4' , 'TOD15*Minute4' , 'TOD16*Minute4' , 'TOD17*Minute4' , 'TOD18*Minute4' , 'TOD19*Minute4' , 'TOD20*Minute4' , 'TOD21*Minute4' ,
                            'TOD22*Minute4' , 'TOD23*Minute4' , 'TOD24*Minute4']                          

    # added for SVD convergae 
    train = train.dropna()
    train = train[~train.isin([np.nan, np.inf, -np.inf]).any(1)]
    test = test.dropna()
    test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]
    
    #remove zero columns from the formula
    zero_cols = train.loc[:, (train == 0).all()]    
    if len(zero_cols.columns) != 0:
        for col in zero_cols:
            if (col.startswith('TOD')): 
                col = col + "*"
            matching = [s for s in formula_array if col in s]
            for item in matching:
                while formula_array.count(item) > 0:
                    formula_array.remove(item)
        
    #remove zero columns from the train and test sets
    train = train.loc[:, (train != 0).any(axis=0)]       
    test = test.loc[:, (test != 0).any(axis=0)]
    #-------------------------------------------------------------------------------------------------------                                     
    formula = "load ~  "         
    for var in formula_array:
        formula += var
        formula += " + "
    formula = formula[:-2]   
    
    if (isAllStops == True) & (stopid == ""):
        newcolumns = [col for col in train if col.startswith('Uniqu_')]
        colsize = len(newcolumns)
        for i in range(colsize):
            formula += " + "
            formula += newcolumns[i]

    model = smf.glm(formula = formula, data=train, family=sm.families.NegativeBinomial()).fit()
    predictedValues = model.predict(test)
    diff = test.load - predictedValues    
    rmse = math.sqrt(st.mean((diff)**2))
    return round(rmse,2), model                        

def trainAll(isRegression, isAllStops, isCrossValidation, isDeep):

    path = "../2-CSV/ModelData/tmp/"
    for filename in os.listdir(path):
        
        if filename != 'desktop.ini':
            print('filename= ', filename)
            df = pd.read_csv(path+filename, sep=",")
            route = filename.split('.')[0]
            stopid = ""
            #---------------------------
            if isDeep == True:
                
                new_df = preapreDataSet(df, isAllStops, stopid, isRegression)
                new_df = new_df.dropna()
                df_train, df_test = trainANDtestSets(new_df)      
                print('length of train and test: ', len(df_train), len(df_test))
                
                logloss, acc  = trainDeepClassificationModel(df_train, df_test)
                
                
                resultPath = 'ModelData/results/selected_routes_direction_DEEP_FS3_meanneurons_results.csv'
                with open(resultPath,'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([route , logloss, acc])
                  
                
            else:    
            #--------------------------------------------   
                if isAllStops == False:
                    
                    stops_path = 'ModelData/selected_stops.csv'
                    df_stops = pd.read_csv(stops_path, sep=",")
                    df_stops = df_stops[df_stops['route'] == route]  
                    stopid = df_stops.iloc[0]['stop_id']
                        
                    new_df = preapreDataSet(df, isAllStops, stopid, isRegression)
                    new_df = new_df.dropna()
                    df_train, df_test = trainANDtestSets(new_df)            
            
                    if isRegression == True:                        
                        rmse , model = trainRegressionModel(df_train, df_test, stopid, isAllStops)
        
                        with open('ModelData/results/selected_routes_stops_NB_results.csv','a') as file:
                            writer = csv.writer(file)
                            writer.writerow([route , stopid , rmse])
                        
                    else:   
                        isMulticlass = True   
                        isRF = True
                        logloss, f1 , model = trainClassificationModel(df_train, df_test, isMulticlass, isAllStops, stopid, isCrossValidation, isRF)
                        
                        if isRF == True:
                            resultPath = 'ModelData/results/selected_routes_stops_RF_results.csv'
                        else:
                             resultPath = 'ModelData/results/selected_routes_stops_LOG_results.csv'
                             
                        with open(resultPath,'a') as file:
                            writer = csv.writer(file)
                            writer.writerow([route , stopid , logloss, f1])
               
                elif isAllStops == True:
                    
                    new_df = preapreDataSet(df, isAllStops, stopid, isRegression)
                    new_df = new_df.dropna()
                    df_train, df_test = trainANDtestSets(new_df)      
                    print('length of train and test: ', len(df_train), len(df_test))
                
                    if isRegression == True:                        
                        rmse , model = trainRegressionModel(df_train, df_test, stopid, isAllStops)
            
                        with open('ModelData/results/selected_routes_direction_NB_results.csv','a') as file:
                            writer = csv.writer(file)
                            writer.writerow([route , rmse])
                            
                    else:   
                        isMulticlass = True     
                        isRF = True
                        logloss, f1 , model = trainClassificationModel(df_train, df_test, isMulticlass, isAllStops, stopid, isCrossValidation, isRF)
                        
                        if isRF == True:
                            resultPath = 'ModelData/results/selected_routes_direction_RF_results.csv'
                        else:
                             resultPath = 'ModelData/results/selected_routes_direction_LOG_results.csv'
                             
                        with open(resultPath,'a') as file:
                            writer = csv.writer(file)
                            writer.writerow([route , logloss, f1])
                    
                    
def filterByRouteAllStops(df, route_id):
    df = df[df['Uniqu'] != '00009999']
    new_df = df[(df['Rout'] == route_id)] 
    new_df['duplicate_flag'] = new_df.duplicated(subset=['_Date_','TripID', 'Uniqu'], keep=False)
    new_df = new_df.loc[new_df['duplicate_flag'] == False]
    path = 'ModelData/' + str(route_id) + '.csv'
    new_df.to_csv(path, sep=',')    


def meanLoadLineChart():
    #Diagram
    import matplotlib.pyplot as plt
    group_df = pd.read_csv('../2-CSV/ModelData/1-61C_allstops_logloss_f1/61C_inbound_stops_mean_load_sorted.csv', sep=",")
    fig1 = plt.figure(figsize=(12, 2))
    ax1 = fig1.add_subplot(111)
    plt.xticks(rotation=90, fontsize = 7)
    x1 = group_df['stop_id']
    y1 = group_df['mean_load']
    plt.xticks(rotation=90)
    plt.xlabel('Stops')
    plt.ylabel('Mean Load')
    plt.title('Mean load for inbound stops')
    #plt.rc('xtick',labelsize=7)
    ax1.plot(x1, y1)   

def loglossf1linechart():

    df = pd.read_csv('../2-CSV/ModelData/1-61C_allstops_logloss_f1/613_geo_allstops_logloss_f1_results.csv', sep=",")
    dff = pd.read_csv('../2-CSV/ModelData/1-61C_allstops_logloss_f1/61C_stops_geo_order.csv', sep=",")
    
    dff1 = dff[dff['direction_id'] == 1]
    df1 = df[df['direction'] == 1]
    
    df1 = df1.set_index('stopid') #sort df2 based on df1 stops
    df1 = df1.reindex(index=dff1['stop_id'])
    df1 = df1.reset_index()
    
    dff0 = dff[dff['direction_id'] == 0]
    df0 = df[df['direction'] == 0]
    
    df0 = df0.set_index('stopid') #sort df2 based on df1 stops
    df0 = df0.reindex(index=dff0['stop_id'])
    df0 = df0.reset_index()
    #--------------------------
    fig1 = plt.figure(figsize=(12, 2))
    ax1 = fig1.add_subplot(111)
    #df1 = df[df['direction'] == 1]
    x1 = df1['stop_id']
    y1 = df1['logloss']
    plt.xticks(rotation=90, fontsize = 7)
    plt.ylim([0,1])
    plt.xlabel('Stops')
    plt.ylabel('Log Loss')
    plt.title('log loss for inbound stops')
    #plt.rc('xtick',labelsize=4)
    ax1.plot(x1, y1)   
    #-----------------
    fig0 = plt.figure(figsize=(12, 2))
    ax0 = fig0.add_subplot(111)
    #df0 = df[df['direction'] == 0]
    x0 = df0['stop_id']
    y0 = df0['logloss']
    plt.xticks(rotation=90, fontsize = 7)
    plt.ylim([0,1])
    plt.xlabel('Stops')
    plt.ylabel('Log Loss')
    plt.title('log loss for outbound stops')
    plt.rc('xtick',labelsize=7)
    ax0.plot(x0, y0)
    #--------------------------
    fig11 = plt.figure(figsize=(12, 2))
    ax11 = fig11.add_subplot(111)
    x11 = df1['stop_id']
    y11 = df1['f1']
    plt.xticks(rotation=90, fontsize = 7)
    plt.ylim([0,1])
    plt.xlabel('Stops')
    plt.ylabel('F1 score')
    plt.rc('xtick',labelsize=7)
    plt.title('f1 for inbound stops')
    ax11.plot(x11, y11)   
    #-----------------
    fig00 = plt.figure(figsize=(12, 2))
    ax00 = fig00.add_subplot(111)
    x00 = df0['stop_id']
    y00 = df0['f1']
    plt.xticks(rotation=90, fontsize = 7)
    plt.ylim([0,1])
    plt.xlabel('Stops')
    plt.ylabel('F1 score')
    plt.title('f1 for inbound stops')
    plt.rc('xtick',labelsize=7)
    ax00.plot(x00, y00)
    
def allModelsScatterPlot(param1, param2, xsize, ysize):

    df1 = pd.read_csv('../2-CSV/ModelData/3-all_models_results/routes_allmodels_f1_logloss_meanload.csv',sep=',')
    if param2 == 'mean_load':
        ylabel = 'mean load'
        
    if param2 == 'logloss':
        ylabel = 'log loss'
        
    if param1 == 'logloss':
        xlabel = 'log loss'   
    else:
        xlabel = param1
    # Initializing a figure
    fig = plt.figure(figsize=(xsize, ysize))
    # Adding labels using a subplot
    ax = plt.subplot(111)
    i = 0
    for xy in zip(df1[param1],df1[param2]):
        ax.annotate(df1['route'][i], xy, textcoords='data')
        i+=1
    # Plotting
    plt.scatter(df1[param1],df1[param2])
    # Formatting graph
    plt.xticks(rotation = 45)
    plt.xlabel(xlabel)
    plt.xlim([0,1])        
    plt.ylabel(ylabel)
    plt.show()
    
def allModelsF1LoglossMeanloadScatterPlot():

    df1 = pd.read_csv('../2-CSV/ModelData/3-all_models_results/routes_allmodels_f1_logloss_meanload.csv',sep=',')

    fig = plt.figure(figsize=(10.5, 8))
    ax = fig.add_subplot(111)
    load_group1 = df1[df1['mean_load'] < 5]
    load_group2 = df1[(df1['mean_load'] >= 5) & (df1['mean_load'] < 10)]
    load_group3 = df1[(df1['mean_load'] >= 10) & (df1['mean_load'] < 15)]
    load_group4 = df1[(df1['mean_load'] >= 15) & (df1['mean_load'] < 20)]
    load_group5 = df1[(df1['mean_load'] >= 20) & (df1['mean_load'] < 25)]
    load_group6 = df1[(df1['mean_load'] >= 25)]

    load_group1.plot(kind='scatter', x='f1', y='logloss', ylim=((0, 1)), xlim=((0, 1)), s=1*2, ax=ax, color='g', label='< 5') 
    load_group2.plot(kind='scatter', x='f1', y='logloss', ylim=((0, 1)), xlim=((0, 1)), s= 8*2, ax=ax, color='b',label='5-10')
    load_group3.plot(kind='scatter', x='f1', y='logloss', ylim=((0, 1)), xlim=((0, 1)), s= 27*2, ax=ax, color='y', label='10-15')
    load_group4.plot(kind='scatter', x='f1', y='logloss', ylim=((0, 1)), xlim=((0, 1)), s= 64*2, ax=ax, color='c', label='15-20')
    load_group5.plot(kind='scatter', x='f1', y='logloss', ylim=((0, 1)), xlim=((0, 1)), s= 125*2, ax=ax, color='m',label='20-25')
    load_group6.plot(kind='scatter', x='f1', y='logloss', ylim=((0, 1)), xlim=((0, 1)), s= 216*2, ax=ax, color='r',label='> 25')
    
    plt.legend(loc="lower left", numpoints=1, fontsize=10, title='Mean Load', labelspacing=1.2, borderpad=1)
    plt.xlabel('F1')
    plt.ylabel('Log Loss')
    plt.show()
    
def allModelsMeanLoad():
    
    df1 = pd.read_csv('../2-CSV/ModelData/3-all_models_results/routes_allstops_model_results.csv',sep=',')
    df2 = pd.read_csv('../2-CSV/ModelData/3-all_models_results/routes_allmodels_meanload.csv',sep=',')
    mergeddf = pd.merge(df1 , df2, on=['route'], how='inner') 
    mergeddf.to_csv('../2-CSV/ModelData/3-all_models_results/routes_allmodels_f1_logloss_meanload.csv', sep=',')
    
def filterByRouteDirection(df, route_id, direction_id):    
    
    direction = 'inbound'
    if direction_id == 0:
        direction = 'outbound'
        
    new_df = df[(df['Rout'] == route_id) & (df['Direction'] == direction_id)] 
    path = 'ModelData/routes-allstops-18-19/' + str(route_id) + '-' + direction + '.csv'
    new_df.to_csv(path, sep=',', index=False)  
    

if __name__ == '__main__':

    isDeep = False
    isRegression = False
    isAllStops = True #true when we have route-direction
    isCrossValidation = False
    trainAll(isRegression, isAllStops,isCrossValidation, isDeep)
    
   