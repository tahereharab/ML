# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:56:54 2019

@author: Tahereh
"""
import pandas as pd
import multiprocessing  
pd.options.mode.chained_assignment = None 


def  filterbyPooling(index):        
        route_id  = df_routes.loc[index, 'route_id']
        route_name  = df_routes.loc[index, 'route_name']
        new_df = df[(df['Rout'] == route_id) & (df['Direction'] == 0)] #direction = 0/1
        new_df['duplicate_flag'] = new_df.duplicated(subset=['_Date_','TripID', 'Uniqu'], keep=False)
        new_df = new_df.loc[new_df['duplicate_flag'] == False]
        path = 'ModelData/routes-allstops/' + route_name + '-outbound'+ '.csv' #inbound/outbound
        new_df.to_csv(path, sep=',')  
        
        
df = pd.read_csv('ModelData/clean/oneYear17-18_clean.csv', sep=",")
df = df[df['Uniqu'] != '00009999']
df_routes = pd.read_csv('ModelData/route_stop/RouteMappings.csv', sep=",") 

num_processes = multiprocessing.cpu_count()
print('number of processors: ', num_processes)
pool = multiprocessing.Pool(processes=num_processes)
seq = [row.Index for row in df_routes[:-1].itertuples()]
results = pool.map(filterbyPooling , seq) 
print('Done!')

#-----------------------------
# after filter data and store,start tarining one by one and save models 
"""
isAllStops = False
isNBModel = False
trainAll(isNBModel , isAllStops)
"""
#-----------------------------