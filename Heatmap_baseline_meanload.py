# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 13:31:57 2019

@author: Tahereh
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# this example is for route = 61C
geo_stops = pd.read_csv('../2-CSV/ModelData/4-baseline/61C_stops_inbound_geo_order.csv')
baseline = pd.read_csv('../2-CSV/ModelData/4-baseline/final_Baseline_15min_with-meanload.csv')
route = 613
baseline =  baseline[(baseline['Rout'] == route)]
stops = geo_stops['stop_id']
stop_names = geo_stops['stop_name']

allloads = []
#i = 0
for hour in range(0,24):
    timeloads = []
    for minute in [0,15,30,45]:
        timeloads = []
        for stop in stops:
            baseline_st = baseline[(baseline['Uniqu'] == stop) & (baseline['Hour'] == hour) & (baseline['Minute'] == minute)]
            if not baseline_st.empty:
                meanload = baseline_st['MeanLoad'].item()
                timeloads.append(meanload)
            else:
                timeloads.append(0)  
        
        allloads.append(timeloads)  
## plot------------------------
y = []    
for hour in range(0,24):
    for minute in [0,15,30,45]:
        time = str(hour) + ':' + str(minute)
        y.append(time)
    
x = stop_names
allloads = np.array(allloads)

plt.figure(num=None, figsize=(18, 10), dpi=80)
matplotlib.rcParams.update({'font.size': 8})
plt.pcolor(allloads, cmap='Greys') 
plt.colorbar() #need a colorbar to show the intensity scale
plt.xticks(np.arange(len(x)), x ,rotation=90)
plt.yticks(np.arange(len(y)), y)    
plt.xlabel('Stops', fontsize=12)
plt.ylabel('Time', fontsize=12)
plt.savefig("../2-CSV/ModelData/4-baseline/61C_baseline_load_heatmap.png")
plt.show()
