# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:34:55 2018

@author: Tahereh
"""
import pandas as pd
from math import radians, cos, sin, asin, sqrt

def within(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # within formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

if __name__ == '__main__':

    df = pd.read_csv('../2-CSV/91/91_AprilWeekday.csv', sep=",")
    stops = pd.read_csv('../2-CSV/91/91_stops_locations.csv', sep=",")
    #add flag
    df['imputedStop_flag'] = 0
    cnt = 0
    for i in range(0,len(df)):
        if df.iloc[i, df.columns.get_loc('Stop_Name')] == 'Not Identified - Cal': 
            #mindistance = 0.1
            for j in range(0, len(stops)):
                if (stops['direction_id'][j] == df.iloc[i, df.columns.get_loc('Direction')]):
                    distance = within(stops['stop_lon'][j], stops['stop_lat'][j],-1*df.iloc[i, df.columns.get_loc('Longitude')], df.iloc[i, df.columns.get_loc('Latitude')])
                    if (distance <= 0.1): 
                        df.iloc[i, df.columns.get_loc('Uniqu')] = stops['stop_id'][j]
                        df.iloc[i, df.columns.get_loc('Stop_Name')] = stops['stop_name'][j]
                        df.iloc[i,df.columns.get_loc('imputedStop_flag')] = 1
    df[1:].to_csv('../2-CSV/91/91_AprilWeekday_clean_filledmissings_withflag_only_within100.csv', sep=',')      
            