# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:12:43 2018

@author: Tahereh
"""

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

center_point = [{'lat': 40.3517979999999, 'lng': -79.860919}] #from GTFS data
test_point = [{'lat': 40.35213, 'lng': -79.85944}] #from historical data

lat1 = center_point[0]['lat']
lon1 = center_point[0]['lng']
lat2 = test_point[0]['lat']
lon2 = test_point[0]['lng']

radius = 0.50 # in kilometer

a = within(lon1, lat1, lon2, lat2)

print('Distance (km) : ', a)
if a <= radius:
    print('Inside the area')
else:
    print('Outside the area')
   