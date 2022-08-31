# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 09:38:47 2018

@author: Tahereh
"""
import pandas as pd
import statistics
import operator


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return round(float(intersection / union),2)

def jaccard_for_all_routes():
    df_routeStop = pd.read_csv('../2-CSV/ModelData/allStopsperRoute_weekdays.csv', sep=",")
    stops_route_ids = {}
    route_ids = []
    for row in df_routeStop[:-1].itertuples():    
        route_id  = df_routeStop.loc[row.Index, 'route_id']
        if (route_id not in route_ids): 
            stoplist_route_id = list(df_routeStop[(df_routeStop["route_id"] == route_id)]["stop_id"])
            stops_route_ids[route_id] = stoplist_route_id
            route_ids.append(route_id)
                 
    sims_all = {}
    sim_i_dict = {}
    sim_i_dict2 = {}
            
    for i in range(0, len(route_ids)):
        sim_i_sum = 0
        for j in range(0, len(route_ids)):
            route_i = route_ids[i].split('-')[0].lstrip("0")  
            route_j = route_ids[j].split('-')[0].lstrip("0")  
            sim_i_j = jaccard_similarity(stops_route_ids[route_ids[i]],stops_route_ids[route_ids[j]])
            sim_i_sum = sim_i_sum + sim_i_j
            sims_all[route_i,route_j] = sim_i_j
            
        sim_i_dict[route_ids[i].split('-')[0].lstrip("0")] = round(sim_i_sum - 1 ,2) #deduct the similarity between each route an ditself
        sim_i_dict2[route_ids[i]] = round(sim_i_sum - 1 ,2) 
    
    
    sorted_sim_i_dict = sorted(sim_i_dict.items(), key=operator.itemgetter(1))     

    maximum = max(sim_i_dict.values())
    minimum = min(sim_i_dict.values())
    average = sum(sim_i_dict.values()) / len(sim_i_dict.values())  
    median = statistics.median(sim_i_dict.values())
    print("max= ", maximum, ", min= ", minimum, ", avg= ", round(average,2) , ", median= ", median)
    #------------------------------
    numberOFRoutesPERstop(sim_i_dict2)
    #--------------------------------
        
def numberOFRoutesPERstop(sim_i_dict):
    df_routeStop = pd.read_csv('../2-CSV/ModelData/allStopsperRoute_weekdays.csv', sep=",")
    stop_ids = []
    stops_route_ids = {}
    stops_sum_sim = {}
    for row in df_routeStop[:-1].itertuples():    
        stop_id  = df_routeStop.loc[row.Index, 'stop_id']
        if (stop_id not in stop_ids):
            stoplist_route_id = list(df_routeStop[(df_routeStop["stop_id"] == stop_id)]["route_id"])
            stops_route_ids[stop_id] = [len(stoplist_route_id), stoplist_route_id]
            stop_ids.append(stop_id)
            sum_simsum = 0
            for route_id in stoplist_route_id:
                    simsum = sim_i_dict[route_id]
                    sum_simsum = sum_simsum + simsum
    
            stops_sum_sim[stop_id] = round(sum_simsum,3)
            
    sorted_dic = sorted(stops_route_ids.items(), key=operator.itemgetter(1))
    sorted_sumsim = sorted(stops_sum_sim.items(), key=operator.itemgetter(1))    
        
### for 61* routes
def getStopLists():
    
    df_allstops = pd.read_csv('../2-CSV/ModelData/allStopsperRoute_weekdays.csv', sep=",")
    stoplist_i = list(df_allstops[(df_allstops["route_id"] == "061D-166")]["stop_id"])
    stoplist_j = list(df_allstops[(df_allstops["route_id"] == "061C-166")]["stop_id"])

    d = [stoplist_i, stoplist_j]
    intersect_i_j  = set(d[0]).intersection(*d)
    print(intersect_i_j)
     
if __name__ == '__main__':
    
    #getStopLists()
    jaccard_for_all_routes()
    
   