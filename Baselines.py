# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:45:59 2018

@author: Tahereh
"""
import pandas as pd
import numpy as np
import csv
import os
import math
import graphviz 
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO  
from IPython.display import Image


def mergeBaselines():
    appended_data = []
    path = 'ModelData/baselines/'
    for filename in os.listdir(path):
        print(filename)
        df = pd.read_csv(path+filename, sep=",")
        appended_data.append(df)
    appended_data = pd.concat(appended_data, axis=1)
    appended_data.to_csv('ModelData/Final_Baseline.csv', sep=',') 
    print('finish writing into csv..')
        

def meanLoadofYear():
    df = pd.read_csv('ModelData/clean/oneYear17-18_clean.csv', sep=",") 
    train_size = int(df.shape[0]*0.75)
    df_test = df.iloc[train_size+1:df.shape[0]]
    df_test['_Date_'] = df_test['_Date_'].astype(str)
    df_test = df_test.loc[((df_test['_Date_'].str[:2] <= '12') | (df_test['_Date_'].str.len() == 5)) & (df_test['_Date_'].str[:] != '0')] 
    avgload = df_test['load'].mean()
    print(avgload) #9.44
    return avgload

def createBaselineSet2(df, route , stop):
    new_df = PrepareDataset(df)
    
    groupby_df = new_df.groupby(["Rout","Uniqu", "Hour", "Minute"])[["load"]].mean().reset_index()
    groupby_df.columns = ['Rout', 'Uniqu', 'Hour', 'Minute', 'MeanLoad']
    
    path = '../2-CSV/ModelData/baseline/' + str(route) + '_' + str(stop) + '_baseline2' + '.csv'
    groupby_df.to_csv(path, sep=',')

def FinalBaselinePerQuarterWithDirection(df):
    
    if df is None:
        df = pd.read_csv('ModelData/clean/oneYear17-18_clean.csv', sep=",")
        print('csv is read!') 
        #clean----------
        
        df = df[(df['Rout'] != 0) & (df['Rout'] != 998) & (df['Uniqu'] != '00009999') & (df['Uniqu'] != '00009998') & (df['Uniqu'] != '00009997') & (df['Uniqu'] != '00009996') & (df['Uniqu'] != '00009995')]
        """
        df['duplicate_flag'] = df.duplicated(subset=['_Date_','TripID', 'Uniqu'], keep=False)
        df = df.loc[df['duplicate_flag'] == False]
        """
        print('dataset is clean!')
        #---------------------
        #df = PrepareDataset(df)
        #--------------------------
    
    groupby_df = df.groupby(["Rout",'Uniqu', "Hour", "Minute", "Direction"])[["load"]].mean().reset_index() 
    groupby_df.columns = ['Rout', 'Uniqu','Hour', 'Minute', "Direction", 'MeanLoad']
    
    groupby_df_A = df.groupby(["Rout", 'Uniqu', "Hour", "Minute", "Direction"])[["LoadFactor"]].count().reset_index()
    #groupby_df_A.add_suffix('_count').reset_index()
    groupby_df_A.columns = ['Rout', 'Uniqu', 'Hour', 'Minute', "Direction", 'ALL']
    #print('groupby_All_size = ' , groupby_df_A.shape[0])
    
    # with 5 categories for fullness-----------------------    
    #Crushed
    groupby_df_B = df[(df['LoadFactor'] >= 1.4)].groupby(["Rout", 'Uniqu', "Hour", "Minute", "Direction"])[["LoadFactor"]].count().reset_index()
    groupby_df_B.columns = ['Rout', 'Uniqu', 'Hour','Minute', "Direction", 'Crushed']
    
    # Many people need to stand
    groupby_df_C = df[(df['LoadFactor'] >= 1.1) & (df['LoadFactor'] < 1.4)].groupby(["Rout", 'Uniqu',"Hour", "Minute", "Direction"])[["LoadFactor"]].count().reset_index()
    groupby_df_C.columns = ['Rout', 'Uniqu', 'Hour','Minute' , "Direction", 'ManyPeopleStand']
   
    # few people need to stand
    groupby_df_D = df[(df['LoadFactor'] >= 0.8) & (df['LoadFactor'] < 1.1)].groupby(["Rout",'Uniqu', "Hour", "Minute", "Direction"])[["LoadFactor"]].count().reset_index()
    groupby_df_D.columns = ['Rout', 'Uniqu', 'Hour','Minute', "Direction", 'FewPeopleStand']
    
    # Few empty seats
    groupby_df_E = df[(df['LoadFactor'] >= 0.5) & (df['LoadFactor'] < 0.8)].groupby(["Rout",'Uniqu', "Hour", "Minute", "Direction"])[["LoadFactor"]].count().reset_index()
    groupby_df_E.columns = ['Rout','Uniqu', 'Hour','Minute', "Direction", 'FewEmptySeats']
   
    #Many empty seats
    groupby_df_F = df[(df['LoadFactor'] < 0.5)].groupby(["Rout", 'Uniqu', "Hour", "Minute", "Direction"])[["LoadFactor"]].count().reset_index()
    groupby_df_F.columns = ['Rout', 'Uniqu', 'Hour','Minute', "Direction", 'ManyEmptySeats']
    
    groupby_df_merge1 = pd.merge(groupby_df_A , groupby_df_B , on=["Rout",'Uniqu', "Hour","Minute", "Direction"], how='outer')
    groupby_df_merge2 = pd.merge(groupby_df_merge1 , groupby_df_C , on=["Rout",'Uniqu', "Hour","Minute", "Direction"], how='outer')
    groupby_df_merge3 = pd.merge(groupby_df_merge2 , groupby_df_D , on=["Rout",'Uniqu', "Hour","Minute", "Direction"], how='outer')
    groupby_df_merge4 = pd.merge(groupby_df_merge3 , groupby_df_E , on=["Rout",'Uniqu', "Hour","Minute", "Direction"], how='outer')
    groupby_df_merge5 = pd.merge(groupby_df_merge4 , groupby_df_F , on=["Rout",'Uniqu', "Hour","Minute", "Direction"], how='outer')
    groupby_df_merge = pd.merge(groupby_df_merge5 , groupby_df , on=["Rout", 'Uniqu', "Hour","Minute", "Direction"], how='outer') 
    
    groupby_df_merge = groupby_df_merge.fillna(0)
    groupby_df_merge['Crushed'] =  round(groupby_df_merge[['Crushed']].div(groupby_df_merge.ALL, axis=0)*100,2)
    groupby_df_merge['ManyPeopleStand'] =  round(groupby_df_merge[['ManyPeopleStand']].div(groupby_df_merge.ALL, axis=0)*100,2)
    groupby_df_merge['FewPeopleStand'] =  round(groupby_df_merge[['FewPeopleStand']].div(groupby_df_merge.ALL, axis=0)*100, 2)
    groupby_df_merge['FewEmptySeats'] =  round(groupby_df_merge[['FewEmptySeats']].div(groupby_df_merge.ALL, axis=0)*100, 2)
    groupby_df_merge['ManyEmptySeats'] =  round(groupby_df_merge[['ManyEmptySeats']].div(groupby_df_merge.ALL, axis=0)*100, 2)
        
    groupby_df_merge['Category'] = 1
    groupby_df_merge.loc[((groupby_df_merge['Crushed'] >= groupby_df_merge['ManyPeopleStand'])  & (groupby_df_merge['Crushed'] >= groupby_df_merge['FewPeopleStand']) & (groupby_df_merge['Crushed'] >= groupby_df_merge['FewEmptySeats']) & (groupby_df_merge['Crushed'] >= groupby_df_merge['ManyEmptySeats'])), 'Category'] = 5
    groupby_df_merge.loc[((groupby_df_merge['ManyPeopleStand'] >= groupby_df_merge['Crushed'])  & (groupby_df_merge['ManyPeopleStand'] >= groupby_df_merge['FewPeopleStand']) & (groupby_df_merge['ManyPeopleStand'] >= groupby_df_merge['FewEmptySeats']) & (groupby_df_merge['ManyPeopleStand'] >= groupby_df_merge['ManyEmptySeats'])), 'Category'] = 4
    groupby_df_merge.loc[((groupby_df_merge['FewPeopleStand'] >= groupby_df_merge['Crushed'])  & (groupby_df_merge['FewPeopleStand'] >= groupby_df_merge['ManyPeopleStand'])& (groupby_df_merge['FewPeopleStand'] >= groupby_df_merge['FewEmptySeats']) & (groupby_df_merge['FewPeopleStand'] >= groupby_df_merge['ManyEmptySeats'])), 'Category'] = 3
    groupby_df_merge.loc[((groupby_df_merge['FewEmptySeats'] >= groupby_df_merge['Crushed'])  & (groupby_df_merge['FewEmptySeats'] >= groupby_df_merge['ManyPeopleStand'])& (groupby_df_merge['FewEmptySeats'] >= groupby_df_merge['FewPeopleStand']) & (groupby_df_merge['FewEmptySeats'] >= groupby_df_merge['ManyEmptySeats'])), 'Category'] = 2
    groupby_df_merge.loc[((groupby_df_merge['ManyEmptySeats'] >= groupby_df_merge['Crushed'])  & (groupby_df_merge['ManyEmptySeats'] >= groupby_df_merge['ManyPeopleStand'])& (groupby_df_merge['ManyEmptySeats'] >= groupby_df_merge['FewPeopleStand']) & (groupby_df_merge['ManyEmptySeats'] >= groupby_df_merge['FewEmptySeats'])), 'Category'] = 1

    print('Merge is done!')
    
    #groupby_df_merge.to_csv('ModelData/baselines/final_Baseline_with_new_categories.csv', sep=',') 
    groupby_df_merge.to_csv('ModelData/baselines/final_Baseline18-19_5category_15min_80_percent_trainset_with_direction_stop.csv', sep=',', index=False) 
    
    print('output csv is done!')       
    return groupby_df_merge

def createDecisionTree():
    
    #df = pd.read_csv('ModelData/baselines/final_Baseline_15min_for_Decisiontree.csv', sep=",")
    df = pd.read_csv('../2-CSV/ModelData/baseline/final_Baseline_15min_for_Decisiontree.csv', sep=",")
    
    y = df.Category
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20, random_state=None)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train[['Hour', 'Minute']], y_train)
    features = clf.feature_importances_ 
    score = clf.score(X=X_test[['Hour', 'Minute']], y=y_test) #Returns the mean accuracy on the given test data and labels.
    print('score=', round(score,2))
    #predicted = clf.predict(X_test)
    
    #---------------
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,  
                      filled=True, rounded=True,  
                      special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())
    
    #--------- RULES
    trainX = X_train[['Hour', 'Minute']]
    tree_to_code(clf, trainX.columns.values)
    
    
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print( "{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)     

def trainModelToCompareWithBaseline():
    ########## couldnot nou use this function
    #df = PrepareDataset()  
    #--------------
    train_size = int(df.shape[0]*0.75)
    df_train = df.iloc[1: train_size]
    df_test = df.iloc[train_size+1:df.shape[0]]
    #--------------
    cols=['Hour', 'Minute']
    from sklearn import linear_model
    trainX=df_train[cols]
    trainy=df_train['LoadFactor']  
    model = linear_model.LinearRegression()
    model.fit(trainX, trainy)   
    print('Coefficients: \n', model.coef_)
    #--------------
    #groupby???
    #--------------
    #test
    import statistics as st
    import math
    df_baseline = df = pd.read_csv('ModelData/baselines/final_Baseline_15min_colors_percent_final.csv', sep=",")
    diff = df_test['LoadFactor'].values - df_baseline['LoadFactor'].values
    rmse = math.sqrt(st.mean((diff)**2))
    print("RMSE= " , rmse , '\n')


def OverloadPercentagePerRoute():
    
    df = pd.read_csv('ModelData/clean/oneYear17-18_clean.csv', sep=",")
    print('csv is read!')
    #clean----------
    df = df[(df['Rout'] != 0) & (df['Rout'] != 998) & (df['Uniqu'] != '00009999') & (df['Uniqu'] != '00009998') & (df['Uniqu'] != '00009997') & (df['Uniqu'] != '00009996') & (df['Uniqu'] != '00009995')]
    df['duplicate_flag'] = df.duplicated(subset=['_Date_','TripID', 'Uniqu'], keep=False)
    df = df.loc[df['duplicate_flag'] == False]
    print('dataset is clean!')
    #-----------    
    df['Full'] = df['Full'].astype(int)
    df['load'] = df['load'].astype(int)
    df['LoadFactor'] =  df[['load']].div(df.Full , axis=0)
    df['LoadFactor'] = df['LoadFactor'].astype(float)    
    print('LoadFactor is added!')
            
      
    groupby_df_A = df.groupby(["Rout"])[["LoadFactor"]].count().reset_index()
    groupby_df_A.columns = ['Rout', 'ALL']

    
    groupby_df_B = df[(df['LoadFactor'] > 1)].groupby(["Rout"])[["LoadFactor"]].count().reset_index()
    groupby_df_B.columns = ['Rout', 'Overload']    
    
    groupby_df_merge = pd.merge(groupby_df_A , groupby_df_B , on=["Rout"], how='outer')
    groupby_df_merge = groupby_df_merge.fillna(0)
    # number of all trips for all routes during the year: 89,812,851
    TotalTrips = groupby_df_merge['ALL'].sum()
    #print(TotalTrips)
    #TotalLF = groupby_df_merge['Overload'].sum()
    
    groupby_df_merge['Importance'] = round(groupby_df_merge[['ALL']].div(TotalTrips, axis="index"),2)
    groupby_df_merge['Importance*Overload'] = groupby_df_merge[['Overload']].multiply(groupby_df_merge['Importance'], axis="index")
    #groupby_df_merge['Final'] = round(groupby_df_merge[['Overload']].div(groupby_df_merge.ALL, axis="index")*100,2)  
    groupby_df_merge.sort_values(["Importance*Overload"], inplace=True, ascending=False)
   
    print('Merge is done!')
    groupby_df_merge.to_csv('ModelData/baselines/Overload_Freq_Per_Route.csv', sep=',') 
    print('output csv is done!')   


def OverloadPercentagePerStop():
    
    df = pd.read_csv('ModelData/clean/oneYear17-18_clean.csv', sep=",")
    print('csv is read!')
    #clean----------
    df = df[(df['Rout'] != 0) & (df['Rout'] != 998) & (df['Uniqu'] != '00009999') & (df['Uniqu'] != '00009998') & (df['Uniqu'] != '00009997') & (df['Uniqu'] != '00009996') & (df['Uniqu'] != '00009995')]
    df['duplicate_flag'] = df.duplicated(subset=['_Date_','TripID', 'Uniqu'], keep=False)
    df = df.loc[df['duplicate_flag'] == False]
    print('dataset is clean!')
    #-----------    
    df['Full'] = df['Full'].astype(int)
    df['load'] = df['load'].astype(int)
    df['LoadFactor'] =  df[['load']].div(df.Full , axis=0)
    df['LoadFactor'] = df['LoadFactor'].astype(float)    
    print('LoadFactor is added!')
        
    groupby_df_A = df.groupby(["Uniqu"])[["LoadFactor"]].count().reset_index()
    groupby_df_A.columns = ['Uniqu', 'ALL']    
    groupby_df_B = df[(df['LoadFactor'] > 1)].groupby(["Uniqu"])[["LoadFactor"]].count().reset_index()
    groupby_df_B.columns = ['Uniqu', 'Overload']
    groupby_df_merge = pd.merge(groupby_df_A , groupby_df_B , on=["Uniqu"], how='outer')
    groupby_df_merge = groupby_df_merge.fillna(0)
    # number of all trips for all routes during the year: 89812851
    TotalTrips = groupby_df_merge['ALL'].sum()
    #TotalLF = groupby_df_merge['Overload'].sum()
    
    groupby_df_merge['Importance'] = round(groupby_df_merge[['ALL']].div(TotalTrips, axis="index"),5)
    groupby_df_merge['Importance*Overload'] = groupby_df_merge[['Overload']].multiply(groupby_df_merge['Importance'], axis="index")
    #groupby_df_merge['Final'] = round(groupby_df_merge[['Overload']].div(groupby_df_merge.ALL, axis="index")*100,2)  
    groupby_df_merge.sort_values(["Importance*Overload"], inplace=True, ascending=False)
   
    print('Merge is done!')
    groupby_df_merge.to_csv('ModelData/baselines/Overload_Freq_Per_Stop.csv', sep=',') 
    print('output csv is done!')   
    
def OverloadPercentagePerTime():
    
    df = pd.read_csv('ModelData/clean/oneYear17-18_clean.csv', sep=",")
    print('csv is read!')
    #clean----------
    df = df[(df['Rout'] != 0) & (df['Rout'] != 998) & (df['Uniqu'] != '00009999') & (df['Uniqu'] != '00009998') & (df['Uniqu'] != '00009997') & (df['Uniqu'] != '00009996') & (df['Uniqu'] != '00009995')]
    df['duplicate_flag'] = df.duplicated(subset=['_Date_','TripID', 'Uniqu'], keep=False)
    df = df.loc[df['duplicate_flag'] == False]
    print('dataset is clean!')
    #-----------   
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
    df.loc[ (df.Arrive >= 260000) & (df.Arrive < 270000) , 'Hour'] = 2 
    df.loc[ (df.Arrive >= 30000) & (df.Arrive < 40000) , 'Hour'] = 3
    df.loc[ (df.Arrive >= 40000) & (df.Arrive < 50000) , 'Hour'] = 4 
    print('Hour is added!')
    ##########
    ##seperating minute from time
    df['Min'] = 0
    df['Arrive'] = df['Arrive'].astype(str)
    df['Min'] = np.where(((df['Hour'] >= 3) & (df['Hour'] <= 9)), df.Arrive.str[1:3] , df.Arrive.str[2:4])
    print('Min is added!')
    
    df['Minute'] = 0
    df['Min'] = df['Min'].astype(int)
    df.loc[((df['Min'] >= 00) & (df['Min'] < 15)), 'Minute'] = 0
    df.loc[((df['Min'] >= 15) & (df['Min'] < 30)), 'Minute'] = 15
    df.loc[((df['Min'] >= 30) & (df['Min'] < 45)), 'Minute'] = 30
    df.loc[((df['Min'] >= 45) & (df['Min'] <= 59)), 'Minute'] = 45
    print('Minute is added!')
    
    df['Full'] = df['Full'].astype(int)
    df['load'] = df['load'].astype(int)
    df['LoadFactor'] =  df[['load']].div(df.Full , axis=0)
    df['LoadFactor'] = df['LoadFactor'].astype(float)    
    print('LoadFactor is added!')
    
    groupby_df_A = df.groupby(["Hour", "Minute"])[["LoadFactor"]].count().reset_index()
    groupby_df_A.columns = ["Hour", "Minute", 'ALL']
        
    groupby_df_B = df[(df['LoadFactor'] > 1)].groupby(["Hour", "Minute"])[["LoadFactor"]].count().reset_index()
    groupby_df_B.columns = ["Hour", "Minute", 'Overload']
    
    groupby_df_merge = pd.merge(groupby_df_A , groupby_df_B , on=["Hour", "Minute"], how='outer')
    groupby_df_merge = groupby_df_merge.fillna(0)
    
    # number of all trips for all routes during the year: 89812851
    TotalTrips = groupby_df_merge['ALL'].sum()
    
    groupby_df_merge['Importance'] = round(groupby_df_merge[['ALL']].div(TotalTrips, axis="index"),2)
    groupby_df_merge['Importance*Overload'] = groupby_df_merge[['Overload']].multiply(groupby_df_merge['Importance'], axis="index")
    #groupby_df_merge['Final'] = round(groupby_df_merge[['Overload']].div(groupby_df_merge.ALL, axis="index")*100,2)  
    groupby_df_merge.sort_values(["Importance*Overload"], inplace=True, ascending=False)
   
    print('Merge is done!')
    groupby_df_merge.to_csv('ModelData/baselines/Overload_Freq_Per_Time.csv', sep=',') 
    print('output csv is done!')   
      

def Barchart():
    
    df = pd.read_csv('../2-CSV/ModelData/baseline/Overload_Freq_Per_Route.csv', sep=",")
    fig, ax = plt.subplots(figsize = (10,4))
    df['Rout'] = df['Rout'].astype(str)
    x = df['Rout']
    y = df['Overload']
    ax.bar(x , y, color='blue')
    
    plt.xticks(np.arange(len(x)+1), x ,rotation=90)
    #plt.yticks(np.arange(len(y)+1), y)
    ax.set_xlabel("Route", fontsize=12)
    ax.set_ylabel("Overload", fontsize=12)
    #ax.set_xticklabels(df['Rout'], rotation = 'vertical', ha="right" , fontsize=6)
    plt.savefig("../2-CSV/ModelData/baseline/Overload_Freq_Per_Route.png")
    
    
def NumberofStopsPerRouteDirection():
    
    df_stopsperroute = pd.read_csv('../2-CSV/ModelData/numberofStopsPerRoute.csv', sep=",") 
    bins = [0,10, 20, 30, 40, 50,60,70,80,90,100,110, 120,130,140,150, 160, 170,180]
    group_names = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '101-110', '111-120', '121-130', '131-140', '141-150', '151-160', '161-170', '171-180']
    df_stopsperroute['Cat_in'] = pd.cut(df_stopsperroute['inbound'], bins=bins, labels=group_names)
    df_stopsperroute['Cat_out'] = pd.cut(df_stopsperroute['outbound'], bins=bins, labels=group_names)
    #print(df_stopsperroute.head(10))
    print(pd.value_counts(df_stopsperroute['Cat_in']) , '\n')
    print(pd.value_counts(df_stopsperroute['Cat_out']))
        
def PartitionRoutesInBaselineBothDirections():
    import operator
    df = pd.read_csv('../2-CSV/ModelData/4-baseline/final_Baseline_with_new_categories.csv', sep=",")
    df_routeMapping = pd.read_csv('../2-CSV/ModelData/RouteMappings.csv', sep=",") 
    df_stops = pd.read_csv('../2-CSV/ModelData/allStopsperRoute_weekdays_with_direction.csv', sep=",") 
    
    ManyEmptySeats_in = {}   #{route_i, #green}
    FewEmptySeats_in = {}
    FewPeopleStand_in = {}
    ManyPeopleStand_in = {}
    Crushed_in = {}
    
    ManyEmptySeats_out = {}   
    FewEmptySeats_out = {}
    FewPeopleStand_out = {}
    ManyPeopleStand_out = {}
    Crushed_out = {}
    
    for route in df_routeMapping['route_id']:
        route_fullname =  df_routeMapping.loc[df_routeMapping['route_id'] == route, 'route_fullname'].iloc[0]
        route_name =  df_routeMapping.loc[df_routeMapping['route_id'] == route, 'route_name'].iloc[0]
        stop_count_inbound = df_stops[(df_stops['route_id'] == route_fullname) & (df_stops['direction_id'] == 1)].shape[0]
        stop_count_outbound = df_stops[(df_stops['route_id'] == route_fullname)& (df_stops['direction_id'] == 0)].shape[0]
        #records_total = df[df['Rout'] == route].shape[0]
        #print(route_name , stop_count, records_total)

        if (stop_count_inbound != 0):        
            ManyEmptySeats_in[route_name] = round((df[(df['Rout'] == route) & (df['Category'] == 1)].shape[0])/stop_count_inbound, 3)
            FewEmptySeats_in[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 2)].shape[0])/stop_count_inbound, 3)
            FewPeopleStand_in[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 3)].shape[0])/stop_count_inbound, 3)
            ManyPeopleStand_in[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 4)].shape[0])/stop_count_inbound, 3)
            Crushed_in[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 5)].shape[0])/stop_count_inbound, 3)
            
        if (stop_count_outbound != 0):        
            ManyEmptySeats_out[route_name] = round((df[(df['Rout'] == route) & (df['Category'] == 1)].shape[0])/stop_count_outbound, 3)
            FewEmptySeats_out[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 2)].shape[0])/stop_count_outbound, 3)
            FewPeopleStand_out[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 3)].shape[0])/stop_count_outbound, 3)
            ManyPeopleStand_out[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 4)].shape[0])/stop_count_outbound, 3)
            Crushed_out[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 5)].shape[0])/stop_count_outbound, 3)

    sorted_ManyEmptySeats_in = sorted(ManyEmptySeats_in.items(), key=operator.itemgetter(1), reverse=True)
    sorted_FewEmptySeats_in = sorted(FewEmptySeats_in.items(), key=operator.itemgetter(1), reverse=True)
    sorted_FewPeopleStand_in = sorted(FewPeopleStand_in.items(), key=operator.itemgetter(1), reverse=True)
    sorted_ManyPeopleStand_in = sorted(ManyPeopleStand_in.items(), key=operator.itemgetter(1), reverse=True)
    sorted_Crushed_in = sorted(Crushed_in.items(), key=operator.itemgetter(1), reverse=True)
    
    
    sorted_ManyEmptySeats_out = sorted(ManyEmptySeats_out.items(), key=operator.itemgetter(1), reverse=True)
    sorted_FewEmptySeats_out = sorted(FewEmptySeats_out.items(), key=operator.itemgetter(1), reverse=True)
    sorted_FewPeopleStand_out = sorted(FewPeopleStand_out.items(), key=operator.itemgetter(1), reverse=True)
    sorted_ManyPeopleStand_out = sorted(ManyPeopleStand_out.items(), key=operator.itemgetter(1), reverse=True)
    sorted_Crushed_out = sorted(Crushed_out.items(), key=operator.itemgetter(1), reverse=True)
    
    
    print('inbound list: \n', sorted_ManyEmptySeats_in, '\n\n', sorted_FewEmptySeats_in, '\n\n', sorted_FewPeopleStand_in, '\n\n', sorted_ManyPeopleStand_in , '\n\n', sorted_Crushed_in)
    print('outbound list: \n', sorted_ManyEmptySeats_out, '\n\n', sorted_FewEmptySeats_out, '\n\n', sorted_FewPeopleStand_out, '\n\n', sorted_ManyPeopleStand_out , '\n\n', sorted_Crushed_out)
    #sort dictionaries
    
def PartitionRoutesInBaseline():
    import operator
    df = pd.read_csv('../2-CSV/ModelData/4-baseline/final_Baseline_with_new_categories.csv', sep=",")
    df_routeMapping = pd.read_csv('../2-CSV/ModelData/RouteMappings.csv', sep=",") 
    df_stops = pd.read_csv('../2-CSV/ModelData/allStopsperRoute_weekdays_with_direction.csv', sep=",") 
    
    ManyEmptySeats = {}   #{route_i, #green}
    FewEmptySeats = {}
    FewPeopleStand = {}
    ManyPeopleStand = {}
    Crushed = {}
    
    for route in df_routeMapping['route_id']:
        route_fullname =  df_routeMapping.loc[df_routeMapping['route_id'] == route, 'route_fullname'].iloc[0]
        route_name =  df_routeMapping.loc[df_routeMapping['route_id'] == route, 'route_name'].iloc[0]
        stop_count = df_stops[(df_stops['route_id'] == route_fullname)].shape[0]
        #records_total = df[df['Rout'] == route].shape[0]
        #print(route_name , stop_count, records_total)

        if (stop_count != 0):        
            ManyEmptySeats[route_name] = round((df[(df['Rout'] == route) & (df['Category'] == 1)].shape[0])/stop_count, 3)
            FewEmptySeats[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 2)].shape[0])/stop_count, 3)
            FewPeopleStand[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 3)].shape[0])/stop_count, 3)
            ManyPeopleStand[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 4)].shape[0])/stop_count, 3)
            Crushed[route_name] =  round((df[(df['Rout'] == route) & (df['Category'] == 5)].shape[0])/stop_count, 3)

    sorted_ManyEmptySeats = sorted(ManyEmptySeats.items(), key=operator.itemgetter(1), reverse=True)
    sorted_FewEmptySeats = sorted(FewEmptySeats.items(), key=operator.itemgetter(1), reverse=True)
    sorted_FewPeopleStand = sorted(FewPeopleStand.items(), key=operator.itemgetter(1), reverse=True)
    sorted_ManyPeopleStand = sorted(ManyPeopleStand.items(), key=operator.itemgetter(1), reverse=True)
    sorted_Crushed = sorted(Crushed.items(), key=operator.itemgetter(1), reverse=True)
    
    print(sorted_ManyEmptySeats, '\n\n', sorted_FewEmptySeats, '\n\n', sorted_FewPeopleStand, '\n\n', sorted_ManyPeopleStand , '\n\n', sorted_Crushed)
    #sort dictionaries
    
def Compute15minBaselineRMSE():
    #comapre baseline15-min with original data records   
    df = pd.read_csv('ModelData/clean/oneYear17-18_clean_20_percent_testset.csv', sep=",")
    baseline = pd.read_csv('ModelData/baselines/final_Baseline_15min_80_percent_trainset.csv', sep=",")
    #--------------------------
    #routes = [612,714, 903, 281, 444, 613] 
    route = 56
    stop = 'E25085'    
    df = df[(df['Rout'] == route) & (df['Uniqu'] == stop)]     
    RMSE = 0
    diff = 0
    trueCount = 0
    
    rows  = baseline[(baseline['Rout'] == route) & (baseline['Uniqu'] == stop)]
    for row in rows[:-1].itertuples(): 
        hour  = rows.loc[row.Index, 'Hour']
        minute  = rows.loc[row.Index, 'Minute']
        baselineLoad  = rows.loc[row.Index, 'MeanLoad']
        df_match  = df[(df['Hour'] == hour) & (df['Minute'] == minute)]
        trueloads = df_match['load']
        
        if len(trueloads) != 0:
            trueCount = trueCount + len(trueloads)
            for trueload in trueloads:
                diff = diff + (trueload - baselineLoad)**2
                
    RMSE = math.sqrt(diff / trueCount)
    print('Baseline RMSE for ', route , 'is ', round(RMSE,2))    
    

def CreateBaselineTrainSet():
    #comapre baseline15-min with original data records   
    
    # first part: create baseline train set 
    df = pd.read_csv('ModelData/NewData/csv/clean/oneYear18-19_clean.csv', sep=",")
    print('csv is read!')
    #--------------------------
    df = df[(df['Rout'] != 0) & (df['Rout'] != 998) & (df['Uniqu'] != '00009999') & (df['Uniqu'] != '00009998') & (df['Uniqu'] != '00009997') & (df['Uniqu'] != '00009996') & (df['Uniqu'] != '00009995')]
    #---------------------
    #df = PrepareDataset(df)
    #--------------------   
    y = df.load
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20, random_state=None) 
    #train_size = int(df.shape[0]*0.8)
    #df_test = df.iloc[train_size+1:df.shape[0]]
    df = X_test         
    df.to_csv('ModelData/NewData/csv/clean/oneYear18-19_clean_20_percent_testset.csv', sep=',')  
    #--------------------------
    FinalBaselinePerQuarterWithDirection(X_train)   
    ############################################
    
def Compute15minBaselineLogLoss():
    #--------------------------
    df = pd.read_csv('ModelData/clean/oneYear17-18_clean_20_percent_testset.csv', sep=",")
    baseline = pd.read_csv('ModelData/baselines/final_Baseline_5category_15min_80_percent_trainset.csv', sep=",")
    #--------------------------
    #--------------------------
    route = 281
    stop = 'P02430' 
    df = df[(df['Rout'] == route) & (df['Uniqu'] == stop)]    
    logloss = 0
    sumlog = 0
    trueCount = 0
    target = 0
    predict_prob = 0
    matchcount = 0
    accuracy = 0
    
    rows  = baseline[(baseline['Rout'] == route) & (baseline['Uniqu'] == stop)]
    for row in rows[:-1].itertuples(): 
        hour  = rows.loc[row.Index, 'Hour']
        minute  = rows.loc[row.Index, 'Minute']
        PredictedCategory  = rows.loc[row.Index, 'Category']
        
        df_match  = df[(df['Hour'] == hour) & (df['Minute'] == minute)]
        trueCategories = df_match['load_category']
        
        if len(trueCategories) != 0:
            trueCount = trueCount + len(trueCategories)
            for trueCategory in trueCategories:
                predict_prob = list(trueCategories).count(trueCategory) / len(trueCategories)
                sumlog+=  - np.log(predict_prob) 
                # cross entropy loss = log loss (-target {1 if true label=predicted label , 0 otherwise} * log (predict_prob))
                if (trueCategory == PredictedCategory):
                    matchcount = matchcount + 1
                
           
    logloss = sumlog / trueCount 
    accuracy = matchcount / trueCount
    print('Baseline Log Loss for ', route , 'at ', stop , ' is ', round(logloss,3))    
    print('Baseline Accuracy for ', route , 'at ', stop , ' is ', round(accuracy,3)) #accuracy = F1 micro


def ComputeBaselineLogLossForOneRouteAllStops():
    
    df = pd.read_csv('ModelData/clean/oneYear17-18_clean_20_percent_testset.csv', sep=",")
    baseline = pd.read_csv('ModelData/baselines/final_Baseline_5category_15min_80_percent_trainset.csv', sep=",")

    route = 612
    df = df[(df['Rout'] == route)]    
    logloss = 0
    sumlog = 0
    trueCount = 0
    predict_prob = 0
    matchcount = 0
    accuracy = 0
    
    rows  = baseline[(baseline['Rout'] == route)]
    for row in rows[:-1].itertuples(): 
        hour  = rows.loc[row.Index, 'Hour']
        minute  = rows.loc[row.Index, 'Minute']
        PredictedCategory  = rows.loc[row.Index, 'Category']
        
        df_match  = df[(df['Hour'] == hour) & (df['Minute'] == minute)]
        trueCategories = df_match['load_category']
        
        if len(trueCategories) != 0:
            trueCount = trueCount + len(trueCategories)
            for trueCategory in trueCategories:
                predict_prob = list(trueCategories).count(trueCategory) / len(trueCategories)
                sumlog+=  - np.log(predict_prob) 
                # cross entropy loss = log loss (-target {1 if true label=predicted label , 0 otherwise} * log (predict_prob))
                if (trueCategory == PredictedCategory):
                    matchcount = matchcount + 1
                
           
    logloss = sumlog / trueCount 
    accuracy = matchcount / trueCount
    print('Baseline Log Loss for ', route , ' is ', round(logloss,3))    
    print('Baseline Accuracy for ', route , ' is ', round(accuracy,3)) #accuracy = F1 micro

        
def ComputeBaselineLogLossForOneRouteAllStopsWithDirection(df, baseline, route, direction):

    new_df = df[(df['Rout'] == route) & (df['Direction'] == direction)]    
    logloss = 0
    sumlog = 0
    trueCount = 0
    predict_prob = 0
    matchcount = 0
    accuracy = 0
    
    rows  = baseline[(baseline['Rout'] == route) & (baseline['Direction'] == direction)]
    for row in rows[:-1].itertuples(): 
        hour  = rows.loc[row.Index, 'Hour']
        minute  = rows.loc[row.Index, 'Minute']
        PredictedCategory  = rows.loc[row.Index, 'Category']
        
        df_match  = new_df[(new_df['Hour'] == hour) & (new_df['Minute'] == minute)]
        trueCategories = df_match['load_category']
        
        if len(trueCategories) != 0:
            trueCount = trueCount + len(trueCategories)
            for trueCategory in trueCategories:
                predict_prob = list(trueCategories).count(trueCategory) / len(trueCategories)
                sumlog+=  - np.log(predict_prob) 
                # cross entropy loss = log loss (-target {1 if true label=predicted label , 0 otherwise} * log (predict_prob))
                if (trueCategory == PredictedCategory):
                    matchcount = matchcount + 1
                
           
    logloss = sumlog / trueCount 
    accuracy = matchcount / trueCount
    print('Baseline Log Loss for ', route , ' ' , direction , ' is ', round(logloss,2))    
    print('Baseline F1 score for ', route , ' ' , direction , ' is ', round(accuracy,2)) #accuracy = F1 micro 
    
    import csv
    with open('ModelData/selected_routes_baseline_logloss_f1.csv','a') as file:
        writer = csv.writer(file)
        writer.writerow([route, direction , round(logloss,2), round(accuracy,2)])
     
def ComputeBaselineRMSEForOneRouteAllStopsWithDirection(df, baseline, route, direction):    
    new_df = df[(df['Rout'] == route) & (df['Direction'] == direction)]  
    
    RMSE = 0
    diff = 0
    trueCount = 0
    
    rows  = baseline[(baseline['Rout'] == route) & (baseline['Direction'] == direction)]
    for row in rows[:-1].itertuples(): 
        hour  = rows.loc[row.Index, 'Hour']
        minute  = rows.loc[row.Index, 'Minute']
        baselineLoad  = rows.loc[row.Index, 'MeanLoad'] #load1 instead of mean load
        df_match  = new_df[(new_df['Hour'] == hour) & (new_df['Minute'] == minute)]
        trueloads = df_match['load']
        
        if len(trueloads) != 0:
            trueCount = trueCount + len(trueloads)
            for trueload in trueloads:
                diff = diff + (trueload - baselineLoad)**2

    RMSE = math.sqrt(diff / trueCount)
    print('Baseline RMSE for ', route , ' ' , direction ,  'is ', round(RMSE,2))  
    
    import csv
    with open('ModelData/selected_routes_baseline_rmse.csv','a') as file:
        writer = csv.writer(file)
        writer.writerow([route, direction , round(RMSE,2)])

def ComputeBaselineRMSEForOneRouteWithStopWithDirection(df, baseline, route, direction):  
    
    new_df = df[(df['Rout'] == route) & (df['Direction'] == direction)]
    rows  = baseline[(baseline['Rout'] == route) & (baseline['Direction'] == direction)]
    stops =  new_df['Uniqu'].unique()
    stop_size = len(stops)
    print(route, direction, stop_size) 
      
    sumRMSE = 0
    
    for stop in stops:
    
        true_df = new_df[new_df['Uniqu'] == stop]    
        predict_df  = rows[(rows['Uniqu'] == stop)]
        
        RMSE = 0
        diff = 0
        trueCount = 0
    
        for row in predict_df[:-1].itertuples(): 
            hour  = predict_df.loc[row.Index, 'Hour']
            minute  = predict_df.loc[row.Index, 'Minute']
            baselineLoad  = predict_df.loc[row.Index, 'MeanLoad'] #load1 instead of mean load
            df_match  = true_df[(true_df['Hour'] == hour) & (true_df['Minute'] == minute)]
            trueloads = df_match['load']
        
            if len(trueloads) != 0:
                trueCount = trueCount + len(trueloads)
                for trueload in trueloads:
                    diff = diff + (trueload - baselineLoad)**2
                    
        if trueCount != 0:            
            RMSE = math.sqrt(diff / trueCount)
            sumRMSE = sumRMSE + RMSE
        
    print('Baseline RMSE for ', route , ' ' , direction ,  'is ', round(sumRMSE/stop_size,2))  
    
    import csv
    with open('ModelData/selected_routes_baseline18-19_RMSE_avg_stops.csv','a') as file:
        writer = csv.writer(file)
        writer.writerow([route, direction , round(sumRMSE/stop_size,2)])
        
    
def ComputeBaselineLogLossForOneRouteWithStopWithDirection(df, baseline, route, direction):
    
    new_df = df[(df['Rout'] == route) & (df['Direction'] == direction)]
    rows  = baseline[(baseline['Rout'] == route) & (baseline['Direction'] == direction)]
    stops =  new_df['Uniqu'].unique()
    stop_size = len(stops)
    print(route, ' ', direction, ' ', stop_size)
    
    sumlogloss = 0
    sumacc = 0
    
    for stop in stops:
        
        true_df = new_df[new_df['Uniqu'] == stop]    
        predict_df  = rows[(rows['Uniqu'] == stop)]
        
        logloss = 0
        sumlog = 0
        trueCount = 0
        predict_prob = 0
        matchcount = 0
        accuracy = 0
        
        for row in predict_df[:-1].itertuples(): 
            hour  = predict_df.loc[row.Index, 'Hour']
            minute  = predict_df.loc[row.Index, 'Minute']
            PredictedCategory  = predict_df.loc[row.Index, 'Category']
            
            df_match  = true_df[(true_df['Hour'] == hour) & (true_df['Minute'] == minute)]
            trueCategories = df_match['load_category']
            
            if len(trueCategories) != 0:
                trueCount = trueCount + len(trueCategories)
                for trueCategory in trueCategories:
                    predict_prob = list(trueCategories).count(trueCategory) / len(trueCategories)
                    sumlog+=  - np.log(predict_prob) 
                    # cross entropy loss = log loss (-target {1 if true label=predicted label , 0 otherwise} * log (predict_prob))
                    if (trueCategory == PredictedCategory):
                        matchcount = matchcount + 1
                    
        if trueCount != 0:       
            
            logloss = sumlog / trueCount 
            accuracy = matchcount / trueCount
            
            sumlogloss = sumlogloss + logloss
            sumacc = sumacc + accuracy
            
        else:
            print('trueCount for ', stop, 'is zero')
        
    print('Baseline Log Loss for ', route , ' '  , direction , ' is ', round(sumlogloss/stop_size,2))    
    print('Baseline F1 score for ', route , ' '  , direction , ' is ', round(sumacc/stop_size,2)) #accuracy = F1 micro 

    with open('ModelData/selected_routes_baseline18-19_logloss_f1_avg_stops.csv','a') as file:
        writer = csv.writer(file)
        writer.writerow([route, direction , round(sumlogloss/stop_size,2), round(sumacc/stop_size,2)])

def ComputeBaselineLogLossForOneStopAllRoutes():
    
    df = pd.read_csv('ModelData/clean/oneYear17-18_clean_20_percent_testset.csv', sep=",")
    baseline = pd.read_csv('ModelData/baselines/final_Baseline_5category_15min_80_percent_trainset.csv', sep=",")

    stop = 'E01000'
    df = df[(df['Uniqu'] == stop)]    
    logloss = 0
    sumlog = 0
    trueCount = 0
    predict_prob = 0
    matchcount = 0
    accuracy = 0
    
    rows  = baseline[(baseline['Uniqu'] == stop)]
    for row in rows[:-1].itertuples(): 
        hour  = rows.loc[row.Index, 'Hour']
        minute  = rows.loc[row.Index, 'Minute']
        PredictedCategory  = rows.loc[row.Index, 'Category']
        
        df_match  = df[(df['Hour'] == hour) & (df['Minute'] == minute)]
        trueCategories = df_match['load_category']
        
        if len(trueCategories) != 0:
            trueCount = trueCount + len(trueCategories)
            for trueCategory in trueCategories:
                predict_prob = list(trueCategories).count(trueCategory) / len(trueCategories)
                sumlog+=  - np.log(predict_prob) 
                # cross entropy loss = log loss (-target {1 if true label=predicted label , 0 otherwise} * log (predict_prob))
                if (trueCategory == PredictedCategory):
                    matchcount = matchcount + 1
                
           
    logloss = sumlog / trueCount 
    accuracy = matchcount / trueCount
    print('Baseline Log Loss for ', stop , ' is ', round(logloss,2))    
    print('Baseline Accuracy for ', stop , ' is ', round(accuracy,2)) #accuracy = F1 micro    

def preparebaseline(df):
    df['load1'] = np.where((df.ID == 0) , 0 , df['load'].shift(1))
    #df['load2'] = np.where(((df.ID == 0) | (df.ID == 1)) , 0 , df['load'].shift(2))
    #df['load3'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2)), 0 , df['load'].shift(3))
    #df['load4'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3)), 0 , df['load'].shift(4))
    df['load5'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4)), 0 , df['load'].shift(5))
   
    #df['load6'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4) | (df.ID == 5)), 0 , df['load'].shift(6))
    #df['load7'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4) | (df.ID == 5)| (df.ID == 6)), 0 , df['load'].shift(7))
    #df['load8'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4) | (df.ID == 5) | (df.ID == 6)| (df.ID == 7)), 0 , df['load'].shift(8))
    #df['load9'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4) | (df.ID == 5) | (df.ID == 6) | (df.ID == 7) | (df.ID == 8)), 0 , df['load'].shift(9))
    df['load10'] = np.where(((df.ID == 0) | (df.ID == 1) | (df.ID == 2) | (df.ID == 3) | (df.ID == 4) | (df.ID == 5) | (df.ID == 6) | (df.ID == 7) | (df.ID == 8) | (df.ID == 9)), 0 , df['load'].shift(10))
        
    df = df.dropna()
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    
    df['Full'] = df['Full'].astype(int)
    df['load10'] = df['load10'].astype(int)
    df['LoadFactor'] =  df[['load10']].div(df.Full , axis=0) #------------- load1 <= load
    df['LoadFactor'] = df['LoadFactor'].astype(float)
    #--------------
    # isFull is used in Logistic Regression
    df['isFull'] = np.where((df.LoadFactor > 0.75), 1, 0)   
    #----------------
    df['load_category'] = 1
    df.loc[(df['LoadFactor'] < 0.5), 'load_category'] = 1
    df.loc[((df['LoadFactor'] < 0.8) & (df['LoadFactor'] >= 0.5)), 'load_category'] = 2
    df.loc[((df['LoadFactor'] < 1.1) & (df['LoadFactor'] >= 0.8)), 'load_category'] = 3
    df.loc[((df['LoadFactor'] < 1.4) & (df['LoadFactor'] >= 1.1)), 'load_category'] = 4
    df.loc[(df['LoadFactor'] >= 1.4), 'load_category'] = 5
         
    return df

def trainANDtestSets(df):
    y = df.load
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20, random_state=200)  
    return X_train, X_test

def  baselinebyPooling(df_train, df_test, index):        

    sumlog = 0
    trueCount = 0
    predict_prob = 0
    matchcount = 0
    stops = []

    stop = df_train.loc[index, 'Uniqu']  
    if stop not in stops:
        
        stops.append(stop)          
        PredictedCategory  = df_train.loc[index, 'load_category']               
                   
        df_match  = df_test[(df_test['Uniqu'] == stop)] 
        trueCategories = df_match['load_category']
        
        if len(trueCategories) != 0:
            trueCount = trueCount + len(trueCategories)
            for trueCategory in trueCategories:
                predict_prob = list(trueCategories).count(trueCategory) / len(trueCategories)
                sumlog+=  - np.log(predict_prob) 
                # cross entropy loss = log loss (-target {1 if true label=predicted label , 0 otherwise} * log (predict_prob))
                if (trueCategory == PredictedCategory):
                    matchcount = matchcount + 1
            
    return  sumlog, trueCount, matchcount
   

def baselineOnlyLoad():
    path = 'ModelData/routes-allstops/'
    for filename in os.listdir(path):
        if filename != 'desktop.ini':
            print('filename= ', filename)
            df = pd.read_csv(path+filename, sep=",")
            route = filename.split('.')[0]
            logloss = 0
            sumlog = 0
            trueCount = 0
            predict_prob = 0
            matchcount = 0
            accuracy = 0
            stops = []
            
            new_df = preparebaseline(df)
            new_df = new_df.dropna()
            df_train, df_test = trainANDtestSets(new_df)
            
            
            for row in df_train[:-1].itertuples(): 
                stop = df_train.loc[row.Index, 'Uniqu']
                
                if stop not in stops: #there is no big difference between the results if I dont ignore the stops that have been already seen but this way is much faster.
                    stops.append(stop)
                    PredictedCategory  = df_train.loc[row.Index, 'load_category']
                    
                    #df_match  = df_test[(df_test['Uniqu'] == stop) & (df_test['Year'] == year) & (df_test['Month'] == month) & (df_test['Day'] == day) & (df_test['Hour'] == hour) & (df_test['Minute'] == minute)] 
                    df_match  = df_test[(df_test['Uniqu'] == stop)] 
                    trueCategories = df_match['load_category']
                    
                    if len(trueCategories) != 0:
                        trueCount = trueCount + len(trueCategories)
                        for trueCategory in trueCategories:
                            predict_prob = list(trueCategories).count(trueCategory) / len(trueCategories)
                            sumlog+=  - np.log(predict_prob) 
                            # cross entropy loss = log loss (-target {1 if true label=predicted label , 0 otherwise} * log (predict_prob))
                            if (trueCategory == PredictedCategory):
                                matchcount = matchcount + 1
                        
                   
            logloss = sumlog / trueCount 
            accuracy = matchcount / trueCount
            print('Baseline Log Loss for ', route , ' is ', round(logloss,2))    
            print('Baseline F1 score for ', route , ' is ', round(accuracy,2)) #accuracy = F1 micro 
            
            import csv
            with open('ModelData/selected_routes_baseline_logloss_f1_only_10th_Pload_2.csv','a') as file:
                writer = csv.writer(file)
                writer.writerow([route, round(logloss,2), round(accuracy,2)]) 

def aggregateCategoriesBaseline_faster():
    
    df_5min = pd.read_csv('../2-CSV/ModelData/4-baseline/test_baseline_5min.csv', sep=",")
    df_15min = pd.read_csv('../2-CSV/ModelData/4-baseline/test_baseline_15min.csv', sep=",")
    routes = pd.read_csv('../2-CSV/ModelData/4-baseline/RouteMappings.csv',sep=",")
    
    df_agg = pd.DataFrame(columns=['Rout', 'Uniqu', 'Hour', 'Minute', 'ALL', 'Crushed', 'ManyPeopleStand', 'FewPeopleStand'
                                   , 'FewEmptySeats', 'ManyEmptySeats', 'MeanLoad', 'Category'])
    
    for route in routes['route_id']:
        
        new_df = df_5min[(df_5min['Rout'] == route)]
        stops =  new_df['Uniqu'].unique()
        minutes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        
        for stop in stops:
            
            for hour in range(0,24):
                
                for minute in minutes:
                    
                    
                    new_df_h = new_df[(new_df['Uniqu'] == stop) & (new_df['Hour'] == hour)]

                    if len(new_df_h) != 0:
                        new_df_m = new_df_h[(new_df_h['Minute'] == minute)]
                
                        if len(new_df_m) != 0:
                            
                            count = new_df_m['ALL'].iloc[0]
                            #print(route, stop, hour, minute , count)
                            
                            if (count < 10): 
                                
                                if (minute >= 0) & (minute < 15):
                                    minute_15min = 0
                                elif (minute >= 15) & (minute < 30):
                                    minute_15min = 15
                                elif (minute >= 30) & (minute < 45):
                                    minute_15min = 30
                                elif (minute >= 45) & (minute <= 59):
                                    minute_15min = 45    
                                    
                                related_row_15min = df_15min[(df_15min['Rout'] == route) & (df_15min['Uniqu'] == stop) & (df_15min['Hour'] == hour) & (df_15min['Minute'] == minute_15min)]  
                                
                                if len(related_row_15min) != 0:
                                    
                                    count_15min = related_row_15min['ALL'].iloc[0]
                                    
                    
                                    if count_15min >= 10: #if count_15min < 10 => dont report
                                    
                                        crushed_15min = related_row_15min['Crushed'].iloc[0]
                                        manyPeopleStand_15min = related_row_15min['ManyPeopleStand'].iloc[0]
                                        fewPeopleStand_15min = related_row_15min['FewPeopleStand'].iloc[0]
                                        fewEmptySeats_15min = related_row_15min['FewEmptySeats'].iloc[0]
                                        manyEmptySeats_15min = related_row_15min['ManyEmptySeats'].iloc[0]
                                        meanLoad_15min = related_row_15min['MeanLoad'].iloc[0]
                                        category_15min = related_row_15min['Category'].iloc[0]
                                        
                                        #-------------------
                                        if minute_15min == 0:
                                            related_rows_5min = new_df_h[(new_df_h['Minute'] >= 0) & (new_df_h['Minute'] < 15)]
                                        elif minute_15min == 15:
                                            related_rows_5min = new_df_h[(new_df_h['Minute'] >= 15) & (new_df_h['Minute'] < 30)]
                                        elif minute_15min == 30:
                                            related_rows_5min = new_df_h[(new_df_h['Minute'] >= 30) & (new_df_h['Minute'] < 45)]
                                        elif minute_15min == 45:
                                             related_rows_5min = new_df_h[(new_df_h['Minute'] >= 45) & (new_df_h['Minute'] <= 59)]
                                              
                                             
                                        related_rows_5min['ALL'] = count_15min
                                        related_rows_5min['Crushed'] = crushed_15min
                                        related_rows_5min['ManyPeopleStand'] = manyPeopleStand_15min
                                        related_rows_5min['FewPeopleStand'] = fewPeopleStand_15min
                                        related_rows_5min['FewEmptySeats'] = fewEmptySeats_15min
                                        related_rows_5min['ManyEmptySeats'] = manyEmptySeats_15min
                                        related_rows_5min['MeanLoad'] = meanLoad_15min 
                                        related_rows_5min['Category'] = category_15min
                                        
                                        #print(related_rows_5min['ALL'], related_rows_5min['Crushed'],related_rows_5min['Category'] )
                                        df_agg = df_agg.append(related_rows_5min)
                                        
                            else: # count >= 10
                
                              df_agg = df_agg.append(pd.Series([new_df_m['Rout'].iloc[0], new_df_m['Uniqu'].iloc[0],
                              new_df_m['Hour'].iloc[0], new_df_m['Minute'].iloc[0] , new_df_m['ALL'].iloc[0] , 
                              new_df_m['Crushed'].iloc[0], 
                              new_df_m['ManyPeopleStand'].iloc[0], new_df_m['FewPeopleStand'].iloc[0],
                              new_df_m['FewEmptySeats'].iloc[0], new_df_m['ManyEmptySeats'].iloc[0],
                              new_df_m['MeanLoad'].iloc[0], new_df_m['Category'].iloc[0]]), 
                              ignore_index=True)            
    

    #df_agg['ALL'] = df_agg['ALL'].astype(int)   
    df_agg_final = df_agg.groupby(['Rout', 'Uniqu', 'Hour', 'Minute'], group_keys=False).apply(lambda x: x.loc[x.ALL.idxmax()])
    df_agg_final.to_csv('../2-CSV/ModelData/4-baseline/test_baseline_aggregated_fast.csv', sep=',', index=False)  
    print('aggregate is done!')    
    #df_agg_final.to_csv('ModelData/baselines/final_Baseline_5category_5min_aggregated.csv', sep=',', index=False)                             
    
if __name__ == '__main__':
     
    print()
    

    


    
    
    
    