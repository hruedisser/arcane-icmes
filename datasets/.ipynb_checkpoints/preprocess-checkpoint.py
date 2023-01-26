import pandas as pds
import datetime
import numpy as np
from datasets import event as evt
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import pickle 
from datasets import features
import matplotlib
import logging

logger = logging.getLogger()

def compare_lists(dicoflists):
    '''
    compare number of events in lists
    '''
    plt.figure()
    years = ['2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
    df = pds.DataFrame(columns = dicoflists, index = years)
    for lists in dicoflists:
        for year in years:
            df[lists][year] = len([x for x in dicoflists[lists] if (str(x.begin.year)==year)])
    ax = df.plot()
    ax.set_ylabel('Number of Events')
    ax.set_xlabel('Year')
    
    print(df)

def getyeardata(years,data):
    '''
    get data for specific years
    '''
    for i, year in enumerate(years):
        if i == 0:
            result = data[data.index.year == year]
        else:
            result = pds.concat([result, data[data.index.year == year]], sort=True)
    return result.sort_index()

def printpercentage(y):
    '''
    print percentage of positive labels
    '''
    return(np.sum(y)/len(y))

def getdatas(train,test,val,data_scaled,truelabel):
    '''
    get split dataset
    '''
    
    X_test = getyeardata(test,data_scaled)
    Y_test = getyeardata(test,truelabel)
    
    X_val = getyeardata(val,data_scaled)
    Y_val = getyeardata(val,truelabel)
    
    X_train = getyeardata(train,data_scaled)
    Y_train = getyeardata(train,truelabel)
    
    logger.info('Percentage of true labels:')    
    logger.info('TEST: %f' %printpercentage(Y_test))
    logger.info('TRAIN: %f' %printpercentage(Y_train))
    logger.info('VAL: %f' %printpercentage(Y_val))
    
    return X_test, Y_test, X_val, Y_val, X_train, Y_train

import heapq

def sublist_creator(lst, n):
    lists = [[] for _ in range(n)]
    totals = [(0, i) for i in range(n)]
    heapq.heapify(totals)
    for value in lst:
        total, index = heapq.heappop(totals)
        lists[index].append(value)
        heapq.heappush(totals, (total + value, index))
    return lists

def automaticsplitter(evtlist, eventyears, groupn):
    """
    Automatically creates balanced splits.
    """
    
    groups = []
    
    eventyearscount = []
    
    eventyearsn = eventyears
    
    for year in eventyearsn:
        
        eventyearscount.append(len([i for i in evtlist if (i.begin.year == year)]))
    
    sublist = sublist_creator(eventyearscount, groupn)
    
    for sub in sublist:
        subgroup = []
        
        for item in sub:
            flag = False
            
            for year in eventyearsn:
                
                if len([i for i in evtlist if (i.begin.year == year)]) == item:
                    if flag == False:
                        eventyearsn.remove(year)
                        subgroup.append(year)
                        flag = True
                
        groups.append(subgroup)

    return groups

def getautomaticsplit(split, evtlist, eventyears):
    """
    Automatically creates balanced splits.
    """
    groups = automaticsplitter(evtlist, eventyears, 5)
    logger.info('Splits consist of the following groups:')  
    logger.info(groups)
    
    if split == 1:

        test = groups[0]
        val = groups[1]
        train = list(set(groups[2] + groups[3] + groups[4]))

    if split == 2:

        test = groups[4]
        val = groups[0]
        train = list(set(groups[1] + groups[2] + groups[3]))
        
    if split == 3:

        test = groups[3]
        val = groups[4]
        train = list(set(groups[0] + groups[1] + groups[2]))
    
    if split == 4:

        test = groups[2]
        val = groups[3]
        train = list(set(groups[4] + groups[0] + groups[1]))
        
    if split == 5:

        test = groups[1]
        val = groups[2]
        train = list(set(groups[3] + groups[4] + groups[0]))
        
    return test, val, train



def clearempties(evtlist, data):
    """
    Clear events from a list for which no data exists.
    """
    evtlistnew = []

    for i in evtlist:
        if len(data[i.begin:i.end]) > 6:
            evtlistnew.append(i)
            
    return evtlistnew

def get_truelabel(data,events, feature):
    '''
    get the true label for each point in time
    '''
    
    x = pds.to_datetime(data.index)
    y = np.zeros(np.shape(data)[0])
    
    for e in events:
        n_true = np.where((x >= e.begin) & (x <= e.end))
        y[n_true] = 1
    
    label = pds.DataFrame(y, index = data.index, columns = [feature])
    
    return label

def get_truelabel_all(data,eventdic, feature):
    '''
    get the true label for each point in time
    '''
    
    x = pds.to_datetime(data.index)
    y = np.zeros(np.shape(data)[0])
    
    for i in eventdic:
        for e in eventdic[str(i)]:
            z = np.zeros(np.shape(data)[0])
            n_true = np.where((x >= e.begin) & (x <= e.end))
            z[n_true] = 1/len(eventdic)
            y = np.add(y,z)
    
    label = pds.DataFrame(y, index = data.index, columns = [feature])
    
    return label


def get_weights(serie, bins):
    a, b = np.histogram(serie, bins=bins)
    weights = 1/(a[1:]/np.sum(a[1:]))
    weights = np.insert(weights, 0,1)
    weights_Serie = pds.Series(index = serie.index, data=1)
    for i in range(1, bins):
        weights_Serie[(serie>b[i]) & (serie<b[i+1])] = weights[i]
    return weights_Serie