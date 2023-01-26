import logging

import numpy as np
import pandas as pds
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
from datasets import event as evt
import matplotlib
from datasets import features

from sklearn.preprocessing import StandardScaler
from datasets import preprocess as pp

from datasets import utils

logger = logging.getLogger()

class singlespacecraft(object):
    """
    Dataset class for dealing with in situ data from a single spacecraft. 
    
    Arguments:
        spacecraft        data from which spacecraft should be loaded
        config            entries in the config file
    
    Output:
        self.all_df       pandas dataframe containing the resampled and preprocessed dataset
        self.eventyears   years that contain events
        self.evtlist      dictionary containing the different event lists
        self.labels_df    pandas dataframe containing the labels calculated from the catalog
        self.X_test       Data used for testing
        self.Y_test       Labels used for testing
        self.X_train      Data used for training
        self.Y_train      Labels used for training
        self.X_val        Data used for training
        self.Y_val        Labels used for training
    """
    
    def __init__(self, spacecraft, config):
              
        if config['catalog'] == 'helcat':
            logger.info("Loading Helio4cast catalog...")
            
            cat = evt.get_catevents(config['data_dir'] + 'HELIO4CAST_ICMECAT_v21_pandas.p',config['spacecraft'])
        
        elif config['catalog'] == 'nguyen':
            logger.info("Loading Nguyen catalog...")
            
            cat = evt.read_csv('listOfICMEs.csv', index_col=None)
        
        elif config['catalog'] == 'allcat':
            logger.info("Loading all catalogs...")
            
            catdic = evt.getallcat(config)
            
        logger.info("Loading data...")
        
        alldata = loadalldata(spacecraft, config['data_dir'])
        
        logger.info("Preprocessing data...")

        data = preprocessdataset(config['resampling'],alldata)
        
        logger.info("Deleting empty events and removing eventless data...")
        
        if config['catalog'] == 'allcat':
            self.all_df, self.eventyears = evt.cleardata(catdic['helcats'], data)
            self.all_df, self.eventyears = evt.cleardata(catdic['nguyen'], self.all_df)
            self.all_df, self.eventyears = evt.cleardata(catdic['chinchilla'], self.all_df)
            self.all_df, self.eventyears = evt.cleardata(catdic['richardson'], self.all_df)
            self.all_df, self.eventyears = evt.cleardata(catdic['chi'], self.all_df)
            
            self.evtlist = {}
            self.evtlist['helcats'] = evt.clearempties(catdic['helcats'],self.all_df)
            self.evtlist['nguyen'] = evt.clearempties(catdic['nguyen'],self.all_df)
            self.evtlist['chinchilla'] = evt.clearempties(catdic['chinchilla'],self.all_df)
            self.evtlist['richardson'] = evt.clearempties(catdic['richardson'],self.all_df)
            self.evtlist['chi'] = evt.clearempties(catdic['chi'],self.all_df)
       
        else:
            self.all_df, self.eventyears = evt.cleardata(cat, data)
        
            self.evtlist = evt.clearempties(cat,self.all_df)
        
        logger.info("Creating label...")

        if config['catalog'] == 'allcat':
            self.labels_df = pds.DataFrame(pp.get_truelabel_all(self.all_df, self.evtlist, config['feature']))
            
       
        else:
            self.labels_df = pds.DataFrame(pp.get_truelabel(self.all_df, cat, config['feature']))
        
        logger.info("Scaling data...")

        data_scaled = scaledata(self.all_df)
        
        logger.info("Splitting data...")
        
        if config['catalog'] == 'allcat':
            self.test, self.val, self.train = pp.getautomaticsplit(config['split'], self.evtlist['chi'], self.eventyears)
        else:
            self.test, self.val, self.train = pp.getautomaticsplit(config['split'], self.evtlist, self.eventyears)
        
        self.X_test, self.Y_test, self.X_val, self.Y_val, self.X_train, self.Y_train = pp.getdatas(self.train,self.test,self.val,data_scaled,self.labels_df)
        
        
        
        
class multispacecraft(object):
    """
    Dataset class for dealing with in situ data from multiple spacecraft. 
    
    Arguments:
        config            entries in the config file
    
    Output:
        self.Dataw        singlespacecraft Data for Wind
        self.Dataa        singlespacecraft Data for STEREO-A
        self.Datab        singlespacecraft Data for STEREO-A
        self.X_test       Data used for testing containing data from all three spacecraft
        self.Y_test       Labels used for testing containing data from all three spacecraft
        self.X_train      Data used for training containing data from all three spacecraft
        self.Y_train      Labels used for training containing data from all three spacecraft
        self.X_val        Data used for training containing data from all three spacecraft
        self.Y_val        Labels used for training containing data from all three spacecraft
    """
    
    def __init__(self, config):
              
        logger.info("Wind Data...")
        
        self.Dataw = singlespacecraft(config['resampling'], 'Wind',config['split'], config['feature'], config['data_dir'])
        
        logger.info("STEREO-A Data...")
        
        self.Dataa = singlespacecraft(config['resampling'], 'STEREO-A',config['split'], config['feature'], config['data_dir'])
        
        logger.info("STEREO-B Data...")
        
        self.Datab = singlespacecraft(config['resampling'], 'STEREO-B',config['split'], config['feature'], config['data_dir'])
        
        logger.info("Creating combined dataset...")
        
        self.X_train = self.Dataw.X_train.append(self.Dataa.X_train, sort=False)
        self.X_train = self.X_train.append(self.Datab.X_train, sort = False)
        
        self.Y_train = self.Dataw.Y_train.append(self.Dataa.Y_train, sort=False)
        self.Y_train = self.Y_train.append(self.Datab.Y_train, sort = False)
        
        self.X_val = self.Dataw.X_val.append(self.Dataa.X_val, sort=False)
        self.X_val = self.X_val.append(self.Datab.X_val, sort = False)
        
        self.Y_val = self.Dataw.Y_val.append(self.Dataa.Y_val, sort=False)
        self.Y_val = self.Y_val.append(self.Datab.Y_val, sort = False)
        
        self.X_test = self.Dataw.X_test.append(self.Dataa.X_test, sort=False)
        self.X_test = self.X_test.append(self.Datab.X_test, sort = False)
        
        self.Y_test = self.Dataw.Y_test.append(self.Dataa.Y_test, sort=False)
        self.Y_test = self.Y_test.append(self.Datab.Y_test, sort = False)
        

def loadalldata(spacecraft, data_dir): 
    """
    Loads the in situ data from a .p file. 
    
    Arguments:
        spacecraft        data from which spacecraft should be loaded
        data_dir          path where the data files are located
    
    Output:
        alldata           in situ data
    """
    if spacecraft == 'Wind':
        [alldata, dataheader] = pickle.load(open(data_dir + "wind_2007_2021_heeq_ndarray.p", "rb"))
    elif spacecraft == "STEREO-A":
        [alldata, dataheader] = pickle.load(open(data_dir + "stereoa_2007_2021_sceq_ndarray.p", "rb"))
    elif spacecraft == "STEREO-B":
        [alldata, attb, bheader] = pickle.load(open(data_dir + "stereob_2007_2014_sceq_ndarray.p", "rb"))

    return alldata



def preprocessdataset(resample, dataset):
    """
    Resamples and preprocesses a dataset. Additional features are computed (Beta, Pdyn and Texrat)
    
    Arguments:
        resample          Sampling rate
        dataset           dataset to be preprocessed
    
    Output:
        data              resampled and preprocessed dataset
    """
    # pre process on the data set

    data = pds.DataFrame(dataset)
    data['time'] = matplotlib.dates.num2date(data['time'], tz=None) 
    data['time'] = pds.to_datetime(data['time'], format="%Y/%m/%d %H:%M")
    data.set_index('time',  inplace=True)
    data.index.name = None
    data.index = data.index.tz_localize(None)
    data.drop(['x', 'y', 'z', 'r', 'lat', 'lon'], axis = 1,  inplace=True)

    # compute additional features

    features.computeBetawiki(data)
    features.computePdyn(data)
    features.computeTexrat(data)

    # resample data
    data = data.resample(resample).mean().dropna()

    return data

def scaledata(data):
    """
    Scales a dataset.
    
    Arguments:
        data              dataset to be scaled
    
    Output:
        data_scaled       scaled dataset
    """
    scale = StandardScaler()
    scale.fit(data)

    data_scaled = pds.DataFrame(index = data.index, columns = data.columns, data = scale.transform(data))

    return data_scaled




def data_factory_fun(config):
    
    """
    Data factory when calling from a jupyter notebook.
    
    Arguments:
        config            entries in the config file
    
    Output:
        data              data object
    """
    
    data_class = {'all': multispacecraft,
                'Wind': singlespacecraft,
                'STEREO-A': singlespacecraft,
                'STEREO-B': singlespacecraft}
    
    spacecraft = data_class[config['spacecraft']]
    
    data = spacecraft(config['spacecraft'], config)
    
    return data


data_factory = {'all': multispacecraft,
                'Wind': singlespacecraft,
                'STEREO-A': singlespacecraft,
                'STEREO-B': singlespacecraft}