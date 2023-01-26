import scipy.constants as constants
import pandas as pds
import datetime
import numpy as np
import logging

logger=logging.getLogger()

def computeBeta(data):
    '''
    compute the evolution of the Beta for data
    data is a Pandas dataframe
    The function assume data already has ['Np','B','Vth'] features
    '''
    try:
        data['Beta'] = 1e6 * data['Vth']*data['Vth']*constants.m_p*data['Np']*1e6*constants.mu_0/(1e-18*data['B']*data['B'])
    except KeyError:
        print('Error computing Beta,B,Vth or Np'
              ' might not be loaded in dataframe')
    return data

def computeBetawiki(data):
    '''
    compute Beta according to wikipedia
    '''
    try:
        data['beta'] = 1e6*data['np']*constants.Boltzmann*data['tp']/(np.square(1e-9*data['bt'])/(2*constants.mu_0))
    except KeyError:
        print('KeyError')
        
    return data
                                                               
def computePdyn(data):
    '''
    compute the evolution of the Beta for data
    data is a Pandas dataframe
    the function assume data already has ['Np','V'] features
    '''
    try:
        data['Pdyn'] = 1e12*constants.m_p*data['np']*data['vt']**2
    except KeyError:
        print('Error computing Pdyn, V or Np might not be loaded '
              'in dataframe')
    return data
        
def computeTexrat(data):
    '''
    compute the ratio of Tp/Tex
    '''
    try:
        data['texrat'] = data['tp']*1e-3/(np.square(0.031*data['vt']-5.1))
    except KeyError:
        print( 'Error computing Texrat')
    
    return data
       