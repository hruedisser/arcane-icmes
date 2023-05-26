import logging
import pandas as pds
import numpy as np
import datetime
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, roc_auc_score
#from sklearn.metrics import precision_recall_curve, auc, plot_precision_recall_curve
from numpy.lib.stride_tricks import as_strided
from datasets import event as evt
import seaborn as sns

logger = logging.getLogger()

'''
A lot of these functions borrow heavily from 
https://github.com/gautiernguyen/Automatic-detection-of-ICMEs-at-1-AU-a-deep-learning-approach
'''
def predict(data,model,config):
    ## Generating the result
    image_size = (config['data_window_len'],1,10)
    X_test_windowed = make_views(data.X_test, win_size = config['data_window_len'], step_size = config['data_window_len'] ,writeable = False)
    Y_test_windowed = make_views(data.Y_test, win_size = config['data_window_len'], step_size = config['data_window_len'] ,writeable = False)

    for i, test in tqdm(enumerate(X_test_windowed), total=len(X_test_windowed)):
    
        df_mask = pds.DataFrame(Y_test_windowed[i])
        df_mask = df_mask.set_index(df_mask[1])
        df_mask = df_mask.iloc[: , :-1]
        image = pds.DataFrame(test)
        image = image.set_index(image[10])
        image = image.iloc[: , :-1]
        predict_mask = model.predict(np.expand_dims(np.asarray(np.expand_dims(image, axis=0)).astype('float64'),2))[0]
        df_mask['pred'] = np.squeeze(predict_mask)
        df_mask.columns = ['true', 'pred']
        if i == 0:
            result = df_mask
        else:
            result = pds.concat([result, df_mask], sort=True)

    result = result.sort_index()
    result['true'] = np.asarray(result['true']).astype('float64')
    result.index = pds.to_datetime(result.index)
    resultbin = make_binary(result['pred'], 0.5)
    events = makeEventList(resultbin, 1, 10)
    ICMEs = removeCreepy(events, config['creepy'])
    test_clouds = [x for x in data.evtlist if (x.begin.year in data.test)]
    
    logger.info('{} Score:'.format(config['spacecraft']))

    TP, FN, FP, detected = evaluate(ICMEs, test_clouds, thres=0.1)
    logger.info('Precision is %f:'%(len(TP)/(len(TP)+len(FP))))
    logger.info('Recall is: %f'%(len(TP)/(len(TP)+len(FN))))
    logger.info('True Positives %i' %(len(TP)))
    logger.info('False Negatives %i' %(len(FN)))
    logger.info('False Positives %i' %(len(FP)))
    
    result.to_csv(config['pred_dir']+'/prediction.csv')
    
    logger.info("Predictions saved to '{}'".format(config['pred_dir']))
    
    return TP, FN, FP, detected, result
    
def generate_result(data, model,config):
    ## Generating the result
    image_size = (config['data_window_len'],1,10)
    X_test_windowed = make_views(data.X_test, win_size = config['data_window_len'], step_size = config['data_window_len'] ,writeable = False)
    Y_test_windowed = make_views(data.Y_test, win_size = config['data_window_len'], step_size = config['data_window_len'] ,writeable = False)

    for i, test in tqdm(enumerate(X_test_windowed), total=len(X_test_windowed)):
    
        df_mask = pds.DataFrame(Y_test_windowed[i])
        df_mask = df_mask.set_index(df_mask[1])
        df_mask = df_mask.iloc[: , :-1]
        image = pds.DataFrame(test)
        image = image.set_index(image[10])
        image = image.iloc[: , :-1]
        predict_mask = model.predict(np.expand_dims(np.asarray(np.expand_dims(image, axis=0)).astype('float64'),2))[0]
        df_mask['pred'] = np.squeeze(predict_mask)
        df_mask.columns = ['true', 'pred']
        if i == 0:
            result = df_mask
        else:
            result = pds.concat([result, df_mask], sort=True)

    result = result.sort_index()
    result['true'] = np.asarray(result['true']).astype('float64')
    result.index = pds.to_datetime(result.index)
    
    return result

def make_views(arr,win_size,step_size,writeable = False):
    """
    
    see https://krbnite.github.io/Memory-Efficient-Windowing-of-Time-Series-Data-in-Python-3-Memory-Strides-in-Pandas/
    
    arr: any 2D array whose columns are distinct variables and 
    rows are data records at some timestamp t
    win_size: size of data window (given in data points along record/time axis)
    step_size: size of window step (given in data point along record/time axis)
    writable: if True, elements can be modified in new data structure, which will affect
    original array (defaults to False)
  
    Note that step_size is related to window overlap (overlap = win_size - step_size), in 
    case you think in overlaps.
  
    This function can work with C-like and F-like arrays, and with DataFrames.  Yay.
    
   
    """
  
    # If DataFrame, use only underlying NumPy array
    if type(arr) == type(pds.DataFrame()):
        arr['index'] = arr.index
        arr = arr.values
  
    # Compute Shape Parameter for as_strided
    n_records = arr.shape[0]
    n_columns = arr.shape[1]
    remainder = (n_records - win_size) % step_size 
    num_windows = 1 + int((n_records - win_size - remainder) / step_size)
    shape = (num_windows, win_size, n_columns)
  
    # Compute Strides Parameter for as_strided
    next_win = step_size * arr.strides[0]
    next_row, next_col = arr.strides
    strides = (next_win, next_row, next_col)

    new_view_structure = as_strided(arr,shape = shape,strides = strides,writeable = writeable)
    return new_view_structure

def removeCreepy(eventList, thres=3.5):
    '''
    For a given list, remove the element whose duration is under the threshold
    '''
    return [x for x in eventList if x.duration > datetime.timedelta(hours=thres)]

def make_binary(serie, thresh):
    
    serie = (serie > thresh)*1
    serie = serie.interpolate()
    
    return serie

def makeEventList(y, label, delta=2):
    '''
    Consider y as a pandas series, returns a list of Events corresponding to
    the requested label (int), works for both smoothed and expected series
    Delta corresponds to the series frequency (in our basic case with random
    index, we consider this value to be equal to 2)
    '''
    listOfPosLabel = y[y == label]
    if len(listOfPosLabel) == 0:
        return []
    deltaBetweenPosLabel = listOfPosLabel.index[1:] - listOfPosLabel.index[:-1]
    deltaBetweenPosLabel.insert(0, datetime.timedelta(0))
    endOfEvents = np.where(deltaBetweenPosLabel > datetime.timedelta(minutes=delta))[0]
    indexBegin = 0
    eventList = []
    for i in endOfEvents:
        end = i
        eventList.append(evt.Event(listOfPosLabel.index[indexBegin], listOfPosLabel.index[end]))
        indexBegin = i+1
    eventList.append(evt.Event(listOfPosLabel.index[indexBegin], listOfPosLabel.index[-1]))
    return eventList

def get_roc(result):

    # calculate roc curve

    fpr, tpr, thresholds = roc_curve(result['true'], result['pred'])

    auc_roc = roc_auc_score(result['true'], result['pred'])


    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'r', label = 'WIND - AUC = %0.2f' % auc_roc)
    plt.plot([0, 1], [0, 1],'g--',label = 'No skill')
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def get_pr(result):

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(result['true'], result['pred'])
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.title('Precision vs Recall')
    plt.plot(recall, precision, 'r', label = 'WIND - AUC = %0.2f' % pr_auc)
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()   
    


def plot_durations(TP,FN,FP):
       
    tp = np.zeros(len(TP))
    fn = np.zeros(len(FN))
    fp = np.zeros(len(FP))
    
    for i in range(len(tp)):
        tp[i] = TP[i].duration.total_seconds()/60/60
        
    for i in range(len(fp)):
        fp[i] = FP[i].duration.total_seconds()/60/60
        
    for i in range(len(fn)):
        fn[i] = FN[i].duration.total_seconds()/60/60
    
    dic = {'TP':tp, 'FP':fp, 'FN':fn}
    df = pds.DataFrame.from_dict(dic, orient='index').transpose()
    
    ax = df.plot.hist(bins = 25, alpha =0.4)
    ax.set_xlabel('Duration')
    ax.set_title('Duration of True Positives, False Positives and False Negatives')
    
    
def evaluate(predicted_list, test_list, thres=0.51, durationCreepies=2.5):
    '''
    for each cloud of validation_list, gives the list of clouds in the
    predicted_list that overlap the cloud among the threshold
    '''
    TP = []
    FN = []
    FP = []
    detected = []
    for event in test_list:
        corresponding = evt.find(event, predicted_list, thres, 'best')
        if corresponding is None:
            FN.append(event)
        else:
            TP.append(corresponding)
            detected.append(event)
    FP = [x for x in predicted_list if max(evt.overlapWithList(x, test_list, percent=True)) == 0]
    seum = [x for x in FP if x.duration < datetime.timedelta(hours=durationCreepies)]
    for event in seum:
        FP.remove(event)
        predicted_list.remove(event)

    return TP, FN, FP, detected

def plotrealtime(data, result):
    x = 1

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib

from matplotlib.colors import ListedColormap

def predict_realtime(data,model,config):
    ## Generating the result
    image_size = (config['data_window_len'],1,10)
    X_test_windowed = make_views(data.X_test, win_size = config['data_window_len'], step_size = 1000, writeable = False)
    Y_test_windowed = make_views(data.Y_test, win_size = config['data_window_len'], step_size = 1000, writeable = False)
    
    #sns.set_style('darkgrid')
    #sns.set_context('paper')
    
    colors = [(0, 0, 0, 0.5),(0, 1, 0, 0.5),(1, 0, 0, 0.5)]
    cmap = ListedColormap(colors)
        
    fig=plt.figure(figsize=(15,6)) 
    
    for i, test in tqdm(enumerate(X_test_windowed), total=len(X_test_windowed)):
    
        df_mask = pds.DataFrame(Y_test_windowed[i])
        df_mask = df_mask.set_index(df_mask[1])
        df_mask = df_mask.iloc[: , :-1]
        image = pds.DataFrame(test)
        image = image.set_index(image[10])
        image = image.iloc[: , :-1]
        predict_mask = model.predict(np.expand_dims(np.asarray(np.expand_dims(image, axis=0)).astype('float64'),2))[0]
        df_mask['pred'] = np.squeeze(predict_mask)
        df_mask.columns = ['true', 'pred']
        
        df_mask.to_csv(config['pred_dir']+'/realtime_prediction'+ str(i)+'.csv')
        
        plt.clf()
        
        ax = plt.subplot(111) 

        
        ax.set_ylim(-15,15)
        ax.tick_params(axis='x', which='major', labelsize=8, pad=10)
        

        # create background color
        for j in range(len(image.index)-1):
            if df_mask.loc[image.index[j], 'true'] == 1 and df_mask.loc[image.index[j], 'pred'] < .6:
                ax.axvspan(image.index[j], image.index[j + 1], color='grey', alpha=0.2)  # transparent grey
            elif df_mask.loc[image.index[j], 'true'] == 1 and df_mask.loc[image.index[j], 'pred'] > .6:
                ax.axvspan(image.index[j], image.index[j + 1], color='green', alpha=0.2)  # transparent green
            elif df_mask.loc[image.index[j], 'true'] == 0 and df_mask.loc[image.index[j], 'pred'] > .6:
                ax.axvspan(image.index[j], image.index[j + 1], color='red', alpha=0.2)  # transparent red
        
        #plt.plot(x, image[start:end])
        ax.plot_date(image.index, image[0],'-r',label='Bx',linewidth=0.5)
        ax.plot_date(image.index, image[1],'-g',label='By',linewidth=0.5)
        ax.plot_date(image.index, image[2],'-b',label='Bz',linewidth=0.5)
        ax.plot_date(image.index, image[3],'-k',label='Btotal',lw=0.5)
        plt.ylabel('B [nT]')
        plt.legend(loc=3,ncol=4,fontsize=8)
    
        #fig.canvas.draw()
        #fig.canvas.flush_events()
        plt.tight_layout()
        #plt.show()
        # save plot
        plt.savefig(config['pred_dir']+'/realtime_prediction_image'+ str(i)+'.png')

    #plt.rcParams['animation.ffmpeg_path'] = r'/Users/hannahruedisser/Downloads/ffmpeg'
    #FFwriter=animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
    #ani = animation.FuncAnimation(fig, animate, frames=len(X_test_windowed), interval=100)
    #ani.save(config['pred_dir']+'/realtime_prediction.mp4', writer=FFwriter)
   
    logger.info("Predictions saved to '{}'".format(config['pred_dir']))
    
    return 

