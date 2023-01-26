
import pandas as pds
import numpy as np
import datetime
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, plot_precision_recall_curve
from numpy.lib.stride_tricks import as_strided
from datasets import event as evt

'''
A lot of these functions borrow heavily from 
https://github.com/gautiernguyen/Automatic-detection-of-ICMEs-at-1-AU-a-deep-learning-approach
'''

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