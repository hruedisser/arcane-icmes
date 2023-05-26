import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

import logging

logger = logging.getLogger()


class UnetGen(Sequence):
    
    """Utility class for generating batches of temporal data.
    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.
    # Arguments
        data: Indexable generator (such as list or Numpy array)
            containing consecutive data points (timesteps).
            The data should be at 2D, and axis 0 is expected
            to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have same length as `data`.
        length: Length of the output sequences (in number of timesteps).
        sampling_rate: Period between successive individual timesteps
            within sequences. For rate `r`, timesteps
            `data[i]`, `data[i-r]`, ... `data[i - length]`
            are used for create a sample sequence.
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index: Data points earlier than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Data points later than `end_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        reverse: Boolean: if `true`, timesteps in each output sample will be
            in reverse chronological order.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one).
    # Returns
        A [Sequence](/utils/#sequence) instance.
    # Examples
    ```python
    from keras.preprocessing.sequence import TimeseriesGenerator
    import numpy as np
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])
    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   batch_size=2)
    assert len(data_gen) == 20
    batch_0 = data_gen[0]
    x, y = batch_0
    assert np.array_equal(x,
                          np.array([[[0], [2], [4], [6], [8]],
                                    [[1], [3], [5], [7], [9]]]))
    assert np.array_equal(y,
                          np.array([[10], [11]]))
    ```
    """   
    
    def __init__(self, scenario, config, data):
        
        if scenario == 'train':
            self.data = data.X_train
            self.targets = data.Y_train
            self.stride = config['stride']
            logger.info('Initializing data generator for training...')
            
        elif scenario == 'val':
            self.data = data.X_val
            self.targets = data.Y_val
            self.stride = config['stride']
            logger.info('Initializing data generator for validation...')
            
        elif scenario == 'test':
            self.data = data.X_test
            self.targets = data.Y_test
            logger.info('Initializing data generator for testing...')
            if config['test_stride'] is not None:
                self.stride = config['test_stride']
            else:
                self.stride = config['data_window_len']
                logger.info('No teststride given, using window length.')
                
        '''elif scenario == 'realtime':
            self.data = data.X_test
            self.targets = data.Y_test
            logger.info('Initializing data generator for testing in realtime mode...')
            self.stride = 1'''
                
        self.task = config['task']
        #self.switch = config['switch_output']
        self.shift = config['shift']
        self.length = config['data_window_len']
        self.sampling_rate = config['sampling_rate']
        self.stride = config['stride']
        self.start_index = self.length
        if self.task == 'classification':
            self.end_index = len(self.data) - 1 - self.shift
        else:
            self.end_index = len(self.data) - 1
        if scenario == 'test':
            self.shuffle = False
        else:
            self.shuffle = config['shuffle']
        self.batch_size = config['batch_size']
        self.model = config['model']
          
        
        if self.task == 'seq2seq':
            if len(self.data) != len(self.targets):
                logger.error('Data and targets have to be of same length. Data length is {}'.format(len(self.data)) + ' while target length is {}'.format(len(self.targets)))
                
        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

    def __len__(self):
        return (self.end_index - self.start_index +
                self.batch_size * self.stride) // (self.batch_size * self.stride)

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size *
                                    self.stride, self.end_index + 1), self.stride)
        if (self.task == 'seq2seq') or (self.task == 'imputation'):
            samples = np.array([self.data[row-self.length:row:self.sampling_rate]
                                for row in rows], dtype = 'float64')
            
            '''if self.switch == 'True':
                targets = np.array([self.targets[row-self.length:row:self.sampling_rate].iloc[-1]
                                for row in rows], dtype = 'float64')            
            else:'''
            targets = np.array([self.targets[row-self.length:row:self.sampling_rate]
                                for row in rows], dtype = 'float64')
            
            if self.task == 'seq2seq':
                samples = np.expand_dims(samples, 2) 
                targets = np.expand_dims(targets, 2)
            else:
                samples = np.expand_dims(samples, 2)
                targets = np.squeeze(targets)
                
        elif self.task == 'classification':
            raise NotImplementedError('Not Implemented.')
            
            #samples = np.array([self.data[row - self.length:row:self.sampling_rate]
                  #          for row in rows], dtype = 'float64')
            #targets = np.array([np.squeeze(self.targets)[row+self.shift] for row in rows], dtype = 'float64')
            #samples = np.expand_dims(samples, 2)
            #targets = np.squeeze(targets)
        else:
            raise NotImplementedError('Not Implemented.')
        return samples, targets