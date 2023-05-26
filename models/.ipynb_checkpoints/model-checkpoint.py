import pandas as pds
from models.resunet import ResUnetPlusPlus
import logging
import numpy as np

from utils.metrics import dice_coef, dice_loss, true_skill_score
from utils.learning import CyclicLR, lr_scheduler_warmup

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU, Accuracy, RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import keras

import datetime

logger = logging.getLogger()

def model_factory(config, data):
        
    task = config['task']
    
    if config['spacecraft'] == 'all':
        feat_dim = data.Dataw.all_df.shape[1]
    else:
        feat_dim = data.all_df.shape[1] # dimensionality of data features
        
    logger.info('Feature dimension equals %i' %feat_dim)
    max_seq_len = config['data_window_len']
    logger.info('Sequence length equals %i' %max_seq_len)
    image_size = (max_seq_len, 1, feat_dim)
    logger.info('Image size equals %s' %str(image_size))
    '''
    if (config['finetune'] == 'True') and (config['train'] == 'True'):
        logger.info('Loading pretrained model')
        model = load_model(config['model_dir_pretrained'], compile = False)
        
    elif (config['finetune'] == 'True') and (config['test'] == 'True'):
        logger.info('Loading trained model')
        model = load_model(config['model_dir'], compile = False)
        
    else:
    '''
    if config['model'] == 'resunet':
        if config['model'] == 'resunet':
            logger.info('Choosing model architecture: ResUNet')
            arch = ResUnetPlusPlus(input_shape = image_size)
            if task == 'seq2seq':
                logger.info('Building model for Seq2Seq')
                model = arch.build_seq(config['n_filters_max'],config['res_depth'],config['reduce_factor']) #arch.build_model() 
            else:
                raise NotImplementedError('Not Implemented.')

                #model = arch.build_classifier(config['n_filters_max'],config['res_depth'],config['reduce_factor'])
            logger.info('Created ResUNet!')      
        else:
            raise NotImplementedError('Not Implemented.')

            
    if config['switch_output'] == 'True':
        
        x = model.layers[-1].output
        x = Flatten(name='switch_flatten')(x)
        x = Dense(1, name='switch_dense')(x)
        x = Activation("sigmoid", name='switch_activation')(x)
        x = tf.expand_dims(x, 2) 
        x = tf.expand_dims(x, 2) 
        model = Model(inputs = model.input, outputs = x)
        print(model.summary())
        
    
    logger.info('Selecting optimizer')
    optimizer = selectoptimizer(config)
    logger.info('Selecting metrics')
    metrics = selectmetrics(config)
    logger.info('Selecting loss')
    loss = selectloss(config)

    if config['model'] == 'informer':
        raise NotImplementedError('Not Implemented.') 
    else:
        logger.info('Compiling model...')
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics) 
    logger.info('Setting callbacks...')
    callbacks = selectcallbacks(config)
    
    if config['test'] == 'False':
        epochs = config['epochs']
        logger.info('Model will train for %i epochs' %epochs)
       
    return model, callbacks

def selectoptimizer(config):
    
    if config['optimizer'] == 'Adam':
        if config['schedule'] == 'simple':
            optimizer = Adam(config['lr'])
        else:
            optimizer = Adam()
                             
        logger.info('Setting Adam optimizer...')
    return optimizer

def selectmetrics(config):
    
    if (config['task'] =='classification') or (config['task'] =='seq2seq'):
        metrics = [Accuracy(), Recall(), Precision(), dice_coef,true_skill_score, MeanIoU(num_classes=2)]
    else:
        metrics =  [Accuracy(), Recall(), Precision(), dice_coef,true_skill_score, MeanIoU(num_classes=2)]
    
    return metrics

def selectloss(config):
    
    if (config['task'] =='classification') or (config['task'] =='seq2seq'):
        if config['loss'] == 'dice_loss':
            loss = dice_loss
        elif config['loss'] == 'mse':
            loss = MeanSquaredError()
    return loss

def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.1
    if epoch > 10:
        learning_rate = 0.01
    if epoch > 20:
        learning_rate = 0.001
    if epoch > 30:
        learning_rate = 0.0001

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

def selectcallbacks(config):
    
    if config['finetune'] == 'True':
        model_path = config['save_dir'] + '/' + config['model'] +'_'+ config['task'] +'_'+ str(config['split']+'-finetuned')
    else:
        model_path = config['save_dir'] + '/' + config['model'] +'_'+ config['task'] +'_'+ str(config['split'])

    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)

    callbacks = []
    
    callbacks.append(checkpoint)

    
    if config['schedule'] == 'cyclic':
        expclr = CyclicLR(base_lr=config['base_lr'], max_lr=config['max_lr'], step_size=config['step_size'])
        callbacks.append(expclr)
        logger.info('Using cyclic learning rate...')
    elif config['schedule'] == 'warmup':
        warm = keras.callbacks.LearningRateScheduler(partial(lr_scheduler_warmup, ...), verbose=0)
        callbacks.append(warm)
        logger.info('Using learning rate warmup...')
    
    
    if config['schedule'] == 'custom_schedule':
        #log_dir = config['save_dir'] + '/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        #file_writer.set_as_default()

        #lr_callback = LearningRateScheduler(lr_schedule)
        #tensorboard_callback = TensorBoard(log_dir=log_dir)
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 12)))
        
    #else:
    log_dir = config['save_dir'] + '/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    callbacks.append(tensorboard_callback)

        
    if config['reduce_lr'] is not None:
        rlpat =  int(config['reduce_lr'])
        rl = ReduceLROnPlateau(monitor='val_loss', patience= rlpat, min_delta=1e-6, factor = 0.1)
        callbacks.append(rl)
        logger.info('Learning rate will be reduced after %i epochs of no improvement' % rlpat)
        
    if config['earlystopping_patience'] is not None:
        espat = int(config['earlystopping_patience'])
        es = EarlyStopping(monitor='val_loss', patience= espat, restore_best_weights=True)
        callbacks.append(es)
        logger.info('Early Stopping will be done after %i epochs of no improvement' % espat)
        
    
    return callbacks
            

