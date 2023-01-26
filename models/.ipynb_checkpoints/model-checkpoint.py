import pandas as pds
from models.resunet import ResUnetPlusPlus
import logging

from utils.metrics import dice_coef, dice_loss, true_skill_score
from utils.learning import CyclicLR, lr_scheduler_warmup

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU, Accuracy, RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

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
    
    epochs = config['epochs']
    logger.info('Model will train for %i epochs' %epochs)
       
    return model, callbacks

def selectoptimizer(config):
    
    if config['optimizer'] == 'Adam':
        if config['schedule'] == 'cyclic':
            optimizer = Adam()
        else:
            optimizer = Adam(config['lr'])
                             
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

def selectcallbacks(config):
    
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
    
    if config['earlystopping_patience'] is not None:
        es = EarlyStopping(monitor='val_loss', patience=config['earlystopping_patience'], restore_best_weights=True)
        callbacks.append(es)
        logger.info('Early Stopping will be done after %i epochs of no improvement' %config['earlystopping_patience'])
        
    
    return callbacks
            
    