import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

logger.info("Loading packages...")

from running import setup
from options import Options
import argparse

import numpy as np
import pandas as pds
import random 

import time
import os

from datasets.data import data_factory_fun
from models.model import model_factory
from utils.datagen import UnetGen
from datasets import event as evt
from utils import postprocess

import tensorflow as tf
from tensorflow.keras import backend as K

def main(config):
    
    total_start_time = time.time()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=config['GPU']

    tf.config.list_physical_devices('GPU')
    
    if config['seed'] is not None:
        os.environ['PYTHONHASHSEED']= '0'
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        tf.random.set_seed(config['seed'])
    
    data = data_factory_fun(config)
    
    model, callbacks = model_factory(config, data)
    
    if (config['train'] == 'True') or (config['pretrain'] == 'True'):
        train_gen = UnetGen('train', config, data)
        val_gen = UnetGen('val', config, data)
        model.fit(train_gen, validation_data=val_gen,epochs=config['epochs'],callbacks=callbacks)
        model.save(config['model_dir'])
        
    if config['test'] == 'True':
        model = tf.keras.models.load_model(conf['model_dir'])
        test_gen = UnetGen('test', config, data)
        model.evaluate(test_gen, verbose=1)
        TP, FN, FP, detected, result = postprocess.predict(data, model, config)

if __name__ == '__main__':
    
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)