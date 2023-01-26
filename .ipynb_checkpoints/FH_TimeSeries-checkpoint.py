#!/usr/bin/env python
# coding: utf-8

# # Time Series Event Detection
# 
# This notebook was created to be shown at FH in Graz during a lecture on time series event detection. It is running on Python 3.8.5 with dependencies listed in the requirements.txt file.
# 

# In[2]:


#only run if GPU available

#Set devices_id
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf
tf.config.list_physical_devices('GPU')

import sys

print(sys.path)
# In[3]:


# Don't print warnings
import warnings
warnings.filterwarnings('ignore')

import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

logger.info("Loading packages...")

from config import get_config
from models.model import model_factory
from datasets.data import data_factory_fun
from utils.datagen import UnetGen
from datasets import event as evt


conf = get_config("")
conf.update(spacecraft='Wind')

catdic = evt.getallcat(conf)
evt.compare_lists(catdic)
# In[4]:


data = data_factory_fun(conf)


# In[5]:


model, callbacks = model_factory(conf, data)


# In[6]:


model.summary()


# In[ ]:


train_gen = UnetGen('train', conf, data)
val_gen = UnetGen('val', conf, data)


# In[ ]:


import tensorflow as tf
#tf.config.run_functions_eagerly(True)

model.fit(train_gen,validation_data=val_gen,epochs=conf['epochs'],callbacks=callbacks)


# In[ ]:




