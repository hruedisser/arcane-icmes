"""
ResUNet++ architecture in Keras TensorFlow
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 1), padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 1), padding="same")(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 1), padding="same", strides=strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 1), padding="same", strides=1)(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x

def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 1), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 1), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 1), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 1), padding="same")(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding="same")(y)
    return y

def attetion_block(g, x,reduce_factor):
    """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
    """

    filters = x.shape[-1]

    g_conv = BatchNormalization()(g)
    g_conv = Activation("relu")(g_conv)
    g_conv = Conv2D(filters, (3, 3), padding="same")(g_conv)

    g_pool = MaxPooling2D(pool_size=(2, 1), strides=(reduce_factor, 1))(g_conv) 

    x_conv = BatchNormalization()(x)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(filters, (3, 3), padding="same")(x_conv)

    gc_sum = Add()([g_pool, x_conv])

    gc_conv = BatchNormalization()(gc_sum)
    gc_conv = Activation("relu")(gc_conv)
    gc_conv = Conv2D(filters, (3, 3), padding="same")(gc_conv)

    gc_mul = Multiply()([gc_conv, x])
    return gc_mul

class ResUnetPlusPlus:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self):
        n_filters = [16,32,64, 128, 256, 512]
        inputs = Input((self.input_shape))
        c0 = inputs
        c1 = stem_block(c0, n_filters[2], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[3], strides=4)
        c3 = resnet_block(c2, n_filters[4], strides=4)
        
        #Bridge
        b1 = aspp_block(c3, n_filters[5])

        ## Decoder
        d1 = attetion_block(c2, b1)
        d1 = UpSampling2D((4, 1))(d1)
        d1 = Concatenate()([d1, c2])
        d1 = resnet_block(d1, n_filters[4])
        
        d2 = attetion_block(c1, d1)
        d2 = UpSampling2D((4, 1))(d2)
        d2 = Concatenate()([d2, c1])
        d2 = resnet_block(d2, n_filters[3])
        
        
        ## output
        outputs = aspp_block(d2, n_filters[2])
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs, name = 'ResUNetPlusPlusModel')
        return model
    
    
    def build_classifier(self, n_filters_max,res_depth, reduce_factor):
        
        inputs = Input((self.input_shape))
        clist = []
        c0 = inputs
        clist.append(c0)
        
        c = stem_block(c0, n_filters_max/2**(res_depth+1), strides=1)
        clist.append(c)
        
        ## Encoder
        
        for i in range(res_depth,0,-1):
            
            c = resnet_block(c, n_filters_max/2**(i),strides=reduce_factor)
            clist.append(c)
            
        #Bridge
        b = aspp_block(clist[res_depth+1], n_filters_max)

        ## Decoder
        
        d = attetion_block(clist[res_depth],b,reduce_factor)
        d = UpSampling2D((reduce_factor,1))(d)
        d = Concatenate()([d,clist[res_depth]])
        d = resnet_block(d, n_filters_max/2)
        
        for i in range(res_depth-1,0,-1):
            
            n = 2
            d = attetion_block(clist[i],d,reduce_factor)
            d = UpSampling2D((reduce_factor,1))(d)
            d = Concatenate()([d,clist[i]])
            d = resnet_block(d, n_filters_max/2**n)
            n = n+1
                             
        
        ## output
        outputs = aspp_block(d, n_filters_max/2**(res_depth+1))
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Flatten()(outputs)
        outputs = Dense(1)(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs, name = 'ResUNetPlusPlusClassifier')
        return model
    
    def build_seq(self, n_filters_max,res_depth, reduce_factor):
        
        n_filters_max = int(n_filters_max)
        res_depth = int(res_depth)
        reduce_factor = int(reduce_factor)
        
        
        inputs = Input((self.input_shape))
        clist = []
        c0 = inputs
        clist.append(c0)
        
        c = stem_block(c0, n_filters_max/2**(res_depth+1), strides=1)
        clist.append(c)
        
        ## Encoder
        
        for i in range(res_depth,0,-1):
            
            c = resnet_block(c, n_filters_max/2**(i),strides=reduce_factor)
            clist.append(c)
            
        #Bridge
        b = aspp_block(clist[res_depth+1], n_filters_max)

        ## Decoder
        
        d = attetion_block(clist[res_depth],b,reduce_factor)
        d = UpSampling2D((reduce_factor,1))(d)
        d = Concatenate()([d,clist[res_depth]])
        d = resnet_block(d, n_filters_max/2)
        
        for i in range(res_depth-1,0,-1):
            
            n = 2
            d = attetion_block(clist[i],d,reduce_factor)
            d = UpSampling2D((reduce_factor,1))(d)
            d = Concatenate()([d,clist[i]])
            d = resnet_block(d, n_filters_max/2**n)
            n = n+1
                             
        
        ## output
        outputs = aspp_block(d, n_filters_max/2**(res_depth+1))
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs, name = 'ResUNetPlusPlusSeq2Seq')
        return model
    