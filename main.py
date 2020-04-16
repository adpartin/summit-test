from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pprint, pformat
from glob import glob

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    if int(tf.__version__.split('.')[0]) < 2:
        import keras
        from keras.models import load_model
        from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
        from keras.utils import plot_model
    else:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
        from tensorflow.keras.utils import plot_model

        from tensorflow.keras import backend as K
        from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
        from tensorflow.keras import optimizers
        from tensorflow.keras.optimizers import SGD, Adam
        from tensorflow.keras.models import Sequential, Model
except:
    print('Could not import tensorflow.')

def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]
    
def reg_go_arch(input_dim, dr_rate=0.1):
    DR = dr_rate
    inputs = Input(shape=(input_dim,))
    x = Dense(250, activation='relu')(inputs)
    x = Dropout(DR)(x)
    x = Dense(125, activation='relu')(x)
    x = Dropout(DR)(x)
    x = Dense(60, activation='relu')(x)
    x = Dropout(DR)(x)
    x = Dense(30, activation='relu')(x)
    x = Dropout(DR)(x)
    outputs = Dense(1, activation='relu')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def reg_go_callback_def(outdir, ref_metric='val_loss'):
    """ Required for lrn_crv.py """
    checkpointer = ModelCheckpoint( str(outdir/'model_best.h5'), monitor='val_loss', verbose=0,
                                    save_weights_only=False, save_best_only=True )
    csv_logger = CSVLogger( outdir/'training.log' )
    reduce_lr = ReduceLROnPlateau( monitor=ref_metric, factor=0.75, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=3, min_lr=0.000000001 )
    early_stop = EarlyStopping( monitor=ref_metric, patience=50, verbose=1, mode='auto' )
    return [checkpointer, csv_logger, early_stop, reduce_lr]


# File path
filepath = Path(__file__).resolve().parent
# data = pd.read_parquet('ml.ADRP-ADPR_pocket1_round1_dock.dsc.parquet')
# data = data.sample(n=100000, random_state=0).reset_index(drop=True)
data = pd.read_parquet('data.parquet')

# Get features (x), target (y), and meta
# fea_list = ['mod']
# trg_name = 'reg'
# xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep='.')
# meta = data.drop( columns=xdata.columns )
# ydata = meta[ trg_name ]
# del data
xdata = data.iloc[:,1:]
ydata = data.iloc[:,0]

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler = StandardScaler()
cols = xdata.columns
xdata = pd.DataFrame( scaler.fit_transform(xdata), columns=cols, dtype=np.float32 )    

outdir = os.makedirs('./out', exist_ok=True)
input_dim = xdata.shape[1]
model = reg_go_arch(input_dim, dr_rate=0.1)

opt = SGD(lr=0.0001, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.summary()

from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(xdata, ydata, test_size=0.3)
eval_set = (xte, yte)
ml_fit_kwargs = {'epochs': 50, 'batch_size': 32, 'verbose': 1}
ml_fit_kwargs['validation_data'] = eval_set
history = model.fit(xtr, ytr, **ml_fit_kwargs)

print('Done.')

