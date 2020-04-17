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

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Main.')
parser.add_argument('--ii', type=int, help='Index of the run.')                                                                                                 
parser.add_argument('--ep', type=int, default=5, help='Number of epochs.')                                                                    
parser.add_argument('--gout', type=str, help='Global output.')                                                                    
# args, other_args = parser.parse_known_args()
args = parser.parse_args()
args = vars(args)

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

def model_arch(input_dim, dr_rate=0.1):
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

def model_callback_def(outdir, ref_metric='val_loss'):
    """ ... """
    checkpointer = ModelCheckpoint( str(outdir/'model_best.h5'), monitor='val_loss', verbose=0,
                                    save_weights_only=False, save_best_only=True )
    csv_logger = CSVLogger( outdir/'training.log' )
    reduce_lr = ReduceLROnPlateau( monitor=ref_metric, factor=0.75, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=3, min_lr=0.000000001 )
    early_stop = EarlyStopping( monitor=ref_metric, patience=50, verbose=1, mode='auto' )
    return [checkpointer, csv_logger, early_stop, reduce_lr]

# File path
filepath = Path(__file__).resolve().parent
print('Loading data ...')
data = pd.read_parquet('data.parquet')
xdata = data.iloc[:,1:]
ydata = data.iloc[:,0]

scaler = StandardScaler()
cols = xdata.columns
xdata = pd.DataFrame( scaler.fit_transform(xdata), columns=cols, dtype=np.float32 )    

run_id = args['ii']
outdir = Path(args['gout'])/'out{}'.format(run_id)
os.makedirs(outdir, exist_ok=True)
input_dim = xdata.shape[1]

# Model
model = model_arch(input_dim, dr_rate=0.1)
opt = SGD(lr=0.0001, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.summary()

# Fit kwargs
xtr, xte, ytr, yte = train_test_split(xdata, ydata, test_size=0.3)
eval_set = (xte, yte)
callbacks = model_callback_def(outdir)
ml_fit_kwargs = {'epochs': args['ep'], 'batch_size': 32, 'verbose': 1}
ml_fit_kwargs['validation_data'] = eval_set
ml_fit_kwargs['callbacks'] = callbacks

# Train
history = model.fit(xtr, ytr, **ml_fit_kwargs)

# Predict
yte_pred = model.predict(xte)

# Dump preds
pd.DataFrame(yte_pred).to_csv(outdir/'yte_preds.csv', index=False)

print('Done.')

