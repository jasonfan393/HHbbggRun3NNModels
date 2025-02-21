#Todo: cleanup imports
import os
import pandas as pd
import awkward as ak
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from matplotlib import pyplot as plt
import numpy as np
import datetime
from tensorflow.keras.optimizers import Adam
import vector
from scipy.optimize import leastsq
import mplhep as hep

import utils
import argparse
import json

def build_model(input_shape, learning_rate=0.001, loss = 'mean_squared_error', N = 100, layers = 3):
    normalization_layer = Normalization()
    normalization_layer.adapt(X_train)  # Compute the mean and variance of the training data
    if layers == 3:
        model = Sequential([
            normalization_layer,
            Dense(N, activation='relu', input_shape=(input_shape,)),
            Dropout(0.1),
            Dense(N/2, activation='relu'),
            Dropout(0.1),
            Dense(N/4, activation='relu'),
            Dropout(0.1),
            Dense(1)  # Single output for regression with linear activation
        ])
    elif layers == 2:
        model = Sequential([
            normalization_layer,
            Dense(N, activation='relu', input_shape=(input_shape,)),
            Dropout(0.1),
            Dense(N/2, activation='relu'),
            Dropout(0.1),
            Dense(1)  # Single output for regression with linear activation
        ])
    elif layers == 4:
        model = Sequential([
            normalization_layer,
            Dense(N, activation='relu', input_shape=(input_shape,)),
            Dropout(0.1),
            Dense(N/2, activation='relu'),
            Dropout(0.1),
            Dense(N/4, activation='relu'),
            Dropout(0.1),
            Dense(N/8, activation='relu'),
            Dropout(0.1),
            Dense(1)  # Single output for regression with linear activation
        ])
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=[loss])  # MSE loss and MAE metric for regression
    return model




#mjj_trainer processing begins here:
parser = argparse.ArgumentParser(
    prog='Mjj Regressor Trainer', description='Script traings mjj regression, produces relevant performance plots')
parser.add_argument('--vars', default='configs/variables_Cornell_mjj.json')
parser.add_argument('--out_dir', default='mjj_regressor_performance')
parser.add_argument('--model', default='mjj_regressor_model')
parser.add_argument('--training_set', default='/afs/cern.ch/user/j/jafan/eosjfan/public/mbbTraining/GluGluToHH_with_full_PNet_info.parquet')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

#open input variable json
with open(args.vars, 'r') as file:
    data = json.load(file)
input_vars = data["input_variables"]
target_var = data["target"]
allvars = [*input_vars, target_var]

#Open training parquet, apply some quality cuts
df = utils.load_parquet_file(args.training_set, loadAll = True)
print("# of events (Pre-cuts): " + str(len(df.index)))

df = df.drop(df[np.abs(df["nonRes_lead_bjet_genFlav"]) != 5].index)
df = df.drop(df[np.abs(df["nonRes_sublead_bjet_genFlav"]) != 5].index)

print("# of events (Post-cuts): " + str(len(df.index)))

#Create calculated columns not in base parquets
df = utils.add_PNetCorrections(df)
df_in, extravars = utils.mjj_input_df(df, allvars)

df_train,df_test = train_test_split(df_in, test_size=0.3)
X_train = df_train[input_vars]
X_test = df_test[input_vars]
y_train = df_train[target_var]
y_test = df_test[target_var]

print("Training variables:")
for i in input_vars:
    print(i)
print('# of variables: ' + str(len(input_vars)))

#TODO port gridsearch into this script, runs very slow on lxplus

#results of most recent grid search:
lr = .00001
epochs = 25
batchsize = 32
N = 200
L = 4
loss = 'mean_squared_error'
print("Hyperparameters: ")
print("lr: " + str(lr) + " ep: " + str(epochs) + " ba: " + str(batchsize) + " loss: " + loss + "N: " + str(N) + "L:" + str(L))
# Initialize the model
model = build_model(X_train.shape[1], lr, loss, N, L)

# Train the mode - this is very slow on lxplus
history = model.fit(X_train, y_train, epochs=epochs,batch_size=batchsize, validation_split=0.3)

model.save("Mjj_regressor_model")


