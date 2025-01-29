from tensorflow.keras.models import load_model
import pandas as pd
import awkward as ak
# import tensorflow as tf
# from tensorflow.keras.layers.experimental.preprocessing import Normalization
# from sklearn import preprocessing
# from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np
import vector
import scipy
import json
import argparse
import utils
import os
# outer-most directory containing merged parquets to apply mjj on
# FIXME change this to use json with file locations
preamble = '/eos/user/e/evourlio/HiggsDNA'

parser = argparse.ArgumentParser(
    prog='ProgramName', description='placeholder', epilog='placeholder')
parser.add_argument('--vars', default='configs/variables_Cornell_mjj.json')
parser.add_argument('--output_location', default='mjj_regressor_output')
parser.add_argument('--attach_inputs', default = False)
args = parser.parse_args()
output = args.output_location
if not os.path.exists(output):
    os.makedirs(output)

with open(args.vars, 'r') as file:
    data = json.load(file)
input_vars = data["input_variables"]

print("Please verify regressor is trained with selected input variables: ")
for input_var in input_vars:
    print(input_var)
print("# of variables: "+str(len(input_vars)))

file_paths = []
for subdir, dirs, files in os.walk(preamble):
    for file in files:
        if "merged.parquet" not in file:
            continue
        file_path = os.path.join(subdir, file)
        file_paths.append(file_path)

# load your Mjj regressor here
model = load_model("Mjj_regressor_model")

# main evaluator loop

for file_path in file_paths:

    # load parquet with all variables
    df_all = utils.load_parquet_file(file_path, loadAll=True)

    #add PNET variables (unnecessary and to be removed in future)
    df_all = utils.add_PNetCorrections(df_all)

    df_input, extravars = utils.mjj_input_df(df_all, input_vars)
    if args.attach_inputs:
        df_all = pd.concat([df_all,extravars],axis=1)

    df_all["mjj_regressed"] = model.predict(df_input)

    destination_path = output + file_path.replace(preamble, "")
    destination_dir = destination_path.replace(
        "NOTAG_merged.parquet", "")  # FIXME this is a hack
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    df_all.to_parquet(destination_path)
    print("output saved in: " + destination_path)

