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

from tensorflow.keras.models import load_model

# outer-most directory containing merged parquets to apply mjj on
# FIXME change this to use json with file locations

parser = argparse.ArgumentParser(
    prog='ProgramName', description='placeholder', epilog='placeholder')
parser.add_argument('--vars', default='configs/variables_Cornell_mjj.json')
parser.add_argument('--output_location', default='mjj_regressor_output')
parser.add_argument('--attach_inputs', default = False, action = 'store_true')
parser.add_argument('--attach_pNet', default = False, action = 'store_true')
parser.add_argument('--input_location', default = '/eos/user/e/evourlio/HiggsDNA')

args = parser.parse_args()
preamble = args.input_location
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
        if ".parquet" not in file:
            continue
        file_path = os.path.join(subdir, file)
        file_paths.append(file_path)

# load your Mjj regressor here
model = load_model("Mjj_regressor_model")

# main evaluator loop

for file_path in file_paths:
    print("Loading file: " + file_path)
    # load parquet with all variables
    df_all = utils.load_parquet_file(file_path, loadAll=True)
    original_columns = df_all.columns

    #add PNET variables (unnecessary and to be removed in future)
    df_PNet = utils.add_PNetCorrections(df_all)
    if args.attach_pNet:
        df_all = df_PNet

    df_input, extravars = utils.mjj_input_df(df_PNet, input_vars)

    if args.attach_inputs:
        df_all = pd.concat([df_all,extravars],axis=1)
    df_is_not_placeholder = df_all["nonRes_sublead_bjet_phi"]>-999
    df_all["mjj_regressor_correction"] = model.predict(df_input)
    df_all["mjj_regressed"] = (df_all["mjj_regressor_correction"]*df_PNet["nonRes_dijet_corr_mass"]) + df_PNet["nonRes_dijet_corr_mass"]

    for column in df_all.columns:
        if column not in original_columns:
            df_all[column] = np.where(df_is_not_placeholder, df_all[column], -999)

    destination_path = output + file_path.replace(preamble, "")
    file_name = destination_path.split("/")[-1]
    destination_dir = destination_path.replace(file_name,"")
    new_file_name = file_name.split(".")[0] + "_with_mbb_reg" + "." +  file_name.split(".")[1]
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    df_all.to_parquet(destination_dir + new_file_name)
    print("output saved in: " + destination_dir + new_file_name)

