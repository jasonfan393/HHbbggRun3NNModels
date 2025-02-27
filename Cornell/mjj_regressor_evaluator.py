import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import awkward as ak
import tensorflow as tf
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
parser.add_argument('--attach_inputs', default=False, action='store_true')
parser.add_argument('--attach_pNet', default=False, action='store_true')
parser.add_argument('--input_location', default='mjj_regressor_input')
parser.add_argument('--model', default='Mjj_regressor_model')

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
        if not "merged" in file:
            # does not work with merged parquets
            continue
        file_path = os.path.join(subdir, file)
        file_paths.append(file_path)

# load your Mjj regressor here
model = load_model(args.model, custom_objects={
                   'huber_loss': tf.keras.losses.Huber})

# main evaluator loop
for file_path in file_paths:

    print("Loading file: " + file_path)
    # load parquet with all variables
    chunknum = 0
    chunk_size = 100000
    pqfile = pq.ParquetFile(file_path)
    batches = pqfile.iter_batches(batch_size=chunk_size)
    schema = pq.read_schema(file_path)
    writer = None

    destination_path = output + file_path.replace(preamble, "")
    file_name = destination_path.split("/")[-1]
    destination_dir = destination_path.replace(file_name, "")
    new_file_name = file_name.split(
        ".")[0] + "_mbb_reg" + "." + file_name.split(".")[1]
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    output_file = destination_dir + new_file_name
    for chunk in batches:
        chunknum = chunknum + 1

        df_chunk = chunk.to_pandas()
        print("Processing chunk: " + str(chunknum))
        original_columns = df_chunk.columns
        # add PNET variables (unnecessary and to be removed in future)
        df_chunk = utils.add_PNetCorrections(df_chunk)

        df_input, extravars = utils.mjj_input_df(df_chunk, input_vars)

        if args.attach_inputs:
            # attach input variables, only necessary for mbb_regressor studies
            df_chunk = pd.concat([df_chunk, extravars], axis=1)
        correction_term_pred = model.predict(df_input)
        if not args.attach_pNet:  # attach PNET corrections to parquet, unnecessary in latest parquets
            df_chunk = df_chunk[original_columns]
        df_chunk["mjj_regressor_correction"] = correction_term_pred
        df_chunk["mjj_regressed"] = (df_chunk["mjj_regressor_correction"] *
                                     df_chunk["nonRes_dijet_mass_PNet_all"]) + df_chunk["nonRes_dijet_mass_PNet_all"]

        # check for placeholder values, and insert in new columns (all new columns require a valid sublead jet)
        df_is_not_placeholder = df_chunk["nonRes_sublead_bjet_phi"] > -999
        for column in df_chunk.columns:
            if column not in original_columns:
                df_chunk[column] = np.where(
                    df_is_not_placeholder, df_chunk[column], -999)

        table = pa.Table.from_pandas(df_chunk)
        if writer is None:
            schema = schema.append(
                pa.field('mjj_regressor_correction', pa.float32()))
            schema = schema.append(pa.field('mjj_regressed', pa.float64()))
            writer = pq.ParquetWriter(output_file, schema)
        table = table.cast(schema)
        writer.write_table(table)

    if writer:
        writer.close()
        print("output saved in: " + output_file)
