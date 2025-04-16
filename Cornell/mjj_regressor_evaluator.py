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
from mjj_trainer import plot_input_vars
from tensorflow.keras.models import load_model


parser = argparse.ArgumentParser(
    prog='ProgramName', description='placeholder', epilog='placeholder')
parser.add_argument('--vars', default='configs/variables_mjj_alljets.json')
parser.add_argument('--output_location', default='mjj_regressor_output')
parser.add_argument('--attach_inputs', default=False, action='store_true')
parser.add_argument('--attach_pNet', default=False, action='store_true')
parser.add_argument('--input_location', default='mjj_regressor_input')
# outer-most directory containing merged parquets to apply mjj on
# FIXME change this to use json with file locations
parser.add_argument('--model', default='mjj_model_2022_MET')
parser.add_argument('--year', default='2022')

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
        if "merged.parquet" not in file:
            continue
        if "nominal" not in subdir:
            continue
        if args.year not in subdir:
            continue
        file_path = os.path.join(subdir, file)
        file_paths.append(file_path)

# load your Mjj regressor here

years = {"2022preEE": 0,
         "2022postEE": 0,
         "2023preBPix": 1,
         "2023postBPix": 1
         }

# main evaluator loop
for file_path in file_paths:
    model = load_model(args.model, custom_objects={
        'huber_loss': tf.keras.losses.Huber})

    print("Loading file: " + file_path)
    # load parquet with all variables
    chunknum = 0
    chunk_size = 100000
    pqfile = pq.ParquetFile(file_path)
    batches = pqfile.iter_batches(batch_size=chunk_size)
    schema = pq.read_schema(file_path)
    writer = None
    for year in years:
        if year in file_path:
            sample_year = years[year]
    destination_path = output + file_path.replace(preamble, "")
    file_name = destination_path.split("/")[-1]
    destination_dir = destination_path.replace(file_name, "")
    new_file_name = file_name.split(
        ".")[0] + "_mbb_reg_noMetCorr" + "." + file_name.split(".")[1]
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    output_file = destination_dir + new_file_name
    for chunk in batches:
        chunknum = chunknum + 1

        df_chunk = chunk.to_pandas()
        print("Processing chunk: " + str(chunknum))
        original_columns = df_chunk.columns
        for ANType in ["Res", "nonRes"]:
            new_vars = []
            for var in input_vars:
                var = var.replace("Res_", ANType+"_", 1)
                new_vars.append(var)
            df_chunk = utils.add_PNetCorrections(df_chunk, ANType)
            new_vars.remove("year")
            df_input, extravars = utils.mjj_input_df(
                df_chunk, new_vars, ANType, sample_year)

            if args.attach_inputs:
                df_chunk = pd.concat([df_chunk, extravars], axis=1)
            correction_term_pred = model.predict(df_input)
            if not args.attach_pNet:  # attach PNET corrections to parquet, unnecessary in latest parquets
                if ANType == "Res":
                    df_chunk = df_chunk[original_columns]
                else:
                    mod_columns = list(original_columns)
                    mod_columns.append("Res_mjj_regressor_correction")
                    mod_columns.append("Res_mjj_regressed")
                    df_chunk = df_chunk[mod_columns]
            df_chunk[ANType + "_mjj_regressor_correction"] = correction_term_pred
            df_chunk[ANType + "_mjj_regressed"] = (df_chunk[ANType + "_mjj_regressor_correction"] *
                                                   df_chunk[ANType + "_dijet_mass"]) + df_chunk[ANType + "_dijet_mass"]
        # check for placeholder values, and insert in new columns (all new columns require a valid sublead jet)
        for ANType in ["Res", "nonRes"]:
            df_is_not_placeholder = df_chunk[ANType +
                                             "_sublead_bjet_phi"] > -999
            for column in df_chunk.columns:
                if ANType not in column:
                    continue
                if column not in original_columns:
                    df_chunk[column] = np.where(
                        df_is_not_placeholder, df_chunk[column], -999)

        table = pa.Table.from_pandas(df_chunk)
        if writer is None:
            schema = schema.append(
                pa.field('Res_mjj_regressor_correction', pa.float32()))
            schema = schema.append(pa.field('Res_mjj_regressed', pa.float64()))
            schema = schema.append(
                pa.field('nonRes_mjj_regressor_correction', pa.float32()))
            schema = schema.append(
                pa.field('nonRes_mjj_regressed', pa.float64()))
            for field in schema:
                if field.name == "__index_level_0__":
                    fields_to_keep = [
                        field_ for field_ in schema if field_.name != "__index_level_0__"]
                    schema = pa.schema(fields_to_keep)
                    schema = schema.append(
                        pa.field('__index_level_0__', pa.int64()))
            writer = pq.ParquetWriter(output_file, schema)
        table = table.cast(schema)
        writer.write_table(table)

    if writer:
        writer.close()
        print("output saved in: " + output_file)
