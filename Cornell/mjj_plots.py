import pandas as pd
from matplotlib import pyplot as plt
import utils
import numpy as np
import json
import argparse
import os
parser = argparse.ArgumentParser(
    prog='ProgramName', description='placeholder', epilog='placeholder')

parser.add_argument('--vars', default='configs/variables_Cornell_mjj.json')
args = parser.parse_args()
with open(args.vars, 'r') as file:
    data = json.load(file)
input_vars = data["input_variables"]

preamble = "/afs/cern.ch/user/j/jafan/eosjfan/HHbbggParquets/withPNetAndMjjReg_01_30_25/"
file_paths = []
for subdir, dirs, files in os.walk(preamble):
    for file in files:
        if ".parquet" not in file:
            continue
        file_path = os.path.join(subdir, file)
        file_paths.append(file_path)

vars_to_plot = ["nonRes_dijet_mass","nonRes_dijet_mass_PNet_all","mjj_regressed"]
for file in file_paths:
    print(file)
    df = utils.load_parquet_file(file, columns = vars_to_plot)
    print("# of events: "+ str(len(df[vars_to_plot[0]])))
    savefile = file.replace(preamble,"")
    savefile = savefile.replace(".parquet",".png")
    bins = np.linspace(0,300)
    #plt.hist(df["mjj_regressed"],label = 'mjj_regressed',histtype='step',bins=bins,density=True)
    var1 = "nonRes_dijet_mass"
    var2 = "nonRes_dijet_mass_PNet_all"

    plt.hist(df["mjj_regressed"],histtype='step',bins=bins, label = 'Mjj_regressed',density=True)
    plt.hist(df[var1],histtype='step',bins=bins, label = var1,density=True)
    plt.hist(df[var2],histtype='step',bins=bins, label = var2,density=True)
    plt.legend()
    plt.savefig("/afs/cern.ch/user/j/jafan/www/hhbbggplots/plots_jan_30_25/"+savefile)
    plt.clf()

