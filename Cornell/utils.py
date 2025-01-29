# file contains some common utility functions for mjj regression tasks
import pandas as pd
import vector
import numpy as np


def load_parquet_file(file_path, columns=[], loadAll=False):
    if loadAll:
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_parquet(file_path, columns=columns)
    df = df.fillna(0)
    return df


def add_PNetCorrections(df):
    # function adds PNetCorrections, will not be necessary as this will be added to the main parquets.
    PNetCorr_lead_bjet_pt = df['nonRes_lead_bjet_pt']*(1-df['nonRes_lead_bjet_rawFactor']) * \
        df['nonRes_lead_bjet_PNetRegPtRawCorr'] * \
        df['nonRes_lead_bjet_PNetRegPtRawCorrNeutrino']
    PNetCorr_sublead_bjet_pt = df['nonRes_sublead_bjet_pt']*(1-df['nonRes_sublead_bjet_rawFactor']) * \
        df['nonRes_sublead_bjet_PNetRegPtRawCorr'] * \
        df['nonRes_sublead_bjet_PNetRegPtRawCorrNeutrino']
    PNetCorr_lead_bjet_mass = df['nonRes_lead_bjet_mass']*(1-df['nonRes_lead_bjet_rawFactor']) * \
        df['nonRes_lead_bjet_PNetRegPtRawCorr'] * \
        df['nonRes_lead_bjet_PNetRegPtRawCorrNeutrino']
    PNetCorr_sublead_bjet_mass = df['nonRes_sublead_bjet_mass']*(
        1-df['nonRes_sublead_bjet_rawFactor'])*df['nonRes_sublead_bjet_PNetRegPtRawCorr']*df['nonRes_sublead_bjet_PNetRegPtRawCorrNeutrino']

    jet1 = vector.array({
        "pt": PNetCorr_lead_bjet_pt,
        "phi": df["nonRes_sublead_bjet_phi"],
        "eta": df["nonRes_sublead_bjet_eta"],
        "mass": PNetCorr_lead_bjet_mass
    })
    jet2 = vector.array({
        "pt": PNetCorr_sublead_bjet_pt,
        "phi": df["nonRes_lead_bjet_phi"],
        "eta": df["nonRes_lead_bjet_eta"],
        "mass": PNetCorr_sublead_bjet_mass
    })
    dijet = jet1 + jet2
    PNet_dijet_corr_mass = dijet["mass"]

    lead_delta_pt = PNetCorr_lead_bjet_pt - df['nonRes_lead_bjet_pt']
    sublead_delta_pt = PNetCorr_sublead_bjet_pt - df['nonRes_sublead_bjet_pt']

    lead_delta = vector.array({
        "rho": lead_delta_pt,
        "phi": df['nonRes_lead_bjet_phi']
    })
    sublead_delta = vector.array({
        "rho": sublead_delta_pt,
        "phi": df['nonRes_sublead_bjet_phi']
    })
    MET_Pt = vector.array({
        "rho": df['puppiMET_pt'],
        "phi": df['puppiMET_phi']
    })
    corr_MET = MET_Pt - (lead_delta + sublead_delta)
    # correct sumET
    lead_delta_Et = np.sqrt(PNetCorr_lead_bjet_pt**2 + PNetCorr_lead_bjet_mass**2) - \
        np.sqrt(df['nonRes_lead_bjet_pt']**2 + df['nonRes_lead_bjet_mass']**2)
    sublead_delta_Et = np.sqrt(PNetCorr_sublead_bjet_pt**2 + PNetCorr_sublead_bjet_mass**2) - \
        np.sqrt(df['nonRes_sublead_bjet_pt']**2 +
                df['nonRes_sublead_bjet_mass']**2)
    corr_MET_sumET = df['puppiMET_sumEt'] - (lead_delta_Et + sublead_delta_Et)

    # add columns to df
    df["nonRes_lead_bjet_PNetCorr_pt"] = PNetCorr_lead_bjet_pt
    df["nonRes_sublead_bjet_PNetCorr_pt"] = PNetCorr_sublead_bjet_pt
    df["nonRes_lead_bjet_PNetCorr_mass"] = PNetCorr_lead_bjet_mass
    df["nonRes_sublead_bjet_PNetCorr_mass"] = PNetCorr_sublead_bjet_mass
    df["nonRes_corr_MET_pt"] = corr_MET["rho"]
    df["nonRes_corr_MET_phi"] = corr_MET["phi"]
    df["nonRes_corr_MET_SumET"] = corr_MET_sumET
    df["nonRes_dijet_corr_mass"] = PNet_dijet_corr_mass
    return df


def calc_var(df_in, input_var):
    if input_var in df_in.columns:
        raiseException(
            "Should not be using this function for variable already in df")
    elif "over" in input_var:
        if "mjj" in input_var:
            base_var = input_var.replace("_over_mjj", "")
            return df_in[base_var]/df_in["nonRes_dijet_corr_mass"]
    elif "projectphi" in input_var:
        num = input_var.split("projectphi")[1][0]
        return np.cos(df_in["nonRes_DeltaPhi_j" + str(num) + "MET"])
    elif "eta_sign" in input_var:
        base_var = input_var.replace("_sign", "")
        return df_in[base_var] * np.sign(df_in["nonRes_lead_bjet_eta"])
    else:
        print("variable not yet supported")
        return None


def mjj_input_df(df_in, input_vars):

    df_out = pd.DataFrame([])
    extravars = pd.DataFrame([])
    for input_var in input_vars:
        if input_var in df_in.columns:
            df_out[input_var] = df_in[input_var]
        else:
            df_out[input_var] = calc_var(df_in, input_var)
            extravars[input_var] = calc_var(df_in, input_var)
    return df_out, extravars
