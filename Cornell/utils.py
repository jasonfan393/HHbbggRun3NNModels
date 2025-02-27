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


def deltaPhi(phi1, phi2):
    phases = phi1 - phi2
    return (phases + np.pi) % (2 * np.pi) - np.pi


def add_PNetCorrections(df):

    lead_delta_pt = df['nonRes_lead_bjet_pt_PNet_all'] - \
        df['nonRes_lead_bjet_pt']
    sublead_delta_pt = df['nonRes_sublead_bjet_pt_PNet_all'] - \
        df['nonRes_sublead_bjet_pt']

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
    lead_delta_Et = np.sqrt(df['nonRes_lead_bjet_pt_PNet_all']**2 + df['nonRes_lead_bjet_mass_PNet_all']**2) - \
        np.sqrt(df['nonRes_lead_bjet_pt']**2 + df['nonRes_lead_bjet_mass']**2)
    sublead_delta_Et = np.sqrt(df['nonRes_sublead_bjet_pt_PNet_all']**2 + df['nonRes_sublead_bjet_mass_PNet_all']**2) - \
        np.sqrt(df['nonRes_sublead_bjet_pt']**2 +
                df['nonRes_sublead_bjet_mass']**2)
    corr_MET_sumET = df['puppiMET_sumEt'] - (lead_delta_Et + sublead_delta_Et)

    # add columns to df
    df["nonRes_corr_MET_pt"] = corr_MET["rho"]
    df["nonRes_corr_MET_phi"] = corr_MET["phi"]
    df["nonRes_corr_MET_SumET"] = corr_MET_sumET

    # recalculate deltaPhi with new MET

    df["deltaPhi_j1MET_corr"] = deltaPhi(
        df["nonRes_corr_MET_phi"], df["nonRes_lead_bjet_phi"])
    df["deltaPhi_j2MET_corr"] = deltaPhi(
        df["nonRes_corr_MET_phi"], df["nonRes_sublead_bjet_phi"])

    return df


def calc_var(df_in, input_var):
    if input_var in df_in.columns:
        raiseException(
            "Should not be using this function for variable already in df")
    elif "over" in input_var:
        if "mjj" in input_var:
            base_var = input_var.replace("_over_mjj", "")
            return df_in[base_var]/df_in["nonRes_dijet_mass_PNet_all"]
    elif "projectphi" in input_var:
        num = input_var.split("projectphi")[1][0]
        return np.cos(df_in["deltaPhi_j" + str(num) + "MET_corr"])
    elif "eta_sign" in input_var:
        base_var = input_var.replace("_sign", "")
        return df_in[base_var] * np.sign(df_in["nonRes_lead_bjet_eta"])
    elif "Mjj_Corr_Ratio" == input_var:
        return (df_in["nonRes_gen_dijet_mass_neutrino"]-df_in["nonRes_dijet_mass_PNet_all"])/df_in["nonRes_dijet_mass_PNet_all"]
    else:
        print("WARNING: Variable not yet supported")
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
