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


def corrMET(MET_pt, MET_phi, sumET, jet_pt_old, jet_m_old, jet_pt_new, jet_m_new, jet_phi):
    MET = vector.array({
        "rho": MET_pt,
        "phi": MET_phi
    })
    delta_pt = jet_pt_new - jet_pt_old
    delta_vector = vector.array({
        "rho": delta_pt,
        "phi": jet_phi
    })

    delta_ET = np.sqrt(jet_pt_new**2 + jet_m_new**2) - \
        np.sqrt(jet_pt_old**2 + jet_m_old**2)

    corr_MET = MET - delta_vector
    corr_MET_sumET = sumET - delta_ET

    # change input MET columns, do nothing if jet is placeholder
    MET_pt = np.where(jet_pt_old == -999, MET_pt, corr_MET["rho"])
    MET_phi = np.where(jet_pt_old == -999, MET_phi, corr_MET["phi"])
    sumET = np.where(jet_pt_old == -999, sumET, corr_MET_sumET)


def add_PNetCorrections(df):

    corr_MET_pt = df["puppiMET_pt"]
    corr_MET_phi = df["puppiMET_phi"]
    corr_MET_sumET = df["puppiMET_sumEt"]
    jet_res_sum = np.zeros_like(df["puppiMET_pt"])

    # jet_prefixes = [f"jet{i}" for i in range(1, 11)]
    jet_prefixes = ["nonRes_lead_bjet", "nonRes_sublead_bjet"]
    for jet_prefix in jet_prefixes:  # correct MET
        corrMET(
            corr_MET_pt,
            corr_MET_phi,
            corr_MET_sumET,
            df[jet_prefix+"_pt"],
            df[jet_prefix+"_mass"],
            df[jet_prefix+"_pt_PNet_all"],
            df[jet_prefix+"_mass_PNet_all"],
            df[jet_prefix+"_phi"])
        jet_res_sum = jet_res_sum + df[jet_prefix + "_PNetRegPtRawRes"]**2
    # add columns to df

    df["jet_res_sum"] = np.sqrt(jet_res_sum)
    df["nonRes_corr_MET_pt"] = corr_MET_pt
    df["nonRes_corr_MET_phi"] = corr_MET_phi
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
    elif "modified" in input_var:
        base_var = input_var.replace("_modified", "")
        if "sublead" in input_var:
            return df_in[base_var]*df_in["nonRes_sublead_bjet_pt_PNet_all"]/df_in["nonRes_dijet_mass_PNet_all"]
        else:
            return df_in[base_var]*df_in["nonRes_lead_bjet_pt_PNet_all"]/df_in["nonRes_dijet_mass_PNet_all"]
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
