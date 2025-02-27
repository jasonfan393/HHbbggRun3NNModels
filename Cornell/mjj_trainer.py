# Todo: cleanup imports
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, make_scorer
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
from tensorflow.keras.optimizers import Adam
import vector
from scipy.optimize import curve_fit
import mplhep as hep
from tensorflow.keras.models import load_model
import utils
import argparse
import json


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


p0 = [.5, 125, .2]


def build_model(input_shape, learning_rate=0.001, loss='mean_squared_error', N=100, layers=3):
    normalization_layer = Normalization()
    # Compute the mean and variance of the training data
    normalization_layer.adapt(X_train)
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
    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])
    return model


def plot_mjj_distr(df, corr_term, save_location, do_fits=True):
    """
    plot_mjj_distr plots dijet mass distribution comparison
    compares raw HiggsDNA output, PNet regressed dijet and mjj regressor output

    :param df: pandas dataframe containing dijet masses
    :param corr_term: correction term prediction from jj_regressor
    :param save_location: directory to save plot
    :param do_fits: bool sets whether or not to attempt to fit dijet masses to gaussians
    """
    min_mjj = 70
    max_mjj = 190
    num_bins = 100
    colors = ['tab:blue', 'orange', 'green']
    m_vars = ["nonRes_dijet_mass_PNet_all", "nonRes_dijet_mass"]
    df = df[m_vars]
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tight_layout()
    hep.cms.label("Preliminary", ax=ax, loc=0)

    bins_hist = np.linspace(min_mjj, max_mjj, num=num_bins)

    print(df["nonRes_dijet_mass_PNet_all"].shape)
    print(corr_term.shape)
    corr_term = corr_term.flatten()
    mjj_reg = (corr_term * df["nonRes_dijet_mass_PNet_all"]
               ) + df["nonRes_dijet_mass_PNet_all"]

    masses_dict = {
        'HiggsDNA Reco.': df["nonRes_dijet_mass"],
        'PNet Reg.': df["nonRes_dijet_mass_PNet_all"],
        'PNet Reg. + mjj Reg.': mjj_reg
    }
    i = 0
    for key, distr in masses_dict.items():
        plt.hist(distr, bins=bins_hist, histtype='step',
                 density=True, color=colors[i], label=key)
        if do_fits:
            bins = np.linspace(min_mjj, max_mjj, num=300)
            hist, bin_edges = np.histogram(distr, density=True, bins=bins)
            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
            mean = np.mean(distr)
            std = np.std(distr)
            lower = mean - std
            upper = mean + std
            hist_range = hist[(bin_centres < upper) & (bin_centres > lower)]
            bin_centres_range = bin_centres[(
                bin_centres < upper) & (bin_centres > lower)]
            p0 = [.5, mean, std]
            coeff, var_matrix = curve_fit(
                gauss, bin_centres_range, hist_range, p0=p0)
            mu = coeff[1]
            sig = coeff[2]
            lower = mu - abs(sig)
            upper = mu + abs(sig)
            hist_range = hist[(bin_centres < upper) & (bin_centres > lower)]
            bin_centres_range = bin_centres[(
                bin_centres < upper) & (bin_centres > lower)]
            coeff, var_matrix = curve_fit(
                gauss, bin_centres_range, hist_range, p0=p0)
            mu = coeff[1]
            sig = coeff[2]
            lower = mu - abs(sig)
            upper = mu + abs(sig)
            hist_range = hist[(bin_centres < upper) & (bin_centres > lower)]
            bin_centres_range = bin_centres[(
                bin_centres < upper) & (bin_centres > lower)]
            coeff, var_matrix = curve_fit(
                gauss, bin_centres_range, hist_range, p0=p0)
            hist_fit = gauss(bin_centres_range, *coeff)
            plt.plot(bin_centres_range, hist_fit, color=colors[i], label='$\mu$ = ' + str(
                coeff[1])[:5] + ", $\sigma$ = " + str(abs(coeff[2]))[:4], linestyle='--')
            i = i+1
    plt.legend()
    plt.xlabel("$M_{jj}$ (GeV)")
    plt.xlim(min_mjj, max_mjj)
    plt.savefig(save_location+"/mjj_distribution.png")
    plt.clf()


def plot_history(history, loss, save_location):
    print(history.history.keys())
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tight_layout()

    hep.cms.label("Preliminary", ax=ax, loc=0)
    plt.plot(history.history["loss"], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Huber)')
    plt.legend()

    plt.savefig(save_location + "/lossplot.png")
    plt.clf()


def predict(model, X):
    return model.predict(X).flatten()  # Ensures output is a 1D NumPy array


def custom_scorer(y_true, y_pred):
    # Negative MSE for consistency with sklearn scoring
    return -mean_squared_error(y_true, y_pred)


class WrappedModel:
    def __init__(self, model):
        self.model = model  # Store the loaded model

    def predict(self, X):
        return predict(model, X)  # Uses the predict function

    def fit(self, X, Y):
        pass


def feature_importance(model, training_vars, X_test, Y_test, metric, save_location):
    wrapped_model = WrappedModel(model)
    perm_importance_results = permutation_importance(
        wrapped_model, X_test, Y_test, scoring=make_scorer(custom_scorer, greater_is_better=True), n_repeats=1, random_state=42)

    feature_importance = perm_importance_results.importances_mean
    feature_names = training_vars

    # hep.style.use("CMS") FIXME this plot does not look good with default mplhep style
    fig, ax = plt.subplots(figsize=(16, 8))
    # hep.cms.label("Preliminary", ax=ax, loc=0)
    plt.barh(feature_names, feature_importance,
             color="royalblue", edgecolor="black", alpha=0.75)
    plt.xlabel("Feature Importance Score", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()  # Invert y-axis so most important feature is at the top
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    fig.subplots_adjust(left=0.4)
    plt.savefig(save_location + "/feature_importance.png")
    plt.clf()


# mjj_trainer processing begins here:
parser = argparse.ArgumentParser(
    prog='Mjj Regressor Trainer', description='Script trains mjj regression, produces relevant performance plots')
parser.add_argument('--vars', default='configs/variables_Cornell_mjj.json')
parser.add_argument('--out_dir', default='mjj_regressor_performance')
parser.add_argument('--model', default='mjj_regressor_model')
parser.add_argument(
    '--training_set', default='/afs/cern.ch/user/j/jafan/eosjfan/public/mbbTraining/GluGluToHH_with_full_PNet_info.parquet')
parser.add_argument('--plotsOnly', default=False, action='store_true')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# open input variable json
with open(args.vars, 'r') as file:
    data = json.load(file)
input_vars = data["input_variables"]
target_var = data["target"]
allvars = [*input_vars, target_var]

# Open training parquet
df = utils.load_parquet_file(args.training_set, loadAll=True)
# Create calculated columns not in base parquets
df = utils.add_PNetCorrections(df)

df_train, df_test = train_test_split(df, test_size=0.5)
print("# of training events (Pre-cuts): " + str(len(df_train.index)))

df_train = df_train.drop(
    df_train[np.abs(df_train["nonRes_lead_bjet_genFlav"]) != 5].index)
df_train = df_train.drop(
    df_train[np.abs(df_train["nonRes_sublead_bjet_genFlav"]) != 5].index)

print("# of training events (Post-cuts): " + str(len(df_train.index)))

df_train_in, extravars = utils.mjj_input_df(df_train, allvars)
df_test_in, extravars = utils.mjj_input_df(df_test, allvars)

X_train = df_train_in[input_vars]
X_test = df_test_in[input_vars]
y_train = df_train_in[target_var]
y_test = df_test_in[target_var]

print("Training variables:")
for i in input_vars:
    print(i)
print('# of variables: ' + str(len(input_vars)))

# TODO port gridsearch into this script, runs very slow on lxplus

# results of a recent grid search #FIXME to import from a config:
lr = .000005
epochs = 50
batchsize = 32
N = 400
L = 4
loss = tf.keras.losses.Huber()

if args.plotsOnly:
    model = load_model(args.model,
                       custom_objects={'huber_loss': tf.keras.losses.Huber})
else:
    # Initialize the model
    model = build_model(X_train.shape[1], lr, loss, N, L)
    # Train the model - this is very slow on lxplus
    history = model.fit(X_train, y_train, validation_split=0.2,
                        epochs=epochs, batch_size=batchsize)
    model.save(args.model)
    # Plot loss function
    plot_history(history, 'huber_loss', args.out_dir)
# Run performance plots, etc

mjj_reg_corr_term = model.predict(df_test_in[input_vars])

feature_importance(model, input_vars, X_test, y_test,
                   tf.keras.losses.MeanSquaredError, args.out_dir)

plot_mjj_distr(df_test, mjj_reg_corr_term, args.out_dir)
