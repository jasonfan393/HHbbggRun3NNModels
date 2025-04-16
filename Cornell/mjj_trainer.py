
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
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
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


def novosibirsk(x, A, x0, sigma, tau):
    """
    Computes the Novosibirsk function.

    Parameters:
    x     : float or ndarray : Input value(s)
    A     : float : Amplitude (scales the function)
    x0    : float : Peak position
    sigma : float : Width parameter (sigma, must be > 0)
    tau   : float : Tail parameter (controls asymmetry)

    Returns:
    float or ndarray : Computed function values
    """
    if abs(tau) < 1e-7:  # If tau is very small, function reduces to a Gaussian
        return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

    # Compute Lambda
    ln4 = np.log(4)
    lambda_ = np.sinh(tau * np.sqrt(ln4)) / (abs(sigma) * tau * np.sqrt(ln4))

    # Compute the logarithmic term safely
    arg = 1 + lambda_ * tau * (x - x0)
    arg = np.clip(arg, 1e-10, None)  # Prevent log of zero/negative values
    log_term = np.log(arg) / tau

    return A * np.exp(-0.5 * (log_term ** 2))


def double_crystal_ball(x, A, mean, sigma, alpha1, n1, alpha2, n2):
    """
    Double-sided Crystal Ball function.

    Parameters:
    x      : float or ndarray : Input value(s)
    A      : float : Amplitude (scales the function)
    mean   : float : Peak position
    sigma  : float : Width parameter
    alpha1 : float : Left-side transition parameter
    n1     : float : Left-side power-law exponent
    alpha2 : float : Right-side transition parameter
    n2     : float : Right-side power-law exponent

    Returns:
    float or ndarray : Computed function values
    """

    abs_alpha1 = abs(alpha1)  # Ensure positive
    abs_alpha2 = abs(alpha2)

    z = (x - mean) / sigma
    C1 = (n1 / abs_alpha1) ** n1 * np.exp(-0.5 * abs_alpha1 ** 2)
    C2 = (n2 / abs_alpha2) ** n2 * np.exp(-0.5 * abs_alpha2 ** 2)
    D1 = n1 / abs_alpha1 - abs_alpha1
    D2 = n2 / abs_alpha2 - abs_alpha2

    result = np.where(
        z < -abs_alpha1,
        A * C1 * (D1 - z) ** -n1,  # Left tail
        np.where(
            z > abs_alpha2,
            A * C2 * (D2 + z) ** -n2,  # Right tail
            A * np.exp(-0.5 * z ** 2)  # Gaussian core
        )
    )

    return result


def build_model(input_shape, X_train, optimizer='adam', N=200, activation='relu', layers=2, dropout=0.1, lr=0.001):
    loss = tf.keras.losses.Huber()
    normalization_layer = Normalization()
    # Compute the mean and variance of the training data
    normalization_layer.adapt(X_train)
    model = Sequential()
    model.add(normalization_layer)
    model.add(Dense(N, activation=activation, input_shape=(input_shape,)))
    model.add(Dropout(dropout))
    for i in range(layers-1):
        model.add(Dense(N/(2+i), activation=activation))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    if optimizer == 'adam':
        opt = Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=lr)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model


def plot_input_vars(df, input_vars, save_location):

    hep.style.use("CMS")
    fig, ax = plt.subplots()
    hep.cms.label("Preliminary", ax=ax, loc=0)

    if not os.path.exists(save_location + "/input_vars"):
        os.makedirs(save_location + "/input_vars")

    for input_var in input_vars:
        plt.hist(df[input_var])
        plt.xlabel(input_var)
        plt.savefig(save_location+"/input_vars/" + input_var+".png")
        plt.clf()
import seaborn as sns
def corr_matrix(df, save_location):

# Compute correlation matrix
    corr_matrix = df.corr()
# Plot the correlation matrix
    new_names = []
    for i in df.columns.tolist():
        i = (i.replace("Res_",""))
        i = (i.replace("corr_MET","MET"))
        new_names.append(i.replace("puppiMET","MET"))

    plt.figure(figsize=(14, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='crest', fmt=".2f", linewidths=0.5)
    # Improve label readability
    plt.xticks(ticks=np.arange(len(new_names)) + 0.5,rotation=90, ha='right', fontsize=15, labels = new_names)
    plt.yticks(ticks=np.arange(len(new_names)) + 0.5,fontsize=15, labels = new_names)

    #plt.title('Improved Correlation Matrix', fontsize=14)
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(save_location + "/corrmatrix.png")

def plot_mjj_distr(df_in, corr_term, save_location, do_fits=True, METcut = 0):
    """
    plot_mjj_distr plots dijet mass distribution comparison
    compares raw HiggsDNA output, PNet regressed dijet and mjj regressor output

    :param df: pandas dataframe containing dijet masses
    :param corr_term: correction term prediction from jj_regressor
    :param save_location: directory to save plot
    :param do_fits: bool sets whether or not to attempt to fit dijet masses to gaussians
    """
    df = df_in.copy()
    min_mjj = 70
    max_mjj = 190
    num_bins = 100
    colors = ['tab:blue', 'orange', 'green', 'purple']
    #m_vars = ["Res_dijet_massPNetCorr","Res_gen_dijet_mass_neutrino"]
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    hep.cms.label("Preliminary", ax=ax, loc=0)

    bins_hist = np.linspace(min_mjj, max_mjj, num=num_bins)

    print(df["Res_dijet_massPNetCorr"].shape)
    print(corr_term.shape)
    corr_term = corr_term.flatten()
    df["mjj_reg"] = (corr_term * df["Res_dijet_mass"]
               ) + df["Res_dijet_mass"]

    if METcut == 1:
        df = df[df["Res_corr_MET_pt"]>40]
    if METcut == 2:
        df = df[df["Res_corr_MET_pt"]<40]

    masses_dict = { 'PNet Reg.': df["Res_dijet_mass"],
        'mjj Reg.': df["mjj_reg"],
        #'test': df["Res_dijet_massPNetCorr"],
#        'target' : df["Res_gen_dijet_mass_neutrino"]
    }
    i = 0
    for key, distr in masses_dict.items():
        plt.hist(distr, bins=bins_hist, histtype='step',
                 density=True, color=colors[i], label=key)
        if do_fits:
            bins = np.linspace(min_mjj, max_mjj, num=100)
            hist, bin_edges = np.histogram(distr, density=True, bins=bins)
            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
            mean = np.mean(distr)
            std = np.std(distr)
            lower = min_mjj
            upper = max_mjj
            p0 = [np.max(hist), 120, 20, 1, 1, 1, 1]
            bounds = (
                [0, 80, 1, 0.1, 1, 0.1, 1],  # Lower bounds
                [np.inf, 160, 60, np.inf, np.inf, np.inf, np.inf]  # Upper bounds
            )
            coeff, var_matrix = curve_fit(
                double_crystal_ball, bin_centres, hist, p0=p0, bounds = bounds)
            print(coeff)
            hist_fit = double_crystal_ball(bin_centres, *coeff)
            plt.plot(bin_centres, hist_fit, color=colors[i], label='$\mu$ = ' + str(
                coeff[1])[:5] + ", $\sigma$ = " + str(abs(coeff[2]))[:4], linestyle='--')
            i = i+1
    plt.legend()
    plt.xlabel("$M_{jj}$ (GeV)")
    plt.xlim(min_mjj, max_mjj)
    plt.savefig(save_location+"/mjj_distribution"+str(METcut)+".png")
    plt.clf()

    if True:
#        df = df.drop(
#            df[np.abs(df["Res_lead_bjet_genFlav"]) != 5].index)
#        df = df.drop(
#            df[np.abs(df["Res_sublead_bjet_genFlav"]) != 5].index)
        df = df.drop(
            df[np.abs(df["Res_lead_bjet_genMatched"]) != 1].index)
        df = df.drop(
            df[np.abs(df["Res_sublead_bjet_genMatched"]) != 1].index)

        hep.style.use("CMS")
        fig, ax = plt.subplots()
        hep.cms.label("Preliminary", ax=ax, loc=0)
        mjj_reco_over_gen = df["Res_dijet_mass"]/df["Res_gen_dijet_mass_neutrino"]
        mjj_reg_over_gen = df["mjj_reg"]/df["Res_gen_dijet_mass_neutrino"]
        mjj_over_gen = df["Res_dijet_mass"]/df["Res_gen_dijet_mass_neutrino"]
        mean_reco = str(np.mean(mjj_reco_over_gen))[:5]
        mean_reg = str(np.mean(mjj_reg_over_gen))[:5]
        std_reco = str(np.std(mjj_reco_over_gen))[:5]
        std_reg = str(np.std(mjj_reg_over_gen))[:5]
        bins = np.linspace(0,2,num=50)
        plt.hist([mjj_reco_over_gen,mjj_reg_over_gen],label = ["PNet Reg. \n $\mu$ = " + mean_reco + " $\sigma$ = "+ std_reco,"PNet Reg. + mjj Reg. \n $\mu$ = " + mean_reg + " $\sigma$ = "+ std_reg],histtype = 'step', density = True,bins = bins)
        #plt.hist(mjj_over_gen, label = 'reco.', histtype = 'step', density = True)
        plt.xlabel("$M_{jj} / M_{jj}^{target}$")
        plt.legend()
        plt.savefig(save_location+"/mjj_ratio.png")
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
    fig, ax = plt.subplots(figsize = (12,12))
    # hep.cms.label("Preliminary", ax=ax, loc=0)
    for i in range(len(feature_importance)):
        if feature_importance[i]<0:
            feature_importance[i]=0
    new_names = []
    for i in feature_names:
        i = (i.replace("Res_",""))
        i = (i.replace("corr_MET","MET"))
        new_names.append(i.replace("puppiMET","MET"))
    plt.barh(new_names, feature_importance,
             color="royalblue", edgecolor="black", alpha=0.75)
    plt.xlabel("Feature Importance Score", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()  # Invert y-axis so most important feature is at the top
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    fig.subplots_adjust(left=0.5)
    plt.savefig(save_location + "/feature_importance.png")
    plt.clf()

if __name__ == '__main__':
    # mjj_trainer processing begins here:
    parser = argparse.ArgumentParser(
        prog='Mjj Regressor Trainer', description='Script trains mjj regression, produces relevant performance plots')
    parser.add_argument('--vars', default='configs/variables_Cornell_mjj.json')
    parser.add_argument('--out_dir', default='mjj_regressor_performance')
    parser.add_argument('--model', default='mjj_regressor_model')
    parser.add_argument(
        '--training_set', default='/afs/cern.ch/user/j/jafan/eosjfan/public/mbbTraining_V5/')
    parser.add_argument('--plotsOnly', default=False, action='store_true')
    parser.add_argument('--doGridSearch', default=False, action='store_true')
    parser.add_argument('--hParams', default='configs/hParams.json')
    parser.add_argument('--year', default = 'all')
    args = parser.parse_args()

    doGridSearch = args.doGridSearch
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # open input variable json
    with open(args.vars, 'r') as file:
        data = json.load(file)
    input_vars = data["input_variables"]
    target_var = data["target"]
    allvars = [*input_vars, target_var]

    # Open training parquet

    years = [ "2022preEE",
             "2022postEE",
             "2023preBPix",
             "2023postBPix"
            ]
    i = 0
    df_list = []

    for i, year in enumerate(years, start=1):  # start=1 ensures numbering starts from 1
        year_ = year.split("p")[0]
        year_int = int(year_) - 2022
        if (year_ == args.year) or (args.year == "all"):
            try:
                df_year = utils.load_parquet_file(args.training_set + "/" + year + "/NOTAG_merged.parquet", loadAll=True)
                df_year["year"] = year_int
                df_list.append(df_year)  # Collect all DataFrames in a list
            except:
                print("FILE " + args.training_set + "/" + year + "/NOTAG_merged.parquet is MISSING")
    if args.year != "all":
        input_vars.remove("year")
    df = pd.concat(df_list, axis=0)  
    df = utils.add_PNetCorrections(df,"Res") #FIXME no longer needed?

    df_train, df_test = train_test_split(df, test_size=0.7, random_state = 123)
    print("# of training events (Pre-cuts): " + str(len(df_train.index)))

#    df_train = df_train.drop(
#        df_train[np.abs(df_train["Res_lead_bjet_genFlav"]) != 5].index)
#    df_train = df_train.drop(
#        df_train[np.abs(df_train["Res_sublead_bjet_genFlav"]) != 5].index)
    df_train = df_train.drop(
        df_train[np.abs(df_train["Res_sublead_bjet_genMatched"]) != 1].index)
    df_train = df_train.drop(
        df_train[np.abs(df_train["Res_sublead_bjet_genMatched"]) != 1].index)

    print("# of training events (Post-cuts): " + str(len(df_train.index)))

    df_train_in, extravars = utils.mjj_input_df(df_train, allvars, "Res",df_train["year"])
    df_test_in, extravars = utils.mjj_input_df(df_test, allvars, "Res", df_train["year"])

    plot_input_vars(df_train_in, input_vars, args.out_dir)

    X_train = df_train_in[input_vars]
    X_test = df_test_in[input_vars]

    print("Training variables:")
    for i in input_vars:
        print(i)
    print('# of variables: ' + str(len(input_vars)))

    print("Target Variable: ")
    print(target_var)

    model = []

    if doGridSearch:
        print("Performing randomized search with CV")
        from scikeras.wrappers import KerasRegressor
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        param_dist = {
            'optimizer': ['adam'],
            'N': [128, 256],
            'activation': ['relu'],
            'layers': [1,2,3],
            'batch_size': [16, 32 , 48],
            'epochs': [20, 30, 35],
            'dropout': [0, 0.1],
            'lr':  [0.00002, 0.00001]
        }

        keras_reg = KerasRegressor(
            build_fn=build_model, input_shape=X_train.shape[1], X_train=X_train, verbose=0)

        keras_reg = KerasRegressor(
            build_fn=build_model,
            X_train=X_train,
            input_shape=X_train.shape[1],  # Fix input shape
            optimizer='adam',  # Default optimizer
            N=64,  # Default N
            activation='relu',
            layers=2,
            lr=0.001,
            dropout=0.2,
            verbose=0
        )
        random_search = RandomizedSearchCV(
            estimator=keras_reg,
            param_distributions=param_dist,
            n_iter=100,
            cv=3,
            verbose=1,
            n_jobs=10
        )
        y_train = df_train_in[target_var]

        random_search.fit(X_train, y_train)

        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        best_params = random_search.best_params_

        with open(args.hParams, 'w') as f:
            json.dump(best_params, f)

    if args.plotsOnly:
        model = load_model(args.model,
                           custom_objects={'huber_loss': tf.keras.losses.Huber})
    else:
        if not args.doGridSearch:
            with open(args.hParams, 'r') as f:
                best_params = json.load(f)

        loss = tf.keras.losses.Huber()
        # Initialize the model
        model = build_model(
            X_train.shape[1], X_train,
            optimizer=best_params['optimizer'],
            N=best_params['N'],
            activation=best_params['activation'],
            layers=best_params['layers'],
            lr=best_params['lr'],
            dropout=best_params['dropout']
        )
        y_train = df_train_in[target_var]

        early_stop = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',
          patience=5,
          restore_best_weights=True
        )

        # Train the model - this is very slow on lxplus
        history = model.fit(X_train, y_train, validation_split=0.2,
                            epochs=best_params['epochs'], batch_size=best_params['batch_size'], callbacks = [early_stop])
        model.save(args.model)
        # Plot loss function
        plot_history(history, 'huber_loss', args.out_dir)


    # Run performance plots, etc
    corr_matrix(df_train_in[input_vars], args.out_dir)

    mjj_reg_corr_term = model.predict(df_test_in[input_vars])
    mjj_reg_corr_term_train = model.predict(df_train_in[input_vars])

    #plot_mjj_distr(df_train, mjj_reg_corr_term_train, args.out_dir)
    plot_mjj_distr(df_test, mjj_reg_corr_term, args.out_dir,METcut = 1)
    plot_mjj_distr(df_test, mjj_reg_corr_term, args.out_dir,METcut = 2)
    plot_mjj_distr(df_test, mjj_reg_corr_term, args.out_dir)

    plt.hist(mjj_reg_corr_term)
    plt.yscale('log')
    plt.savefig(args.out_dir + "test.png")
    plt.clf()
    plt.hist(mjj_reg_corr_term)
    plt.yscale('log')
    plt.savefig(args.out_dir + "test2.png")
    plt.clf()


    y_test = df_test_in[target_var]

#    feature_importance(model, input_vars, X_test, y_test,
#                       tf.keras.losses.MeanSquaredError, args.out_dir)
#    if not os.path.exists(args.out_dir + "/train"):
#        os.makedirs(args.out_dir + "/train")
#    feature_importance(model, input_vars, X_train, y_train,
#                       tf.keras.losses.MeanSquaredError, args.out_dir+"/train/")



