import numpy as np
# import numpy.random as npr
import torch
from sklearn.decomposition import FastICA, PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import explained_variance_score, silhouette_score
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
# from sklearn.feature_selection import f_regression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
# from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
# from tqdm.auto import trange 

from ssm import HMM # keep this if using ssm package
# from dynamax.hidden_markov_model import GaussianHMM # keep this if using dynamax package
# from jax import random as jr # keep this if using dynamax package
# from jax import devices # keep this if using dynamax package
# # from ssm.messages import hmm_sample
# print(devices())
# print(devices()[0].platform)
# if devices()[0].platform == 'cpu':
#     cpu_mode = True
# else:
#     cpu_mode = False


from joblib import Parallel, delayed
# import multiprocessing
# NumThread=(multiprocessing.cpu_count()-1)*2 # sets number of workers based on cpus on current machine
# print('Parallel processing with '+str(NumThread)+' cores')

# from util_factorial_hmm import gibbs

import os

os.makedirs("fig_modelsel", exist_ok=True)
os.makedirs("data_modelsel", exist_ok=True)


if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# --- Step 1: Define a function that computes CV R^2 and significance for one neuron and one covariate ---

def process_neuron_multi(neuron_activity, behavior_df, covariate_cols, n_splits=5, p_thresh=0.05):
    """
    For a single neuron, perform a simultaneous multiple linear regression of its activity on all behavioral covariates,
    with an intercept term. Then, for each covariate, estimate the unique (partial) hold-out variance explained (R²)
    via cross-validation. If the coefficient (beta) for a covariate is not significant in the full model (p >= p_thresh),
    set its R² to NaN.
    
    Parameters
    ----------
    neuron_activity : np.ndarray, shape (T,)
        The activity time series for one neuron.
    behavior_df : pd.DataFrame
        DataFrame containing the behavioral predictors.
    covariate_cols : list of str
        List of column names in behavior_df to use as predictors.
    n_splits : int, optional
        Number of cross-validation folds.
    p_thresh : float, optional
        Significance threshold (default 0.05).
    
    Returns
    -------
    result : dict
        Dictionary mapping each covariate to the estimated cross-validated partial R² if significant, otherwise np.nan.
    """
    # --- Step 1: Z-score the data ---
    # Neuron activity: shape (T,)
    y = zscore(neuron_activity)
    
    # Behavioral predictors: extract and z-score each column (resulting in a matrix X of shape (T, n_covariates))
    X = behavior_df[covariate_cols].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # --- Step 2: Fit full multiple regression on all data to get p-values ---
    X_const = sm.add_constant(X)  # adds intercept
    full_model = sm.OLS(y, X_const).fit()
    p_vals = full_model.pvalues[1:]  # p-values for predictors (skip intercept)
    
    # --- Step 3: Cross-validation to compute partial R^2 for each predictor ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # For each predictor, we will accumulate partial R^2 across folds.
    partial_r2_all = {col: [] for col in covariate_cols}
    
    for train_idx, test_idx in kf.split(y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        # Fit full model (without intercept adjustment since we assume data are z-scored, but we use intercept anyway)
        lr_full = LinearRegression().fit(X_train, y_train)
        full_r2 = lr_full.score(X_test, y_test)
        
        # For each predictor, remove that column and fit a reduced model.
        for i, col in enumerate(covariate_cols):
            X_train_red = np.delete(X_train, i, axis=1)
            X_test_red = np.delete(X_test, i, axis=1)
            lr_red = LinearRegression().fit(X_train_red, y_train)
            red_r2 = lr_red.score(X_test_red, y_test)
            # The unique variance explained by predictor i is the drop in R².
            partial_r2 = full_r2 - red_r2
            # Ensure no negative values (if negative, set to zero)
            partial_r2 = max(partial_r2, 0)
            partial_r2_all[col].append(partial_r2)
    
    # Average the partial R^2 values over folds.
    avg_partial_r2 = {col: np.mean(partial_r2_all[col]) for col in covariate_cols}
    
    # --- Step 4: Create output dictionary: assign NaN for predictors not significant ---
    result = {col: (avg_partial_r2[col] if p_vals[i] < p_thresh/len(covariate_cols) else np.nan)
              for i, col in enumerate(covariate_cols)}
    
    return result

def get_covariates_from_session(save_session,Option):
    
    # Define the mapping of session names to covariate columns
    covariate_dict = {'131124': ['walk_ser','lft_whisk_ser', 'lft_ppl_ser'], '131657': ['walk_ser','rt_whisk_ser', 'rt_ppl_ser'], '132249': ['walk_ser', 'lft_whisk_ser','lft_ppl_ser'],
          '132836': ['walk_ser','rt_whisk_ser', 'rt_ppl_ser'], '144153': ['walk_ser', 'rt_whisk_ser','rt_ppl_ser'], '144441': ['walk_ser','lft_whisk_ser', 'lft_ppl_ser'], 
          '114033': ['walk_ser','rt_whisk_ser', 'rt_ppl_ser'], '130648': ['walk_ser', 'lft_whisk_ser','lft_ppl_ser']}

    frame_rate = {'131124': 3.06, '131657': 3.06, '132249': 3.06, '132836': 3.06, '144153': 3.38, '144441': 3.38, '114033': 3.06, '130648': 3.06}

    if Option == 'covariates':
        dict = covariate_dict
    elif Option == 'timescales':
        dict = frame_rate
    else:
        raise ValueError("Option must be 'covariates' or 'timescales'")
    
    for key in dict:
        if key in save_session:
            return dict[key]
    return None  # or raise an error if preferred

# --- Function to process all neurons in parallel ---
def compute_r2_all_neurons_multi(activity_df_fa, behavior_df, covariate_cols, n_splits=5, p_thresh=0.05):
    """
    For each neuron (each row in activity_df_fa), perform the multiple regression analysis
    using process_neuron_multi and return a DataFrame with results and corresponding Macroarea.
    
    Parameters
    ----------
    activity_df_fa : pd.DataFrame
        DataFrame with at least columns:
          'activity': each element is a 1D np.array of length T,
          'Macroarea': the macroarea label.
    behavior_df : pd.DataFrame
        DataFrame with behavioral predictors.
    covariate_cols : list of str
        List of behavioral covariate column names.
    n_splits : int, optional
        Number of CV folds.
    p_thresh : float, optional
        Significance threshold.
    
    Returns
    -------
    r2_df : pd.DataFrame
        DataFrame with one row per neuron and columns for each covariate's partial R^2,
        plus a 'Macroarea' column.
    """
    results = []
    for idx, row in activity_df_fa.iterrows():
        result = process_neuron_multi(row['activity'], behavior_df, covariate_cols, n_splits, p_thresh)
        results.append(result)
    
    # results = Parallel(n_jobs=-1)(
    #     delayed(process_neuron_multi)(row['activity'], behavior_df, covariate_cols, n_splits, p_thresh)
    #     for idx, row in activity_df_fa.iterrows()
    # )
    macroareas = activity_df_fa['Macroarea'].values
    r2_df = pd.DataFrame(results)
    r2_df['Macroarea'] = macroareas
    return r2_df

# --- Plotting function for box plots by Macroarea ---
def plot_r2_by_macroarea(r2_df, covariate_cols, title="Hold-out Variance Explained by Behavioral Covariates"):
    """
    Create a box plot showing, for each behavioral covariate, the distribution of hold-out R^2 values
    grouped by Macroarea.
    """
    melt_df = r2_df.melt(id_vars="Macroarea", value_vars=covariate_cols,
                         var_name="Behavior", value_name="r2")
    fig = plt.figure(figsize=(12,8))
    ax = sns.boxplot(data=melt_df, x="Behavior", y="r2", hue="Macroarea", palette="tab10")
    ax.set_title(title, fontsize=30)
    ax.set_xlabel("Behavior", fontsize=30)
    ax.set_ylabel("Hold-out Variance Explained (R²)", fontsize=30)
    ax.tick_params(axis="both", labelsize=30)
    plt.legend(title="Macroarea", fontsize=30, title_fontsize=30)
    plt.tight_layout()
    # plt.show()
    return fig

# --- Plotting function for fraction of significant regressions ---
def compute_fraction_significant(r2_df, covariate_cols):
    """
    For each Macroarea and each behavioral covariate, compute the fraction of neurons with significant regression (non-NaN R^2).
    Returns a DataFrame with columns: Macroarea, Behavior, fraction_significant.
    """
    macroareas = r2_df['Macroarea'].unique()
    frac_data = []
    for macro in macroareas:
        sub_df = r2_df[r2_df['Macroarea'] == macro]
        for col in covariate_cols:
            frac_sig = sub_df[col].notna().mean()
            frac_data.append({"Macroarea": macro, "Behavior": col, "fraction_significant": frac_sig})
    return pd.DataFrame(frac_data)

def plot_fraction_significant(frac_df, title="Fraction of Significant Regressions by Macroarea and Behavior"):
    """
    Create a bar plot showing, for each behavioral covariate and Macroarea, the fraction of neurons with significant regression.
    """
    fig = plt.figure(figsize=(12,8))
    ax = sns.barplot(data=frac_df, x="Behavior", y="fraction_significant", hue="Macroarea", palette="tab10")
    ax.set_ylabel("Fraction of Significant Regressions", fontsize=30)
    ax.set_xlabel("Behavior", fontsize=30)
    ax.set_title(title, fontsize=30)
    ax.tick_params(axis="both", labelsize=30)
    plt.legend(title="Macroarea", fontsize=30, title_fontsize=30)
    plt.tight_layout()
    # plt.show()
    return fig



def getStandardFrames(ccf_dict, npy_dict_path=None):
    """
    Create a data dictionary from the standard frames data and add the mappings defined in the data dictionary.
    
    Parameters
    ----------
    ccf_dict : pd.DataFrame
        DataFrame containing the mappings (e.g., with columns 'Area_Number', 'med_x', 'med_y', etc.).
    npy_dict_path : str
        Path to the file containing the standard frames data.
    
    Returns
    -------
    dfof : (pd.DataFrame or np.ndarray)
        Upsampled dfof data.
    df_merged : pd.DataFrame
        DataFrame resulting from merging the ROI data with the CCF mapping.
    """
    # If no file path is specified, prompt the user.
    if npy_dict_path is None:
        npy_dict_path = input("Enter full path and filename of standard frames file to load, without quotes: ")
    
    # Determine file extension.
    ext = os.path.splitext(npy_dict_path)[1].lower()
    if ext == ".npy":
        # Load using np.load with allow_pickle.
        loaded_obj = np.load(npy_dict_path, allow_pickle=True)
        # If loaded_obj is an ndarray with object dtype, extract the contained object.
        if isinstance(loaded_obj, np.ndarray) and loaded_obj.dtype == object:
            loaded_obj = loaded_obj.item()
    elif ext == ".pkl":
        # Use pandas read_pickle for robust unpickling of pandas objects.
        loaded_obj = pd.read_pickle(npy_dict_path)
    # elif ext == ".h5":
    #     # Use h5py to load the file; build a dictionary from datasets.
    #     with h5py.File(npy_dict_path, 'r') as f:
    #         loaded_obj = { key: f[key][()] for key in f.keys() }
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # Now, handle two cases: loaded_obj is either a dictionary or a pandas DataFrame.
    if isinstance(loaded_obj, dict):
        data_dict = loaded_obj
    elif isinstance(loaded_obj, pd.DataFrame):
        # If the DataFrame contains the required keys, extract them.
        required_keys = {'dfof_upsamp_frame', 'roi_info_frame', 'iscell_masks_cell'}
        if required_keys.issubset(set(loaded_obj.columns)):
            data_dict = { key: loaded_obj[key] for key in required_keys }
        else:
            # Otherwise, convert the DataFrame to a dict (using orient='list')
            data_dict = loaded_obj.to_dict(orient='list')
    else:
        raise ValueError("Loaded data is of an unsupported type: expected dict or pandas DataFrame.")
    
    # Extract the standardized fluorescence data.
    dfof = data_dict['dfof_upsamp_frame']
    # Extract the ROI info data.
    roi_df = data_dict['roi_info_frame']
    # Replace roi_df['iscell_masks_cell'] with the adjusted one.
    roi_df['iscell_masks_cell'] = data_dict['iscell_masks_cell']
    # Merge the ROI DataFrame with the CCF mapping.
    df_merged = pd.merge(left=roi_df, right=ccf_dict, left_on='iscell_masks_cell', right_on='Area_Number', how='left')
    df_merged = df_merged.assign(neg_med_y = -1 * df_merged['med_y'],
                                  neg_med_x = -1 * df_merged['med_x'])
    
    return dfof, df_merged

def plot_transition_matrices(transition_matrices):
    """
    Plots a grid of heatmaps for each transition probability matrix in the provided array.
    
    Parameters:
    transition_matrices (np.ndarray): Array of shape (n_factors, n_states, n_states) representing 
                                      transition probability matrices.
    """
    n_factors, n_states, _ = transition_matrices.shape
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(n_factors, 1, figsize=(5 * n_states, 5 * n_factors))
    
    if n_factors == 1:
        axes = [axes]
    
    for i in range(n_factors):
        ax = axes[i]
        matrix = transition_matrices[i]
        
        # Plot the heatmap
        cax = ax.matshow(matrix, cmap='viridis')
        
        # Add colorbar
        fig.colorbar(cax, ax=ax)
        
        # Set axis labels
        ax.set_xticks(np.arange(n_states))
        ax.set_yticks(np.arange(n_states))
        ax.set_xticklabels(np.arange(n_states))
        ax.set_yticklabels(np.arange(n_states))
        ax.set_xlabel('State To')
        ax.set_ylabel('State From')
        ax.set_title(f'Transition Matrix {i+1}')
        
        # Annotate the cells with the transition probabilities
        for (j, k), val in np.ndenumerate(matrix):
            ax.text(k, j, f'{val:.4f}', ha='center', va='center', color='white' if val < 0.5 else 'black')
    
    plt.tight_layout()
    # plt.show()
    
def plot_fHMM_model(fHMM_model, lower=0, upper=None, tickspace=20):
    """
    Plots the state traces, expected emission traces, and actual emission traces of your synthetic fHMM data dictionary.
    
    Parameters:
    fHMM_model (dictionary): Data dictionary containing synthetic fHMM ground truth data derived from generate_fHMM_data()
    lower (int): the timesteps that you want your plots to start at
    upper (int): the timesteps that you want your plots to end at
    tickspace(int): the space between your y axis tick marks for the emission traces
    """
    if upper is None:
        upper = fHMM_model['hypers']['n_timesteps']
        
    fig, axs = plt.subplots(3, 1, sharex=True)
    plt.setp(axs, yticks=list(range(fHMM_model['hypers']['emission_dim'])))
    plt.sca(axs[0])
    plt.yticks(list(range(fHMM_model['hypers']['n_factors'])))
    
    # Visualize the true states
    true_states = fHMM_model['fHMM_data']['true_states'].T
    unique_states = np.unique(true_states)
    cmap_greys = plt.cm.Greys
    norm_greys = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(unique_states) + 0.5, 1), ncolors=256)
    img1 = axs[0].imshow(true_states, aspect="auto", interpolation="none", cmap=cmap_greys, norm=norm_greys)
    axs[0].set_title('true states')
    axs[0].set_ylabel('factor')
    cbar1 = fig.colorbar(img1, ax=axs[0], orientation='vertical', ticks=unique_states)
    cbar1.ax.set_yticklabels([str(int(state)) for state in unique_states])
    
    # Find the global min and max for the emissions
    expec_emissions = fHMM_model['fHMM_data']['expec_emissions'].T
    emissions = fHMM_model['fHMM_data']['emissions'].T
    vmin = min(np.min(expec_emissions), np.min(emissions))
    vmax = max(np.max(expec_emissions), np.max(emissions))
    
    # Visualize the expected emissions
    img2 = axs[1].imshow(expec_emissions, aspect="auto", interpolation="none", vmin=vmin, vmax=vmax)
    axs[1].set_title('expected emissions')
    axs[1].set_ylabel('component')
    y_ticks = axs[1].get_yticks()
    axs[1].set_yticks(y_ticks[::tickspace])
    axs[1].set_yticklabels([str(int(tick)) for tick in y_ticks[::tickspace]])
    
    # Set colorbar ticks for the expected emissions
    cbar2_ticks = [vmin, 0, vmax] # [vmin, (vmin + vmax) / 2, vmax]
    cbar2 = fig.colorbar(img2, ax=axs[1], orientation='vertical', ticks=cbar2_ticks)
    cbar2.ax.set_yticklabels([f'{tick:.2f}' for tick in cbar2_ticks])
    
    # Visualize the actual emissions
    img3 = axs[2].imshow(emissions, aspect="auto", interpolation="none", vmin=vmin, vmax=vmax)
    axs[2].set_title('true emissions')
    axs[2].set_ylabel('component')
    y_ticks = axs[2].get_yticks()
    axs[2].set_yticks(y_ticks[::tickspace])
    axs[2].set_yticklabels([str(int(tick)) for tick in y_ticks[::tickspace]])
    
    # Set colorbar ticks for the actual emissions
    cbar3_ticks = [vmin, 0, vmax] # [vmin, (vmin + vmax) / 2, vmax]
    cbar3 = fig.colorbar(img3, ax=axs[2], orientation='vertical', ticks=cbar3_ticks)
    cbar3.ax.set_yticklabels([f'{tick:.2f}' for tick in cbar3_ticks])
    
    plt.xlim(lower, upper)
    fig.supxlabel('time')
    
    plt.tight_layout()
    # plt.show()
    
def getList(dictionary):
    """
    Iterates through a dictionary and places each key into a list
    
    Parameters
    ----------
    dictionary: python dictionary
    
    Returns
    -------
    list
        a list object containing the keys of the dictionary
    """
    # ls = [] # instantiate empty list
    # for key in dictionary.keys(): # iterate through keys of dictionary
    #     ls.append(key) # append each key to list
    ls = list(dictionary.keys())
    return ls

def area2region(units, field):
    """
    Function to take session information dataframe and add an area column based on the region column.
    
    Parameters
    ----------
    units (pandas df): dataframe containing information on each neuron for the given session
    field (str): the field on which we want to map areas to brain regions
    
    Returns
    -------
    df (pandas df): the mapped dataframe
    """
    dict = {'FrontalCortex': ['FRP', 'PL', 'ACAd', 'MOs', 'MOp'],
            'VisualCortex': ['VISp', 'VISpl', 'VISpor', 'VISl', 'VISli', 'VISal', 'VISrl', 'VISa', 'VISam', 'VISpm'],
            'AuditoryCortex': ['AUDp', 'AUDv', 'AUDd', 'AUDpo'],
            'SomatosensoryCortex': ['SSp', 'SSp-tr', 'SSp-ll', 'SSp-ul', 'SSp-m', 'SSp-n', 'SSp-un', 'SSp-bfd', 'SSs'],
            'PosteriorCortex': ['RSPv', 'RSPd', 'RSPagl', 'TEa']
            }
    df = pd.DataFrame.from_dict(dict.items())
    df = df.explode(1)
    df = df.rename(columns= {0:'region', 1:'area'})
    df = df.merge(units, left_on = 'area', right_on = field)
    # df = df.drop(columns='area_x')
    df = df.rename(columns={'area_y': 'area'})
    return df

def expected_emissions(posteriors, means):
    # output expec_emissions is a 2d array of size (num_timesteps, emission_dim)
    gammat = posteriors_list2array(posteriors)
    means_2d = np.vstack(means)
    expec_emissions = gammat @ means_2d
    return expec_emissions

def true2onehot_states(true_states):
    # Convert true_states to one-hot encoding
    num_steps, num_chains = true_states.shape
    num_states = np.max(true_states) + 1
    one_hot_states = np.zeros((num_steps, num_chains, num_states))

    for i in range(num_steps):
        for j in range(num_chains):
            state = true_states[i, j]
            one_hot_states[i, j, state] = 1

    # Flatten the one_hot_states to (num_steps x (num_chains * num_states))
    flat_one_hot_states = one_hot_states.reshape(num_steps, -1)
    return flat_one_hot_states


def plot_trans_matrix(gen_trans_mat):
          plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
          num_states=gen_trans_mat.shape[0]
          for i in range(gen_trans_mat.shape[0]):
                    for j in range(gen_trans_mat.shape[1]):
                              text = plt.text(j, i, str(np.around(gen_trans_mat[i, j], decimals=2)), ha="center", va="center",
                                        color="k", fontsize=12)
          plt.xlim(-0.5, num_states - 0.5)
          plt.xticks(range(0, num_states), np.arange(num_states)+1, fontsize=10)
          plt.yticks(range(0, num_states), np.arange(num_states)+1, fontsize=10)
          plt.ylim(num_states - 0.5, -0.5)
          plt.ylabel("state t", fontsize = 15)
          plt.xlabel("state t+1", fontsize = 15)

def find_plateau_point(scores, start_monitor=10, window=5, patience=5):
    """
    Estimate the index in the scores array where the derivative becomes nearly constant.
    
    Process:
      1. Compute the derivative: deriv[i] = scores[i+1] - scores[i].
      2. For each index i starting from start_monitor-1 (so that we have at least 10 scores),
         consider the last 'window' derivative values (i.e. deriv[max(0, i-window+1): i+1]).
      3. Compute the minimum and maximum of that window.
      4. Search the entire derivative (from index 0 up to i) for the first index j whose value lies within [min, max].
         That index j is the candidate plateau point.
      5. Continue sliding the window. If the candidate plateau index remains the same for 'patience'
         consecutive iterations, then stop and return candidate+1 (mapping back to the scores index).
      6. If no plateau is found, return the last index.
    
    Parameters
    ----------
    scores : list or np.array
        Array of CV scores.
    start_monitor : int, optional
        Start monitoring after this many scores (default 10).
    window : int, optional
        Number of derivative points to use in each window (default 5).
    patience : int, optional
        Number of consecutive iterations with the same candidate needed to declare plateau (default 5).
    
    Returns
    -------
    plateau_index : int
        Index (in the original scores array) where the plateau is detected.
    """
    scores = np.array(scores)
    deriv = np.diff(scores)  # Length = len(scores) - 1
    stable_count = 0
    candidate = None
    plateau_index = None
    
    # Only start monitoring if we have at least start_monitor scores
    if len(scores) < start_monitor:
        return len(scores) - 1

    # Slide i over the derivative array (i corresponds to the last index in the current derivative window)
    for i in range(start_monitor - 1, len(deriv)):
        # Define the window: the last 'window' derivative values up to index i
        window_vals = deriv[max(0, i - window + 1): i + 1]
        window_min = np.min(window_vals)
        window_max = np.max(window_vals)
        R = np.abs(window_max - window_min)
        
        # Find the first index in the entire derivative (from 0 to i) that lies within [window_min, window_max]
        candidate_index = None
        for j in range(i + 1):  # search from 0 to i
            if window_min <= deriv[j] <= window_max:
                candidate_index = j
                break
        # print(R)
        # # Additional check: ensure that all derivative values from candidate_index to i are within [window_min, window_max]
        # if candidate_index is not None:
        #     if not np.all((deriv[candidate_index:i+1] >= window_min-R/3) & (deriv[candidate_index:i+1] <= window_max+R/3)):
        #         candidate_index = None            
        
        if candidate_index is not None:
            if candidate is not None and candidate == candidate_index:
                stable_count += 1
            else:
                candidate = candidate_index
                stable_count = 1
        else:
            candidate = None
            stable_count = 0
        
        if stable_count >= patience:
            plateau_index = candidate + 1  # map derivative index back to scores index (j+1)
            break

    if plateau_index is None:
        plateau_index = len(scores) - 1
    return plateau_index


def find_elbow_point(y_values):
    n_points = len(y_values)
    all_coords = np.vstack((range(n_points), y_values)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    elbow_index = np.argmax(dist_to_line)
    return elbow_index + 1  # +1 as index starts from 0

def compute_single_score_pca_fa(n,X):
    pca = PCA(n_components=n, svd_solver="full")
    fa = FactorAnalysis(n_components=n)
    pca_score = np.mean(cross_val_score(pca, X,n_jobs=-1,cv=10))
    fa_score = np.mean(cross_val_score(fa, X,n_jobs=-1,cv=10))
    return pca_score, fa_score

def compute_scores_pca_fa(X, n_components=None, method='max',patience=5):
    """
    Compute CV scores for PCA and FactorAnalysis over an increasing range of components.
    Early stopping is applied separately for PCA and FA: if the optimal number of components 
    (i.e. the one yielding the maximum CV score) does not change for 'patience' consecutive 
    candidate n values, then that branch stops updating.

    in the 'elbow' mode:
        Calculates both the elbow and the argmax, and if the argmax < elbow, sets the elbow=argmax
    
    Parameters
    ----------
    X : array-like
        Input data.
    n_components : iterable, optional
        Iterable of component counts to test. If None, defaults to range(1, min(X.shape)+1).
    patience : int, optional
        Number of consecutive iterations with no change in best n before stopping that branch.
    
    Returns
    -------
    pca_scores : list
        List of PCA scores computed (only until early stopping for PCA).
    fa_scores : list
        List of FA scores computed (only until early stopping for FA).
    tested_n_pca : list
        List of n values tested for PCA.
    tested_n_fa : list
        List of n values tested for FA.
    """
    if n_components is None:
        n_components = list(range(1, min(X.shape) + 1))
    else:
        n_components = list(n_components)
    
    pca_scores = []; fa_scores = []; 
    tested_n_pca = []; tested_n_fa = [];
    
    best_n_pca = None; best_n_fa = None;
    count_pca = 0; count_fa = 0;
    
    # Flags to indicate if a branch has stopped updating.
    pca_done = False; fa_done = False;

    for n in n_components:
        pca_score, fa_score = compute_single_score_pca_fa(n, X)
        
        if not pca_done:
            tested_n_pca.append(n)
            pca_scores.append(pca_score)
            # Update PCA early stopping.
            if n>5:
                current_best_index_pca_max = np.argmax(pca_scores)
                if method=='max':
                    current_best_index_pca = current_best_index_pca_max
                elif method=='elbow':
                    current_best_index_pca = find_elbow_point(pca_scores)
                    current_best_index_pca = np.min([current_best_index_pca,current_best_index_pca_max])
                current_best_n_pca = tested_n_pca[current_best_index_pca]
                if best_n_pca is not None and current_best_n_pca == best_n_pca:
                    count_pca += 1
                else:
                    best_n_pca = current_best_n_pca
                    count_pca = 0
                if count_pca >= patience:
                    pca_done = True
        if not fa_done:
            tested_n_fa.append(n)
            fa_scores.append(fa_score)
            if n>2:
                # Update FA early stopping.
                current_best_index_fa_max = np.argmax(fa_scores)
                if method=='max':
                    current_best_index_fa = current_best_index_fa_max
                elif method=='elbow':
                    current_best_index_fa = find_elbow_point(fa_scores)
                    current_best_index_fa = np.min([current_best_index_fa,current_best_index_fa_max])
                current_best_n_fa = tested_n_fa[current_best_index_fa]
                if best_n_fa is not None and current_best_n_fa == best_n_fa:
                    count_fa += 1
                else:
                    best_n_fa = current_best_n_fa
                    count_fa = 0
                if count_fa >= patience:
                    fa_done = True

        
        # If both branches are done, we can break out early.

        if pca_done and fa_done:
            break
    return pca_scores, fa_scores, tested_n_pca, tested_n_fa, best_n_pca, best_n_fa


def save_cv_scores_plot_pca_fa(pca_scores, fa_scores, n_components_pca, n_components_fa, tested_n_pca, tested_n_fa, file_save, title=None):
    """
    Create a plot of cross-validation (CV) scores for PCA and FactorAnalysis, and save it as a PDF.

    Parameters
    ----------
    tested_n_pca, tested_n_fa : array-like
        Array of number of components (x-axis values).
    pca_scores : array-like
        CV scores from PCA.
    fa_scores : array-like
        CV scores from Factor Analysis.
    n_components_pca : int
        Optimal number of PCA components determined by cross-validation.
    n_components_fa : int
        Optimal number of Factor Analysis components determined by cross-validation.
    file_save : str
        Full path (including filename and .pdf extension) to save the plot.
    title : str, optional
        Title for the plot.
    """
    plt.figure()
    
    # New code:
    plt.plot(tested_n_pca, pca_scores, "b", label="PCA scores")
    plt.plot(tested_n_fa, fa_scores, "r", label="FA scores")
    plt.axvline(n_components_pca, color="b", label=f"PCA CV: {n_components_pca}", linestyle="--")
    plt.axvline(n_components_fa, color="r", label=f"FactorAnalysis CV: {n_components_fa}", linestyle="--")
    plt.xlabel("nb of components")
    plt.ylabel("CV VarExp")
    # plt.ylabel("CV scores")
    plt.legend(loc="lower right")
    if title:
        plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(file_save, format="pdf")
    plt.close()

def compute_scores_ica(X, n_components=None, patience=5):
    """
    Compute CV scores for PCA and FactorAnalysis over an increasing range of components.
    Early stopping is applied separately for PCA and FA: if the optimal number of components 
    (i.e. the one yielding the maximum CV score) does not change for 'patience' consecutive 
    candidate n values, then that branch stops updating.
    
    Parameters
    ----------
    X : array-like
        Input data.
    n_components : iterable, optional
        Iterable of component counts to test. If None, defaults to range(1, min(X.shape)+1).
    patience : int, optional
        Number of consecutive iterations with no change in best n before stopping that branch.
    
    Returns
    -------
    pca_scores : list
        List of PCA scores computed (only until early stopping for PCA).
    fa_scores : list
        List of FA scores computed (only until early stopping for FA).
    tested_n_pca : list
        List of n values tested for PCA.
    tested_n_fa : list
        List of n values tested for FA.
    """
    if n_components is None:
        n_components = list(range(1, min(X.shape) + 1))
    else:
        n_components = list(n_components)
    
    ica_scores = []
    tested_n_ica = []
    
    best_n_ica = None
    count_ica = 0
    
    # Flags to indicate if a branch has stopped updating.
    ica_done = False; 

    for n in n_components:
        
        try:
            ica = FastICA(n_components=n)
            ica_score = np.mean(cross_val_score(ica, X,n_jobs=-1,cv=10))
        except Exception as e:
            print(f"ICA error at n={n}: {e}")
            ica_score = np.nan  # Use NaN to indicate failure; you could also use -np.inf if preferred.
        
        
        
                    


        if not ica_done:
            tested_n_ica.append(n)
            ica_scores.append(ica_score)
            # Update PCA early stopping.
            if n>2:
                # current_best_index_pca = np.argmax(pca_scores)
                current_best_index_ica = find_plateau_point(ica_scores)
                current_best_n_ica = tested_n_ica[current_best_index_ica]
                if best_n_ica is not None and current_best_n_ica == best_n_ica:
                    count_ica += 1
                else:
                    best_n_ica = current_best_n_ica
                    count_ica = 0
                if count_ica >= patience:
                    ica_done = True

        if ica_done:
            break
    return ica_scores, tested_n_ica


def rescale_matrix(matrix):
    """
    Rescales each row of the matrix to have values between 0 and 1.
    
    Parameters:
    - matrix: A numpy array of shape (N, T) where N is the number of features and T is the number of samples.
    
    Returns:
    - A numpy array where each row is scaled to have minimum value 0 and maximum value 1.
    """
    # Calculate the minimum and maximum of each row
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    
    # Rescale each row
    scaled_matrix = (matrix - row_min) / (row_max - row_min)
    
    return scaled_matrix


# def autocorrelation(x):
#     """Compute the autocorrelation of the signal, based on the properties of the
#     power spectral density of the signal."""
#     xp = x - np.mean(x)
#     f = np.fft.fft(xp)
#     p = np.array([np.real(v) * np.real(v) + np.imag(v) * np.imag(v) for v in f])
#     pi = np.fft.ifft(p)
#     return np.real(pi)[:x.size // 2] / np.sum(xp ** 2)

def autocovariance_torch(x):
    """
    Compute the autocorrelation of the signal using PyTorch, leveraging GPU computation on Apple Silicon.
    Parameters:
    - x: A 1D PyTorch tensor of the signal.
    Returns:
    - A 1D PyTorch tensor containing the autocorrelation of the input signal.
    """
    # Ensure input is a float32 tensor for fft and ensure it's on the GPU
    # x = torch.tensor(x).to(dtype=torch.float32).to('mps')  # Convert to float32 and use 'mps' for Apple Silicon GPU
    x = torch.tensor(x, dtype=torch.float32, device=device)    
    # Detrend the signal by removing the mean
    xp = x - torch.mean(x)
    
    # Compute the FFT of the detrended signal
    f = torch.fft.fft(xp)
    
    # Compute the power spectrum density (PSD)
    p = torch.real(f) * torch.real(f) + torch.imag(f) * torch.imag(f)
    
    # Inverse FFT to get the autocorrelation function
    pi = torch.fft.ifft(p)
    
    # Normalize and return the real part of the autocorrelation
    out=torch.real(pi)[:x.size(0) // 2] / torch.sum(xp ** 2)
    return out.cpu().numpy()

def compute_hwhm(acf):
    """Estimate the timescale as the half-width at half-maximum of the ACF's envelope."""
    half_max = np.max(acf) / 2
    for i, val in enumerate(acf):
        if val < half_max:
            return i
    return len(acf) - 1  # Return the max timescale if HWHM is not found

def plot_autocorrelations_with_timescale(matrix,file_save=None,OptLeg=True):
    """Plot the autocorrelations of each time series in the matrix and annotate with HWHM."""
    fig=plt.figure(figsize=(10, 6))
    timescales = []
    for i in range(matrix.shape[0]):
        acf = autocovariance_torch(matrix[i])
        timescale = compute_hwhm(acf)
        timescales.append(timescale)
        plt.plot(acf, label=f"Series {i+1} (Timescale: {timescale})")
    plt.legend() if OptLeg else None
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation of Time Series with Timescales")
    plt.xlim([0, np.max(timescales)*10])
    if file_save is not None:
        plt.savefig(file_save+'AutocorrTimescales.pdf')
    else:
        plt.savefig('fig_modelsel/AutocorrTimescales.pdf')
    # plt.show()
    plt.close()
    return timescales, fig


# Function to compute variance explained by SVD components on the test set
def compute_test_variance(X_train, X_test, n_components):
    ica = FastICA(n_components=n_components, random_state=0)
    # Xtrain = W * Htrain
    # project Xtest on W: W^T Xtest = Htest
    # project Xtest: W
    ica.fit(X_train)
    sources_test = ica.transform(X_test)
    X_test_reconstructed = ica.inverse_transform(sources_test)
    return explained_variance_score(X_test, X_test_reconstructed)

# Function to perform cross-validation for a given number of components
def cross_validate_components(n_components, emissions_data):
    kf = KFold(n_splits=4)
    variances = Parallel(n_jobs=-1)(delayed(compute_test_variance)(emissions_data[train_index], emissions_data[test_index], n_components) for train_index, test_index in kf.split(emissions_data))
    return np.mean(variances)


def evaluate_gmm_k_fold(X, K, n_splits=5):
    """
    Evaluates GMM for a single feature using K-fold cross-validation.

    Parameters:
    - X: Data for the feature (should be a 2D array).
    - K: Number of components for GMM.
    - n_splits: Number of splits for K-fold cross-validation.

    Returns:
    - Average log-likelihood across all folds.
    """
    kf = KFold(n_splits=n_splits)
    log_likelihoods = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        gmm = GaussianMixture(n_components=K, random_state=42).fit(X_train)
        log_likelihood = gmm.score(X_test)
        log_likelihoods.append(log_likelihood)

    return np.mean(log_likelihoods)





def evaluate_hmm_k_fold(synthetic_data, K, n_splits=3):
    """
    Evaluates HMM for a single feature using K-fold cross-validation.

    Parameters:
    - synthetic_data: Data for the feature (should be a 2D array).
    - K: Number of components for GMM.
    - n_splits: Number of splits for K-fold cross-validation.

    Returns:
    - Average log-likelihood across all folds.
    """

    kf = KFold(n_splits=n_splits)
    N_iters = 500 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    TOL=10**-4 # tolerance parameter (see N_iters)
    nRunEM=1 # # of times we run EM for each choice of number of states
    # initialized training and test loglik for model selection, and BIC
    ll_training = np.zeros((n_splits,nRunEM))
    ll_heldout = np.zeros((n_splits,nRunEM))
    results=[]
    for iK, (train_index, test_index) in enumerate(kf.split(synthetic_data)):
        data_in={'training_data':synthetic_data[train_index],'test_data':synthetic_data[test_index],
                'N_iters':N_iters,'TOL':TOL}
        for iRun in range(nRunEM):
            # key = jr.PRNGKey(iRun)
            results.append(xval_func(data_in, K))
        # results = Parallel(n_jobs=NumThread)(
        #     delayed(xval_func)(data_in, num_states0)
        #     for iRun,num_states0 in zip(RunN,stN))
        # unpack
        for i in range(nRunEM):
            ll_training[iK,i]=results[i]['ll_training']
            ll_heldout[iK,i]=results[i]['ll_heldout']
    ll_heldout_mean=np.mean(ll_heldout.flatten())

    return ll_heldout_mean

def xval_func(data_in,num_states0):
    
    training_data=data_in['training_data']
    test_data=data_in['test_data']
    N_iters=data_in['N_iters']
    TOL=data_in['TOL']
    
    obs_dim = len(training_data[0])             # number of observed dimensions: outcome
    nTrain=len(training_data)
    nTest=len(test_data)
    
    out={}
    # fit HMM with ssm
    mle_hmm = HMM(num_states0, obs_dim, 
          observations="gaussian", transitions="standard")
    #fit on training data
    hmm_lls = mle_hmm.fit(training_data, method="em", num_iters=N_iters, tolerance=TOL)                
    # Compute log-likelihood for each dataset
    out['ll_training'] = mle_hmm.log_likelihood(training_data)/nTrain
    out['ll_heldout'] = mle_hmm.log_likelihood(test_data)/nTest
    # ## fit hmm with dynamax
    # run_glmhmm=GaussianHMM(num_states0, obs_dim)
    # key = jr.PRNGKey(0)
    # em_params, em_param_props = run_glmhmm.initialize(key)
    # # em_params, em_param_props = run_glmhmm.initialize(key, method="kmeans", emissions=training_data)
    # em_params, hmm_lls = run_glmhmm.fit_em(em_params, 
    #                               em_param_props,
    #                               training_data, num_iters=N_iters, tolerance=TOL)
    # out['ll_training'] = hmm_lls[-1]/nTrain
    # out['ll_heldout'] = run_glmhmm.marginal_log_prob(em_params, test_data)/nTest    
    return out






def find_best_k(feature_data, option):
    """
    Finds the best K using the elbow method on K-fold cross-validation log-likelihood.

    Parameters:
    - feature_data: Data for a single feature.
    - max_k: Maximum number of components to consider.
    - n_splits: Number of splits for K-fold cross-validation.

    Returns:
    - A dictionary with 'best_k' and 'log_likelihoods'.
    """
    MAX,count,old_optimal_n_components=15,0,0
    average_variances = []
    for n_components in range(1,MAX):
          if option=='GMM':
            n_splits=5
            average_variance=evaluate_gmm_k_fold(feature_data, n_components, n_splits)
          elif option=='HMM':
            n_splits=3
            average_variance=evaluate_hmm_k_fold(feature_data, n_components, n_splits)
          average_variances.append(average_variance)
          # Find the elbow point
          if n_components>5:
                    # diffs = np.diff(average_variances)
                    # threshold = 0.1 * np.abs(diffs[0])
                    # elbows = np.where(np.abs(diffs) < threshold)[0]
                    # new_optimal_n_components = elbows[0] + 1 if len(elbows) > 0 else MAX
                    new_optimal_n_components = find_elbow_point(average_variances)
                    if new_optimal_n_components==old_optimal_n_components:
                              count+=1
                    else: count=0
                    if count>2:
                              break
                    old_optimal_n_components=new_optimal_n_components
    optimal_n_components=new_optimal_n_components
    print(optimal_n_components)
    print(average_variances)
    out_return={'best_k': optimal_n_components, 'log_likelihoods': average_variances}
    return out_return



def find_best_k_FixedMaxk(feature_data, max_k=10, n_splits=5):
    """
    Finds the best K using the elbow method on K-fold cross-validation log-likelihood.

    Parameters:
    - feature_data: Data for a single feature.
    - max_k: Maximum number of components to consider.
    - n_splits: Number of splits for K-fold cross-validation.

    Returns:
    - A dictionary with 'best_k' and 'log_likelihoods'.
    """
    log_likelihoods = [evaluate_gmm_k_fold(feature_data, K, n_splits) for K in range(1, max_k + 1)]
    diffs = np.diff(log_likelihoods)
    threshold = 0.1 * np.abs(diffs[0])
    elbows = np.where(np.abs(diffs) < threshold)[0]
    best_k = elbows[0] + 1 if len(elbows) > 0 else max_k

    return {'best_k': best_k, 'log_likelihoods': log_likelihoods}

# def runHMM(synthetic_data,num_states, obs_dim,N_iters,TOL,key):
def runHMM(synthetic_data,num_states, obs_dim,N_iters,TOL):
    temp={}
    ## this code uses ssm library to fit HMM
    new_hmm = HMM(num_states, obs_dim, 
            observations="gaussian", transitions="standard")
    _ = new_hmm.fit(synthetic_data, method="em", num_iters=N_iters, tolerance=TOL)   
    temp['lls']=new_hmm.log_likelihood(synthetic_data)
    temp['hmm']=new_hmm
    temp['posteriors']=new_hmm.expected_states(synthetic_data)[0]
    temp['mus']=new_hmm.observations.mus
    ## this code uses dynamax to fit HMM
    # run_glmhmm=GaussianHMM(num_states, obs_dim)
    # em_params, em_param_props = run_glmhmm.initialize(key)
    # em_params, hmm_lls = run_glmhmm.fit_em(em_params, 
    #                               em_param_props, 
    #                               synthetic_data,num_iters=N_iters)
    # temp['lls']=hmm_lls[-1]
    # temp['hmm']=run_glmhmm
    # temp['posteriors']=run_glmhmm.smoother(em_params,synthetic_data)
    # temp['mus']=em_params.emissions.means
    return temp

def single_hmm(synthetic_data,obs_dim,num_states):
    """Run numRun hmms to find best hmm for the data, return the best hmm and its posterior probs
    This function is needed to find best local maximum in the EM algorithm
    synthetic_data: T x obs_dim
    obs_dim: int
    num_states: int
    
    """
    N_iters=1000
    TOL=10**-4
    numRun=4
    # key_set=[jr.PRNGKey(iRun) for iRun in range(numRun)]
    # out = Parallel(n_jobs=-1)(delayed(runHMM)(synthetic_data,num_states, obs_dim,N_iters,TOL,key) for (irun,key) in zip(range(numRun),key_set))
    # out = Parallel(n_jobs=-1)(delayed(runHMM)(synthetic_data,num_states, obs_dim,N_iters,TOL) for irun in range(numRun))
    out = []
    for irun in range(numRun):
        result = runHMM(synthetic_data, num_states, obs_dim, N_iters, TOL)
        out.append(result)
        
    lls=[out[irun]['lls'] for irun in range(numRun)]
    best_hmm=out[np.argmax(lls)]['hmm']
    # posterior_probs = best_hmm.expected_states(synthetic_data)[0]
    posterior_probs=out[np.argmax(lls)]['posteriors']
    mus=out[np.argmax(lls)]['mus']
    # mus=best_hmm.observations.mus
    # sigmas=best_hmm.observations.Sigmas
    out={'posterior_probs':posterior_probs,'mus':mus}
    return out


# def reduce_correlated_features(correlation_matrix, threshold=0.95):
#     # Find pairs of features with high correlation
#     high_corr_pairs = np.where((correlation_matrix) > threshold)
#     high_corr_pairs = [(i, j) for i, j in zip(*high_corr_pairs) if i != j]

#     # Group correlated features
#     groups = []
#     for pair in high_corr_pairs:
#         for group in groups:
#             if pair[0] in group or pair[1] in group:
#                 group.add(pair[0])
#                 group.add(pair[1])
#                 break
#         else:
#             groups.append(set(pair))

#     # Print group members
#     for group in groups:
#         print(group)

#     # Keep one feature per group
#     features_to_keep = [list(group)[0] for group in groups]

#     # Add features that didn't belong to any group
#     all_features = set(range(correlation_matrix.shape[0]))
#     grouped_features = set([item for sublist in groups for item in sublist])
#     non_grouped_features = list(all_features - grouped_features)
#     features_to_keep.extend(non_grouped_features)

#     # Remove duplicates
#     features_to_keep = list(set(features_to_keep))

#     # Recompute correlation matrix
#     new_corr_matrix = correlation_matrix[np.ix_(features_to_keep, features_to_keep)]

#     return new_corr_matrix, features_to_keep


def visualize_corr_matrix(corr_matrix, option='nan_diag',option_annot=False):
    """
    Visualizes the correlation matrix.

    Args:
        corr_matrix (numpy.ndarray): Correlation matrix.
        option (str): If 'nan_diag', sets the diagonal values to NaN.

    Returns:
        fig (matplotlib.figure.Figure): Figure object of the heatmap.
    """
    fig = plt.figure(figsize=(10, 10))
    
    # Only visualize the off-diagonal terms if option is not 'diag'
    if option == 'nan_diag':
        np.fill_diagonal(corr_matrix, np.nan)
    
    sns.heatmap(corr_matrix, annot=option_annot, fmt=".2f",center=0, cmap='coolwarm', square=True)
    
    return fig



def average_off_diagonal(correlation_matrix):
    # Create a copy of the correlation matrix
    matrix_copy = correlation_matrix.copy()
    # Set the diagonal elements to nan
    np.fill_diagonal(matrix_copy, np.nan)
    # Calculate the mean of the off-diagonal elements
    average = np.nanmean(matrix_copy)
    return average

def calculate_communities(threshold, correlation_matrix):
    # Convert correlation to distance
    distance_matrix = correlation_matrix+1
    np.fill_diagonal(distance_matrix, 0)  # set diagonal to 0

    # Ensure symmetry
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Hierarchical clustering
    Z = linkage(squareform(distance_matrix), method='average')
    clusters = fcluster(Z, threshold, criterion='distance')
    num_clusters=len(np.unique(clusters))
    cluster_averages = []
    for cluster_label in np.unique(clusters):
          cluster_indices = np.where(clusters == cluster_label)[0]
          matrix_copy=correlation_matrix[cluster_indices][:, cluster_indices]
          cluster_averages.append(average_off_diagonal(matrix_copy))
    
    # Count unique clusters
    return num_clusters, clusters, Z,cluster_averages

# Revised function to find the longest interval considering the specific requirements
def find_most_meaningful_interval(num_clusters,correlation_matrix):
    intervals = {}
    
    # Initialize intervals tracking
    for n_clusters in set(num_clusters):
        intervals[n_clusters] = []

    # Track start and end points of intervals for each number of clusters
    start_index = 0
    for i in range(1, len(num_clusters)):
        if num_clusters[i] != num_clusters[start_index]:
            interval_length = i - start_index
            intervals[num_clusters[start_index]].append((start_index, i - 1, interval_length))
            start_index = i
    # Ensure the last interval is also recorded
    interval_length = len(num_clusters) - start_index
    intervals[num_clusters[start_index]].append((start_index, len(num_clusters) - 1, interval_length))
    
    # Exclude intervals where the number of clusters is 1 or equal to the size of the correlation matrix
    exclude = [1, correlation_matrix.shape[0]]
    for ex in exclude:
        if ex in intervals:
            del intervals[ex]
    
    # Find the longest interval among the remaining ones
    longest_interval = (0, 0, 0)  # (start, end, length)
    for interval_list in intervals.values():
        for interval in interval_list:
            if interval[2] > longest_interval[2]:
                longest_interval = interval
                
    # Return the start, end indices, and number of clusters for the longest meaningful interval
    return longest_interval[0], longest_interval[1], num_clusters[longest_interval[0]]



def model_sel_ICA(emissions_data,MAX_runs=50,file_save=None,method='plateau'):
    # Define the range of components to be tested
    MAX,count,old_optimal_n_components=MAX_runs,0,0
    average_variances = []
    for n_components in range(1,MAX):
          average_variance = cross_validate_components(n_components, emissions_data)
          average_variances.append(average_variance)
          # Find the elbow point
          if n_components>15:
                    if method=='elbow':
                        new_optimal_n_components = find_elbow_point(average_variances)
                    elif method=='plateau':
                        new_optimal_n_components = find_plateau_point(average_variances)
                    elif method=='max':
                        new_optimal_n_components = np.argmax(average_variances)                        
                    if new_optimal_n_components==old_optimal_n_components:
                              count+=1
                    else: count=0
                    if count>10:
                              break
                    old_optimal_n_components=new_optimal_n_components
    optimal_n_components=new_optimal_n_components
    max_n_components=n_components

    print("Optimal number of components based on the plateau method:", optimal_n_components)

    # Plotting the scores
    n_factors_range=np.arange(max_n_components)+1
    plt.figure(figsize=(10, 6))
    plt.plot(n_factors_range, average_variances, marker='o')
    plt.axvline(optimal_n_components, color="r", label=f"optimal n: {optimal_n_components}", linestyle="--")
    # Plot the first derivative for reference.
    deriv = np.diff(average_variances)
    plt.plot(n_factors_range[1:], deriv, "o-", color="k", label="Derivative")
    plt.xlabel('Number of Factors')
    plt.ylabel('Cross-Validated Score')
    plt.title('Number of Sources vs. Cross-Validated Score')
    plt.grid(True)
    plt.legend()
    if file_save is not None:
        plt.savefig(file_save+'.pdf')
    else:
        plt.savefig('fig_modelsel/ModelSel_ICA.pdf')
    # plt.show()
    # plt.close()

    ica = FastICA(n_components=optimal_n_components, random_state=0)
    ica.fit(emissions_data)
    factors = ica.transform(emissions_data)
    ica_mixing = ica.mixing_
    ica_mean = ica.mean_
    
    # compare ICA fit and original data
    # Plot the reconstructed data
    nplots=optimal_n_components
    plot_length=np.min([300,emissions_data.shape[0]])
    fig, axes = plt.subplots(nplots, 1, figsize=(8, 12))
    data_rec=factors@ica_mixing.T+ica_mean
    for idx in range(nplots):
          axes[idx].plot(data_rec[:plot_length, idx], alpha=0.5, label=f'Fit')
          axes[idx].plot(emissions_data[:plot_length, idx], alpha=0.5,label=f'Data')
          axes[idx].legend()
#     plt.xlim([1,plot_length])
    plt.tight_layout()
    if file_save is not None:
        plt.savefig(file_save+'_reconstruction.pdf')
    else:
        plt.savefig('fig_modelsel/ModelSel_ICA_reconstruction.pdf')
    # plt.show()

    return ica,optimal_n_components,n_factors_range,average_variances
    # plt.close()


def fit2sources_findNumStates(factors,option='HMM',file_save=None):
        # Assuming `factors` is your T x N numpy array
        N = factors.shape[1]
        # Parallelize across features
        results = Parallel(n_jobs=-1)(delayed(find_best_k)(factors[:, n].reshape(-1, 1),option) for n in range(N))
        n_states=[results[i]['best_k'] for i in range(len(results))]

        # Plotting
        _,N=factors.shape
        fig, axes = plt.subplots(1, N, figsize=(N*5, 4), sharey=True)
        fig.suptitle('Best K using K-Fold Cross-Validation')
        for n, result in enumerate(results):
                axes[n].plot(range(1, len(result['log_likelihoods'])+1), result['log_likelihoods'], marker='o', linestyle='-')
                axes[n].axvline(x=result['best_k'], color='r', linestyle='--', label=f'Best K = {result["best_k"]}')
                axes[n].set_title(f'Feature {n+1}')
                axes[n].set_xlabel('Number of '+option+' States (K)')
                axes[n].set_ylabel('Log-likelihood')
                axes[n].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if file_save is not None:
            plt.savefig(file_save+'ModelSel_'+option+'BestK.pdf')
        else:
            plt.savefig('fig_modelsel/'+'ModelSel_'+option+'BestK.pdf')
        plt.show()  
        # plt.close()
        return results,n_states





def fit_GMM2sources_findNumStates(factors):
          # Assuming `factors` is your T x N numpy array
          N = factors.shape[1]

          # Parallelize across features
          results = Parallel(n_jobs=-1)(delayed(find_best_k)(factors[:, n].reshape(-1, 1), max_k=10, n_splits=5) for n in range(N))
          n_states=[results[i]['best_k'] for i in range(len(results))]

          # Plotting
          _,N=factors.shape
          fig, axes = plt.subplots(1, N, figsize=(N*5, 4), sharey=True)
          fig.suptitle('Best K using K-Fold Cross-Validation')

          for n, result in enumerate(results):
                    axes[n].plot(range(1, 11), result['log_likelihoods'], marker='o', linestyle='-')
                    axes[n].axvline(x=result['best_k'], color='r', linestyle='--', label=f'Best K = {result["best_k"]}')
                    axes[n].set_title(f'Feature {n+1}')
                    axes[n].set_xlabel('Number of Mixtures (K)')
                    axes[n].set_ylabel('Log-likelihood')
                    axes[n].legend()

          plt.tight_layout(rect=[0, 0, 1, 0.95])
        #   plt.show()    
          plt.savefig('fig_modelsel/'+'ModelSel_GMMBestK.png')
          plt.close()

          return results,n_states


# fit hmm to each factor, extract one-hot state posterior prob
def fit_hmm_after_gmm(factors,n_states,timescales=None,file_save=None):
          out=[]
          data=factors
          _,N=data.shape
          obs_dim=1
        #   posteriors = Parallel(n_jobs=-1)(delayed(single_hmm)(data[:, neuron].reshape(-1, 1),obs_dim,num_states) for (neuron,num_states) in zip(range(data.shape[1]),n_states))
          out = Parallel(n_jobs=-1)(delayed(single_hmm)(data[:, neuron].reshape(-1, 1), obs_dim, num_states) for neuron, num_states in zip(range(data.shape[1]), n_states))

          fig = plt.figure(figsize=(N*5, 5), dpi=80, facecolor='w', edgecolor='k')
          fig.suptitle('Factor time series vs HMM prob')
          for neuron in range(N):
                    posterior_probs=out[neuron]['posterior_probs']
                    mus=out[neuron]['mus']
                    state_seq=np.argmax(posterior_probs,axis=1)
                    plt.subplot(1,N,neuron+1)
                    for k in range(n_states[neuron]):
                            #   plt.plot(np.arange(len(posterior_probs)),posterior_probs[:,k], label="State " + str(k + 1), lw=2)
                              plt.plot(np.arange(len(posterior_probs)),factors[:, neuron], label="factor " + str(neuron), lw=2)
                              plt.plot(np.arange(len(posterior_probs)),mus[state_seq], label="exp values", lw=5)
                              # plt.ylim((-0.05, 1.05))
                              if timescales is not None:
                                plt.xlim((1000, 1000+timescales[neuron]*20))
                                plt.title(f"states={n_states[neuron]},tau={timescales[neuron]}", fontsize = 15)
                              else:
                                plt.title(f"states={n_states[neuron]}", fontsize = 15)
                                plt.xlim((1000, 1300))
                              # plt.yticks([0, 0.5, 1], fontsize = 10)
                              plt.xlabel("trial #", fontsize = 15)
                              plt.ylabel("p(state)", fontsize = 15)
          plt.tight_layout()
          if file_save is not None:
            plt.savefig(file_save+'ModelSel_HMMonICASources.pdf')
          else:
            plt.savefig('fig_modelsel/ModelSel_HMMonICASources.pdf')
        #   plt.show()
        #   plt.close()
          return out

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def infer_timescales(factors, min_clusters=1, max_clusters=10,file_save=None,OptLeg=True):
    """
    Infers timescales by clustering factors and selecting the optimal number of clusters
    based on silhouette scores. Handles the case where only one cluster is needed.

    Args:
        factors (numpy.ndarray): Factorized representation of states (e.g., PCA/ICA components).
        min_clusters (int): Minimum number of clusters (>=1).
        max_clusters (int): Maximum number of clusters to evaluate.

    Returns:
        timescales (numpy.ndarray): Estimated timescales for each factor.
        labels (numpy.ndarray): Cluster labels for each factor.
        optimal_n_timescales (int): Optimal number of clusters.
    """

    timescales, fig = plot_autocorrelations_with_timescale(factors.T,file_save=file_save,OptLeg=OptLeg)
    ax = fig.axes[0]  # Access the first axes
    ax.set_xlim([0, 10 * np.max(timescales)])  # Set the x-axis limits
    # print("Timescales:", timescales)

    # Step 1: Log Transformation
    log_timescales = np.log(timescales)

    # Step 2: Handle the single-cluster case
    unique_values = np.unique(log_timescales)
    if min_clusters == 1 and len(unique_values) == 1:
        # Only one unique timescale, return single label
        return timescales, np.zeros(len(timescales), dtype=int), 1

    # Step 3: Bootstrap Analysis
    N = 1000
    bootstrap_samples = np.random.choice(log_timescales, size=(N,), replace=True)

    # Step 4: Clustering with silhouette score optimization
    silhouette_avg_scores = []
    start_cluster = max(min_clusters, 2)  # Ensure valid clustering
    range_n_clusters = np.arange(start_cluster, min(len(timescales), max_clusters) + 1)

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init=10)
        cluster_labels = clusterer.fit_predict(bootstrap_samples.reshape(-1, 1))

        if len(set(cluster_labels)) > 1:  # Ensure at least 2 distinct clusters
            silhouette_avg = silhouette_score(bootstrap_samples.reshape(-1, 1), cluster_labels)
            silhouette_avg_scores.append(silhouette_avg)
        else:
            silhouette_avg_scores.append(-1)  # Assign a low score if invalid clustering

    # Step 5: Find the optimal number of clusters
    if silhouette_avg_scores:
        optimal_n_timescales = max(range_n_clusters, key=lambda i: silhouette_avg_scores[i - start_cluster])
    else:
        optimal_n_timescales = min_clusters  # Default to minimum if no valid clustering found

    # Step 6: Final Clustering
    clusterer = KMeans(n_clusters=optimal_n_timescales, random_state=10, n_init=10)
    old_labels = clusterer.fit_predict(log_timescales.reshape(-1, 1))

    # Compute the mean value of each cluster
    cluster_means = np.array([log_timescales[old_labels == i].mean() for i in range(optimal_n_timescales)])
    sorted_cluster_indices = np.argsort(cluster_means)

    # Create a mapping from original cluster labels to new labels
    label_mapping = {original: new for new, original in enumerate(sorted_cluster_indices)}
    labels = np.array([label_mapping[label] for label in old_labels])

    # print(f"Optimal number of clusters: {optimal_n_timescales}")
    # print(f"Labels: {labels}")
    # print(f"Timescales: {timescales}")

    # Step 7: Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_avg_scores, marker='o', linestyle='-')
    plt.title('KMeans Silhouette Scores to Determine Optimal Timescale Cluster Number')
    plt.xlabel('Number of Timescale Clusters')
    plt.ylabel('Silhouette Score')
    plt.axvline(x=optimal_n_timescales, color='red', linestyle='--', label=f'Optimal: {optimal_n_timescales} Clusters')
    plt.legend()
    plt.grid(True)
    if file_save is not None:
        plt.savefig(file_save+'AutocorrTimescales_Clusters.pdf')
    else:
        plt.savefig('fig_modelsel/AutocorrTimescales_Clusters.pdf')
    plt.close()
    # plt.show()

    return timescales, labels, optimal_n_timescales



def remove_duplicate_hmms(hmm_fits,threshold=0.9,threshold_low=0.8,threshold_hi=0.1,file_save=None):

    # match states from different factors 
    # for each label of timescale, estimate corr matrix between posteriors of states for each factor
    # labels_unique=np.unique(labels.astype(int))
    # posteriors_filtered=[]; mus_filtered=[]; sigmas_filtered=[]; factors_filtered=[];

    # collect states 
    posterior_probs=[]
    idx=np.arange(len(hmm_fits))
    posterior_probs=np.concatenate([hmm_fits[i]['posterior_probs'] for i in idx],axis=1).T
    data_factors={
            'factor':np.concatenate([[i for i2 in range(len(hmm_fits[i]['posterior_probs'][0]))] for i in idx]), # for each entry in posterior_probs, keep track of which factor it belongs to
            'state':np.concatenate([[i2 for i2 in range(len(hmm_fits[i]['posterior_probs'][0]))] for i in idx]), # for each entry in posterior_probs, keep track of which state it belongs to
            'mus':np.concatenate([hmm_fits[i]['mus'] for i in idx]).flatten(),
            # 'sigmas':np.concatenate([hmm_fits[i]['sigmas'] for i in idx]).flatten(),
            'posterior':posterior_probs,
            # 'labels':np.concatenate([[labels[i] for i2 in range(len(hmm_fits[i]['posterior_probs'][0]))] for i in idx])
            }
    # print(data_factors['mus'])
    correlation_matrix=np.corrcoef(posterior_probs)
    fig=visualize_corr_matrix(correlation_matrix)
    # find pairs of most correlated states
    # plt.show()
    if file_save is not None:
        fig.savefig(file_save+'ModelSel_AllHMM_Corr.pdf')
    else:
        fig.savefig('fig_modelsel/ModelSel_AllHMM_Corr.pdf')
    plt.close()

    new_corr_matrix, features_to_keep=select_least_correlated_features(correlation_matrix,threshold=threshold,threshold_low=threshold_low,threshold_hi=threshold_hi)
    fig=visualize_corr_matrix(new_corr_matrix)
    # plt.show()
    if file_save is not None:
        fig.savefig(file_save+'ModelSel_ReducedHMM_Corr.pdf')
    else:
        fig.savefig('fig_modelsel/ModelSel_ReducedHMM_Corr.pdf')
    plt.close()

    data_factors_to_keep = {}
    for key, value in data_factors.items():
        data_factors_to_keep[key] = value[features_to_keep]

    return data_factors_to_keep,new_corr_matrix,data_factors

def select_least_correlated_features(correlation_matrix,threshold=0.9,threshold_low=0.8, threshold_hi=0.1):
    """
    Identifies and removes highly correlated features while keeping the one 
    with the lowest correlation with ungrouped features.

    Steps:
    1. Form groups where correlation_matrix > 0.9.
    2. Within each group, keep only the feature whose correlation with the ungrouped 
       features is the lowest.
    3. Among the features_to_keep, identify groups where correlation_matrix < -0.9.
       - In each group, eliminate the feature that has correlation > threshold_hi 
         with at least two other features.
    4. Print the selected features.

    Args:
        correlation_matrix (numpy.ndarray): Square matrix of feature correlations.
        threshold_hi (float): Correlation threshold to apply within negative correlation groups.

    Returns:
        new_corr_matrix (numpy.ndarray): Updated correlation matrix after removing redundant features.
        features_to_keep (list): Indices of selected features.
    """

    num_features = correlation_matrix.shape[0]
    correlated_groups = []
    visited = set()

    # Step 1: Identify groups of highly correlated features (corr > threshold)
    for i in range(num_features):
        if i in visited:
            continue
        group = set([i])
        for j in range(i + 1, num_features):
            if correlation_matrix[i, j] > threshold:
                group.add(j)
                visited.add(j)
        if len(group) > 1:
            correlated_groups.append(group)

    # Step 2: Sort groups from smallest to largest (by group size)
    correlated_groups.sort(key=len)

    # Step 3: Identify the best feature in each group
    features_to_keep = set(range(num_features))  # Start by keeping all features

    for group in correlated_groups:
        remaining_features = list(features_to_keep - group)  # Ungrouped features

        if remaining_features:
            # Compute the average correlation of each feature in the group with ungrouped features
            avg_corrs = {
                feature: np.mean(np.abs(correlation_matrix[feature, remaining_features]))
                for feature in group
            }
            best_feature = min(avg_corrs, key=avg_corrs.get)  # Feature with lowest avg correlation
        else:
            best_feature = min(group)  # If no ungrouped features remain, pick the first

        # Keep only the selected feature
        features_to_keep -= group
        features_to_keep.add(best_feature)

    # ===================== Additional Condition =====================

    # Step 4: Identify groups with correlation < -threshold among features_to_keep
    neg_correlated_groups = []
    visited_neg = set()

    for i in features_to_keep:
        if i in visited_neg:
            continue
        group = set([i])
        for j in features_to_keep:
            if i != j and correlation_matrix[i, j] < -threshold_low:
                group.add(j)
                visited_neg.add(j)
        if len(group) > 1:
            neg_correlated_groups.append(group)
    print(f'neg_correlated_groups: {neg_correlated_groups}')
    # Step 5: Within each negatively correlated group, eliminate features 
    #         that have correlation > threshold_hi with at least two other features
    for group in neg_correlated_groups:
        to_remove = []
        for feature in group:
            # take correlation coefficient with the features_to_keep minus the current group
            high_corr_count=np.sum(correlation_matrix[feature, list(features_to_keep - group)] > threshold_hi)
            if high_corr_count >= 2:
                to_remove.append(feature)
        print(f'to_remove: {to_remove}')
        # Remove only the features identified
        features_to_keep -= set(to_remove)

    # Print the selected features
    features_to_keep = sorted(features_to_keep)
    print("Selected features:", features_to_keep)

    # Step 6: Construct the new correlation matrix
    new_corr_matrix = correlation_matrix[np.ix_(features_to_keep, features_to_keep)]

    return new_corr_matrix, features_to_keep


def cluster_hmm(correlation_matrix,data_factors_to_keep,file_save=None,threshold_choice=None):

    # Define thresholds
    thresholds = np.linspace(0.45, 0.9, 50)

    # Calculate communities for each threshold
    num_communities = []; id_communities=[]; Z_communities=[]; mean_corr_communities=[]
    for threshold in thresholds:
        numclust,clust,Z1,cluster_averages=calculate_communities(threshold, correlation_matrix)
        num_communities.append(numclust)
        id_communities.append(clust)
        Z_communities.append(Z1)
        mean_corr_communities.append(cluster_averages)

    # Plotting results
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, num_communities, marker='o')
    plt.xlabel('Threshold=(corr+1)')
    plt.ylabel('Number of Communities')
    plt.title('Number of Overlapping Communities vs. Threshold Level')
    plt.grid(True)
    if file_save is not None:
        plt.savefig(file_save+'ModelSel_reducedHMMcommunities_search.pdf')
    else:
        plt.savefig('fig_modelsel/ModelSel_reducedHMMcommunities_search.pdf')
    # plt.show()
    # plt.close()
    # Apply the revised function
    longest_start_meaningful, longest_end_meaningful, n_clusters_meaningful = find_most_meaningful_interval(
        num_communities,correlation_matrix)
    # Select a threshold from the end of the most meaningful interval for visualization
    # midpoint_meaningful = (longest_start_meaningful + longest_end_meaningful) // 2
    selected_threshold_meaningful = thresholds[longest_start_meaningful]-1

    # alternative, select threshold by hand
    if threshold_choice is not None:
        # find first value in thresholds that is closest to threshold_choice        
        longest_start_meaningful = np.argmin(np.abs(thresholds - (threshold_choice+1)))
        selected_threshold_meaningful = thresholds[longest_start_meaningful]-1

    id_communities_meaningful=id_communities[longest_start_meaningful]
    Z_communities_meaningful=Z_communities[longest_start_meaningful]
    # # rescale the corr in Z_communities to subtract one
    # for i in range(Z_communities_meaningful.shape[0]):
    #           Z_communities_meaningful[i,2]=Z_communities_meaningful[i,2]-1
    mean_corr_communities_meaningful=np.array(mean_corr_communities[longest_start_meaningful])

    # remove communities whose correlation is >-0.2
    # Remove communities whose correlation is not negative enough
    ind_comm=np.where((np.bincount(id_communities_meaningful) > 1))[0]
    # good_corr=np.where(mean_corr_communities_meaningful < -0.2)[0]
    good_corr=np.where(mean_corr_communities_meaningful < -0)[0]
    ind_corr=[np.where(id_communities_meaningful==i+1)[0] for i in good_corr]
    ind_corr = [item for sublist in ind_corr for item in sublist]
    indices_keep = np.union1d(ind_comm, ind_corr).astype(int)
    if len(indices_keep)==0:
            raise Exception("No communities found with negative correlation")
    #
    id_communities = id_communities_meaningful[indices_keep]
    # Z_communities = Z_communities_meaningful[indices_keep, :]
    mean_corr_communities = mean_corr_communities_meaningful[np.unique(id_communities)-1]
    # remove corresponding factors
    data_factors_to_keep_filtered = data_factors_to_keep.copy()
    for key, value in data_factors_to_keep_filtered.items():
        data_factors_to_keep_filtered[key] = value[indices_keep]

    # dendrogram before pruning the communinities
    labels_hmm = [f'({factor_idx}, {state_idx})' for (factor_idx, state_idx) in zip(data_factors_to_keep['factor'], data_factors_to_keep['state'])]
    # Plotting the dendrogram
    plt.figure(figsize=(8, 4))
    dendrogram(Z_communities_meaningful,labels=labels_hmm)
    plt.xticklabels=labels_hmm
    plt.title("Dendrogram for Hierarchical Clustering before pruning communities")
    plt.xlabel("Factors")
    plt.ylabel("Distance")
    plt.axhline(y=selected_threshold_meaningful+1, color='r', linestyle='--')
    plt.text(x=0, y=selected_threshold_meaningful+1, s=f' corr threshold= {selected_threshold_meaningful:.2f}', color='red')
    # Shift y-axis ticks
    plt.gca().set_yticklabels([f'{float(tick.get_text()) - 1:.2f}' for tick in plt.gca().get_yticklabels()])
    # plt.show()
    if file_save is not None:
        plt.savefig(file_save+'ModelSel_HMMdendrogram.pdf')
    else:
        plt.savefig('fig_modelsel/ModelSel_HMMdendrogram.pdf')
    # plt.close()
    # plt.show()

    return id_communities,mean_corr_communities,data_factors_to_keep_filtered


def estimate_transition_matrix_vec(posteriors_chain):
    # # Concatenate all posterior arrays into a single 2D array
    # posteri = np.vstack(posteriors_chain)
    # # Convert posterior probabilities to states
    # states = np.argmax(posteri, axis=0)
    # num_states=posteri.shape[0]
    # # Create pairs of consecutive states
    # pairs = np.vstack((states[:-1], states[1:]))
    # # Use bincount to count the transitions and form the transition matrix
    # transition_matrix_vec = np.bincount(num_states * pairs[0] + pairs[1], minlength=num_states**2)
    # transition_matrix_vec = transition_matrix_vec.reshape((num_states, num_states)).astype(float)
    # # Normalize each row to sum to 1
    # transition_matrix_vec /= transition_matrix_vec.sum(axis=1, keepdims=True)
    n_factors=len(posteriors_chain)
    n_states_list=[len(posteriors_chain[f][0]) for f in range(len(posteriors_chain))]
    posteriors_chain = np.hstack(posteriors_chain) 
    combined_trans = np.einsum('ti,tj->tij', posteriors_chain[1:], posteriors_chain[:-1])
    combined_trans_mean = np.mean(combined_trans, axis=0)  # shape: (total_states, total_states)
    # Partition combined_trans_mean into blocks for each factor.
    transition_matrices_list = []
    start = 0
    for h in range(n_factors):
        s = n_states_list[h]
        block = combined_trans_mean[start:start+s, start:start+s]
        row_sums = np.sum(block, axis=1, keepdims=True)
        block = np.where(row_sums == 0, 1.0/s, block / row_sums)
        transition_matrices_list.append(block)
        start += s

    return transition_matrices_list

def reconstruct_parameters(id_communities,data_factors_to_keep,ica):

    # Nchains = len(set(id_communities)) # number of chains
    Nchains_comm = np.unique(id_communities) # number of chains
    means=[] # mean emissions
    posteriors=[] # posterior probabilities
    transitions=[] # transition matrices
    init=[] # initial distribution
    ica_mixing=ica.mixing_.T
    ica_mean=ica.mean_
    for i in Nchains_comm:
        idx = np.where(id_communities == i)[0]
        means_chain=[]; posteriors_chain=[]
        if len(idx)>1:
            for j in range(len(idx)):
                factor_id=data_factors_to_keep['factor'][idx[j]] # ICA factor where current HMM state originates from 
                means_chain.append(ica_mixing[factor_id,:]*data_factors_to_keep['mus'][idx[j]]+ica_mean)
                posteriors_chain.append(data_factors_to_keep['posterior'][idx[j]])
            means.append(means_chain)
            posteriors.append(posteriors_chain)
            # print(posteriors_chain)
            # transitions.append(estimate_transition_matrix_vec(posteriors_chain))
            init.append([posteriors_chain[i][0] for i in range(len(posteriors_chain))])
    # Reformat posteriors_rec: list[n_factors][n_states][T] -> list[n_factors] with shape (T, n_states)
    posteriors_rec_reformatted = [np.column_stack(factor) for factor in posteriors]
    transitions=estimate_transition_matrix_vec(posteriors_rec_reformatted)
    # print('Reconstructed parameters')
    # print(means)
    # print(transitions)
    # print(init)
    return means,posteriors_rec_reformatted,transitions,init


def compare_reconstructed_true(means_rec, posteriors_rec, data):
    """
    Compare the reconstructed parameters with the true fHMM parameters.
    
    Args:
        means_rec (list): Reconstructed emission means, grouped by factor.
        posteriors_rec (list): Reconstructed posterior probabilities.
        data (dict): Dictionary containing true fHMM parameters and data.
    
    Returns:
        None (displays visualizations and prints correlations).
    """

    # Extract the true transition matrices and emission means
    transition_matrices = data['params']['transition_matrices']
    true_means = data['params']['means']
    num_factors = len(transition_matrices)
    num_states_per_factor = [tm.shape[0] for tm in transition_matrices]  # Extract different num_states per factor

    # Construct one-hot encoding for true states, now handling variable num_states per factor
    true_states = data['true_states'] if 'true_states' in data else data['fHMM_data']['true_states']
    flat_one_hot_states = []
    state_factors = []  # Track the factor each state belongs to

    for f in range(num_factors):
        states_f = np.eye(num_states_per_factor[f])[true_states[:, f]]  # One-hot encode each factor separately
        flat_one_hot_states.append(states_f)
        state_factors.extend([f] * num_states_per_factor[f])  # Track factor origin for each state
    flat_one_hot_states = np.concatenate(flat_one_hot_states, axis=1)  # Concatenate for full representation
    posteriors_conc = np.hstack(posteriors_rec) 
    # posteriors_conc = np.concatenate(posteriors_rec, axis=0).T  # Convert to correct shape (T x factors)

    num_factors_FA = posteriors_conc.shape[1]  # Number of independent factors inferred from ICA

    # Calculate correlations between all inferred factors and all true states
    correlation_matrix = np.zeros((num_factors_FA, flat_one_hot_states.shape[1]))

    for factor_idx in range(num_factors_FA):
        for flat_idx in range(flat_one_hot_states.shape[1]):
            correlation_matrix[factor_idx, flat_idx] = np.corrcoef(posteriors_conc[:, factor_idx], flat_one_hot_states[:, flat_idx])[0, 1]

    fig = visualize_corr_matrix(correlation_matrix,option=False)
    ax = fig.gca()
    ax.set_title('Correlation True vs Reconstructed posteriors')
    ax.set_ylabel('Reconstructed states')
    ax.set_xlabel('True States')
    # Set x-axis tick labels with corresponding factor indices
    xtick_labels = [f"F{state_factors[i]}-S{i}" for i in range(len(state_factors))]
    ax.set_xticks(range(len(state_factors)))
    ax.set_xticklabels(xtick_labels, rotation=90)
    plt.savefig('fig_modelsel/GroundTruth_compare_posteriors.pdf')

    # Iterative matching procedure with index tracking
    unmatched_factors = list(range(num_factors_FA))
    unmatched_states = list(range(flat_one_hot_states.shape[1]))
    matched_pairs = []

    while unmatched_factors and unmatched_states:
        # Find the highest correlation in the reduced matrix
        highest_corr = -np.inf
        for factor_idx in unmatched_factors:
            for flat_state_idx in unmatched_states:
                if correlation_matrix[factor_idx, flat_state_idx] > highest_corr:
                    highest_corr = correlation_matrix[factor_idx, flat_state_idx]
                    best_factor = factor_idx
                    best_true = flat_state_idx
        if highest_corr == -np.inf:
            break
        true_factor = state_factors[best_true]  # Get the true factor of the matched state
        print(f'Best match: ICA Factor {best_factor} with True state {best_true} (Factor {true_factor}) with correlation {highest_corr}')
        # print(f'Best match: Factor {best_factor} with True state {best_true} with correlation {highest_corr}')
        # Match the factor and state
        # matched_pairs.append((best_factor, best_true, highest_corr))
        matched_pairs.append((best_factor, best_true, highest_corr, true_factor))

        # Remove matched factor and state from lists
        unmatched_factors.remove(best_factor)
        unmatched_states.remove(best_true)
        print(f'unmatched_factors {unmatched_factors}')

    # Calculate correlation between true and reconstructed emission means
    means_conc_true = np.concatenate(true_means, axis=0)  # True means, flattened across factors
    means_conc_rec = np.concatenate(means_rec, axis=0)  # Reconstructed means

    corr_mus = []
    plot_matrix = np.zeros((means_conc_true.shape[1], 4 * means_conc_true.shape[0]))
    cnt = 0
    for i in range(len(matched_pairs)):
        # print(f'ICA Factor {matched_pairs[i][0]} is matched with true state {matched_pairs[i][1]} with correlation {matched_pairs[i][2]}')
        print(f'ICA Factor {matched_pairs[i][0]} is matched with True state {matched_pairs[i][1]} (Factor {matched_pairs[i][3]}) with correlation {matched_pairs[i][2]}')
        corr_mus.append(np.corrcoef(means_conc_true[matched_pairs[i][1]], means_conc_rec[matched_pairs[i][0]])[0, 1])
        plot_matrix[:, cnt] = means_conc_true[matched_pairs[i][1]]
        cnt += 1
        plot_matrix[:, cnt] = means_conc_rec[matched_pairs[i][0]]
        cnt += 3  # Leave space between pairs


    fig = plt.figure(figsize=(10, 10))
    im = plt.imshow(plot_matrix, aspect='auto', cmap='bwr', vmin=-np.max(np.abs(plot_matrix)), vmax=np.max(np.abs(plot_matrix)))
    plt.title('ICA-reconstructed (left) vs True (right) emission means')
    plt.colorbar(im, extend='both')
    plt.savefig('fig_modelsel/ICAvsTrueemissions.pdf')
    plt.close()
    print('corr_mus:', corr_mus)
    print(f'Mean correlation of matched pairs: {np.mean(corr_mus)} ± {np.std(corr_mus)}')


    # sorted posteriors
    n_timesteps = true_states.shape[0]
    posterior_conc_sort = np.zeros((n_timesteps, len(matched_pairs)))
    flat_one_hot_states_sort = np.zeros((n_timesteps, len(matched_pairs)))
    for i in range(len(matched_pairs)):
        posterior_conc_sort[:, i] = posteriors_conc[:, matched_pairs[i][0]]
        flat_one_hot_states_sort[:, i] = flat_one_hot_states[:, matched_pairs[i][1]]
    # Normalize columns to zero mean and unit variance
    X_norm = (posterior_conc_sort - posterior_conc_sort.mean(axis=0)) / posterior_conc_sort.std(axis=0)
    Y_norm = (flat_one_hot_states_sort - flat_one_hot_states_sort.mean(axis=0)) / flat_one_hot_states_sort.std(axis=0)

    # Compute N x N correlation matrix
    corr_matrix = (X_norm.T @ Y_norm) / X_norm.shape[0]


    fig = visualize_corr_matrix(corr_matrix,option=False)
    ax = fig.gca()
    ax.set_title('Correlation True vs Reconstructed posteriors')
    ax.set_ylabel('Reconstructed states')
    ax.set_xlabel('True States')
    # Set x-axis tick labels with corresponding factor indices
    xtick_labels = [f"F{state_factors[i]}-S{i}" for i in range(len(state_factors))]
    ax.set_xticks(range(len(state_factors)))
    ax.set_xticklabels(xtick_labels, rotation=90)
    plt.savefig('fig_modelsel/GroundTruth_compare_posteriors_sorted.pdf')
