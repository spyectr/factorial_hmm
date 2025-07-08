
import numpy as np
# import numpy.random as npr
# import glob
import os
import seaborn as sns
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(linewidth=1000)
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import pickle
# import json
# import h5py
# from rastermap import Rastermap, utils
# from scipy.stats import norm, zscore, gaussian_kde, ks_2samp, chi2_contingency

# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from matplotlib.colors import BoundaryNorm
# from matplotlib.colors import LinearSegmentedColormap
# from ipywidgets import FloatProgress
# from tqdm.auto import trange
# import operator
# from operator import itemgetter
import logging
import time
# from collections import Counter


# import ssm
# from ssm.messages import hmm_sample

# from joblib import Parallel, delayed
import multiprocessing
NumThread=(multiprocessing.cpu_count()-1)*2 # sets number of workers based on cpus on current machine
print('Parallel processing with '+str(NumThread)+' cores')

# import sys

# import scipy
# import scipy.cluster.hierarchy as sch
# from scipy.stats import norm
# from sklearn.cluster import KMeans
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import silhouette_score, silhouette_samples
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler
# from tslearn.clustering import TimeSeriesKMeans
# from model_fitting_utility_functions import gibbs, generate_tpm, trace_mapping, state_mapping, generate_fHMM_data #, Rastermap_old
from new_model_fitting_utility_functions import *

import matplotlib as mpl
LETTER_WIDTH = 8.5  # inches
FIG_WIDTH_1_3 = LETTER_WIDTH / 3
FIG_WIDTH_2_3 = 2 * LETTER_WIDTH / 3
FIG_WIDTH_3_3 = LETTER_WIDTH
FIG_HEIGHT = 6

# Neuron‐style global rcParams
FNTSZ = 12
mpl.rcParams.update({
    'font.size':         FNTSZ,
    'axes.titlesize':    FNTSZ,
    'axes.labelsize':    FNTSZ,
    'xtick.labelsize':   FNTSZ,
    'ytick.labelsize':   FNTSZ,
    'legend.fontsize':   FNTSZ,
    'figure.titlesize':  FNTSZ
})
# if not os.path.exists('data/'):
#     os.makedirs('data/')

CORR_CMAP       = 'viridis'
EMISSION_CMAP   = 'plasma'
MACRO_AREA_CMAP = sns.color_palette('tab20', n_colors=20)


def fit_fHMM(n_factors, n_states, emissions, hypers, options, pre_estimated_params=None):
    n_timesteps, emission_dim = emissions.shape
    n_runs = options['n_runs']

    fit_start = time.time()

    # Function to run a single iteration
    def run_single_iteration(iteration,seed1):
        logging.info(f'Starting iteration {iteration + 1}/{n_runs} for model fitting.')
        samples, params_samples, lls, posteriors_new = gibbs(emissions, hypers, options, seed1, pre_estimated_params=pre_estimated_params)

        logging.info(f'Completed iteration {iteration + 1}/{n_runs}. Log-likelihood: {lls[-1]}')

        return {
            'll': lls[-1] / n_timesteps,
            'll_series': lls,
            # 'samples': samples[-1][0],
            # 'samples_all': samples,
            'params': params_samples[-1],
            'params_all': params_samples,
            'posteriors': posteriors_new[-1],
            'posteriors_all': posteriors_new
        }

    # Run iterations in parallel
    seeds = np.random.randint(0, options['NumThread'], size=n_runs)

    results=[]
    # for iteration in range(n_runs):
        # out=run_single_iteration(iteration,seeds[iteration])
        # results.append(out)
    max_workers = min(options["NumThread"], options['n_runs']+1)
    with Parallel(n_jobs=max_workers, backend="loky") as parallel:
        results = parallel(
            delayed(run_single_iteration)(iteration,seeds[iteration]) for iteration in range(n_runs)
    )
    # results = Parallel(
    #     n_jobs=options['NumThread'],
    #     backend='multiprocessing'    # <— change here
    #     )(delayed(run_single_iteration)(it, seeds[it]) for it in range(n_runs))    

    # Aggregate results
    fit_dict = {
        'll_tot': [res['ll'] for res in results],
        'll_tot_tot': [res['ll_series'] for res in results],
        # 'samples_tot': [res['samples'] for res in results],
        # 'samples_tot_tot': [res['samples_all'] for res in results],
        'params_tot': [res['params'] for res in results],
        'params_tot_tot': [res['params_all'] for res in results],
        'n_factors': n_factors,
        'n_states': n_states,
        'fit_time': time.time() - fit_start,
        'params_samples': results[-1]['params_all'],  # Last run parameters
        'posteriors_new': results[-1]['posteriors'],
        'posteriors_new_tot': [res['posteriors_all'] for res in results],
        'hypers': hypers,
        'options': options
    }

    logging.info(f'Total fitting time: {fit_dict["fit_time"]:.2f} seconds')
    return fit_dict

def fit_fHMM_noparallel(n_factors, n_states, emissions, hypers, options, pre_estimated_params=None):
    n_timesteps, emission_dim = emissions.shape
    n_runs = options['n_runs']

    fit_start = time.time()

    # Function to run a single iteration
    def run_single_iteration(iteration,seed1):
        logging.info(f'Starting iteration {iteration + 1}/{n_runs} for model fitting.')
        samples, params_samples, lls, posteriors_new = gibbs(emissions, hypers, options, seed1, pre_estimated_params=pre_estimated_params)

        logging.info(f'Completed iteration {iteration + 1}/{n_runs}. Log-likelihood: {lls[-1]}')

        return {
            'll': lls[-1] / n_timesteps,
            'll_series': lls,
            # 'samples': samples[-1][0],
            # 'samples_all': samples,
            'params': params_samples[-1],
            'params_all': params_samples,
            'posteriors': posteriors_new[-1],
            'posteriors_all': posteriors_new
        }

    # Run iterations in parallel
    seeds = np.random.randint(0, options['NumThread'], size=n_runs)

    results=[]
    for iteration in range(n_runs):
        out=run_single_iteration(iteration,seeds[iteration])
        results.append(out)
    # with Parallel(n_jobs=n_runs+1, backend="loky") as parallel:
    #     results = parallel(
    #         delayed(run_single_iteration)(iteration,seeds[iteration]) for iteration in range(n_runs)
    # )
    # results = Parallel(
    #     n_jobs=options['NumThread'],
    #     backend='multiprocessing'    # <— change here
    #     )(delayed(run_single_iteration)(it, seeds[it]) for it in range(n_runs))    

    # Aggregate results
    fit_dict = {
        'll_tot': [res['ll'] for res in results],
        'll_tot_tot': [res['ll_series'] for res in results],
        # 'samples_tot': [res['samples'] for res in results],
        # 'samples_tot_tot': [res['samples_all'] for res in results],
        'params_tot': [res['params'] for res in results],
        'params_tot_tot': [res['params_all'] for res in results],
        'n_factors': n_factors,
        'n_states': n_states,
        'fit_time': time.time() - fit_start,
        'params_samples': results[-1]['params_all'],  # Last run parameters
        'posteriors_new': results[-1]['posteriors'],
        'posteriors_new_tot': [res['posteriors_all'] for res in results],
        'hypers': hypers,
        'options': options
    }

    logging.info(f'Total fitting time: {fit_dict["fit_time"]:.2f} seconds')
    return fit_dict


def find_best_run_LL(fit_dict):
    """
    Find the run with the best log-likelihood from the fit dictionary.
    
    Parameters
    ----------
    fit_dict: dict
        The dictionary containing the fitting results.
        
    Returns
    -------
    best_run_index: int
        The index of the run with the best log-likelihood.
    """
    # n_iterations=len(fit_dict['ll_tot_tot'])

    best_index_per_iteration = [np.argmax(ll_tot) for ll_tot in fit_dict['ll_tot_tot']]
    best_LL_per_iteration = [np.max(ll_tot) for ll_tot in fit_dict['ll_tot_tot']]
    best_iteration = np.array(best_LL_per_iteration).argmax()
    best_run = best_index_per_iteration[best_iteration]
    print(best_index_per_iteration)
    print(best_LL_per_iteration)
    print(f'Best iteration:{best_iteration}, best run:{best_run}')

    fit_best_run={'hypers': fit_dict['hypers'],'options': fit_dict['options']}
    fit_best_run['ll_tot'] = fit_dict['ll_tot_tot'][best_iteration][best_run]
    # fit_best_run['samples'] = fit_dict['samples_tot_tot'][best_iteration][best_run]
    fit_best_run['params'] = fit_dict['params_tot_tot'][best_iteration][best_run]
    fit_best_run['posteriors'] = fit_dict['posteriors_new_tot'][best_iteration][best_run]

    return fit_best_run

def find_best_run_VarExp(fit_dict,var_explained_runs):
    """
    Find the run with the best log-likelihood from the fit dictionary.
    
    Parameters
    ----------
    fit_dict: dict
        The dictionary containing the fitting results.
        
    Returns
    -------
    best_run_index: int
        The index of the run with the best log-likelihood.
    """
#     n_iterations=len(fit_dict['ll_tot_tot'])

    best_index_per_iteration = [np.argmax(ll_tot) for ll_tot in var_explained_runs]
    best_LL_per_iteration = [np.max(ll_tot) for ll_tot in var_explained_runs]
    best_iteration = np.array(best_LL_per_iteration).argmax()
    best_run = best_index_per_iteration[best_iteration]
    print(best_index_per_iteration)
    print(best_LL_per_iteration)
    print(f'Best iteration:{best_iteration}, best run:{best_run}')

    fit_best_run={'hypers': fit_dict['hypers'],'options': fit_dict['options']}
    fit_best_run['ll_tot'] = fit_dict['ll_tot_tot'][best_iteration][best_run]
    # fit_best_run['samples'] = fit_dict['samples_tot_tot'][best_iteration][best_run]
    fit_best_run['params'] = fit_dict['params_tot_tot'][best_iteration][best_run]
    fit_best_run['posteriors'] = fit_dict['posteriors_new_tot'][best_iteration][best_run]

    return fit_best_run

def plot_fhmm_PosteriorCorrelationRuns(fit_dict, file_save):
    """
    Plot the convergence of posterior correlations over runs, in a grid layout similar to plot_fhmm_VarExp_convergenceOverRuns.
    Each subplot shows one iteration's correlation curve, with mean and std over states.
    """
    Nplot = len(fit_dict['posteriors_new_tot'])
    grid_dims = optimal_subplot_grid(Nplot)
    fig, axs = plt.subplots(
        grid_dims[0], grid_dims[1],
        figsize=(FIG_WIDTH_1_3 * grid_dims[1], FIG_HEIGHT),
        constrained_layout=True
    )
    axs = np.array(axs).reshape(-1)

    # Compute all correlations and store for y-limits
    all_corrs = []
    for i in range(Nplot):
        correlations = []
        correlations_std = []
        posteriors_over_runs = []
        for j in range(len(fit_dict['posteriors_new_tot'][i])):
            temp_post = fit_dict['posteriors_new_tot'][i][j]
            temp_post = posteriors_list2array(temp_post)  # time_steps x (factors * states)
            _, factors_states = temp_post.shape
            posteriors_over_runs.append(np.transpose(temp_post, (1, 0)))
        steps = len(posteriors_over_runs)
        for t in range(steps - 1):
            correlations_states = []
            for k in range(factors_states):
                final_posterior_vector = posteriors_over_runs[-1][k]
                correlation = np.corrcoef(posteriors_over_runs[t][k], final_posterior_vector)[0, 1]
                correlations_states.append(correlation)
            correlations.append(np.mean(correlations_states))
            correlations_std.append(np.std(correlations_states))
        all_corrs.append(correlations)

    # Determine common y-limits
    y_min = min(np.min(c) for c in all_corrs)
    y_min = y_min if (y_min>=-1) else -1
    y_max = max(np.max(c) for c in all_corrs)
    y_max = y_max if (y_max<=1) else 1

    # Plot each subplot
    for i in range(Nplot):
        correlations = all_corrs[i]
        # Recompute std for this subplot
        correlations_std = []
        posteriors_over_runs = []
        for j in range(len(fit_dict['posteriors_new_tot'][i])):
            temp_post = fit_dict['posteriors_new_tot'][i][j]
            temp_post = posteriors_list2array(temp_post)
            _, factors_states = temp_post.shape
            posteriors_over_runs.append(np.transpose(temp_post, (1, 0)))
        steps = len(posteriors_over_runs)
        for t in range(steps - 1):
            correlations_states = []
            for k in range(factors_states):
                final_posterior_vector = posteriors_over_runs[-1][k]
                correlation = np.corrcoef(posteriors_over_runs[t][k], final_posterior_vector)[0, 1]
                correlations_states.append(correlation)
            correlations_std.append(np.std(correlations_states))

        ax = axs[i]
        ax.plot(range(1, steps), correlations, color='C0', linewidth=2)
        ax.fill_between(
            range(1, steps),
            np.array(correlations) - np.array(correlations_std),
            np.array(correlations) + np.array(correlations_std),
            alpha=0.2
        )
        ax.hlines(y=1, xmin=0, xmax=steps, colors='red', linestyles='dashed')
        ax.set_title(f'Iter {i}', fontsize=FNTSZ)
        # Only label left column and bottom row
        if i % grid_dims[1] == 0:
            ax.set_ylabel('Correlation', fontsize=FNTSZ)
        if i >= (grid_dims[0]-1)*grid_dims[1]:
            ax.set_xlabel('Step', fontsize=FNTSZ)
        ax.tick_params(labelsize=FNTSZ)
        ax.set_xticks([0, steps-1])
        ax.set_xticklabels([str(0), str(steps-1)], fontsize=FNTSZ)
        ax.set_ylim(y_min, y_max)

    # Hide extra axes
    for ax in axs[Nplot:]:
        ax.axis('off')
    fig.savefig(file_save, bbox_inches='tight')
    plt.close(fig)

def plot_fhmm_LL_convergenceOverRuns(fit_plot, file_save):
    # Number of plots and grid dimensions
    Nplot = len(fit_plot['ll_tot_tot'])
    grid_dims = optimal_subplot_grid(Nplot)
    # Use 2/3 letter width and global FIG_HEIGHT, enable constrained_layout
    fig_width = FIG_WIDTH_2_3
    fig_height = FIG_HEIGHT
    fig, axs = plt.subplots(
        grid_dims[0], grid_dims[1],
        figsize=(fig_width, fig_height),
        constrained_layout=True
    )
    # Flatten axs to handle cases where axs is a 2D array
    axs = np.array(axs).reshape(-1)
    # Determine the common y-axis range
    y_min = min([np.min(np.array(ll)) for ll in fit_plot['ll_tot_tot']])
    y_max = max([np.max(np.array(ll)) for ll in fit_plot['ll_tot_tot']])
    for iteration, ax in enumerate(axs[:Nplot]):  # Iterate only over the necessary subplots
        ll = np.array(fit_plot['ll_tot_tot'][iteration])
        ll_argmax = np.argmax(ll)
        ll_max = np.max(ll)
        ax.plot(ll)
        ax.set_title(f'Iteration {iteration}\nMax(LL)={ll_max:.1f}, Run={str(ll_argmax)}', fontsize=FNTSZ)
        ax.set_xlabel('Run', fontsize=FNTSZ)
        ax.set_ylabel('Log Likelihood', fontsize=FNTSZ if iteration % grid_dims[1] == 0 else FNTSZ)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='both', which='major', labelsize=FNTSZ)
        # Set x-ticks to the start and end, with correct label size
        ax.set_xticks([0, len(ll)-1])
        ax.set_xticklabels([str(0), str(len(ll)-1)], fontsize=FNTSZ)
    # Hide unused subplots (in case grid is larger than the number of iterations)
    for ax in axs[Nplot:]:
        ax.axis("off")
    # Save with tight bounding box and close
    fig.savefig(file_save, bbox_inches='tight')
    plt.close(fig)


def Var_Explained(fit_plot, emissions_fa):
    """
    Find the run with the best log-likelihood from the fit dictionary.
    
    Parameters
    ----------
    fit_dict: dict
        The dictionary containing the fitting results.
        
    Returns
    -------
    best_run_index: int
        The index of the run with the best log-likelihood.
    """
    # fit_plot = fit_dict
    var_explained_runs = []
    tot_var = np.var(emissions_fa, axis=0)
    # Loop over each iteration from the fit (each element of 'samples_tot_tot')
    for iteration in range(len(fit_plot['posteriors_new_tot'])):
        var_explained_by_run = []
        # Loop over each independent run within the iteration
        for run in range(len(fit_plot['posteriors_new_tot'][iteration])):
            # Retrieve the posterior probabilities (mean_weights) for this run.
            # In the new code, these are returned as a combined 2D array (T x total_states)
            posteriors_run = fit_plot['posteriors_new_tot'][iteration][run]
            # Retrieve the reconstructed means from the parameters.
            # In the new format, these are stored as a list of arrays (one per factor)
            reconstructed_means = fit_plot['params_tot_tot'][iteration][run]['means']
            # IMPORTANT CHANGE: Pass the list n_states so the function can split the combined weights appropriately.
            # reconstructed_emissions = reconstruct_emissions_with_weights(reconstructed_means, posteriors_run, n_states)
            reconstructed_emissions = expected_emissions(posteriors_run, reconstructed_means)
            # print(f'reconstructed_emissions {reconstructed_emissions.shape}')
            # print(f' emissions_fa {len(emissions_fa)} {len(emissions_fa[0])}')
            # Compute the reconstruction error and its variance.
            reconstruction_error = emissions_fa - reconstructed_emissions
            # error_var = np.var(reconstruction_error, axis=0)
            # explained_var = tot_var - error_var
            # var_explained_ratios = explained_var / tot_var
            # overall_var_explained_ratio = np.mean(var_explained_ratios)
            overall_var_explained_ratio = 1 - np.sum(np.var(reconstruction_error, axis=0)) / np.sum(np.var(emissions_fa, axis=0))
            var_explained_by_run.append(overall_var_explained_ratio)
        var_explained_runs.append(var_explained_by_run)
    return var_explained_runs


def plot_fhmm_VarExp_convergenceOverRuns(var_explained_runs, file_name):
    """
    Plot convergence of variance-explained over runs for each iteration.
    Each subplot shows one iteration's variance-explained curve,
    formatted for Neuron: 1/3 page width, uniform FNTSZ fonts,
    minimal xticks, shared y-range, and constrained layout.
    """
    Nplot = len(var_explained_runs)
    grid_dims = optimal_subplot_grid(Nplot)
    # Figure width = 1/3 letter per column, height = FIG_HEIGHT
    fig, axs = plt.subplots(
        grid_dims[0], grid_dims[1],
        figsize=(FIG_WIDTH_1_3 * grid_dims[1], FIG_HEIGHT),
        constrained_layout=True
    )
    # Flatten axes array
    axs = np.array(axs).reshape(-1)

    # Determine common y-axis limits
    y_min = min(min(r) for r in var_explained_runs)
    y_max = max(max(r) for r in var_explained_runs)

    # Plot each iteration
    for i in range(Nplot):
        ax = axs[i]
        curve = var_explained_runs[i]
        ax.plot(curve, color='C0', linewidth=2)
        # Annotate title
        m = np.max(curve)
        r = np.argmax(curve)
        ax.set_title(f"Iter {i}  Max={m:.3f}, run={r}", fontsize=FNTSZ)
        # Label only left column and bottom row
        if i % grid_dims[1] == 0:
            ax.set_ylabel("VarExpl", fontsize=FNTSZ)
        if i >= (grid_dims[0]-1)*grid_dims[1]:
            ax.set_xlabel("Run", fontsize=FNTSZ)
        # Shared y-limits and minimal xticks
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([0, len(curve)-1])
        ax.set_xticklabels([0, len(curve)-1], fontsize=FNTSZ)
        ax.tick_params(labelsize=FNTSZ)

    # Hide any extra subplots
    for ax in axs[Nplot:]:
        ax.axis('off')

    # Supertitle and save
    fig.suptitle("Variance Explained Over Runs", fontsize=FNTSZ)
    fig.savefig(file_name, format='pdf', bbox_inches='tight')
    plt.close(fig)


# Overlay LL convergence curves across iterations on a single axis,
# mark the global maximum with an 'x', and title with its iteration, run, and LL.
def plot_fhmm_LL_overlay_convergence(fit_plot, file_save):
    """
    Overlay LL convergence curves across iterations on a single axis,
    mark the global maximum with an 'x', and title with its iteration, run, and LL.
    """
    # Extract all LL series
    ll_runs = fit_plot['ll_tot_tot']
    # Find global maximum
    best_ll = -np.inf
    best_iter = None
    best_run = None
    for i, ll in enumerate(ll_runs):
        ll_arr = np.array(ll)
        idx = np.argmax(ll_arr)
        val = ll_arr[idx]
        if val > best_ll:
            best_ll = val
            best_iter = i
            best_run = idx

    # Create figure
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_1_3, FIG_WIDTH_1_3),
        constrained_layout=True
    )

    # Plot each iteration's curve
    for i, ll in enumerate(ll_runs):
        ax.plot(
            np.arange(len(ll))+1, ll,
            label=f'Iter {i}', linewidth=2
        )

    # Mark global maximum
    ax.scatter(
        best_run+1, best_ll,
        s=100, marker='x', color='black',
        label=f'Max: Iter {best_iter+1}, Run {best_run+1}'
    )

    # Labels, ticks, title
    ax.set_xlabel('Run', fontsize=FNTSZ)
    ax.set_ylabel('LL', fontsize=FNTSZ)
    ax.set_title(
        f'Max: {best_ll:.1f} '
        f'at Iter {best_iter+1}, Run {best_run+1}',
        fontsize=FNTSZ
    )
    ax.tick_params(labelsize=FNTSZ)
    # Minimal xticks
    n_runs = max([len(ll_runs[i]) for i in range(len(ll_runs))])
    ax.set_xticks([1, n_runs])
    ax.set_xticklabels([1, n_runs ], fontsize=FNTSZ)

    # # Legend outside
    # ax.legend(
    #     fontsize=FNTSZ,
    #     loc='upper left',
    #     # bbox_to_anchor=(1.02, 1)
    # )

    # Save and close
    fig.savefig(file_save, format='pdf', bbox_inches='tight')
    plt.close(fig)


# Overlay variance-explained convergence curves across iterations on a single axis,
# mark the global maximum with a red 'X', and label the run and iteration in the legend.
def plot_fhmm_VarExp_overlay_convergence(varexp_runs, file_save):
    """
    Overlay variance-explained convergence curves across iterations on a single axis,
    mark the global maximum with a red 'X', and label the run and iteration in the legend.
    """
    # Find global maximum
    best_val = -np.inf
    best_iter = None
    best_run = None
    for i, curve in enumerate(varexp_runs):
        arr = np.array(curve)
        idx = np.argmax(arr)
        val = arr[idx]
        if val > best_val:
            best_val = val
            best_iter = i
            best_run = idx

    # Create figure
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_1_3, FIG_WIDTH_1_3),
        constrained_layout=True
    )

    # Plot each iteration's curve
    for i, curve in enumerate(varexp_runs):
        ax.plot(
            np.arange(len(curve))+1, curve,
            label=f'Iter {i}', linewidth=2
        )

    # Mark global maximum with red 'X'
    ax.scatter(
        best_run+1, best_val,
        s=100, marker='x', color='red',
        label=f'Max: Iter {best_iter+1}, Run {best_run+1}'
    )

    # Labels, ticks, title
    ax.set_xlabel('Run', fontsize=FNTSZ)
    ax.set_ylabel('VarExp', fontsize=FNTSZ)
    ax.set_title(
        f'Max={best_val:.3f} at Iter {best_iter+1}, Run {best_run+1}',
        fontsize=FNTSZ
    )
    ax.tick_params(labelsize=FNTSZ)
    # Minimal xticks
    n_runs = max([len(varexp_runs[i]) for i in range(len(varexp_runs))])
    ax.set_xticks([1, n_runs])
    ax.set_xticklabels([1, n_runs], fontsize=FNTSZ)

    # # Legend outside
    # ax.legend(
    #     fontsize=FNTSZ,
    #     loc='upper left',
    #     # bbox_to_anchor=(1.02, 1)
    # )

    # Save and close
    fig.savefig(file_save, format='pdf', bbox_inches='tight')
    plt.close(fig)