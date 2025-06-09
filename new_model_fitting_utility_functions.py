import numpy as np
import numpy.random as npr
import pandas as pd

import matplotlib.pyplot as plt
from tqdm.auto import trange 

import numba

# import ssm
# from ssm.messages import hmm_sample
LOG_EPS = 1e-16

import random

from joblib import Parallel, delayed
import multiprocessing
NumThread=(multiprocessing.cpu_count()-1)*2 # sets number of workers based on cpus on current machine


from scipy.stats import norm

import sys
import logging
import atexit
import datetime


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def setup_logging(base_filename="my_script_log", log_level=logging.INFO):
    """
    Sets up logging such that messages printed via print() or logging are written
    exactly as they appear on the console into a log file. The log file's name 
    is constructed from the base_filename plus the current date and time.
    
    This function configures logging with a single StreamHandler pointing to sys.stdout.
    Because sys.stdout is replaced with a Tee instance, the output goes both to the console
    and to the log file.
    
    Parameters
    ----------
    base_filename : str
        The base name for the log file (default: "my_script_log").
    log_level : int
        Logging level (default: logging.INFO).
    """
    # Get current date and time string in format YYYY-MM-DD_HH-MM-SS.
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{base_filename}_{timestamp}.txt"
    
    # Open the log file.
    log_file = open(log_filename, "w")
    atexit.register(log_file.close)
    
    # Save the original stdout and stderr.
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    
    # Redirect stdout and stderr so output goes to both console and the log file.
    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)
    
    # Configure logging to use only a StreamHandler directed to sys.stdout.
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename

def distribute_elements(N, M):
    # Base size of each group
    base_size = N // M
    # Number of groups that will have one more element
    extra_elements = N % M
    # Initialize the list to store the size of each group
    group_sizes = [base_size + (1 if i < extra_elements else 0) for i in range(M)]
    return group_sizes

def generate_means(n_factors, n_states, emission_dim, mean_scalar = 1):
    group_sizes = distribute_elements(N = emission_dim, M = n_factors)
    # Create a list of arrays, one for each factor, with its specific number of states
    means = [np.zeros((n_states[i], emission_dim)) for i in range(n_factors)]
    
    # Compute start indices for each factor's block in the emission dimension
    start_indices = np.cumsum([0] + group_sizes[:-1])
    
    # For each factor, fill in a block of the mean matrix with random values
    for i, start_idx in enumerate(start_indices):
        end_idx = start_idx + group_sizes[i]
        means[i][:, start_idx:end_idx] = mean_scalar * np.random.standard_normal((n_states[i], group_sizes[i]))
    
    return means

def generate_tpm(K, diagonal = 0.8):
    # K is the number of states
    # diagonal is the probobility of remaining in a given state
    matrix = np.zeros((K, K)) # initiate the blank transition probability matrix
    noise = 1 - diagonal
    np.fill_diagonal(matrix, diagonal + np.random.uniform(-1*noise, noise, K)) # insert the diagonal elements with uniform random noise
    off_diagonals = (1 - np.diag(matrix)) / (K - 1) # evenly distributed the rest of the transition probability among the off-diagonal elements of each row
    matrix += np.outer(off_diagonals, np.ones(K)) - np.diag(off_diagonals) # place all of these probabilities in the tpm
    return matrix

def generate_fHMM_data(n_factors, n_states, emission_dim, n_timesteps, trans_diag, rand_seed = None, selectivity = None, mean_scalar = 1):
    if rand_seed is not None:
        npr.seed(rand_seed)
    
    hypers = dict(n_factors = n_factors,
                  n_states = n_states,
                  emission_dim = emission_dim,
                  n_timesteps = n_timesteps,
                  tpm_diagonal = trans_diag,
                  selectivity = selectivity,
                  mean_scalar = mean_scalar)
    
    # initial distribution: each factor's states are equally likely
    initial_dist = [np.ones(n_states[h]) / n_states[h] for h in range(n_factors)]
    
    # Ensure trans_diag is properly defined
    if trans_diag is None:
        trans_diag = [0 for i in range(n_factors)]
        for i in range(n_factors):
            if i < min(4, n_factors):
                epsi = 10 ** (-i - 1)
            trans_diag[i] = (1 - epsi)
    else:
        if len(trans_diag) != n_factors:
            raise ValueError("trans_diag must have the same length as n_states")
    
    # Populate transition matrices for each factor
    transition_matrices = [np.zeros((s, s)) for s in n_states]
    for i in range(n_factors):
        transition_matrix = np.eye(n_states[i]) * trans_diag[i]
        mask = np.ones_like(transition_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        transition_matrix[mask] = (1 - trans_diag[i]) / (n_states[i] - 1)
        transition_matrices[i] = transition_matrix
            
    # Generate emission means
    if selectivity == 'mixedsel':
        means = [npr.randn(n_states[i], emission_dim) for i in range(n_factors)]
    elif selectivity == 'puresel':
        means = [np.zeros((n_states[i], emission_dim)) for i in range(n_factors)]
        for i in range(n_factors):
            indperm = np.random.permutation(np.arange(emission_dim))
            neuron_per_state = np.floor(emission_dim / n_states[i]).astype(int)
            for i_st in range(n_states[i]):
                start_idx = i_st * neuron_per_state
                end_idx = (i_st + 1) * neuron_per_state if i_st < n_states[i] - 1 else len(indperm)
                means[i][i_st, indperm[start_idx:end_idx]] = npr.randn(len(indperm[start_idx:end_idx]))
    else:
        raise ValueError("Invalid emission_option. Choose 'mixedsel' or 'puresel'.")
    
    variances = 0.5*np.ones(emission_dim) + np.random.uniform(-0.1, 0.1, emission_dim)
    
    params = dict(initial_dist = initial_dist,
                  transition_matrices = transition_matrices,
                  means = means,
                  variances = variances)
    
    true_states = np.zeros((n_timesteps, n_factors), dtype=int)
    emissions = np.zeros((n_timesteps, emission_dim))
    expec_emissions = np.zeros((n_timesteps, emission_dim))
    
    for t in range(n_timesteps):
        for h in range(n_factors):
            if t > 0:
                # Use the h-th factor's state space range
                true_states[t, h] = npr.choice(n_states[h], p=transition_matrices[h][true_states[t-1, h]])
        # Compute expected emissions as the sum over factors (list comprehension)
        expec_emissions[t] = np.sum([means[h][true_states[t, h], :] for h in range(n_factors)], axis=0)
        emissions[t] = expec_emissions[t] + np.sqrt(variances) * npr.randn(emission_dim)
    
    fHMM_data = dict(true_states = true_states,
                     emissions = emissions,
                     expec_emissions = expec_emissions)
    
    fHMM_ground_truth = dict(hypers = hypers,
                             params = params,
                             fHMM_data = fHMM_data)
    
    return fHMM_ground_truth

def expected_emissions(posteriors, means):
    # output expec_emissions is a 2d array of size (num_timesteps, emission_dim)
    gammat = posteriors_list2array(posteriors)
    means_2d = np.vstack(means)
    expec_emissions = gammat @ means_2d
    return expec_emissions

def posteriors_list2array(posteriors):
    # reshape posteriors into an array of shape (num_timesteps, sum(n_states))
    gammat_gibbs_new = np.hstack(posteriors) 
    return gammat_gibbs_new

def posteriors_array2list(gammat_gibbs,n_states):
    # reshape posteriors into an array of shape (num_timesteps, sum(n_states))
    split_indices = np.cumsum(n_states)[:-1]  # Get the split indices
    posteriors_new = np.split(gammat_gibbs, split_indices, axis=1) 
    return posteriors_new

def getList(dictionary):
    """
    Iterates through a dictionary and places each key into a list
    
    Parameters
    ----------
    dict: python dictionary
    
    Returns
    -------
    list
        a list object containing the keys of the dictionary
    """
    ls = list(dictionary.keys())
    return ls

def npy_to_dict(file_name):
    """When you save a dictionary to a .npy file, it's stored as an array.
    This array can be converted back to a dictionary by flattening the array and taking the 0th element.
    This function does just that.
    
    Parameters
    ----------
    file_name: string
        The name of the .npy file to be loaded as a dictionary (and full path if not in working directory).
        
    Returns
    -------
    dictionary: dict
        The .npy file loaded back as a dictionary    
    """
    
    dictionary_arrays = np.load(file_name, allow_pickle = True)
    dictionary = dictionary_arrays.flat[0]
    
    return dictionary

def trace_mapping(true_states, samples_fit):
    n_traces = true_states.shape[1]
    corr_dict = {}
    for i in range(n_traces):
        corr_dict["corr_{0}".format(i)] = np.abs(np.corrcoef(np.hstack((true_states[:,i][:,np.newaxis], samples_fit)), rowvar = False))
    corr_keys = list(corr_dict.keys())
    trace_map = []
    for i in range(n_traces):
        mapped = np.argmax(corr_dict[corr_keys[i]][0][1:n_traces+1])
        trace_map.append((i, mapped))
    for i in range(n_traces):
        print('True factor ' + str(trace_map[i][0]) + ' is most correlated with predicted factor ' + str(trace_map[i][1]))
    return trace_map

def state_mapping(trace_map, params, params_fit):
    corr_dict_states = {}
    map_dict = {}
    # trace map state will be a list of tuples (true_factor, inferred_factor, true_state, inferred_state)
    trace_map_state = []
    for i in range(len(trace_map)):
        param_ind = trace_map[i][0]
        fit_ind = trace_map[i][1]
        for x in range(params['means'][param_ind].shape[0]):
            corr_dict_states['true_factor_{0}_inferred_factor_{1}_true_state_{2}'.format(param_ind, fit_ind, x)] = np.abs(np.corrcoef(params['means'][param_ind][x], params_fit['means'][fit_ind]))
            map_dict['true_factor_{0}_inferred_factor_{1}_true_state_{2}'.format(param_ind, fit_ind, x)] = (param_ind, fit_ind, x)
    corr_state_keys = list(corr_dict_states.keys())
    for j in range(len(corr_dict_states)):
        max_corr = np.argmax(corr_dict_states[corr_state_keys[j]][0][1:])
        map_dict[corr_state_keys[j]] = map_dict[corr_state_keys[j]] + (max_corr,)
        print("true factor " + str(map_dict[corr_state_keys[j]][0]) + " paired with inferred factor " + str(map_dict[corr_state_keys[j]][1]) + "\n" + "true state " + str(map_dict[corr_state_keys[j]][2]) + " paired with inferred state " + str(map_dict[corr_state_keys[j]][3]) + "\n")
    return map_dict

def optimal_subplot_grid(N):
    """
    Compute the optimal (nx, ny) grid dimensions for N subplots, keeping it as square as possible.

    Parameters
    ----------
    N : int
        Number of subplots.

    Returns
    -------
    list
        [nx, ny] where nx is the number of rows and ny is the number of columns.
    """
    ny = int(np.ceil(np.sqrt(N)))  # Start with a square-like layout
    nx = int(np.ceil(N / ny))      # Compute corresponding rows to fit N

    return [nx, ny]

def _compute_changes(params_change, lls_change):
    # Compute changes for transition matrices (list of arrays)
    changes_tpm = []
    for tm1, tm2 in zip(params_change[-1]['transition_matrices'], params_change[-2]['transition_matrices']):
        diff = np.abs(tm1 - tm2).flatten()
        changes_tpm.append(np.max(diff) / len(diff))
    change_tpm = np.max(changes_tpm)
    
    # Compute changes for means (list of arrays)
    changes_means = []
    for m1, m2 in zip(params_change[-1]['means'], params_change[-2]['means']):
        diff = np.abs(m1 - m2).flatten()
        changes_means.append(np.max(diff) / len(diff))
    change_means = np.max(changes_means)
    
    # Compute changes for variances (assumed to be 1D arrays)
    change_variances = np.max(np.abs(params_change[-1]['variances'] - params_change[-2]['variances']).flatten()) \
                       / len(np.array(params_change[-1]['variances']).flatten())
    
    change_lls = np.abs(lls_change[-1] - lls_change[-2]) / lls_change[-1]
    
    return [change_tpm, change_means, change_variances, change_lls]

def _m_step(gammat_gibbs_m, states_outer_gibbs_m, trans_gibbs_m, emissions, hypers):
    """
    Updated M-step using state probabilities estimated from Gibbs sampling,
    supporting different numbers of states per factor.
    
    Assumptions:
      - gammat_gibbs_m: either a 3D array of shape (T, max(n_states), n_factors) (if not yet combined)
                     or a combined 2D array of shape (T, total_states),
                     where total_states = sum(n_states_list) with n_states_list = hypers["n_states"].
      - states_outer_gibbs_m: if 5D, shape (T, max(n_states), max(n_states), n_factors, n_factors);
                            if 3D, already combined: shape (T, total_states, total_states).
      - trans_gibbs_m: should be (T-1, total_states, total_states) if combined.
      - emissions: array of shape (T, emission_dim)
      - hypers: dictionary containing:
           "n_factors": number of factors,
           "n_states": list of number of states per factor,
           "emission_dim": dimension of the emissions.
    """
    # Retrieve hyperparameters.
    n_factors = hypers["n_factors"]
    n_states_list = hypers["n_states"]  # e.g. [n1, n2, ..., n_M]
    emission_dim = emissions.shape[1]
    T = emissions.shape[0]
    total_states = sum(n_states_list)
    # params_m = params.copy()
    params_m = {}
    
    # --- Ensure gammat_gibbs is combined into a (T, total_states) array ---
    if gammat_gibbs_m.ndim == 3:
        # gammat_gibbs_m is of shape (T, max(n_states), n_factors); take valid columns for each factor
        gammat_list = [gammat_gibbs_m[:, :n_states_list[h], h] for h in range(n_factors)]
        gammat_combined_m = np.hstack(gammat_list)  # shape: (T, total_states)
    elif gammat_gibbs_m.ndim == 2:
        gammat_combined_m = gammat_gibbs_m  # already combined
    else:
        raise ValueError("gammat_gibbs_m has unexpected number of dimensions.")
    
    # === Initial state distribution: split first row of combined gamma into blocks ===
    split_indices = np.cumsum(n_states_list)[:-1]
    initial_dist_list = np.split(gammat_combined_m[0, :], split_indices)
    params_m["initial_dist"] = initial_dist_list  # list of arrays per factor
    
    # === Transition matrices ===
    # Here, rather than using trans_gibbs_m directly, we recompute a combined transition matrix
    # using the combined gamma. (This avoids issues with reshaping trans_gibbs_m.)
    combined_trans = np.einsum('ti,tj->tij', gammat_combined_m[1:], gammat_combined_m[:-1])
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
    params_m["transition_matrices"] = transition_matrices_list
    
    # === Emission means update via weighted linear regression ===
    # Compute weighted sum: means_first = emissions.T @ gammat_combined_m --> (emission_dim, total_states)
    means_first = np.matmul(emissions.T, gammat_combined_m)
    # Process states_outer_gibbs_m:
    if states_outer_gibbs_m.ndim == 5:
        outer_list = []
        for t in range(T):
            blocks = []
            for h in range(n_factors):
                row_blocks = []
                for k in range(n_factors):
                    block = states_outer_gibbs_m[t, :n_states_list[h], :n_states_list[k], h, k]
                    row_blocks.append(block)
                blocks.append(np.hstack(row_blocks))
            outer_list.append(np.vstack(blocks))
        states_outer_combined = np.stack(outer_list, axis=0)  # shape: (T, total_states, total_states)
    elif states_outer_gibbs_m.ndim == 3:
        states_outer_combined = states_outer_gibbs_m
    else:
        raise ValueError("states_outer_gibbs_m has unexpected number of dimensions.")
    
    means_second = np.sum(states_outer_combined, axis=0)  # shape: (total_states, total_states)
    means_second_inv = np.linalg.pinv(means_second)
    means_2d = np.matmul(means_first, means_second_inv)  # shape: (emission_dim, total_states)
    
    # === Variance update ===
    Cnew_first = (1/T) * np.matmul(emissions.T, emissions)  # (emission_dim, emission_dim)
    Cnew_sec1 = (1/T) * np.matmul(emissions.T, gammat_combined_m)  # (emission_dim, total_states)
    Cnew_sec = np.matmul(Cnew_sec1, means_2d.T)  # (emission_dim, emission_dim)
    Cnew = Cnew_first - Cnew_sec
    params_m["variances"] = np.diag(Cnew)
    
    # === Partition the combined means back into per–factor means ===
    split_indices = np.cumsum(n_states_list)[:-1]
    means_list = np.split(means_2d, split_indices, axis=1)  # Each: (emission_dim, n_states_i)
    means_final = [m.T for m in means_list]  # Each: (n_states_i, emission_dim)
    params_m["means"] = means_final
    
    # === Log-Likelihood computation ===
    Cinv = np.linalg.inv(np.diag(params_m['variances']))
    Q1 = -(1/2) * np.trace(np.matmul(np.matmul(emissions, Cinv), emissions.T))
    Q21 = np.matmul(Cinv, np.matmul(means_2d, gammat_combined_m.T))
    Q2 = np.trace(np.matmul(emissions, Q21))
    Q31 = np.matmul(states_outer_combined, means_2d.T).transpose(1, 0, 2)
    Q31 = Q31.reshape(total_states, T * emission_dim)
    Q31 = np.matmul(Cinv, np.matmul(means_2d, Q31)).reshape(emission_dim, T, emission_dim)
    Q3 = -(1/2) * np.sum(np.trace(Q31, axis1=0, axis2=2))
    pi1 = np.concatenate(initial_dist_list)  # shape: (total_states,)
    Q4 = np.matmul(gammat_combined_m[0, :], pi1)
    Q51 = np.einsum('ti,tj->tij', gammat_combined_m[1:], gammat_combined_m[:-1])
    Q52 = combined_trans_mean  # shape: (total_states, total_states)
    Q5 = np.sum(np.trace(np.matmul(Q51, Q52), axis1=1, axis2=2))
    lls = -(Q1 + Q2 + Q3 + Q4 + Q5)
    
    return params_m, lls


def _gibbs_sample_states(h, states, emissions, params, hypers):
    """
    Sample the sequence of states for factor h using the current parameter estimates.
    """
    n_factors = hypers["n_factors"]
    n_states_list = hypers["n_states"]
    n_timesteps = states.shape[0]
    means = params["means"]
    variances = params["variances"]

    lls = np.zeros((n_timesteps, n_states_list[h]))
    tmp_states = states.copy()
    for k in range(n_states_list[h]):
        tmp_states[:, h] = k
        expec_emissions = np.zeros_like(emissions)
        for j in range(n_factors):
            expec_emissions += means[j][tmp_states[:, j], :]
        var_emissions = variances
        lls[:, k] = norm(expec_emissions, np.sqrt(var_emissions)).logpdf(emissions).sum(axis=1)
    
    return _hmm_sample(params["initial_dist"][h],
                      params["transition_matrices"][h][None, :, :],
                      lls)


def _gibbs_all_factors(emissions, params, hypers, options, seed1):
    """
    Run Gibbs sampling for all factors.
    Modified to support n_states as a list.
    """
    verbose = options['verbose']
    num_factors = hypers["n_factors"]
    n_states_list = hypers["n_states"]
    num_timesteps = emissions.shape[0]
    random.seed(seed1)
    # Initialize states for each factor using its own range
    states1 = np.column_stack([np.random.randint(0, n_states_list[h], size=num_timesteps)
                              for h in range(num_factors)])
    states_old = states1.copy()
    for itr in range(options["n_gibbssampler"]):
        for h in range(num_factors):
            states1[:, h] = _gibbs_sample_states(h, states1, emissions, params, hypers)
        if np.all(states1 == states_old):
            if verbose:
                print('Convergence at iteration number ' + str(itr))
            break
        states_old = states1.copy()
    
    # Construct gammat as a list for each factor
    # The variable gammat is a Python list with length equal to num_factors. Each element in the list is a NumPy array
    # with shape (\text{num_timesteps},\ n\_states\_list[h]), where h is the factor index.
    gammat1 = [np.zeros((num_timesteps, n_states_list[h])) for h in range(num_factors)]
    for h in range(num_factors):
        tmp_states = states1[:, h]
        gammat1[h][np.arange(num_timesteps), tmp_states] = 1

    # To compute state_outer and trans, combine the list into a block-diagonal-like array.
    # Here we choose to horizontally concatenate gammat for all factors:
    gammat_combined = np.hstack(gammat1)  # shape: (num_timesteps, sum(n_states_list))
    # Compute state_outer using the corrected einsum with distinct factor subscripts:
    # For the combined representation, we lose the factor structure.
    # (An alternative is to compute outer products for each factor pair separately.)
    # Compute combined state_outer and transition samples.
    state_outer = np.einsum('ti,tj->tij', gammat_combined, gammat_combined)
    trans = np.einsum('ti,tj->tij', gammat_combined[1:], gammat_combined[:-1])
    
    out = dict(gammat = gammat_combined,
               state_outer = state_outer,
               trans = trans,
               states = states1)
    return out




def _gibbs_post_prob(emissions, params, hypers, options):
    """This function runs the gibbs sampling for all factors n_gibbs times 
        and returns the posterior probability as the average over the n_gibbs runs
        It can be parallelized
    """
    n_gibbs=options["n_gibbs"]
    params=params.copy()   
    gammat_runs=[]
    states_outer_runs=[]
    trans_runs=[]
    states_run=[]
    # Generate unique seeds for each parallel process
    seeds = np.random.randint(0, options['NumThread'], size=n_gibbs)
    # Run multiple Gibbs chains in parallel
    # if options["parallel"]:
    #     out = Parallel(n_jobs=n_gibbs)(
    #         delayed(my_gibbs_all_factors)(emissions, params, hypers,options,seeds[igibbs])
    #         for igibbs in range(n_gibbs))
    #     # unpack outputs
    #     for i in range(n_gibbs):
    #         gammat_runs.append(out[i]['gammat'])
    #         states_outer_runs.append(out[i]['state_outer'])
    #         trans_runs.append(out[i]['trans'])
    #         states_run.append(out[i]['states'])

    # else:
    for igibbs in range(n_gibbs):
        out=_gibbs_all_factors(emissions, params, hypers,options,seeds[igibbs])
        gammat_runs.append(out['gammat'])
        states_outer_runs.append(out['state_outer'])
        trans_runs.append(out['trans'])
        states_run.append(out['states'])

    # average over independent runs of gibbs to obtain posterior prob for exact m-step below 
    gammat_gibbs1=np.mean(gammat_runs,axis=0) # gammat_gibbs is an array of shape (num_timesteps, sum(n_states_list))
    states_outer_gibbs1=np.mean(states_outer_runs,axis=0)
    trans_gibbs1=np.mean(trans_runs,axis=0)
    
    return gammat_gibbs1,states_outer_gibbs1,trans_gibbs1,states_run


# def gibbs(emissions, hypers, options,seed1,pre_estimated_params=None):
#     n_factors = hypers["n_factors"]
#     n_states = hypers["n_states"]  # now a list
#     n_timesteps, emission_dim = emissions.shape

#     random.seed(seed1)

#     if pre_estimated_params is not None:
#         # params = pre_estimated_params    
#         use_pre=True
#         posteriors = pre_estimated_params['posteriors']
#         # stackpost=posteriors_list2array(posteriors)
#         # stackpost=100*np.random.randn(*stackpost.shape)
#         # posteriors=posteriors_array2list(stackpost,n_states)
#         params0 = {
#             # Generate initial distribution per factor
#             "initial_dist": [np.random.dirichlet(np.ones(n_states[h])) for h in range(n_factors)],
#             # Initialize means for each factor using the mean of emissions (can be refined)
#             "means": [np.tile(np.mean(emissions, axis=0).reshape(1, emission_dim), (n_states[h], 1)) for h in range(n_factors)],
#             # "means": [np.random.randn(n_states[h],emission_dim) for h in range(n_factors)],
#             # Generate transition matrices for each factor
#             "transition_matrices": ([generate_tpm(n_states[h]) for h in range(n_factors)]),
#             "variances": np.var(emissions, axis=0),
#             # Initialize states: for each factor, create a column vector of zeros (integer)
#             # "states": np.column_stack([np.zeros(n_timesteps, dtype=int) for h in range(n_factors)])
#         }
#         # gammat_gibbs, states_outer_gibbs, trans_gibbs, states_out = _gibbs_post_prob(emissions, params0, hypers, options)
#     else:
#         use_pre=False
#         params0 = {
#             # Generate initial distribution per factor
#             "initial_dist": [np.random.dirichlet(np.ones(n_states[h])) for h in range(n_factors)],
#             # Initialize means for each factor using the mean of emissions (can be refined)
#             # "means": [np.tile(np.mean(emissions, axis=0).reshape(1, emission_dim)+0.*np.random.randn(n_states[h],emission_dim), (n_states[h], 1)) for h in range(n_factors)],
#             "means": [np.tile(np.mean(emissions, axis=0).reshape(1, emission_dim), (n_states[h], 1)) for h in range(n_factors)],
#             # "means": [np.random.randn(n_states[h],emission_dim) for h in range(n_factors)],
#             # Generate transition matrices for each factor
#             "transition_matrices": ([generate_tpm(n_states[h]) for h in range(n_factors)]),
#             "variances": np.var(emissions, axis=0),
#             # Initialize states: for each factor, create a column vector of zeros (integer)
#             # "states": np.column_stack([np.zeros(n_timesteps, dtype=int) for h in range(n_factors)])
#         }
    
#     params = params0.copy()
#     samples = []
#     lls = []
#     posteriors_list = []
#     params_samples = [];# params_samples.append(params)
#     for iopt in trange(options["n_em_iter"]):
#         print(f'iteration {iopt}')
#         # gammat_gibbs, states_outer_gibbs, trans_gibbs, states_out = _gibbs_post_prob(emissions, params, hypers, options)
#         if (use_pre and iopt == 0):
#             print('Set expectations to pre_estimated_params values, skipping to m step')
#             gammat_gibbs = np.hstack(posteriors) # reshape posteriors into an array of shape (num_timesteps, sum(n_states_list))
#             states_outer_gibbs = np.einsum('ti,tj->tij', gammat_gibbs, gammat_gibbs) 
#             trans_gibbs = np.einsum('ti,tj->tij', gammat_gibbs[1:], gammat_gibbs[:-1])
#             states_out = np.zeros((n_timesteps, n_factors), dtype=int)
#             for h in range(n_factors):
#                 # set initial_states[:,h] as the argmax of the params['posteriors'] for factor h
#                 states_out[:,h]=np.argmax(posteriors[h][:,:],axis=1)
#             use_pre=False
#         else:
#             gammat_gibbs, states_outer_gibbs, trans_gibbs, states_out = _gibbs_post_prob(emissions, params, hypers, options)
#             if iopt == 0:
#                 gammat_gibbs = np.hstack(options['posteriors']) # reshape posteriors into an array of shape (num_timesteps, sum(n_states_list))
#                 states_outer_gibbs = states_outer_gibbs
#                 trans_gibbs = trans_gibbs
#                 states_out = np.zeros((n_timesteps, n_factors), dtype=int)
#                 for h in range(n_factors):
#                     # set initial_states[:,h] as the argmax of the params['posteriors'] for factor h
#                     states_out[:,h]=np.argmax(options['posteriors'][h][:,:],axis=1)        
        
#         print(gammat_gibbs)
#         params, lls1 = _m_step(gammat_gibbs, states_outer_gibbs, trans_gibbs, emissions, hypers)
#         # reformat gammat_gibbs from (num_timesteps,sum(n_states)) to a list of arrays
#         posteriors_reformat=posteriors_array2list(gammat_gibbs,n_states)
#         # split_indices = np.cumsum(n_states)[:-1]  # Get the split indices
#         # posteriors_reformat = np.split(gammat_gibbs, split_indices, axis=1) 
#         samples.append(states_out)
#         posteriors_list.append(posteriors_reformat)
#         lls.append(lls1)
#         params_samples.append(params)

#         if iopt > 0:
#             changes = _compute_changes(params_samples, lls)
#             print(changes)
#             if np.max(changes) < options['tolerance']:
#                 print(f'Convergence at EM iteration number {iopt+1} of {options["n_em_iter"]}')
#                 break        
    
#     return samples, params_samples, lls, posteriors_list


def gibbs(emissions, hypers, options,seed1, pre_estimated_params=None):
    n_factors = hypers["n_factors"]
    n_states = hypers["n_states"]  # now a list
    n_timesteps, emission_dim = emissions.shape

    
    if pre_estimated_params is not None:
        # params = pre_estimated_params    
        use_pre=True
        posteriors = pre_estimated_params['posteriors']
    else:
        use_pre=False
        random.seed(seed1)
        params = {
            # Generate initial distribution per factor
            "initial_dist": [np.random.dirichlet(np.ones(n_states[h])) for h in range(n_factors)],
            # Initialize means for each factor using the mean of emissions (can be refined)
            "means": [(1+0.1*np.random.randn(n_states[h],emission_dim))*np.tile(np.mean(emissions, axis=0).reshape(1, emission_dim), (n_states[h], 1)) for h in range(n_factors)],
            # "means": [np.random.randn(n_states[h],emission_dim) for h in range(n_factors)],
            # Generate transition matrices for each factor
            "transition_matrices": ([generate_tpm(n_states[h]) for h in range(n_factors)]),
            "variances": 0.5*np.var(emissions, axis=0),
            # Initialize states: for each factor, create a column vector of zeros (integer)
            # "states": np.column_stack([np.zeros(n_timesteps, dtype=int) for h in range(n_factors)])
        }
    
    # params = params.copy()
    samples = []
    lls = []
    posteriors_list = []
    params_samples = [];# params_samples.append(params)
    for iopt in trange(options["n_em_iter"]):
        if (use_pre and iopt == 0):
            print('Set expectations to pre_estimated_params values, skipping to m step')
            gammat_gibbs = np.hstack(posteriors) # reshape posteriors into an array of shape (num_timesteps, sum(n_states_list))
            states_outer_gibbs = np.einsum('ti,tj->tij', gammat_gibbs, gammat_gibbs) 
            trans_gibbs = np.einsum('ti,tj->tij', gammat_gibbs[1:], gammat_gibbs[:-1])
            states_out = np.zeros((n_timesteps, n_factors), dtype=int)
            for h in range(n_factors):
                # set initial_states[:,h] as the argmax of the params['posteriors'] for factor h
                states_out[:,h]=np.argmax(posteriors[h][:,:],axis=1)
            use_pre=False
            # params = pre_estimated_params    
            # gammat_gibbs, states_outer_gibbs, trans_gibbs, states_out = _gibbs_post_prob(emissions, params, hypers, options)
        else:
            # print(f'iteration {iopt}')
            gammat_gibbs, states_outer_gibbs, trans_gibbs, states_out = _gibbs_post_prob(emissions, params, hypers, options)
        # print(gammat_gibbs)
        params, lls1 = _m_step(gammat_gibbs, states_outer_gibbs, trans_gibbs, emissions, hypers)
        # reformat gammat_gibbs from (num_timesteps,sum(n_states)) to a list of arrays
        posteriors_reformat=posteriors_array2list(gammat_gibbs,n_states)
        # split_indices = np.cumsum(n_states)[:-1]  # Get the split indices
        # posteriors_reformat = np.split(gammat_gibbs, split_indices, axis=1) 
        samples.append(states_out)
        posteriors_list.append(posteriors_reformat)
        lls.append(lls1)
        params_samples.append(params)

        if iopt > 0:
            changes = _compute_changes(params_samples, lls)
            # print(changes)
            if np.max(changes) < options['tolerance']:
                print(f'Convergence at EM iteration number {iopt+1} of {options["n_em_iter"]}')
                break        
    
    return samples, params_samples, lls, posteriors_list




# taken from the ssm package

@numba.jit(nopython=True, cache=True)
def logsumexp(x):
    N = x.shape[0]

    # find the max
    m = -np.inf
    for i in range(N):
        m = max(m, x[i])

    # sum the exponentials
    out = 0
    for i in range(N):
        out += np.exp(x[i] - m)

    return m + np.log(out)

@numba.jit(nopython=True, cache=True)
def backward_sample(Ps, log_likes, alphas, us, zs):
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
    assert Ps.shape[1] == K
    assert Ps.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K
    assert us.shape[0] == T
    assert zs.shape[0] == T

    lpzp1 = np.zeros(K)
    lpz = np.zeros(K)

    # Trick for handling time-varying transition matrices
    hetero = (Ps.shape[0] == T-1)

    for t in range(T-1,-1,-1):
        # compute normalized log p(z[t] = k | z[t+1])
        lpz = lpzp1 + alphas[t]
        Z = logsumexp(lpz)

        # sample
        acc = 0
        zs[t] = K-1
        for k in range(K):
            acc += np.exp(lpz[k] - Z)
            if us[t] < acc:
                zs[t] = k
                break

        # set the transition potential
        if t > 0:
            lpzp1 = np.log(Ps[(t-1) * hetero, :, int(zs[t])] + LOG_EPS)

@numba.jit(nopython=True, cache=True)
def forward_pass(pi0,
                 Ps,
                 log_likes,
                 alphas):

    T = log_likes.shape[0]  # number of time steps
    K = log_likes.shape[1]  # number of discrete states

    assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
    assert Ps.shape[1] == K
    assert Ps.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K

    # Check if we have heterogeneous transition matrices.
    # If not, save memory by passing in log_Ps of shape (1, K, K)
    hetero = (Ps.shape[0] == T-1)
    alphas[0] = np.log(pi0) + log_likes[0]
    for t in range(T-1):
        m = np.max(alphas[t])
        alphas[t+1] = np.log(np.dot(np.exp(alphas[t] - m), Ps[t * hetero])) + m + log_likes[t+1]
    return logsumexp(alphas[T-1])


@numba.jit(nopython=True, cache=True)
def _hmm_sample_run(pi0, Ps, ll):
    T, K = ll.shape

    # Forward pass gets the predicted state at time t given
    # observations up to and including those from time t
    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)

    # Sample backward
    us = npr.rand(T)
    zs = -1 * np.ones(T)
    backward_sample(Ps, ll, alphas, us, zs)
    return zs

def _hmm_sample(pi0, Ps, ll):
    return _hmm_sample_run(pi0, Ps, ll).astype(int)