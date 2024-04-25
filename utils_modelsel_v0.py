import numpy as np
# import numpy.random as npr
import torch
from sklearn.decomposition import FastICA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import explained_variance_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt
import seaborn as sns
# from tqdm.auto import trange 

import ssm
# from ssm.messages import hmm_sample

from joblib import Parallel, delayed
# import multiprocessing
# NumThread=(multiprocessing.cpu_count()-1)*2 # sets number of workers based on cpus on current machine
# print('Parallel processing with '+str(NumThread)+' cores')

# from util_factorial_hmm import gibbs





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

def autocorrelation_torch(x):
    """
    Compute the autocorrelation of the signal using PyTorch, leveraging GPU computation on Apple Silicon.
    Parameters:
    - x: A 1D PyTorch tensor of the signal.
    Returns:
    - A 1D PyTorch tensor containing the autocorrelation of the input signal.
    """
    # Ensure input is a float32 tensor for fft and ensure it's on the GPU
    x = torch.tensor(x).to(dtype=torch.float32).to('mps')  # Convert to float32 and use 'mps' for Apple Silicon GPU
    
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

def plot_autocorrelations_with_timescale(matrix):
    """Plot the autocorrelations of each time series in the matrix and annotate with HWHM."""
    fig=plt.figure(figsize=(10, 6))
    timescales = []
    for i in range(matrix.shape[0]):
        acf = autocorrelation_torch(matrix[i])
        timescale = compute_hwhm(acf)
        timescales.append(timescale)
        plt.plot(acf, label=f"Series {i+1} (Timescale: {timescale})")
    plt.legend()
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation of Time Series with Timescales")
    plt.xlim([0, np.max(timescales)*10])
    plt.show()
    plt.savefig('AutocorrTimescales.png')
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
    kf = KFold(n_splits=5)
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

def find_best_k(feature_data, max_k=10, n_splits=5):
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

def runHMM(synthetic_data,num_states, obs_dim,N_iters,TOL):
    temp={}
    new_hmm = ssm.HMM(num_states, obs_dim, 
            observations="gaussian", transitions="standard")
    _ = new_hmm.fit(synthetic_data, method="em", num_iters=N_iters, tolerance=TOL)   
    temp['lls']=new_hmm.log_likelihood(synthetic_data)
    temp['hmm']=new_hmm
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
    out = Parallel(n_jobs=-1)(delayed(runHMM)(synthetic_data,num_states, obs_dim,N_iters,TOL) for irun in range(numRun))
    lls=[out[irun]['lls'] for irun in range(numRun)]
    best_hmm=out[np.argmax(lls)]['hmm']
    out=best_hmm.log_likelihood(synthetic_data)
    posterior_probs = best_hmm.expected_states(synthetic_data)[0]
    mus=best_hmm.observations.mus
    sigmas=best_hmm.observations.Sigmas
    out={'posterior_probs':posterior_probs,'mus':mus,'sigmas':sigmas}
    return out


def reduce_correlated_features(correlation_matrix, threshold=0.95):
    # Find pairs of features with high correlation
    high_corr_pairs = np.where(np.abs(correlation_matrix) > threshold)
    high_corr_pairs = [(i, j) for i, j in zip(*high_corr_pairs) if i != j]

    # Group correlated features
    groups = []
    for pair in high_corr_pairs:
        for group in groups:
            if pair[0] in group or pair[1] in group:
                group.add(pair[0])
                group.add(pair[1])
                break
        else:
            groups.append(set(pair))

    # Print group members
    for group in groups:
        print(group)

    # Keep one feature per group
    features_to_keep = [list(group)[0] for group in groups]

    # Add features that didn't belong to any group
    all_features = set(range(correlation_matrix.shape[0]))
    grouped_features = set([item for sublist in groups for item in sublist])
    non_grouped_features = list(all_features - grouped_features)
    features_to_keep.extend(non_grouped_features)

    # Remove duplicates
    features_to_keep = list(set(features_to_keep))

    # Recompute correlation matrix
    new_corr_matrix = correlation_matrix[np.ix_(features_to_keep, features_to_keep)]

    return new_corr_matrix, features_to_keep

def visualize_corr_matrix(corr_matrix):
          fig = plt.figure(figsize=(10,10))
          sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
          # plt.show()
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



def model_sel_ICA(emissions_data,MAX_runs=50):
    # Define the range of components to be tested
    MAX,count,old_optimal_n_components=MAX_runs,0,0
    average_variances = []
    for n_components in range(1,MAX):
          average_variance = cross_validate_components(n_components, emissions_data)
          average_variances.append(average_variance)
          # Find the elbow point
          if n_components>5:
                    new_optimal_n_components = find_elbow_point(average_variances)
                    if new_optimal_n_components==old_optimal_n_components:
                              count+=1
                    else: count=0
                    if count>2:
                              break
                    old_optimal_n_components=new_optimal_n_components
    optimal_n_components=new_optimal_n_components
    max_n_components=n_components

    print("Optimal number of components based on the elbow method:", optimal_n_components)

    # Plotting the scores
    n_factors_range=np.arange(max_n_components)+1
    plt.figure(figsize=(10, 6))
    plt.plot(n_factors_range, average_variances, marker='o')
    plt.xlabel('Number of Factors')
    plt.ylabel('Cross-Validated Score')
    plt.title('Number of Sources vs. Cross-Validated Score')
    plt.grid(True)
    plt.show()
    plt.savefig('ModelSel_ICA.png')

    ica = FastICA(n_components=optimal_n_components, random_state=0)
    ica.fit(emissions_data)
    factors = ica.transform(emissions_data)
    ica_mixing = ica.mixing_
    ica_mean = ica.mean_
    
    # compare ICA fit and original data
    # Plot the reconstructed data
    nplots=np.min([5,emissions_data.shape[1]])
    plot_length=np.min([300,emissions_data.shape[0]])
    fig, axes = plt.subplots(nplots, 1, figsize=(8, 12))
    data_rec=factors@ica_mixing.T+ica_mean
    for idx in range(nplots):
          axes[idx].plot(data_rec[:plot_length, idx], alpha=0.5, label=f'Fit')
          axes[idx].plot(emissions_data[:plot_length, idx], alpha=0.5,label=f'Data')
          axes[idx].legend()
#     plt.xlim([1,plot_length])
    plt.tight_layout()
    plt.show()
    plt.savefig('ModelSel_ICA_reconstruction.png')


    return ica,optimal_n_components


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
          plt.savefig('ModelSel_GMMBestK.png')
          plt.show()    


          return results,n_states


# fit hmm to each factor, extract one-hot state posterior prob
def fit_hmm_after_gmm(factors,n_states,timescales):
          out=[]
          data=factors
          _,N=data.shape
          obs_dim=1
          # posteriors = Parallel(n_jobs=-1)(delayed(single_hmm)(data[:, neuron].reshape(-1, 1),obs_dim,num_states) for (neuron,num_states) in zip(range(data.shape[1]),n_states))
          for (neuron,num_states) in zip(range(data.shape[1]),n_states):
                    out.append(single_hmm(data[:, neuron].reshape(-1, 1),obs_dim,num_states))

          fig = plt.figure(figsize=(N*5, 5), dpi=80, facecolor='w', edgecolor='k')
          fig.suptitle('Factor time series vs HMM prob')
          for neuron in range(N):
                    posterior_probs=out[neuron]['posterior_probs']
                    mus=out[neuron]['mus']
                    state_seq=np.argmax(posterior_probs,axis=1)
                    plt.subplot(1,N,neuron+1)
                    for k in range(n_states[neuron]):
                              plt.plot(np.arange(len(posterior_probs)),posterior_probs[:,k], label="State " + str(k + 1), lw=2)
                              plt.plot(np.arange(len(posterior_probs)),factors[:, neuron], label="factor " + str(neuron), lw=2)
                              plt.plot(np.arange(len(posterior_probs)),mus[state_seq], label="exp values", lw=5)
                              # plt.ylim((-0.05, 1.05))
                              plt.xlim((1000, 1000+timescales[neuron]*20))
                              # plt.yticks([0, 0.5, 1], fontsize = 10)
                              plt.xlabel("trial #", fontsize = 15)
                              plt.title(f"states={n_states[neuron]},tau={timescales[neuron]}", fontsize = 15)
                              plt.ylabel("p(state)", fontsize = 15)
          plt.savefig('ModelSel_HMMonICASources.png')
          return out

def infer_timescales(factors):

          timescales, fig = plot_autocorrelations_with_timescale(factors.T)
          ax = fig.axes[0]  # Access the first axes
          ax.set_xlim([0,10*np.max(timescales)])  # Set the x-axis limits
          # ax.show()  # Display the plot with the updated xlim
          print("Timescales:", timescales)



          # Step 1: Log Transformation
          # timescales = np.array([482, 5, 3894, 45, 3905, 5, 515, 47])
          log_timescales = np.log(timescales)

          # Step 2: Bootstrap Analysis
          N = 1000
          bootstrap_samples = np.random.choice(log_timescales, size=(N,), replace=True)


          range_n_clusters=np.arange(2,len(timescales)-1)
          silhouette_avg_scores = []
          for n_clusters in range_n_clusters:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                    cluster_labels = clusterer.fit_predict(bootstrap_samples.reshape(-1, 1))    
                    silhouette_avg = silhouette_score(bootstrap_samples.reshape(-1, 1), cluster_labels)
                    silhouette_avg_scores.append(silhouette_avg)
                    # print(f"Number of clusters: {n_clusters}, Silhouette Score: {silhouette_avg}")

          # Find the optimal number of clusters based on silhouette score
          optimal_n_timescales = np.argmax(np.diff(silhouette_avg_scores))+3
          clusterer = KMeans(n_clusters=optimal_n_timescales, random_state=10)
          old_labels = clusterer.fit_predict(log_timescales.reshape(-1, 1))    
          # Compute the mean value of each cluster
          cluster_means = np.array([log_timescales[old_labels == i].mean() for i in range(optimal_n_timescales)])
          # Sort the clusters based on their means and get the sorted indices
          sorted_cluster_indices = np.argsort(cluster_means)
          # Create a mapping from original cluster labels to new labels
          # The new labels are assigned based on the sorted order of cluster means
          label_mapping = {original: new for new, original in enumerate(sorted_cluster_indices)}
          # Apply the mapping to get the new labels
          labels = np.array([label_mapping[label] for label in old_labels])
          # print("Original Labels:", old_labels)
          # print("New Labels:     ", labels)
          print(f"Optimal number of clusters: {optimal_n_timescales}")
          print(f"Labels: {labels}")
          print(f"Timescales: {timescales}")

          # Plotting the results again for reference
          plt.figure(figsize=(10, 6))
          plt.plot(range_n_clusters, silhouette_avg_scores, marker='o', linestyle='-')
          plt.title('KMeans Silhouette Scores to Determine Optimal Timescale Cluster Number')
          plt.xlabel('Number of Timescale Clusters')
          plt.ylabel('Silhouette Score')
          plt.axvline(x=optimal_n_timescales, color='red', linestyle='--', label=f'Optimal: {optimal_n_timescales} Clusters')
          plt.legend()
          plt.grid(True)
          plt.show()
          plt.savefig('AutocorrTimescales_Clusters.png')

          return timescales,labels,optimal_n_timescales



def remove_duplicate_hmms(hmm_fits,labels):

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
            'sigmas':np.concatenate([hmm_fits[i]['sigmas'] for i in idx]).flatten(),
            'posterior':posterior_probs,
            'labels':np.concatenate([[labels[i] for i2 in range(len(hmm_fits[i]['posterior_probs'][0]))] for i in idx])
            }
    correlation_matrix=np.corrcoef(posterior_probs)
    fig=visualize_corr_matrix(correlation_matrix)
    # find pairs of most correlated states
    fig.savefig('ModelSel_AllHMM_Corr.png')

    new_corr_matrix, features_to_keep=reduce_correlated_features(correlation_matrix, threshold=0.95)
    fig=visualize_corr_matrix(new_corr_matrix)
    fig.savefig('ModelSel_ReducedHMM_Corr.png')
    data_factors_to_keep = {}
    for key, value in data_factors.items():
        data_factors_to_keep[key] = value[features_to_keep]

    return data_factors_to_keep,new_corr_matrix,data_factors


def cluster_hmm(correlation_matrix,data_factors_to_keep):

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
    plt.plot(thresholds, num_communities, marker='o')
    plt.xlabel('Threshold=(corr+1)')
    plt.ylabel('Number of Communities')
    plt.title('Number of Overlapping Communities vs. Threshold Level')
    plt.grid(True)
    plt.show()
    plt.savefig('ModelSel_reducedHMMcommunities_search.png')

    # Apply the revised function
    longest_start_meaningful, longest_end_meaningful, n_clusters_meaningful = find_most_meaningful_interval(
        num_communities,correlation_matrix)

    # Select a threshold from the middle of the most meaningful interval for visualization
    # midpoint_meaningful = (longest_start_meaningful + longest_end_meaningful) // 2
    selected_threshold_meaningful = thresholds[longest_start_meaningful]-1
    id_communities_meaningful=id_communities[longest_start_meaningful]
    Z_communities_meaningful=Z_communities[longest_start_meaningful]
    # # rescale the corr in Z_communities to subtract one
    # for i in range(Z_communities_meaningful.shape[0]):
    #           Z_communities_meaningful[i,2]=Z_communities_meaningful[i,2]-1
    mean_corr_communities_meaningful=np.array(mean_corr_communities[longest_start_meaningful])-1
    labels_hmm = [f'({factor_idx}, {state_idx})' for (factor_idx, state_idx) in zip(data_factors_to_keep['factor'], data_factors_to_keep['state'])]
    # Plotting the dendrogram
    plt.figure(figsize=(8, 4))
    dendrogram(Z_communities_meaningful,labels=labels_hmm)
    plt.xticklabels=labels_hmm
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.xlabel("Factors")
    plt.ylabel("Distance")
    plt.axhline(y=selected_threshold_meaningful+1, color='r', linestyle='--')
    plt.text(x=0, y=selected_threshold_meaningful+1, s=f' corr threshold= {selected_threshold_meaningful:.2f}', color='red')
    # Shift y-axis ticks
    plt.gca().set_yticklabels([f'{float(tick.get_text()) - 1:.2f}' for tick in plt.gca().get_yticklabels()])
    plt.show()
    plt.savefig('ModelSel_reducedHMMdendrogram.png')

    return id_communities_meaningful,mean_corr_communities_meaningful,Z_communities_meaningful


def estimate_transition_matrix_vec(posteriors_chain):
    # Concatenate all posterior arrays into a single 2D array
    posteri = np.vstack(posteriors_chain)
    # Convert posterior probabilities to states
    states = np.argmax(posteri, axis=0)
    num_states=posteri.shape[0]
    # Create pairs of consecutive states
    pairs = np.vstack((states[:-1], states[1:]))
    # Use bincount to count the transitions and form the transition matrix
    transition_matrix_vec = np.bincount(num_states * pairs[0] + pairs[1], minlength=num_states**2)
    transition_matrix_vec = transition_matrix_vec.reshape((num_states, num_states)).astype(float)
    # Normalize each row to sum to 1
    transition_matrix_vec /= transition_matrix_vec.sum(axis=1, keepdims=True)
    return transition_matrix_vec

def reconstruct_parameters(id_communities,data_factors_to_keep,ica):

    Nchains = len(set(id_communities)) # number of chains
    means=[] # mean emissions
    posteriors=[] # posterior probabilities
    transitions=[] # transition matrices
    init=[] # initial distribution
    ica_mixing=ica.mixing_.T
    ica_mean=ica.mean_
    for i in range(Nchains):
        idx = np.where(id_communities == i+1)[0]
        means_chain=[]; posteriors_chain=[]
        for j in range(len(idx)):
            factor_id=data_factors_to_keep['factor'][idx[j]] # ICA factor where current HMM state originates from 
            means_chain.append(ica_mixing[factor_id,:]*data_factors_to_keep['mus'][idx[j]]+ica_mean)
            posteriors_chain.append(data_factors_to_keep['posterior'][idx[j]])
        means.append(means_chain)
        posteriors.append(posteriors_chain)
        transitions.append(estimate_transition_matrix_vec(posteriors_chain))
        init.append([posteriors_chain[i][0] for i in range(len(posteriors_chain))])

    return means,posteriors,transitions,init


def compare_reconstructed_true(means_rec,posteriors_rec,data):

    flat_one_hot_states=true2onehot_states(data['true_states'])
    num_chains,num_states=data['params']['transition_matrices'].shape[0],data['params']['transition_matrices'][0].shape[1]
    posteriors_conc=np.concatenate(posteriors_rec,axis=0).T
    num_factors_FA=posteriors_conc.shape[1]

    # Calculate correlations between all factors and all states
    correlation_matrix = np.zeros((num_factors_FA, num_chains * num_states))
    for factor_idx in range(num_factors_FA):
        for chain_idx in range(num_chains):
            for state_idx in range(num_states):
                flat_idx = chain_idx * num_states + state_idx
                correlation_matrix[factor_idx, flat_idx] = np.corrcoef(posteriors_conc[:, factor_idx], flat_one_hot_states[:, flat_idx])[0, 1]

    fig=visualize_corr_matrix (correlation_matrix)
    ax = fig.gca()
    ax.set_title('Correlation True vs Reconstructed states')
    ax.set_ylabel('Reconstructed states')
    ax.set_xlabel('True States')


    # Iterative matching procedure with index tracking
    unmatched_factors = list(range(num_factors_FA))
    unmatched_states = list(range(num_chains * num_states))
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

        # Match the factor and state
        matched_pairs.append((best_factor, best_true,highest_corr))

        # Remove matched factor and state from lists
        unmatched_factors.remove(best_factor)
        unmatched_states.remove(best_true)

    # correlation between emission means
    means=data['params']['means']
    means_conc=np.concatenate(means,axis=0)
    means_conc_rec=np.concatenate(means_rec,axis=0)
    corr_mus=[]; plot_matrix=np.zeros((means_conc.shape[1],4*means_conc.shape[0]))
    cnt=0
    for i in range(len(matched_pairs)):
        print('ICA Factor '+str(matched_pairs[i][0])+' is matched with true state '+str(matched_pairs[i][1])+' with correlation '+str(matched_pairs[i][2]))
        corr_mus.append(np.corrcoef(means_conc[matched_pairs[i][1]],means_conc_rec[matched_pairs[i][0]])[0,1])
        plot_matrix[:,cnt]=means_conc[matched_pairs[i][1]]; cnt=cnt+1
        plot_matrix[:,cnt]=means_conc_rec[matched_pairs[i][0]]; cnt=cnt+3
    fig = plt.figure(figsize=(10,10))
    im = plt.imshow(plot_matrix,aspect='auto',cmap='bwr', vmin=-np.max(np.abs(plot_matrix)), vmax=np.max(np.abs(plot_matrix)))
    plt.title('ICA-reconstructed (left) v True (right) emission means')
    plt.colorbar(im, extend='both')
    plt.savefig('ICAvsTrueemissions.png')

    print('corr_mus is '+str(corr_mus))
    print('Mean correlation of matched pairs is '+str(np.mean(corr_mus))+' with std '+str(np.std(corr_mus)))
