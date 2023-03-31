import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt
from tqdm.auto import trange 

import ssm
from ssm.messages import hmm_sample

from joblib import Parallel, delayed
import multiprocessing
NumThread=(multiprocessing.cpu_count()-1)*2 # sets number of workers based on cpus on current machine


from scipy.stats import norm


"""
In the E-M algorithm for the factorial HMM, the M-step can be performed exactly, 
but the E-step is exactly solvable, so we replace it with an approximate E-step
where the posterior probabilities of the hidden states are obtained via Gibbs sampling 
We follow (Ghahramani and Jordan, 1997):
- Gibbs sampling for the E-step implemented in the gibbs sampling functions below:
    1. '_gibbs_sample_states' runs a single Gibbs sampling sweep 
        for the state sequence for one factor, using the parameters from the previous M-step
        or from the initial values if it's the first E-M iteration. Note: impatient sampling
        recommended, using a small number of steps.
    2.  '_gibbs_all_factors' runs a single Gibbs sampling sweep 
        calling '_gibbs_sample_states' for all factors and 
        converts the sample into 1-hot posterior probabilities 
    3. '_gibbs_post_prob' runs multiple Gibbs samplings (option for parallel computing)
        each one by calling '_gibbs_all_factors'. It then averages the single-sample 1-hot 
        posterior probabilities to get the sample-averaged posterior probabilities
- Exact M-step in Appendix A implemented in the '_m_step' function. It takes the 
    sample-averaged posteriors from the E-step above and computes the parameters.
The code iterates over E-M steps several times. The main issue is the non-convexity of the 
process, so it's important to run it several times to get multiple estimates and then pick the
one with the best likelihood.
"""

def _m_step(gammat_gibbs,states_outer_gibbs, trans_gibbs,emissions, params, hypers):
    """
    Find parameters with exact M-step using state probabilities estimated from Gibbs sampling 
    This function implements Appendix A in (Ghahramani and Jordan, 1997)
    """
    num_factors = hypers["num_factors"]
    num_states = hypers["num_states"]
    emission_dim = emissions.shape[1]
    num_timesteps = gammat_gibbs.shape[0]
    params_sample=params.copy()

    # initial state probabilities
    params_sample["initial_dist"]=gammat_gibbs[0,:,:].T
    aa=np.sum(gammat_gibbs[1:],axis=0) + 1e-32
    transition_matrix=np.nan_to_num(np.sum(trans_gibbs,axis=0)/aa)
    transition_matrix=np.transpose(transition_matrix,(2,0,1))
    for h in range(num_factors):
        P = transition_matrix[h]
        P = np.where(P.sum(axis=0, keepdims=True) == 0, 1.0 / num_states, P)
        transition_matrix[h]=P
    # print(transition_matrix)
    params_sample['transition_matrices']=transition_matrix
    # means
    # reshape arrays for convenience
    gammat_re=np.reshape(gammat_gibbs,(num_timesteps,num_factors*num_states))
    states_outer_gibbs_re=np.reshape(np.transpose(states_outer_gibbs,(0,1,3,2,4)),(num_timesteps,num_states*num_factors,num_factors*num_states))
    means_first=np.matmul(emissions.T,gammat_re) # emission_dim x num_states*num_factors
    means_second=np.sum(states_outer_gibbs_re,axis=0) # num_states*num_factors  x num_states*num_factors
    # # check that means_second is symmetric
    # a=means_second-means_second.T
    # print("means_second is symmetric up to "+str(a.max()))

    means_second_inv=np.linalg.pinv(means_second)
    # # check that means_second is symmetric
    # a=means_second_inv-means_second_inv.T
    # print("means_second_inv is symmetric up to "+str(a.max()))
    means_2d=np.matmul(means_first,means_second_inv) # emission_dim x num_states*num_factors
    

    # variances
    Cnew_first=(1/num_timesteps)*np.matmul(emissions.T,emissions)
    Cnew_sec1=(1/num_timesteps)*np.matmul(emissions.T,gammat_re) # D x num_states*num_factors
    Cnew_sec=np.matmul(Cnew_sec1,means_2d.T)
    # # check that Cnew_sec is symmetric
    # a=Cnew_sec-Cnew_sec.T
    # print("Cnew_sec is symmetric up to "+str(a.max()))
    Cnew=Cnew_first-Cnew_sec # D x D

    means=np.transpose(np.reshape(means_2d,(emission_dim,num_states,num_factors)),(2,1,0))
    # print('means.shape',means.shape)
    params_sample["means"]=means
    params_sample["variances"]=np.diag(Cnew)

    """Log-likelihood
    """

    # first factor
    Cinv=np.linalg.inv(np.diag(params_sample['variances']))
    Q1=-(1/2)*np.trace(np.matmul(np.matmul(emissions,Cinv),emissions.T))
    # print(Q1)
    # second factor
    Q21=np.matmul(Cinv,np.matmul(means_2d,gammat_re.T)) # emissions_dim x time
    Q2=np.trace(np.matmul(emissions,Q21))
    # print(Q2)
    # third factor
    Q31=np.matmul(states_outer_gibbs_re,means_2d.T).transpose(1,0,2).reshape(num_factors*num_states,num_timesteps*emission_dim)
    Q31=np.matmul(Cinv,np.matmul(means_2d,Q31)).reshape(emission_dim,num_timesteps,emission_dim)
    Q3=-(1/2)*np.sum(np.trace(Q31, axis1=0, axis2=2))
    # print(Q1+Q2+Q3)
    # fourth factor
    pi1=np.reshape(params_sample["initial_dist"].T,num_states*num_factors)
    Q4=np.matmul(gammat_re[0,:],pi1)
    # print(Q4)
    # fifth factor
    # print(trans_gibbs.shape)
    Q51=trans_gibbs.reshape(num_timesteps-1,num_states,num_states*num_factors) # last dim=state at t-1
    Q52=transition_matrix.transpose(1,2,0).reshape(num_states,num_states*num_factors).T
    Q5=np.sum(np.trace(np.matmul(Q51,Q52),axis1=1,axis2=2))
    # print(Q5)
    lls=-(Q1+Q2+Q3+Q4+Q5)

    return params_sample,lls


def _gibbs_sample_states(h, states, emissions, params, hypers):
    """Sample sequence of states for factor h given the other factors.
    """
    num_factors = hypers["num_factors"]
    num_states = hypers["num_states"]
    num_timesteps = states.shape[0]
    means = params["means"]
    variances = params["variances"]

    lls = np.zeros((num_timesteps, num_states))
    tmp_states = states.copy()
    for k in range(num_states):
        tmp_states[:, h] = k
        expec_emissions = np.zeros_like(emissions)
        var_emissions = np.zeros_like(emissions)
        for j in range(num_factors):
            expec_emissions += means[j, tmp_states[:, j], :]
        var_emissions = variances

      
        lls[:, k] = norm(expec_emissions, np.sqrt(var_emissions)).logpdf(emissions).sum(axis=1)

    return hmm_sample(params["initial_dist"][h],
                             params["transition_matrices"][h][None, :, :],
                             lls)  

def _gibbs_all_factors(initial_states, emissions, params, hypers,options):
    """Gibbs samples of all factors, and it gives as outputs posterior prob obtained from a single Gibbs sample
    needed for the m-step inference
    """
    num_factors = hypers["num_factors"]
    num_states = hypers["num_states"]
    num_timesteps = initial_states.shape[0]
    states=initial_states.copy()
    # run num_iters steps of gibbs for states convergence
    for itr in range(options["num_iters"]):
        for h in range(num_factors):
            states[:, h] = _gibbs_sample_states(h, states, emissions, params, hypers)
    # turn state sample at the end of the gibbs run into one-hot posterior probability distribution
    # gammat=post prob [S_t^(m)]_k for time t, factor m, state k
    # state_outer=post prob [S_t^(m1)]_k1 [S_t^(m2)]_k2 for time t, factors m1,m2, states k1,k2
    # states=true_states.copy()
    gammat=np.zeros((num_timesteps,num_states,num_factors))
    for h in range(num_factors):
            tmp_states=states[:,h]
            gammat_h = np.zeros((tmp_states.size, num_states))
            gammat_h[np.arange(tmp_states.size), tmp_states] = 1
            gammat[:,:,h]=gammat_h
    state_outer=np.zeros((num_timesteps,num_states,num_states,num_factors,num_factors))
    for h1 in range(num_factors):
            for h2 in range(num_factors):
                    for t in range(num_timesteps):
                        state_outer[t,:,:,h1,h2]=np.outer(gammat[t,:,h1],gammat[t,:,h2])

    trans=np.zeros((num_timesteps-1,num_states,num_states,num_factors))
    for h in range(num_factors):
            for t in range(num_timesteps-1):
                        trans[t,:,:,h]=np.outer(gammat[t+1,:,h],gammat[t,:,h])    
    out=dict(gammat=gammat,
             state_outer=state_outer,
             trans=trans,
             states=states)
    return out

def _gibbs_post_prob(initial_states, emissions, params, hypers, options):
    """This function runs the gibbs sampling for all factors num_gibbs times 
        and returns the posterior probability as the average over the num_gibbs runs
        It can be parallelized
    """
    num_gibbs=options["num_gibbs"]
    params=params.copy()   
    gammat_runs=[]
    states_outer_runs=[]
    trans_runs=[]
    states_run=[]
    if options["parallel"]:
        out = Parallel(n_jobs=num_gibbs)(
            delayed(_gibbs_all_factors)(initial_states, emissions, params, hypers,options)
            for igibbs in range(num_gibbs))
        # unpack outputs
        for i in range(num_gibbs):
            gammat_runs.append(out[i]['gammat'])
            states_outer_runs.append(out[i]['state_outer'])
            trans_runs.append(out[i]['trans'])
            states_run.append(out[i]['states'])

    else:
        for igibbs in range(num_gibbs):
            out=_gibbs_all_factors(initial_states, emissions, params, hypers,options)
            gammat_runs.append(out['gammat'])
            states_outer_runs.append(out['state_outer'])
            trans_runs.append(out['trans'])
            states_run.append(out['states'])

    # average over independent runs of gibbs to obtain posterior prob for exact m-step below 
    gammat_gibbs=np.mean(gammat_runs,axis=0)
    states_outer_gibbs=np.mean(states_outer_runs,axis=0)
    trans_gibbs=np.mean(trans_runs,axis=0)
    
    return gammat_gibbs,states_outer_gibbs,trans_gibbs,states_run





def gibbs(initial_states, emissions, params, hypers, options):
    num_factors = hypers["num_factors"]
    num_states = hypers["num_states"]
    emission_dim=emissions.shape[1]
    params=params.copy()
    samples = []
    lls = []
    posteriors = []
    params_samples = []
    for iopt in trange(options["num_runs"]):
#         print("iopt",iopt)
#         params['means'] = 3*npr.randn(num_factors, num_states, emission_dim)
        gammat_gibbs,states_outer_gibbs,trans_gibbs,states=_gibbs_post_prob(initial_states, emissions, params, hypers, options)

        # get parameters using m_step
        params,lls1 = _m_step(gammat_gibbs,states_outer_gibbs, trans_gibbs,emissions, params, hypers)

        samples.append(states)
        posteriors.append(gammat_gibbs)
        lls.append(lls1)
        params_samples.append(params)
    
    return samples, params_samples,lls, posteriors