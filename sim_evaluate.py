"""Simulation study 1: calculate inferential utility metrics of original and synthetic datasets"""

import os
import pandas as pd
import numpy as np
import itertools
from itertools import product
from time import time
from utils.eval import avg_inv_KLdiv, common_rows_proportion, estimate_estimator, CI_coverage
from utils.disease import ground_truth

def load_data(n_samples: int, n_runs: int, sim_dir: str):
    """
    Load source files containing original and synthetic datasets.
    """
    
    # Create empty data structure for storage
    data = {}
    for n_sample in n_samples:
        data[f'n_{n_sample}'] = {}
        for i in range(n_runs):
            data[f'n_{n_sample}'][f'run_{i}'] = {}
    
    # Load datasets
    data_dir = [os.path.join(root, name) for root, dirs, files in os.walk(sim_dir) for name in files]
    for dir in data_dir:
        
        if '.ipynb_checkpoint' in dir:
            continue # skip dot files
        
        dir_structure = dir[len(sim_dir):].split('/') # remove sim_dir prefix from name and split directory by '/'
        try: # load datasets per n_sample and per Monte Carlo run
            data[dir_structure[0]][dir_structure[1]][dir_structure[2][:-len('.csv')]] = pd.read_csv(dir, engine='pyarrow') # also remove '.csv' suffix from name
        except Exception as e:
            print(f'File {dir} not stored in \'data\' object because file has aberrant directory structure')
    
    # Remove empty simulations
    for n_sample in n_samples:
        for i in range(n_runs):
            if not data[f'n_{n_sample}'][f'run_{i}']:
                del data[f'n_{n_sample}'][f'run_{i}']
    
    return data

def create_metadata(data: pd.DataFrame, export: bool=False, sim_dir: str=None):
    """
    Create meta_data (or update if additional generative models have been added) per n_sample and per Monte Carlo run.
    """
    
    for n_sample in data:
               
        for run in data[n_sample]:
           
            all_data = data[n_sample][run] # all datasets per run and n_sample
            meta_data = pd.DataFrame({'dataset_name': [name for name in all_data if all(substring not in name for substring in ['meta', 'bias', 'expected'])],
                                      'n': [all_data[name].shape[0] for name in all_data if all(substring not in name for substring in ['meta', 'bias', 'expected'])],
                                      'run': np.repeat(run, len([name for name in all_data if all(substring not in name for substring in ['meta', 'bias', 'expected'])])),
                                      'generator': ['_'.join(name.split('_')[:-1]) for name in all_data if all(substring not in name for substring in ['meta', 'bias', 'expected'])]})
            data[n_sample][run]['meta_data'] = meta_data # store meta data per simulation
            
            if export:
                meta_data.to_csv(sim_dir + n_sample + '/' + run + '/meta_data.csv', index=False) # export meta_data to .csv
    
    return data

def create_metadiagnostics(data: pd.DataFrame, export: bool=False, sim_dir: str=None):
    """
    Create meta_diagnostics (or update if additional generative models have been added) per n_sample and per Monte Carlo run.
    """

    for n_sample in data:

        for run in data[n_sample]:
            
            all_data = data[n_sample][run] # all datasets per run and n_sample

            # Initialize meta data
            meta_diagnostics = pd.DataFrame({'dataset_name': [name[:-len('_bias')] for name in all_data if 'bias' in name], # remove '_bias' suffix
                                             'n': np.repeat(int(n_sample[len('n_'):]), len([name for name in all_data if 'bias' in name])), # remove 'n_' prefix
                                             'run': np.repeat(run, len([name for name in all_data if 'bias' in name])),
                                             'generator': ['_'.join(name.split('_')[:-2]) for name in all_data if 'bias' in name]})
            
            # Bias terms
            bias_data = pd.DataFrame({})
            for name in all_data:
                if 'bias' in name:
                    bias_data = pd.concat([bias_data, all_data[name]], axis=0)
           
            # P_hat_n nuisance parameters
            expected_data = pd.DataFrame({})
            for name in all_data:
                if 'expected' in name:
                    expected_dataset = pd.DataFrame(np.array(all_data[name].iloc[:,1:]).reshape(-1)).transpose()
                    expected_dataset.columns = ['E_' + column + '_given_' + all_data[name].columns[0] + '_' + x 
                                                for x in all_data[name].iloc[:,0] 
                                                for column in all_data[name].columns[1:]]
                    expected_data = pd.concat([expected_data, expected_dataset], axis=0)
            
            # Create meta data
            meta_diagnostics = pd.concat([meta_diagnostics.reset_index(),
                                          bias_data.reset_index(), 
                                          expected_data.reset_index()], axis=1).drop(columns=['index'])
            data[n_sample][run]['meta_diagnostics'] = meta_diagnostics

            # Export
            if export:
                meta_diagnostics.to_csv(sim_dir + n_sample + '/' + run + '/meta_diagnostics.csv', index=False) # export meta_diagnostics to .csv
    
    return data

def create_metadiagnostics2(data: pd.DataFrame, Y: str=None, A: str=None, X: str=None, export: bool=False, sim_dir: str=None):
    """
    Create meta_diagnostics2 (or update if additional generative models have been added) per n_sample and per Monte Carlo run.
    """

    for n_sample in data:
                
        for run in data[n_sample]:
            
            meta_diagnostics2 = pd.DataFrame({})
            
            for dataset_name in [name for name in data[n_sample][run] if all(substring not in name for substring in ['original', 'targeted', 'meta', 'bias', 'expected'])]:
            
                # Define Y, A, X
                Y, A, X = 'bp', 'therapy', 'stage'
                
                # Synthetic dataset
                dataset = data[n_sample][run][dataset_name]
              
                # P_tilde_m  
                E_hat_P_tilde_m_Y_given_X = pd.DataFrame({'E_hat_P_tilde_m_Y_given_X': dataset.groupby(X)[Y].mean()}) # calculate mean Y for each level of X
                E_hat_P_tilde_m_Y_given_X.index = E_hat_P_tilde_m_Y_given_X.index.astype(str) # set to string to allow for indexing in case X is boolean variable
                E_hat_P_tilde_m_A_given_X = pd.DataFrame({'E_hat_P_tilde_m_A_given_X': dataset.groupby(X)[A].mean()}) # calculate mean A for each level of X
                E_hat_P_tilde_m_A_given_X.index = E_hat_P_tilde_m_A_given_X.index.astype(str) # set to string to allow for indexing in case X is boolean variable                
                
                # P_hat_n
                E_hat_P_hat_n_Y_given_X  = []
                E_hat_P_hat_n_A_given_X  = []
                indices = []
                for Xi in dataset[X].unique():
                    if dataset_name in list(data[n_sample][run]['meta_diagnostics']['dataset_name']): # debiasing successful
                        E_hat_P_hat_n_Y_given_X.append(float(data[n_sample][run]['meta_diagnostics'].query(f'dataset_name==\'{dataset_name}\'')['_'.join(['E',Y,'given',X,Xi])]))
                        E_hat_P_hat_n_A_given_X.append(float(data[n_sample][run]['meta_diagnostics'].query(f'dataset_name==\'{dataset_name}\'')['_'.join(['E',A,'given',X,Xi])]))
                    else: # debiasing not successful
                        E_hat_P_hat_n_Y_given_X.append(np.nan)
                        E_hat_P_hat_n_A_given_X.append(np.nan)
                    indices.append(str(Xi)) # set to string to allow for indexing in case X is boolean variable    
                E_hat_P_hat_n_Y_given_X = pd.DataFrame({'E_hat_P_hat_n_Y_given_X': E_hat_P_hat_n_Y_given_X})
                E_hat_P_hat_n_Y_given_X.index = indices
                E_hat_P_hat_n_A_given_X = pd.DataFrame({'E_hat_P_hat_n_A_given_X': E_hat_P_hat_n_A_given_X})
                E_hat_P_hat_n_A_given_X.index = indices
                
                # Merge
                E_hat = pd.concat([E_hat_P_tilde_m_Y_given_X,
                                   E_hat_P_tilde_m_A_given_X,
                                   E_hat_P_hat_n_Y_given_X,
                                   E_hat_P_hat_n_A_given_X], axis=1)  
                
                # Calculate difference between P_tilde_m and P_hat_n
                diff_E_hat = pd.DataFrame({f'{Y}': (E_hat['E_hat_P_tilde_m_Y_given_X'] - E_hat['E_hat_P_hat_n_Y_given_X']),
                                           f'{A}': (E_hat['E_hat_P_tilde_m_A_given_X'] - E_hat['E_hat_P_hat_n_A_given_X'])}).reset_index(names=X)
                diff_E_hat_reshape = pd.DataFrame(np.array(diff_E_hat.iloc[:,1:]).reshape(-1)).transpose()
                diff_E_hat_reshape.columns = ['diff_E_hat_' + column + '_given_' + diff_E_hat.columns[0] + '_' + x
                                              for x in diff_E_hat.iloc[:,0] 
                                              for column in diff_E_hat.columns[1:]]
                diff_E_hat_reshape[['dataset_name', 'run', 'n', 'generator']] = dataset_name, run, int(n_sample[len('n_'):]), '_'.join(dataset_name.split('_')[:-1])
                
                # Create meta data
                meta_diagnostics2 = pd.concat([meta_diagnostics2, diff_E_hat_reshape], axis=0)
                data[n_sample][run]['meta_diagnostics2'] = meta_diagnostics2
            
    return data

def create_dummies(data: pd.DataFrame):
    """
    Create dummy variables from categorical variables.
    """
    
    for n_sample in data:
        
        for run in data[n_sample]:
            
            for dataset_name in [name for name in data[n_sample][run] if all(substring not in name for substring in ['meta', 'bias', 'expected'])]:
                dataset = data[n_sample][run][dataset_name]
                dataset['stage'] = pd.Categorical(dataset['stage'], categories=['I', 'II', 'III', 'IV'], ordered=True) # define all theoretical categories
                dataset_dummies = pd.get_dummies(dataset) # create dummies
                dataset['stage'] = pd.Categorical(dataset['stage'], categories=np.unique(dataset['stage']), ordered=True) # make stage ordinal for only observed categories again (so stage can be used as outcome in polr model)
                data[n_sample][run][dataset_name] = dataset.merge(dataset_dummies) # merge dummies
    
    return data

def calculate_estimates(data: pd.DataFrame, cv_folds=0):
    """
    Calculate estimates (store in meta data per n_sample and per Monte Carlo run).
    """
    
    for n_sample in data:
        
        for run in data[n_sample]:
            
            # Model-based sample mean and SE
            for var in ['age']:
                for estimator in ['mean', 'mean_se']:
                    data[n_sample][run]['meta_data'][var + '_' + estimator] = estimate_estimator(
                        data=data[n_sample][run], var=var, estimator=estimator)
                    
            # Debiased sample mean and influence curve SE
            for var in ['age']:
                estimator = 'meanic'
                data[n_sample][run]['meta_data']['_'.join([var,estimator])], data[n_sample][run]['meta_data']['_'.join([var,estimator,'se'])] = estimate_estimator(
                    data=data[n_sample][run], var=var, estimator=estimator, cv_folds=cv_folds)
            
            # Model-based OLS coefficient and SE
            for var in [('bp', 'stage_II', 'stage_III', 'stage_IV', 'therapy')]: # blood pressure is outcome
                estimator = 'ols'
                tmp = np.array(estimate_estimator(data=data[n_sample][run], var=var, estimator=estimator)).transpose() # calculate regression coefficients
                columns = ['_'.join([var[0],y,x]) for x, y in list(product([estimator, estimator + '_se'], var[1:]))] # define names for regression coefficients
                for i in range(len(columns)):
                    data[n_sample][run]['meta_data'][columns[i]] = list(tmp[i]) # assign corresponding names and values
                    
            # Debiased regression estimator and influence curve SE
            for var in [('bp', 'therapy', 'stage')]: # blood pressure is outcome, therapy is exposure, stage is covariate 
                estimator = 'olsic'
                data[n_sample][run]['meta_data']['_'.join([var[0],var[1],estimator])], data[n_sample][run]['meta_data']['_'.join([var[0],var[1],estimator,'se'])] = estimate_estimator(
                    data=data[n_sample][run], var=var, estimator=estimator, cv_folds=cv_folds)
    
    return data

def combine_metadata(data: pd.DataFrame, target: str='meta_data', export: bool=False, sim_dir: str=None):
    """
    Combine all meta data over n_sample and Monte Carlo runs.
    """
    
    data[target] = pd.DataFrame({})
    for n_sample in data:
        if 'meta' in n_sample: # if overall meta_data or meta_diagnostics were already created by previously calling combine_metadata() function
            continue
        for run in data[n_sample]:
            data[target] = pd.concat([data[target],
                                      data[n_sample][run][target]],
                                     ignore_index=True)
    
    if export:
        data[target].to_csv(sim_dir + f'{target}.csv', index=False) # export meta_data to .csv
    
    return data

def sanity_check(data: pd.DataFrame):
    """
    Calculate sanity checks.
    """
    
    data['meta_data']['sanity_common_rows_proportion'] = data['meta_data'].apply(
        lambda i: common_rows_proportion(data['n_' + str(i['n'])][i['run'].replace(' ', '_')]['original_data'],
                                         data['n_' + str(i['n'])][i['run'].replace(' ', '_')][i['dataset_name']]), axis=1)
    
    data['meta_data']['sanity_IKLD'] = data['meta_data'].apply(
        lambda i: avg_inv_KLdiv(data['n_' + str(i['n'])][i['run'].replace(' ', '_')]['original_data'],
                                data['n_' + str(i['n'])][i['run'].replace(' ', '_')][i['dataset_name']]), axis=1)
    
    return data

def estimates_check(meta_data: pd.DataFrame):
    """
    Check non-estimable estimates (too large or too small SE).
    """
    
    select_se = [column for column in meta_data.columns if column[-3:]=='_se'] # select columns with '_se' suffix
    
    for var_se in select_se:
        
        # Very small SE: set estimate and SE to np.nan
        meta_data[var_se[:-len('_se')]].mask(meta_data[var_se] < 1e-10, np.nan, inplace=True) 
        meta_data[var_se].mask(meta_data[var_se] < 1e-10, np.nan, inplace=True)
            
        # Very large SE: set estimate and SE to np.nan
        meta_data[var_se[:-len('_se')]].mask(meta_data[var_se] > 1e2, np.nan, inplace=True)
        meta_data[var_se].mask(meta_data[var_se] > 1e2, np.nan, inplace=True)
        
    return meta_data

def inferential_utility(meta_data: pd.DataFrame, ground_truth):
    """
    Calculate inferential utility metrics.
    """
    
    # Sample mean and OLS regression coefficient
    for estimator in ['age_mean',
                      'age_meanic',
                      'bp_therapy_ols',
                      'bp_therapy_olsic']:
        
        # Bias of population parameter
        meta_data[estimator + '_bias'] = meta_data.apply(
            lambda i: i[estimator] - ground_truth[estimator], axis=1) # population parameter

        # Coverage of population parameter
        meta_data[estimator + '_coverage'] = meta_data.apply(
            lambda i:
            CI_coverage(estimate=i[estimator],
                        se=i[estimator + '_se'],
                        ground_truth=ground_truth[estimator], # population parameter
                        distribution='standardnormal' if 'ic' in estimator else 't', # IC: standard normal, model-based: t
                        df=i['n']-1 if 'mean' in estimator else i['n']-5, # mean: df=n-1, OLS: df=n_sample-n_parameters
                        quantile=0.975), axis=1) # naive SE
        meta_data[estimator + '_coverage_corrected'] = meta_data.apply(
            lambda i:
            CI_coverage(estimate=i[estimator],
                        se=i[estimator + '_se'],
                        ground_truth=ground_truth[estimator], # population parameter
                        se_correct_factor=np.sqrt(2) if i['dataset_name']!='original_data' else 1, # corrected SE
                        distribution='standardnormal' if 'ic' in estimator else 't', # IC: standard normal, model-based: t
                        df=i['n']-1 if 'mean' in estimator else i['n']-5, # mean: df=n-1, OLS: df=n_sample-n_parameters
                        quantile=0.975), axis=1)
        
        # Rejection of population parameter (NHST type 1 error)
        meta_data[estimator + '_NHST_type1'] = meta_data.apply(
            lambda i:
            not CI_coverage(estimate=i[estimator],
                            se=i[estimator + '_se'],
                            ground_truth=ground_truth[estimator], # null hypothesis: mu = ground_truth
                            distribution='standardnormal' if 'ic' in estimator else 't', # IC: standard normal, model-based: t
                            df=i['n']-1 if 'mean' in estimator else i['n']-5,  # mean: df=n-1, OLS: df=n_sample-n_parameters
                            quantile=0.975), axis=1) # naive SE
        meta_data[estimator + '_NHST_type1_corrected'] = meta_data.apply(
            lambda i:
            not CI_coverage(estimate=i[estimator],
                            se=i[estimator + '_se'],
                            ground_truth=ground_truth[estimator], # null hypothesis: mu = ground_truth
                            se_correct_factor=np.sqrt(2) if i['dataset_name']!='original_data' else 1, # corrected SE
                            distribution='standardnormal' if 'ic' in estimator else 't', # IC: standard normal, model-based: t
                            df=i['n']-1 if 'mean' in estimator else i['n']-5, # mean: df=n-1, OLS: df=n_sample-n_parameters
                            quantile=0.975), axis=1)
        
        # Non-rejection of small effect (NHST type 2 error)
        meta_data[estimator + '_NHST_type2'] = meta_data.apply(
            lambda i:
            CI_coverage(estimate=i[estimator],
                        se=i[estimator + '_se'],
                        ground_truth=ground_truth[estimator]*0.98, # null hypothesis: mu = 0.98 * ground_truth
                        distribution='standardnormal' if 'ic' in estimator else 't', # IC: standard normal, model-based: t
                        df=i['n']-1 if 'mean' in estimator else i['n']-5, # mean: df=n-1, OLS: df=n_sample-n_parameters
                        quantile=0.975), axis=1) # naive SE
        meta_data[estimator + '_NHST_type2_corrected'] = meta_data.apply(
            lambda i:
            CI_coverage(estimate=i[estimator],
                        se=i[estimator + '_se'],
                        ground_truth=ground_truth[estimator]*0.98, # null hypothesis: mu = 0.98 * ground_truth
                        se_correct_factor=np.sqrt(2) if i['dataset_name']!='original_data' else 1, # corrected SE
                        distribution='standardnormal' if 'ic' in estimator else 't', # IC: standard normal, model-based: t
                        df=i['n']-1 if 'mean' in estimator else i['n']-5, # mean: df=n-1, OLS: df=n_sample-n_parameters
                        quantile=0.975), axis=1)
            
    return meta_data

if __name__ == "__main__":

    # Presets
    n_samples = [50,160,500,1600,5000] # number of observations per original data set
    n_runs = 250 # number of Monte Carlo runs per number of observations
    sim_dir = 'simulation_study1/' # output of simulations
    cv_folds = 5 # cross-fitting of estimator and influence curve (set to 0 if no cross-fitting needed)
    start_time = time()
    
    # Data preparation
    data = load_data(n_samples, n_runs, sim_dir) # load data
    data = create_metadata(data, export=False) # create (single) meta_data per n_sample and Monte Carlo run
    data = create_metadiagnostics(data, export=False) # create (single) meta_diagnostics (containing biases and P_hat_n nuisance parameters) per n_sample and Monte Carlo run
    data = create_metadiagnostics2(data, export=False) # create (single) meta_diagnostics2 (containing L2 loss between P_tilde_m and P_hat_n) per n_sample and Monte Carlo run
    data = create_dummies(data) # create dummy variables from categorical variables (needed for estimating regression coefficients)
    
    # Calculate estimates per data set
    data = calculate_estimates(data, cv_folds)

    # Combine (overall) meta data over n_sample and Monte Carlo runs 
    data = combine_metadata(data, target='meta_data', export=False) # export later after calculating inferential utility metrics
    data = combine_metadata(data, target='meta_diagnostics', export=True, sim_dir=sim_dir) # export meta_diagnostics to .csv
    data = combine_metadata(data, target='meta_diagnostics2', export=True, sim_dir=sim_dir) # export meta_diagnostics2 to .csv

    # Calculate sanity checks
    data = sanity_check(data)
    
    # Only go further with meta_data (to reduce RAM memory)
    meta_data = data['meta_data']
    del data
    
    # Check non-estimable estimates (set to np.nan if too large or too small SE)
    meta_data = estimates_check(meta_data)
    
    # Calculate inferential utility metrics
    data_gt, _ = ground_truth()
    meta_data = inferential_utility(meta_data, ground_truth=data_gt)

    # Save to file
    meta_data.to_csv(sim_dir + 'meta_data.csv', index=False) # export meta_data to .csv
    
    # Print run time
    print(f'Total run time: {(time()-start_time):.3f} seconds')