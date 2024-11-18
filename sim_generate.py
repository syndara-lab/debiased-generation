"""Simulation study 1: sample original data and generate synthetic version(s) debiased w.r.t. target parameter"""

import numpy as np
import pandas as pd
import os
import warnings
from time import time
from utils.disease import sample_disease
from utils.custom_ctgan import CTGAN
from utils.custom_tvae import TVAE
from sklearn.model_selection import KFold
import statsmodels.api as sm
from statsmodels.api import OLS
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
import torch
import random
import sys

def generative_models():
    """
    Specify generative models used in simulation study.
    """
       
    # Define generative models (assign synthcity Plugins() to synthcity_models object prior to calling this function)
    models = [
        synthcity_models.get('ctgan'), # default hyperparameters
        CTGAN(), # default hyperparameters
        synthcity_models.get('tvae'), # default hyperparameters
        TVAE() # default hyperparameters
    ] 
    
    return models   

def train_model(original_data: pd.DataFrame, generative_model: any, discrete_columns: list[str]=None, seed: int=2024): 
    """
    Wrapper function for training generative model.
    """
    
    # Fix seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Use custom plugins (adapted from SDV)
    if 'custom' in generative_model.name():  
        
        with warnings.catch_warnings(): 
            # ignore warning on line 132 of rdt/transformers/base.py: 
            # FutureWarning: Future versions of RDT will not support the 'model_missing_values' parameter. 
            # Please switch to using the 'missing_value_generation' parameter to select your strategy.
            warnings.filterwarnings("ignore", lineno=132)
            generative_model.fit(original_data, discrete_columns=discrete_columns, seed=seed)
            
    # Use synthcity plugins
    else:
        loader = GenericDataLoader(original_data, sensitive_features=list(original_data.columns))
        generative_model.fit(loader) # seed is fixed within synthcity (random_state=0)
        
    return generative_model

def sample_synthetic(generative_model: any, m: int=1, seed: int=2024) -> pd.DataFrame:
    """
    Wrapper function for sampling synthetic data.
    """
    
    # Fix seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Use custom plugins (adapted from SDV)
    if 'custom' in generative_model.name():
        synthetic_data = generative_model.sample(n=m, seed=seed) # use condition_column=X, condition_value=Xi arguments for conditional sampling
    
    # Use synthcity plugins
    else:
        synthetic_data = generative_model.generate(count=m, random_state=seed).dataframe()
    
    return synthetic_data   

            
def E_hat_Y_and_A_given_X(condition_sample: pd.DataFrame, generative_model: any, 
                          Y: str=None, A: str=None, X: str=None) -> list:
    """
    Generate a very large synthetic sample (that is different from the target sample, so use different seed) of A and Y given observed level Xi.
    """

    # Calculate mean Y and mean A for each level of X
    synthetic_sample = sample_synthetic(generative_model, m=int(1e+5), seed=2024)
    E_hat_Y_given_X = pd.DataFrame(synthetic_sample.groupby(X)[Y].mean())
    E_hat_A_given_X = pd.DataFrame(synthetic_sample.groupby(X)[A].mean())
    E_hat_Y_given_X.index = E_hat_Y_given_X.index.astype(str) # set to string to allow for indexing in case X is boolean variable
    E_hat_A_given_X.index = E_hat_A_given_X.index.astype(str) # set to string to allow for indexing in case X is boolean variable
    
    # Map Xi to E_hat_Y_given_X and to E_hat_A_given_X
    E_hat_Y_given_Xi = E_hat_Y_given_X.loc[condition_sample[X].astype(str), Y].to_numpy() # set to string to allow for indexing in case X is boolean variable
    E_hat_A_given_Xi = E_hat_A_given_X.loc[condition_sample[X].astype(str), A].to_numpy() # set to string to allow for indexing in case X is boolean variable

    return E_hat_Y_given_Xi, E_hat_A_given_Xi

def bias_estimand(original_sample: pd.DataFrame, generative_model: any,
                  variable: str=None, variable_bias: list[float]=[0], Y: str=None, Y_bias: list[float]=[0], A: str=None, X: str=None, estimand: str=None) -> float:
    """
    Calculating bias term of sample mean and sample OLS regression coefficient.
    """
        
    if estimand == 'mean':
        """
        Use variable argument.
        """
                    
        # 1. Generate very large synthetic sample (that is different from target_sample, so use different seed)
        synthetic_sample = sample_synthetic(generative_model, m=int(1e+5), seed=2024)
        synthetic_sample[variable] += np.sum(variable_bias) # for repeated debiasing (shift synthetic variable)

        # 2. Calculate regularization bias
        O_bar = np.mean(original_sample[variable])
        Y_bar = np.mean(synthetic_sample[variable]) 
        bias = O_bar - Y_bar
        
        return bias
    
    elif estimand == 'ols':
        """
        Use Y, A and X arguments.
        """
        # 1. Calculate E_hat_Y_given_Xi and E_hat_A_given_Xi
        E_hat_Y_given_Xi, E_hat_A_given_Xi = E_hat_Y_and_A_given_X(original_sample, generative_model, Y, A, X)
        #E_hat_Y_given_Xi += np.sum(Y_bias) * (original_sample[A] - E_hat_A_given_Xi) # not necessary for repeatead debiasing - debiasing does not influence predictions of generative model!
        
        # 2. Calculate theta_P_hat
        synthetic_sample = sample_synthetic(generative_model, m=int(1e+5), seed=2024)
        E_hat_Y_given_Xi_tilde, E_hat_A_given_X_tilde = E_hat_Y_and_A_given_X(synthetic_sample, generative_model, Y, A, X)
        synthetic_sample[Y] += np.sum(Y_bias)*(synthetic_sample[A] - E_hat_A_given_X_tilde) # for repeated debiasing (shift synthetic outcome)
        #E_hat_Y_given_Xi_tilde += np.sum(Y_bias) * (synthetic_sample[A] - E_hat_A_given_X_tilde) # not necessary for repeatead debiasing - debiasing does not influence predictions of generative model!
        theta_P_hat = np.mean((synthetic_sample[A] - E_hat_A_given_X_tilde) * (synthetic_sample[Y] - E_hat_Y_given_Xi_tilde))
        theta_P_hat /= np.mean((synthetic_sample[A] - E_hat_A_given_X_tilde)**2)
        
        # 3. Calculate regularization bias
        bias = np.mean((original_sample[A] - E_hat_A_given_Xi) * (original_sample[Y] - E_hat_Y_given_Xi - theta_P_hat * (original_sample[A] - E_hat_A_given_Xi)))
        bias /= np.mean((original_sample[A] - E_hat_A_given_Xi)**2)
        
        return bias, theta_P_hat

    else:
        
        bias = np.nan
     
        return bias   

def debiasing_cv(original_sample: pd.DataFrame, generative_model: any, target_sample: pd.DataFrame, cv_folds: int=0, n_steps: int=1,
                 variable: str=None, Y: str=None, A: str=None, X: str=None, discrete_columns: list[str]=None) -> pd.DataFrame:
    """
    Debiasing of sample mean and sample OLS regression coefficient.
    """
    
    # 1. Calculate regularization bias                          
    
    # Empty objects for storage
    samplemean_bias_per_fold = []       
    ols_bias_per_fold = []
    ols_thetaPhat_per_fold = []

    # Cross-fitting presets
    if cv_folds==0: 
        # Without K-fold cross-fitting
        loop = iter([(original_sample.index, original_sample.index)]) # original indices will be used for both (training_idx, val_idx)
    else: 
        # With K-fold cross-fitting
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=2024) 
        loop = kf.split(original_sample) # gives (training_idx, val_idx) per fold

    # Cross-fitting of bias term
    for k, (train_idx, val_idx) in enumerate(loop):
                
        # Split in train and validation sets
        train_data = original_sample.loc[train_idx, :] # training data (= in-fold original data)
        val_data = original_sample.loc[val_idx, :] # validation data (= out-of-fold original data)
           
        # Train (untargeted) generative submodel on in-fold original data
        if cv_folds==0: 
            # Without K-fold cross-fitting
            generative_submodel = generative_model # no need to retrain global model
        else: 
            # With K-fold cross-fitting
            generative_submodel = (CTGAN() if 'ctgan' in generative_model.name() else TVAE()) \
                if 'custom' in generative_model.name() else synthcity_models.get(generative_model.name())
            train_model(train_data, generative_submodel, discrete_columns, seed=2024)                                                         
        
        # Repeat debiasing step for n_steps
        samplemean_bias_per_step = [0]
        ols_bias_per_step = [0]
        ols_thetaPhat_per_step = []
        
        for step in range(n_steps):
                   
            # Calculate bias on out-of-fold original data
            if variable: # when variable argument is defined
                samplemean_bias = bias_estimand(original_sample=val_data, 
                                                generative_model=generative_submodel, 
                                                variable=variable,
                                                variable_bias=samplemean_bias_per_step[:(step+1)], # incorporate prior bias terms for repeated debiasing
                                                estimand='mean')
                samplemean_bias_per_step.append(samplemean_bias)
            if all([Y, A, X]): # when Y, A, X arguments are defined
                ols_bias, ols_thetaPhat = bias_estimand(original_sample=val_data,
                                                        generative_model=generative_submodel,
                                                        Y=Y, A=A, X=X,
                                                        Y_bias=ols_bias_per_step[:(step+1)], # incorporate prior bias for repeated debiasing
                                                        estimand='ols')
                ols_bias_per_step.append(ols_bias)
                ols_thetaPhat_per_step.append(ols_thetaPhat)
                
        # Sum out-of-fold bias over steps
        samplemean_bias_per_fold.append(np.sum(samplemean_bias_per_step))
        ols_bias_per_fold.append(np.sum(ols_bias_per_step))
        ols_thetaPhat_per_fold.append(ols_thetaPhat_per_step[-1]) # take latest theta_P_hat update (only used to export)
        
    # Average out-of-fold bias over folds
    average_samplemean_bias = np.mean(samplemean_bias_per_fold)
    average_ols_bias = np.mean(ols_bias_per_fold)
    average_ols_thetaPhat = np.mean(ols_thetaPhat_per_fold) # only used to export

    # 2. Debiasing of target_sample
    
    # Sample mean: by shifting variable with bias
    if variable: # when variable argument is defined
        target_sample[variable] += average_samplemean_bias
    
    # OLS coefficient: by adding b*{A_tilde − E(A|X_tilde)}
    if all([Y, A, X]): # when Y, A, X arguments are defined
        E_hat_Y_given_X_tilde, E_hat_A_given_X_tilde = E_hat_Y_and_A_given_X(target_sample, generative_model, Y, A, X)
        target_sample[Y] += average_ols_bias * (target_sample[A] - E_hat_A_given_X_tilde) 
    
    # 3. Export biases and expected values
    
    bias_data = pd.DataFrame({})
    expected_data = pd.DataFrame({})
    if variable: # when variable argument is defined
        bias_data[f'{variable}_mean_bias'] = [average_samplemean_bias]
    if all([Y, A, X]): # when Y, A, X arguments are defined
        bias_data[f'{Y}_{A}_ols_bias'] = [average_ols_bias]
        bias_data[f'{Y}_{A}_ols_thetaPhat'] = [average_ols_thetaPhat]
        expected_data = pd.DataFrame({f'{X}': target_sample[X], f'{Y}': E_hat_Y_given_X_tilde, f'{A}': E_hat_A_given_X_tilde}).drop_duplicates(ignore_index=True)
     
    return target_sample, bias_data, expected_data

def simulation_study1(n_samples, start_run, n_runs, n_sets, sim_dir, discrete_columns, cv_folds, n_steps):
    """
    Setup of simulation study.
    """
    
    # OUTER loop over number of observations per original data set
    for n_sample in n_samples: 
        
        # Define output folder to save files
        n_dir = f'n_{n_sample}/'
        if not os.path.exists(sim_dir + n_dir): 
            os.mkdir(sim_dir + n_dir) # create n_dir folder if it does not exist yet
            
        # INNER loop over Monte Carlo runs
        for i in range(start_run, start_run+n_runs):
                
            # Print progress
            print(f'[n={n_sample}, run={i}] start')
            
            # Define output folder to save files
            out_dir = sim_dir + n_dir + f'run_{i}/'
            if not os.path.exists(out_dir): 
                os.mkdir(out_dir) # create out_dir folder if it does not exist yet
            
            # Simulate toy data
            original_data = sample_disease(n_sample, seed=i)
            original_data.to_csv(out_dir + 'original_data.csv', index=False) # export file

            # Define generative models
            models = generative_models()
            
            # Train generative models
            for model in models:
                
                try:
                    train_model(original_data, model, discrete_columns, seed=2024)
                except Exception as e:
                    print(f'[n={n_sample}, run={i}] error with fitting {model.name()}: {e}')
           
            # Generate synthetic data
            for model in models:
                
                for j in range(n_sets):
                    
                    # Default synthetic dataset
                    try:
                        synthetic_data = sample_synthetic(model, m=n_sample, seed=j) # generated synthetic data = size of original data
                        synthetic_data.to_csv(out_dir + model.name() + f'_{j}.csv', index=False) # export file
                    except Exception as e:
                        print(f'[n={n_sample}, run={i}, set={j}] error with generating from {model.name()}: {e}')
                    
                    # Debiased synthetic dataset
                    try:
                        targeted_synthetic_data = synthetic_data.copy()
                        targeted_synthetic_data, bias_data, expected_data = debiasing_cv(original_sample=original_data,
                                                                                         generative_model=model,
                                                                                         target_sample=targeted_synthetic_data,
                                                                                         cv_folds=cv_folds,
                                                                                         n_steps=n_steps,
                                                                                         variable='age', # debiasing of age_mean
                                                                                         Y='bp', A='therapy', X='stage', # debiasing of bp_therapy_ols
                                                                                         discrete_columns=discrete_columns)
                        targeted_synthetic_data.to_csv(out_dir + model.name() + f'_targeted_{j}.csv', index=False) # export file
                        bias_data.to_csv(out_dir + model.name() + f'_{j}_bias.csv', index=False) # export file
                        expected_data.to_csv(out_dir + model.name() + f'_{j}_expected.csv', index=False) # export file
                    except Exception as e:
                        print(f'[n={n_sample}, run={i}, set={j}] error with debiasing {model.name()}: {e}')
                                   
if __name__ == "__main__":
       
    # Presets
    n_samples = [50,160,500,1600,5000] # number of observations per original dataset
    start_run = 0 # start index of Monte Carlo runs (0 when single job submission; int(sys.argv[1]) when parallel job submission)
    n_runs = 250 # number of Monte Carlo runs per number of observations
    n_sets = 1 # number of synthetic datasets generated per generative model
    
    sim_dir = 'simulation_study1/' # output of simulations
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir) # create sim_dir folder if it does not exist yet
        
    discrete_columns = ['stage', 'therapy'] # columns with discrete values (argument needed for SDV modules)
    
    cv_folds = 0 # cross-fitting for bias calculation (set to 0 if no sample splitting needed)
    n_steps = 1 # number of debiasing steps (one step suffices)

    # Disable tqdm as default setting (used internally in synthcity plugins)
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    
    # Load synthcity plugins 
    synthcity_models = Plugins()
    
    # Run simulation study
    start_time = time()
    simulation_study1(n_samples, start_run, n_runs, n_sets, sim_dir, discrete_columns, cv_folds, n_steps)
    print(f'Total run time: {(time()-start_time):.3f} seconds')