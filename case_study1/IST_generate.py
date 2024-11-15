"""Case study 1: IST original data and generate synthetic version(s) targeted to estimand of interest"""
""" Proportion of death """

import numpy as np
import pandas as pd
import os
import warnings
from time import time
from sklearn.model_selection import KFold
import statsmodels.api as sm
from statsmodels.api import OLS
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
import torch
import random
import sys
sys.path.append("..")
from utils.custom_ctgan import CTGAN
from utils.custom_tvae import TVAE


def HPC_job():
    # Convert the first command-line argument to an integer (the seed)
    arrayID = int(sys.argv[1])
    return arrayID

def binary_mapping_outcome(interest):
    if interest == 'death':
        binary_mapping = {
            0: None,  # None will facilitate the drop of these entries later
            1: 1,
            2: 0,
            3: 0,
            4: 0,
            9: None
        }
    elif interest == 'dependent':
        binary_mapping = {
            0: None,  # None will facilitate the drop of these entries later
            1: 0,
            2: 1,
            3: 0,
            4: 0,
            9: None
        }
    elif interest == 'not_recovered':
        binary_mapping = {
            0: None,  # None will facilitate the drop of these entries later
            1: 0,
            2: 0,
            3: 1,
            4: 0,
            9: None
        }
    elif interest == 'recovered':
        binary_mapping = {
            0: None,  # None will facilitate the drop of these entries later
            1: 0,
            2: 0,
            3: 0,
            4: 1,
            9: None
        }
   
    return binary_mapping

def multi_mapping_outcome():
    multi_mapping = {
        0: None,  # None will facilitate the drop of these entries later
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        9: None
    }
    return multi_mapping

def preproces_IST_heparin(IST_gt):
    """
    Pre-process IST dataset
    """    
    # Heparin
    category_mapping = {
        'N': 'No Heparin',
        'L': 'Heparin',
        'M': 'Heparin',
        'H': 'Heparin'
    }
    IST_gt['RXHEP'] = IST_gt['RXHEP'].replace(category_mapping).astype('category')
    IST_gt['RXHEP'] = pd.Categorical(IST_gt['RXHEP'],
                                     categories=['No Heparin', 'Heparin'],
                                     ordered=False)
    # Ensure RXHEP is treated as an object
    IST_gt['RXHEP'] = IST_gt['RXHEP'].astype(object)

    # Apply the mapping directly to OCCODE and drop missing values
    IST_gt['OCCODE'] = IST_gt['OCCODE'].map(multi_mapping_outcome()).astype('Int64')  # Use 'Int64' to allow NaN handling
    IST_gt.dropna(subset=['OCCODE'], inplace=True)  # Drop rows where OCCODE is None (previously missing)

    # Ensure OCCODE is treated as an integer column if not already understood as such
    IST_gt['OCCODE'] = IST_gt['OCCODE'].astype(object)
    print(f'Data is loaded and preprocessed') 
    
    return IST_gt
    
def preproces_IST_aspirin(IST_gt):
    """
    Pre-process IST dataset
    """    
    # Aspirin
    category_mapping = {
        'N': 'No aspirin',
        'Y': 'Aspirin'
    }
    # Apply the updated category mapping
    IST_gt['RXASP'] = IST_gt['RXASP'].replace(category_mapping).astype('category')
    # Redefine the categories to combine low and medium heparin
    IST_gt['RXASP'] = pd.Categorical(IST_gt['RXASP'],
                                     categories =['Aspirin', 'No aspirin'],
                                     ordered=False)
    # Ensure RXASP is treated as an object
    IST_gt['RXASP'] = IST_gt['RXASP'].astype(object)
    
    # Apply the mapping directly to OCCODE and drop missing values
    IST_gt['OCCODE'] = IST_gt['OCCODE'].map(multi_mapping_outcome()).astype('Int64')  # Use 'Int64' to allow NaN handling
    IST_gt.dropna(subset=['OCCODE'], inplace=True)  # Drop rows where OCCODE is None (previously missing)

    # Ensure OCCODE is treated as an object column if not already understood as such
    IST_gt['OCCODE'] = IST_gt['OCCODE'].astype(object)
    print(f'Data is loaded and preprocessed') 
    
    return IST_gt
    
def generative_models(n_sample: int=200):
    """
    Specify generative models used in simulation study
    """
       
    # Define generative models (assign synthcity Plugins() to synthcity_models object prior to calling this function)
    models = [synthcity_models.get('ctgan'), # default hyperparameters
              CTGAN(), # default
              synthcity_models.get('tvae'), # default
              TVAE()] # default hyperparameters
    
    return models   

def train_model(original_data: pd.DataFrame, generative_model: any, discrete_columns: list[str]=None, seed: int=2024): 
    """
    Wrapper function for training generative model 
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
    Wrapper function for sampling synthetic data
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

   
def targeting_mean(original_sample, generative_model, target_sample, variable: str, group: str):
    """
    Debiasing of sample mean.
    """
    # 1. Generate very large synthetic sample (that is different from target_sample, so use different seed)
    synthetic_sample = sample_synthetic(generative_model, m=int(1e+5), seed=2024)
    # Split this up according to the group you are looking at
    synthetic_sample_group = synthetic_sample[synthetic_sample['RXASP'] == group[0]].copy() # take first element from the group argument since group will always be one element
    # create dummy variables and force all dummies to be in there!
    possible_values = [1, 2, 3, 4]
    dummies_ss = pd.get_dummies(synthetic_sample_group['OCCODE'], columns=possible_values, prefix='OCCODE_dummy')
    # Add dummy columns for any missing categories
    for value in possible_values:
        if f'OCCODE_dummy_{value}' not in dummies_ss.columns:
            dummies_ss[f'OCCODE_dummy_{value}'] = 0
    # Reorder columns
    dummies_ss = dummies_ss[[f'OCCODE_dummy_{value}' for value in possible_values]]
    dummies_ss = dummies_ss.astype('object')
    synthetic_sample_group = synthetic_sample_group.join(dummies_ss)      
        
    bias_data = pd.DataFrame({})
    for dummy in variable:
        # 2. Calculate regularization bias
        O_bar = np.mean(original_sample[dummy])
        Y_bar = np.mean(synthetic_sample_group[dummy])
        bias = O_bar - Y_bar
    
        # 3. Debiasing of variable in the target_sample
        target_sample.loc[:, dummy] += bias
        
        # 4. Export biases
        bias_data[f'{dummy}_mean_bias'] = [bias]
    
    return target_sample, bias_data    
    
def case_study_IST(n_samples, n_runs, n_sets, case_dir, discrete_columns, seed_job):
    """
    Setup of simulation study
    """
    
    # OUTER loop over number of observations per original data set
    for n_sample in n_samples: 
        
        # Define output folder to save files
        n_dir = f'n_{n_sample}/'
        if not os.path.exists(case_dir + n_dir): 
            os.mkdir(case_dir + n_dir) # create n_dir folder if it does not exist yet
            
        # INNER loop over Monte Carlo runs
        for i in range(n_runs):            
            # Print progress
            print(f'[n={n_sample}, run={i+seed_job}] start')
            
            # Define output folder to save files
            out_dir = case_dir + n_dir + f'run_{i+seed_job}/'
            if not os.path.exists(out_dir): 
                os.mkdir(out_dir) # create out_dir folder if it does not exist yet
            
            # Simulate toy data -> not really applicable here, we just use the complete original dataset
            # take a sample from the "big" dataset, depending on n_sample loop you are in.
            # we also define a set seed, which is the monte carlo run iteration. (This is important when running parallel jobs)
            original_data = IST_gt.sample(n=n_sample, random_state=i+seed_job)
            original_data.to_csv(out_dir + 'original_data.csv', index=False) # export file 
            
            # Define generative models
            models = generative_models(n_sample) # batch_size hyperparameter depends on sample size

            # Train generative models
            for model in models:
                print(model.name())
                try:
                    train_model(original_data, model, discrete_columns, seed=2024) 
                    print(f'training [n={model.name()}, run={i+seed_job}] done')
                except Exception as e:
                    print(f'[n={n_sample}, run={i+seed_job}] error with fitting {model.name()}: {e}')
               
            # Now that training is done, create dummies for the original data (needs to be outside the loop, otherwise problems with overwriting)
            # For the original data
            # Force all dummies to be in there
            possible_values = [1, 2, 3, 4]
            dummies = pd.get_dummies(original_data['OCCODE'], columns=possible_values, prefix='OCCODE_dummy')
            # Add dummy columns for any missing categories
            for value in possible_values:
                if f'OCCODE_dummy_{value}' not in dummies.columns:
                    dummies[f'OCCODE_dummy_{value}'] = 0
            # Reorder columns
            dummies = dummies[[f'OCCODE_dummy_{value}' for value in possible_values]]
            dummies = dummies.astype('object')
            original_data = original_data.join(dummies)

            # Generate synthetic data 
            for model in models:
                print(model.name())
                for j in range(n_sets):
                    # untargeted synthetic dataset: no dummies yet!
                    try:
                        synthetic_data = sample_synthetic(model, m=n_sample, seed=j) # generated synthetic data = size of original data  
                        synthetic_data.to_csv(out_dir + model.name() + f'_{j}.csv', index=False)

                    except Exception as e:
                        print(f'[n={n_sample}, run={i+seed_job}] error with generating from {model.name()}: {e}')

                    # Targeted synthetic dataset. Enter the dummies here!
                    # For the synthetic data
                    synthetic_data_dummies = synthetic_data.copy()
                    # Force all dummies to be in there
                    possible_values = [1, 2, 3, 4]
                    dummies = pd.get_dummies(synthetic_data_dummies['OCCODE'], columns=possible_values, prefix='OCCODE_dummy')
                    # Add dummy columns for any missing categories
                    for value in possible_values:
                        if f'OCCODE_dummy_{value}' not in dummies.columns:
                            dummies[f'OCCODE_dummy_{value}'] = 0
                    # Reorder columns
                    dummies = dummies[[f'OCCODE_dummy_{value}' for value in possible_values]]
                    dummies = dummies.astype('object')
                    synthetic_data_dummies = synthetic_data_dummies.join(dummies)
                    
                    # Targeted synthetic dataset for Aspirin
                    targeted_synthetic_data_asp = synthetic_data_dummies.copy()
                    try:
                        targeted_synthetic_data_asp, bias_data_asp = targeting_mean(original_sample=original_data[original_data['RXASP'] == 'Aspirin'].copy(),
                                                                                    generative_model=model, 
                                                                                    target_sample=targeted_synthetic_data_asp[targeted_synthetic_data_asp['RXASP'] == 'Aspirin'].copy(),
                                                                                    variable=['OCCODE_dummy_1', 'OCCODE_dummy_2', 'OCCODE_dummy_3', 'OCCODE_dummy_4'], # debiasing of OCCODE, all dummies
                                                                                    group = ['Aspirin'])
                    except Exception as e:
                        print(f'[n={n_sample}, run={i+seed_job}] error with targeting aspirin from {model.name()}: {e}')
                    
                    
                    # Targeted synthetic dataset for no Aspirin
                    targeted_synthetic_data_noasp = synthetic_data_dummies.copy()
                    try:
                        targeted_synthetic_data_noasp, bias_data_noasp = targeting_mean(original_sample=original_data[original_data['RXASP'] == 'No aspirin'].copy(),
                                                                                        generative_model=model, 
                                                                                        target_sample=targeted_synthetic_data_noasp[targeted_synthetic_data_noasp['RXASP'] == 'No aspirin'].copy(),
                                                                                        variable=['OCCODE_dummy_1', 'OCCODE_dummy_2', 'OCCODE_dummy_3', 'OCCODE_dummy_4'], # debiasing of OCCODE, all dummies
                                                                                        group = ['No aspirin'])  
                    except Exception as e:
                        print(f'[n={n_sample}, run={i+seed_job}] error with targeting no aspirin from {model.name()}: {e}')
                    
                    
                    # export files
                    targeted_synthetic_data_asp.to_csv(out_dir + model.name() + f'Asp_targeted_{j}.csv', index=False)
                    bias_data_asp.to_csv(out_dir + model.name() + f'Asp_{j}_bias.csv', index=False)
                    targeted_synthetic_data_noasp.to_csv(out_dir + model.name() + f'No_Asp_targeted_{j}.csv', index=False)
                    bias_data_noasp.to_csv(out_dir + model.name() + f'No_Asp_{j}_bias.csv', index=False)

                        
                        
            
if __name__ == "__main__":
    # If you want to run parallel jobs, use the code below
    # take  $PBS_ARRAYID number to define monte carlo run
    # seed_job = HPC_job()  
    seed_job = 0 # when you don't run parallel jobs
    
    # Presets 
    n_sample_original = 19435
    n_samples = [50, 160, 500, 1600, 5000] # number of observations per original dataset
    n_runs = 100 # number of Monte Carlo runs per number of observations, in case of parallel jobs, put this on 1 (because you then use multiple jobarrays)
    n_sets = 1 # number of synthetic datasets generated per generative model
    
    case_dir = 'case_study_IST_data/' # output of simulations
    if not os.path.exists(case_dir): 
        os.mkdir(case_dir) # create case_dir folder if it does not exist yet
        
    # Load IST dataset (for this case study, we also use the pilot observations)
    features = ['RXASP',
                'OCCODE']
    IST_gt = pd.read_csv(f'{case_dir}IST_dataset_orig.csv',engine='python', usecols = features) # here you need to specify the path and file name where you stored the original dataset
     
    ### Pre-processing. 
    IST_gt = preproces_IST_aspirin(IST_gt) 
    n_sample_original = 19285 # since deletion missing outcome-subjects
    
    discrete_columns = ['RXASP', 'OCCODE'] # columns with discrete values

    # Disable tqdm as default setting (used internally in synthcity plugins)
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # Load synthcity plugins 
    synthcity_models = Plugins()
    
    # Run simulation study
    start_time = time()
    case_study_IST(n_samples, n_runs, n_sets, case_dir, discrete_columns, seed_job)

    print(f'Total run time: {(time()-start_time):.3f} seconds')
    
                            