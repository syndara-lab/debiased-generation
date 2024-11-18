"""Case study 1: generate default and debiased synthetic version(s) from Adult Census Income dataset using TVAE"""

import pandas as pd
import numpy as np
import os
from time import time
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
import scipy.stats as ss
import statsmodels.api as sm
from statsmodels.api import OLS
import torch
import random
import sys
sys.path.append("..")
from sim_generate import debiasing_cv

def load_adult(data_dir):  
    """
    Load Adult Census Income dataset.
    """

    features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status',
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                'hours_per_week', 'native_country', 'income'] 
    adult = pd.concat([pd.read_csv(data_dir + 'adult.data', sep=r'\s*,\s*', engine='python', names=features, na_values='?'),
                       pd.read_csv(data_dir +'adult.test', skiprows=1, sep=r'\s*,\s*', engine='python', names=features, na_values='?')]) 
    adult.drop(['education_num'], axis=1, inplace=True) # drop redundant column
    adult.dropna(axis=0, how='any', inplace=True) # only retain complete cases
    adult['income'] = [income.replace('.', '') for income in adult['income']] # remove '.' suffix From test labels
    adult['income'] = list(map(lambda x: 1 if x=='>50K' else 0, adult['income'] )) # create dummy
    
    return adult

def generate_synthetic(adult, data_dir, n_samples):
    """
    Generate (default and debiased) synthetic version(s) from Adult Census Income dataset using TVAE.
    """
    
    for n_sample in n_samples:

        # Print progress
        print(f'[n={n_sample}] start generation')

        # Fix seed for reproducibility
        random.seed(2024)
        np.random.seed(2024)
        torch.manual_seed(2024)

        # Sample original data
        original_data = adult.sample(n=n_sample, replace=False, random_state=2024)
        loader = GenericDataLoader(adult, sensitive_features=list(adult.columns))
        original_data.to_csv(data_dir + f'adult_original_n_{n_sample}.csv', index=False) # export file

        # Train generative model
        generative_model = synthcity_models.get('tvae')
        generative_model.fit(loader)

        # Generate default synthetic dataset
        synthetic_data = generative_model.generate(count=1000000, random_state=2024).dataframe() # synthetic dataset of size m>>>n 
        synthetic_data.to_csv(data_dir + f'adult_synthetic_n_{n_sample}.csv', index=False) # export file

        # Debias synthetic dataset
        targeted_synthetic_data = synthetic_data.copy()
        targeted_synthetic_data, _, _ = debiasing_cv(original_sample=original_data,
                                                     generative_model=generative_model,
                                                     target_sample=targeted_synthetic_data,
                                                     cv_folds=0, # cross-fitting for bias calculation (set to 0 if no sample splitting needed)
                                                     n_steps=1, # number of debiasing steps (one step suffices)
                                                     variable='hours_per_week', # debiasing of mean
                                                     Y='age', A='income', X='sex') # debiasing of ols regression coefficient
        targeted_synthetic_data.to_csv(data_dir + f'adult_targeted_n_{n_sample}.csv', index=False) # export file
        
def calculate_estimates(data_dir, n_samples):
    """
    Calculate sample mean of 'hours_per_week' and OLS for effect 'income' on 'age' adjusted for 'sex'.
    """    
    
    # Create empty data structure for storage
    meta_data = {}
    meta_data['all'] = pd.DataFrame({})
       
    # OUTER loop over number of observations per original data set
    for n_sample in n_samples:

        # Print progress
        print(f'[n={n_sample}] start estimation')

        # Create empty data structure for storage
        meta_data[f'n_{n_sample}'] = pd.DataFrame({'n': np.repeat(n_sample, 3),
                                                   'dataset_type': ['original', 'synthetic', 'targeted']})
        samplemeans, samplemean_ll, samplemean_ul = [], [], []
        ols_coefs, ols_coef_ll, ols_coef_ul = [], [], []

        # INNER loop over dataset_types within n_sample
        for dataset_type in meta_data[f'n_{n_sample}']['dataset_type']:

            # Load data
            dataset = pd.read_csv(data_dir + f'adult_{dataset_type}_n_{n_sample}.csv')

            # Correction factor
            correction_factor = np.sqrt(1+dataset.shape[0]/n_sample) if dataset_type != 'original' else 1

            # Sample mean of hours_per_week
            samplemean = np.mean(dataset['hours_per_week'])
            samplemean_se = np.std(dataset['hours_per_week'])/np.sqrt(dataset.shape[0]-1)
            samplemean_q = ss.t.ppf(0.975, df=dataset.shape[0]-1)
            samplemeans.append(samplemean)
            samplemean_ll.append(samplemean - samplemean_q*correction_factor*samplemean_se)
            samplemean_ul.append(samplemean + samplemean_q*correction_factor*samplemean_se)

            # OLS for effect income on age adjusted for sex
            dataset_dummies = pd.get_dummies(dataset)
            ols_model = OLS(dataset_dummies['age'], sm.add_constant(dataset_dummies[['income','sex_Female']])).fit()
            income_index = ols_model.cov_params().index.get_indexer(['income'])
            ols_coef = float(ols_model.params.to_numpy()[income_index])
            ols_se = float(np.sqrt(ols_model.cov_params().to_numpy().diagonal()[income_index]))
            ols_q = ss.t.ppf(0.975, df=dataset.shape[0]-3)
            ols_coefs.append(ols_coef)
            ols_coef_ll.append(ols_coef - ols_q*correction_factor*ols_se)
            ols_coef_ul.append(ols_coef + ols_q*correction_factor*ols_se)

        # Store in meta_data per n_sample
        meta_data[f'n_{n_sample}']['samplemean'] = samplemeans
        meta_data[f'n_{n_sample}']['samplemean_ll'] = samplemean_ll
        meta_data[f'n_{n_sample}']['samplemean_ul'] = samplemean_ul
        meta_data[f'n_{n_sample}']['ols_coef'] = ols_coefs
        meta_data[f'n_{n_sample}']['ols_coef_ll'] = ols_coef_ll
        meta_data[f'n_{n_sample}']['ols_coef_ul'] = ols_coef_ul

        # Store in overall meta_data
        meta_data['all'] = pd.concat([meta_data['all'],  meta_data[f'n_{n_sample}']], axis=0)

    # Export overall meta_data
    meta_data['all'].to_csv(data_dir + 'meta_data.csv', index=False) # export file

if __name__ == "__main__":
        
    # Presets
    n_samples = np.geomspace(start=50, stop=45222, num=5, endpoint=True, dtype=int) # varying log-uniformly between 50 and 45222
    data_dir = 'data/'
    if not os.path.exists(data_dir): 
        os.mkdir(data_dir) # create data_dir folder if it does not exist yet
 
    # Disable tqdm as default setting (used internally in synthcity's plugins)
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    
    # Load synthcity plugins 
    synthcity_models = Plugins()
    
    # Run case study
    start_time = time()
    adult = load_adult(data_dir)
    generate_synthetic(adult, data_dir, n_samples)
    calculate_estimates(data_dir, n_samples)
    print(f'Total run time: {(time()-start_time):.3f} seconds')