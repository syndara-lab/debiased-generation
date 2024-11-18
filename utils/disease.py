"""Simulation of toy data"""

import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm
from statsmodels.api import OLS

def sample_disease(nsamples, seed=2024) -> pd.DataFrame:
    """
    Sample from imaginary disease
    """
    
    # Fix seed
    rng = np.random.default_rng(seed)
    
    # Age (normal distribution)
    age = rng.normal(loc=50, scale=10, size=nsamples)
    
    # Stage (proportional odds linear regression model) 
    stage_tags = ['I', 'II', 'III', 'IV']
    stage_intercepts = [2, 3, 4] # intercepts of stages are spaced evenly
    stage_beta_age = -0.05 # a negative value for beta means that with increasing values for x, the odds increase of being more than a given value k -> easier to exceed intercept 
    stage_logodds_1, stage_logodds_2, stage_logodds_3 = np.array(stage_intercepts).reshape(len(stage_intercepts),1) + np.vstack([stage_beta_age*age]*len(stage_intercepts)) # get logodds from age 
    stage_cumprob_1, stage_cumprob_2, stage_cumprob_3 = expit([stage_logodds_1, stage_logodds_2, stage_logodds_3]) # cumulative probability of exceeding each of the intercepts
    stage_probs = np.stack((stage_cumprob_1, stage_cumprob_2-stage_cumprob_1, stage_cumprob_3-stage_cumprob_2, 1-stage_cumprob_3), axis=1) # transform cumulative probability of exceeding intercept into probability of being in a certain stage, shape (nsamples, 4)
    stage = np.array([rng.choice(stage_tags, size=1, p=stage_prob) for stage_prob in stage_probs]).flatten() # sample from stage probabilities (categorical distribution)
    
    # Therapy (binary)
    p_therapy = 0.5 # 50% chance of receiving therapy
    therapy = rng.choice([False, True], size=nsamples, p=[1-p_therapy, p_therapy])
    
    # Blood pressure (normal distribution)
    bp_intercept = 120 # blood pressure in reference category
    bp_beta_stage = [0, 10, 20, 30] # increasing blood pressure with stage
    stage_tag_to_bp_beta = np.vectorize(dict(zip(stage_tags, bp_beta_stage)).get, otypes=[float]) # map stage tag to beta value
    bp_beta_therapy = -20 # decreasing blood pressure with therapy
    bp_betas = bp_intercept + stage_tag_to_bp_beta(stage) + bp_beta_therapy*therapy
    bp = rng.normal(loc=bp_betas, scale=10, size=nsamples)
    
    # Aggregate in dataframe
    data = pd.DataFrame({'age': age, 'stage': stage, 'therapy': therapy, 'bp': bp})
    
    return data

def ground_truth(nsamples=1000000) -> dict:
    """
    Population parameters
    """
    
    # Use large sample estimate to numerically approximate population parameter (for analytically intractable estimators)
    # draw large sample
    large_sample = sample_disease(nsamples, seed=2024)
    large_sample['stage'] = pd.Categorical(large_sample['stage'], categories=['I', 'II', 'III', 'IV'], ordered=True) # define all theoretical categories 
    large_sample_dummies = pd.get_dummies(large_sample) # to use 'stage' dummies as predictor in OLS
    large_sample = large_sample.merge(large_sample_dummies) # merge dummies
    # fit regression models
    large_sample_OLS = OLS(large_sample['bp'], sm.add_constant(large_sample[['stage_II', 'stage_III', 'stage_IV', 'therapy']].astype(float))).fit()

    # Population parameter
    ground_truth = {'age_mean': 50,
                    'age_meanic': 50,
                    'age_sd': 10,
                    'bp_therapy_ols': -20,
                    'bp_therapy_olsic': -20}
    
    # Rescale empirical SE with asymptotic variance (constant c) for plotting
    # SE = c*n^{-a} <-> log(SE) = log(c)-a*log(n) <-> log(SE/c) = -a*log(n)
    unit_rescale = {'age_mean': ground_truth['age_sd'],
                    'age_meanic': ground_truth['age_sd'],
                    'bp_therapy_ols': np.sqrt(large_sample_OLS.cov_params().loc['therapy', 'therapy'])*np.sqrt(nsamples),
                    'bp_therapy_olsic': np.sqrt(large_sample_OLS.cov_params().loc['therapy', 'therapy'])*np.sqrt(nsamples)}
    
    return ground_truth, unit_rescale

if __name__ == "__main__":
    
    # Presets
    n_samples = 20
    seed = 2024
    
    # Sample toy data
    data = sample_disease(n_samples, seed=seed)
    print('Imaginary disease (original data)')
    print('\nhead\n', data.head(10))
    print('\ndtypes\n', data.dtypes)
    print('\ndescriptives\n', data.describe(include='all')) 
    
    # Population parameters and rescaling constants
    data_gt, data_rescale = ground_truth(10000)
    print('\npopulation parameters\n', data_gt)
    print('\nasymptotic variance\n', data_rescale)