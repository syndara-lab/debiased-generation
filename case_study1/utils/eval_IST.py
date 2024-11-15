import os
import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.stats import entropy
import statsmodels.api as sm
from statsmodels.api import OLS
import itertools
from itertools import product
import plotnine
from plotnine import *
from mizani.formatters import percent_format 
from mizani.transforms import trans
from sklearn.model_selection import KFold
from time import time


# eval functions for IST case study
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
    
def expand_grid(data_dict):
    """
    Expand grid
    """ 
    rows = product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def load_data(n_samples: int, n_runs: int, sim_dir: str):
    """
    Load source files containing original and synthetic datasets
    """

    # Create empty data structure for storage
    data = {}
    for n_sample in n_samples:
        data[f'n_{n_sample}'] = {}
        for i in range(n_runs):
            data[f'n_{n_sample}'][f'run_{i+1}'] = {}

    # Load data sets
    data_dir = [os.path.join(root, name) for root, dirs, files in os.walk(sim_dir) for name in files]
    for dir in data_dir:
        if '.ipynb_checkpoint' in dir:
            continue # skip dot files

        dir_structure = dir[len(sim_dir):].split('/') # remove sim_dir prefix in name and split directory by '/'
        try: # load datasets per n_sample and per Monte Carlo run
            # print(dir_structure)
            data[dir_structure[0]][dir_structure[1]][dir_structure[2][:-4]] = pd.read_csv(dir, engine='pyarrow') # also remove '.csv' suffix in name; use pyarrow engine for multithreading
        except Exception as e:
            print(f'File {dir} not stored in \'data\' object because file has aberrant directory structure')
            print('Expected files are empty here because we are not targeting OLS estimate.')

    # Remove empty simulations
    for n_sample in n_samples:
        for i in range(n_runs):
            if not data[f'n_{n_sample}'][f'run_{i+1}']:
                print(f'There was a problem with n_{n_sample},run_{i+1}')
                del data[f'n_{n_sample}'][f'run_{i+1}']

    return(data)

def create_dummiesdata(data: pd.DataFrame, list_generators: str=None):
    """
    Create dummies for original datasets and untargeted datasets 
    """
    if not list_generators:
      raise ValueError("Error: 'list_generators' cannot be empty!")

    for n_sample in data:

        for run in data[n_sample]:
            names = [name for name in data[n_sample][run] if all(substring not in name for substring in ['meta', 'bias', 'expected','targeted'])] # only want original and untargeted datasets!

            for name in names:
                possible_values = [1, 2, 3, 4]
                dummies = pd.get_dummies(data[n_sample][run][name]['OCCODE'], columns=possible_values, prefix='OCCODE_dummy')
                # Add dummy columns for any missing categories
                for value in possible_values:
                    if f'OCCODE_dummy_{value}' not in dummies.columns:
                        dummies[f'OCCODE_dummy_{value}'] = 0
                # Reorder columns
                dummies = dummies[[f'OCCODE_dummy_{value}' for value in possible_values]]
                dummies = dummies.astype('object')
                data[n_sample][run][name] = data[n_sample][run][name].join(dummies)

    return data

# Make sure all dummies are in the dataframes!
def check_all_dummies(data: pd.DataFrame):
    for n_sample in data:

        for run in data[n_sample]:
            names = [name for name in data[n_sample][run] if all(substring not in name for substring in ['meta', 'bias', 'expected'])] 

            for name in names:
                for var in ['OCCODE_dummy_1', 'OCCODE_dummy_2', 'OCCODE_dummy_3', 'OCCODE_dummy_4']:
                    try:
                        # Check if the variable is in the dataframe
                        if var not in data[n_sample][run][name].columns:
                            raise KeyError(f"{var} not found in dataframe columns.")
                            # add an extra column with all 0's
                            # data[n_sample][run][name][var] = 0
                    except KeyError as e:
                        print(e)  # Print the error message
                        # add an extra column with all 0's
                        data[n_sample][run][name][var] = 0
                        continue  # Skip the current iteration and continue with the next one
    return data

def create_metadata(data: pd.DataFrame, export: bool=False, sim_dir: str=None, list_generators: list=[]):
    """
    Create meta_data (or update if additional generative models have been added) per n_sample and per Monte Carlo run
    """
    if not list_generators:
      raise ValueError("Error: 'list_generators' cannot be empty!")

    for n_sample in data:

        for run in data[n_sample]:

            combined_list = [f"{generator}_untargeted" for generator in list_generators] + \
                            [f"{generator}_targeted" for generator in list_generators]
            row_names = ['Original_data'] + combined_list
            # Create an empty DataFrame
            meta_data = pd.DataFrame(index=row_names, columns=['run', 'n', 'generator', 
                                                               'Prop_Aspirin_Death', 'Prop_No_Aspirin_Death', 'Var_Prop_Aspirin_Death', 'Var_Prop_No_Aspirin_Death',
                                                               'Prop_Aspirin_Dependent', 'Prop_No_Aspirin_Dependent', 'Var_Prop_Aspirin_Dependent', 'Var_Prop_No_Aspirin_Dependent'])
            meta_data['n'] = np.repeat(n_sample[2:], len(row_names)) # just take the sample size, delete the 'n_'
            meta_data['run'] = np.repeat(run, len(row_names))
            meta_data['generator'] = row_names

            data[n_sample][run]['meta_data'] = meta_data # store meta data per simulation

            if export:
                meta_data.to_csv(sim_dir + n_sample + '/' + run + '/meta_data.csv', index=False) # export meta_data to .csv

    return data
      
def combine_metadata(data: pd.DataFrame, target: str='meta_data', export: bool=False, sim_dir: str=None):
    """
    Combine all meta data over n_sample and Monte Carlo runs
    """

    data[target] = pd.DataFrame({})
    for n_sample in data:
        if 'meta' in n_sample: # if overall meta_data or meta_diagnostics were already created by previously calling combine_metadata() function
            continue
        for run in data[n_sample]:
            data[target] = pd.concat([data[target],
                                      data[n_sample][run][target]],
                                     ignore_index=False)

    if export:
        data[target].to_csv(sim_dir + f'{target}.csv', index=False) # export meta_data to .csv

    return data

def detect_keywords_in_element(element, list_generators: list=[]):
    matched_keyword = None
    for keyword in list_generators:
        if keyword in element:
            if matched_keyword is None or len(keyword) > len(matched_keyword):
                matched_keyword = keyword
    return matched_keyword

def calculate_estimates(data: pd.DataFrame, cv_folds=0, list_generators: list=[]):
    """
    Calculate estimates (store in meta data per n_sample and per Monte Carlo run)
    """

    for n_sample in data:
        # print(n_sample)
        for run in data[n_sample]:
            # print(run)

            # Model-based sample mean and SE
            for var in ['OCCODE_dummy_1', 'OCCODE_dummy_2']: # I only want it for dead (dummy 1) and dependent (dummy 2) for now
                column_meta_data = 'Death' if var == 'OCCODE_dummy_1' else 'Dependent'
                for estimator in ['prop']:
                    names = [name for name in data[n_sample][run] if all(substring not in name for substring in ['meta', 'bias', 'expected'])]
                    total_targeted_list = [s for s in names if 'targeted' in s] # look whether it is a targeted dataset or not (then it is original or untargeted synthetic)
                    No_Asp_targeted_list = [s for s in total_targeted_list if 'No_Asp_targeted' in s] # is it data for group no aspirine
                    Asp_targeted_list = [s for s in total_targeted_list if s not in No_Asp_targeted_list] # or is it data for group aspirine
                    
                    for name in names:
                        # print(name)
                        if estimator == 'prop':
                            if name in No_Asp_targeted_list: # if targeted dataset, the dataset is already split by group, so easy calculation
                                result = detect_keywords_in_element(name, list_generators)
                                prop_tmp = np.mean(data[n_sample][run][name][var])
                                n_obs = len(data[n_sample][run][name][var])
                                if n_obs == 0:
                                    print(f"There were no observations generated for sample size {n_sample}, in {run} and generator {name}")
                                data[n_sample][run]['meta_data'].loc[(f"{result}_targeted"), (f"Prop_No_Aspirin_{column_meta_data}")] = np.mean(data[n_sample][run][name][var])
                                try:
                                    data[n_sample][run]['meta_data'].loc[(f"{result}_targeted"), (f"Var_Prop_No_Aspirin_{column_meta_data}")] = ((prop_tmp)*(1-prop_tmp))/n_obs
                                except Exception as e:
                                    print(f'[n={n_sample}, run={run}, generator = {name}] error in division because: {e}')
                            elif name in Asp_targeted_list: # if targeted dataset, the dataset is already split by group, so easy calculation
                                result = detect_keywords_in_element(name, list_generators)
                                prop_tmp = np.mean(data[n_sample][run][name][var])
                                n_obs = len(data[n_sample][run][name][var])
                                if n_obs == 0:
                                    print(f"There were no observations generated for sample size {n_sample}, in {run} and generator {name}")
                                data[n_sample][run]['meta_data'].loc[(f"{result}_targeted"), (f"Prop_Aspirin_{column_meta_data}")] = np.mean(data[n_sample][run][name][var])
                                try:
                                    data[n_sample][run]['meta_data'].loc[(f"{result}_targeted"), (f"Var_Prop_Aspirin_{column_meta_data}")] = ((prop_tmp)*(1-prop_tmp))/n_obs
                                except Exception as e:
                                    print(f'[n={n_sample}, run={run}, generator = {name}] error in division because: {e}')
                            elif name == 'original_data': # I first need to split the data by using the variable RXASP
                                data_tmp = data[n_sample][run][name]
                                data_split_no_asp = data_tmp[data_tmp['RXASP'] == 'No aspirin']
                                prop_tmp = np.mean(data_split_no_asp[var])
                                n_obs = len(data_split_no_asp[var])
                                if n_obs == 0:
                                    print(f"There were no observations generated for sample size {n_sample}, in {run} and generator {name}")
                                data[n_sample][run]['meta_data'].loc['Original_data', (f"Prop_No_Aspirin_{column_meta_data}")] = np.mean(data_split_no_asp[var])
                                try:
                                    data[n_sample][run]['meta_data'].loc['Original_data', (f"Var_Prop_No_Aspirin_{column_meta_data}")] = ((prop_tmp)*(1-prop_tmp))/n_obs
                                except Exception as e:
                                    print(f'[n={n_sample}, run={run}, generator = {name}] error in division because: {e}')
                                data_split_asp = data_tmp[data_tmp['RXASP'] == 'Aspirin']
                                prop_tmp = np.mean(data_split_asp[var])
                                n_obs = len(data_split_asp[var])
                                if n_obs == 0:
                                    print(f"There were no observations generated for sample size {n_sample}, in {run} and generator {name}")
                                data[n_sample][run]['meta_data'].loc['Original_data', (f"Prop_Aspirin_{column_meta_data}")] = np.mean(data_split_asp[var])
                                try:
                                    data[n_sample][run]['meta_data'].loc['Original_data', (f"Var_Prop_Aspirin_{column_meta_data}")] = ((prop_tmp)*(1-prop_tmp))/n_obs
                                except Exception as e:
                                  print(f'[n={n_sample}, run={run}, generator = {name}] error in division because: {e}')
                            else: # All others are untargeted synthetic datasets! I also first need to split the data by using the variable RXASP
                                result = detect_keywords_in_element(name, list_generators)
                                data_tmp_untargeted = data[n_sample][run][name]
                                data_split_no_asp_untargeted = data_tmp_untargeted[data_tmp_untargeted['RXASP'] == 'No aspirin']
                                prop_tmp = np.mean(data_split_no_asp_untargeted[var])
                                n_obs = len(data_split_no_asp_untargeted[var])
                                if n_obs == 0:
                                    print(f"There were no observations generated for sample size {n_sample}, in {run} and generator {name}")
                                data[n_sample][run]['meta_data'].loc[(f"{result}_untargeted"), (f"Prop_No_Aspirin_{column_meta_data}")] = np.mean(data_split_no_asp_untargeted[var])
                                try:
                                    data[n_sample][run]['meta_data'].loc[(f"{result}_untargeted"), (f"Var_Prop_No_Aspirin_{column_meta_data}")] = ((prop_tmp)*(1-prop_tmp))/n_obs
                                except Exception as e:
                                  print(f'[n={n_sample}, run={run}, generator = {name}] error in division because: {e}')
                                data_split_asp_untargeted = data_tmp_untargeted[data_tmp_untargeted['RXASP'] == 'Aspirin']
                                prop_tmp = np.mean(data_split_asp_untargeted[var])
                                n_obs = len(data_split_asp_untargeted[var])
                                if n_obs == 0:
                                    print(f"There were no observations generated for sample size {n_sample}, in {run} and generator {name}")
                                data[n_sample][run]['meta_data'].loc[(f"{result}_untargeted"), (f"Prop_Aspirin_{column_meta_data}")] = np.mean(data_split_asp_untargeted[var])
                                try:
                                    data[n_sample][run]['meta_data'].loc[(f"{result}_untargeted"), (f"Var_Prop_Aspirin_{column_meta_data}")] = ((prop_tmp)*(1-prop_tmp))/n_obs
                                except Exception as e:
                                  print(f'[n={n_sample}, run={run}, generator = {name}] error in division because: {e}')

    return data

def plot_bias(meta_data: pd.DataFrame, select_estimators: list=[], plot_outliers: bool=True, name_original: list=[], list_generators: list=[], figure_size: tuple=(), 
              unit_rescale: dict={}, plot_estimates: bool=False, ground_truth: dict={}, labels_plot: list=[]):
    """
    Plot bias. Note that pd.DataFrame.groupby.mean ignores missing values when calculating the mean (desirable).
    If plot_estimates==True, then ground_truth should be given.
    """     
    suffix = ''     
    # Average bias of estimator for population parameter per generator (over sets) for each n and run
    # this is actually not necessary since I only have 1 synthetic sample per run and per generator.
    bias_data = meta_data.groupby(['n', 'run', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'run', 'generator'],
        var_name='estimator',
        value_name='bias')

    # Don't plot outliers based on IQR of empirical SE (set plot_outliers = False)
    if plot_outliers:
        bias_data['plot'] = True
    else:
        iqr = meta_data.groupby(['n', 'generator'])[select_estimators].agg([q1, q3]).reset_index()
        iqr.columns = ['_'.join(column) if column[1] != '' else column[0] for column in iqr.columns.to_flat_index()] # rename hierarchical columns
        iqr = iqr.melt(
            id_vars=['n', 'generator'],
            var_name='estimator',
            value_name='value')
        iqr['quantile'] = iqr.apply(lambda x: 'q1' if 'q1' in x['estimator'] else 'q3', axis=1)
        iqr['estimator'] = list(map(lambda x: x[:-3], iqr['estimator']))
        iqr = iqr.pivot(index=['n', 'generator', 'estimator'], columns='quantile', values='value').reset_index()
        bias_data = bias_data.merge(iqr, how='left', on=['n', 'generator', 'estimator'])
        bias_data['plot'] = bias_data.apply(
            lambda x: x['bias'] > x['q1'] - 1.5*(x['q3']-x['q1']) and x['bias'] < x['q3'] + 1.5*(x['q3']-x['q1']), axis=1) # non-outlier is > Q1-1.5*IQR and < Q3+1.5*IQR
        
    # Make a list of all generators (original, untargeted and targeted)
    # order_generators = name_original + \
    # [f"{generator}_untargeted" for generator in list_generators] + \
    # [f"{generator}_targeted" for generator in list_generators]
    
    order_generators = name_original + [
    gen for pair in zip(
        [f"{generator}_untargeted" for generator in list_generators],
        [f"{generator}_targeted" for generator in list_generators]
    ) for gen in pair
]
    print(order_generators)
    
    
    # Change plotting order (non-alphabetically) of estimator and generator
    bias_data['estimator'] = pd.Categorical(list(map(lambda x: x[:(-len(suffix) or None)], bias_data['estimator'])), # remove suffix; 'or None' is called suffix=''
                                            categories=[estimator[:(-len(suffix) or None)] for estimator in list(bias_data['estimator'].unique())]) # change order (non-alphabetically); 'or None' is called suffix=''
    if len(order_generators)!=0:
        bias_data['generator'] = pd.Categorical(bias_data['generator'], 
                                                categories=order_generators) # change order (non-alphabetically)

    # Root-n consistency funnel
    root_n_consistency = expand_grid({'x': np.arange(np.min(bias_data['n']), np.max(bias_data['n'])), 'estimator': bias_data['estimator'].unique()})
    root_n_consistency['estimator'] = pd.Categorical(root_n_consistency['estimator'], categories=root_n_consistency['estimator'].unique()) # change order (non-alphabetically)
    root_n_consistency['unit_sd'] = root_n_consistency.apply(lambda x: unit_rescale[x['estimator']], axis=1) # scale estimate to measurument unit
    root_n_consistency['y'] =  root_n_consistency.apply(lambda x: ground_truth[x['estimator']], axis=1) if plot_estimates else 0 
    root_n_consistency['y_ul'] = root_n_consistency['y'] + ss.norm.ppf(0.975)*root_n_consistency['unit_sd']/np.sqrt(root_n_consistency['x'])
    root_n_consistency['y_ll'] = root_n_consistency['y'] - ss.norm.ppf(0.975)*root_n_consistency['unit_sd']/np.sqrt(root_n_consistency['x'])
    
    # Plot labs
    plot_title = '' if plot_estimates else 'Bias of estimator'
    plot_y_lab = 'Estimate' if plot_estimates else 'Bias'
    
    # Default figure size
    if len(figure_size) == 0:
        figure_size = (1.5+len(bias_data['generator'].unique())*1.625, 1.5+len(bias_data['estimator'].unique())*1.625)
    
    # Plot average bias and root-n consistency funnel
    plot = ggplot(bias_data.query('plot==True'), aes(x='n', y='bias', colour='generator')) +\
        geom_line(data=root_n_consistency, mapping=aes(x='x', y='y'), linetype='dashed', colour='black') +\
        geom_line(data=root_n_consistency, mapping=aes(x='x', y='y_ul'), linetype='dashed', colour='black') +\
        geom_line(data=root_n_consistency, mapping=aes(x='x', y='y_ll'), linetype='dashed', colour='black') +\
        geom_point(alpha=0.20) +\
        stat_summary(geom='line') +\
        scale_x_continuous(breaks=list(bias_data['n'].unique()), labels=list(bias_data['n'].unique()), trans='log') +\
        facet_grid('estimator ~ generator', scales='free', labeller=labels_plot) +\
        scale_colour_manual(values={'original': '#808080', 
                                    'tvae_untargeted': '#F1A42B', 'custom_tvae_untargeted': '#DC4E28', 'ctgan_untargeted': '#1E64C8', 'custom_ctgan_untargeted': '#71A860',
                                    'tvae_targeted': '#F1A42B', 'custom_tvae_targeted': '#DC4E28', 'ctgan_targeted': '#1E64C8', 'custom_ctgan_targeted': '#71A860'}) +\
        labs(x='n (log scale)', y=plot_y_lab) +\
        theme_bw() +\
        theme(axis_title=element_text(size=14), # axis title size
              strip_text=element_text(size=11), # facet_grid title size
              axis_text=element_text(size=9), # axis labels size
              legend_position='none',
              figure_size=figure_size)
    
    return plot

def plot_coverage_paper(meta_data: pd.DataFrame, select_estimators: list=[], figure_size: tuple=(), name_original: list=[], list_generators: list=[], labels_plot: list=[]):
    """
    Plot confidence interval coverage
    """     
         
    # Average coverage of population parameter per generator (over sets and runs) for each n
    coverage_data_plot = meta_data.groupby(['n', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'generator'],
        var_name='estimator',
        value_name='coverage')
    # print(coverage_data_plot)
    
    # Make a list of all generators (original, untargeted and targeted)
    order_generators = name_original + \
    [f"{generator}_untargeted" for generator in list_generators] + \
    [f"{generator}_targeted" for generator in list_generators]
    
    # Change plotting order (non-alphabetically)
    coverage_data_plot['corrected'] = pd.Categorical(list(map(lambda x: 'corrected SE' if 'corrected' in x else 'model-based SE', coverage_data_plot['estimator'])), # change label
                                                     categories=['model-based SE', 'corrected SE']) # change order (non-alphabetically)
    if len(order_generators)!=0:
        coverage_data_plot['generator'] = pd.Categorical(coverage_data_plot['generator'],
                                                         categories=order_generators) # change order (non-alphabetically)      
    # Targeted indicator
    coverage_data_plot['targeted'] = coverage_data_plot.apply(lambda i: 'targeted' if '_targeted' in i['generator'] else 'default', axis=1)
    # Generator class
    coverage_data_plot['generator_class'] = coverage_data_plot.apply(lambda i: i['generator'][:-len('_targeted')] if '_targeted' in i['generator'] else (i['generator'][:-len('_untargeted')] if '_untargeted' in i['generator'] else i['generator']), axis=1)
    list_tmp = name_original + list_generators
    coverage_data_plot['generator_class'] = pd.Categorical(coverage_data_plot['generator_class'], categories=[generator for generator in list_tmp if '_untargeted' not in generator])

     # I don't want the rows with uncorrected, correction Raab should always be used!
    coverage_data_plot_Raab = coverage_data_plot[coverage_data_plot['corrected'] == 'corrected SE']

    # Default figure size
    if len(figure_size) == 0:
        figure_size = (10, 1.5+len(coverage_data_plot['estimator'].unique())*1.625)
        
    # Plot
    plot = ggplot(coverage_data_plot_Raab, aes(x='n', y='coverage', group ='generator', colour='generator_class', linetype='targeted')) +\
        geom_hline(yintercept=0.95, linetype='dashed') +\
        geom_line() +\
        scale_x_continuous(breaks=list(coverage_data_plot['n'].unique()), labels=list(coverage_data_plot['n'].unique()), trans='log') +\
        scale_y_continuous(limits=(0,1), labels=percent_format()) +\
        scale_colour_manual({'Original_data': '#808080', 
                            'custom_ctgan': '#71A860', 'ctgan': '#1E64C8', 'custom_tvae': '#DC4E28', 'tvae': '#F1A42B'},
                           labels=labels_plot)+\
        scale_linetype_manual(values={'default': 'solid', 'targeted': 'dashed'},
                             labels=['No', 'Yes']) +\
        labs(x='n (log scale)', y='Coverage',colour='Generator', linetype='Debiased') +\
        guides(colour=guide_legend(nrow=3),
               linetype=guide_legend(nrow=2)) +\
        theme_bw() +\
        theme(aspect_ratio=1,
              axis_title=element_text(size=10), # axis title size
              axis_text=element_text(size=8), # axis labels size
              figure_size=figure_size,
              legend_position='right',
              legend_title=element_text(size=10), # legend title size
              legend_title_align='left',
              legend_text=element_text(size=7), 
              legend_key=element_blank()) 
        

    return plot

def plot_type_I_paper(meta_data: pd.DataFrame, select_estimators: list= [], name_original: list=[], list_generators: list=[], use_power: bool=False, figure_size: tuple=(), labels_plot: list=[]):
    """
    Plot type I and II error rate
    """     

    # Average type 1 error rate per generator (over sets and runs) for each n
    NHST_data_plot = meta_data.groupby(['n', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'generator'],
        var_name='error',
        value_name='probability')   

    # Make a list of all generators (original, untargeted and targeted)
    order_generators = name_original + \
    [f"{generator}_untargeted" for generator in list_generators] + \
    [f"{generator}_targeted" for generator in list_generators]
    
    # Change plotting order (non-alphabetically)
    NHST_data_plot['corrected'] = pd.Categorical(list(map(lambda x: 'corrected SE' if 'corrected' in x else 'model-based SE', NHST_data_plot['error'])), # change label
                                                 categories=['model-based SE', 'corrected SE']) # change order (non-alphabetically)
    if len(order_generators)!=0:
        NHST_data_plot['generator'] = pd.Categorical(NHST_data_plot['generator'],
                                                     categories=order_generators) # change order (non-alphabetically)    
       
    # Additional plot formatting 
    NHST_data_plot['error'] = pd.Categorical(list(map(lambda x: 'type 1 error' if 'type1' in x else ('power' if use_power else 'type 2 error'), NHST_data_plot['error'])), # change label
                                             categories=['type 1 error', 'power' if use_power else 'type 2 error']) # change order (non-alphabetically) 
    NHST_data_plot['probability'] = NHST_data_plot.apply(lambda x:  1-x['probability'] if x['error']=='power' else x['probability'], axis=1) # power = 1 - type 2 error
    
    # Targeted indicator
    NHST_data_plot['targeted'] = NHST_data_plot.apply(lambda i: 'targeted' if '_targeted' in i['generator'] else 'default', axis=1)
    # Generator class
    NHST_data_plot['generator_class'] = NHST_data_plot.apply(lambda i: i['generator'][:-len('_targeted')] if '_targeted' in i['generator'] else (i['generator'][:-len('_untargeted')] if '_untargeted' in i['generator'] else i['generator']), axis=1)
    list_tmp = name_original + list_generators
    NHST_data_plot['generator_class'] = pd.Categorical(NHST_data_plot['generator_class'], categories=[generator for generator in list_tmp if '_untargeted' not in generator])
    
    # I don't want the rows with uncorrected, correction Raab should always be used!
    NHST_data_plot_Raab = NHST_data_plot[NHST_data_plot['corrected'] == 'corrected SE']

    # Default figure size
    if len(figure_size) == 0:
        height = 1
        figure_size = (3+height*1.625, 3+len(NHST_data_plot['error'].unique())*1.625)

    # Plot
    plot = ggplot(NHST_data_plot_Raab, aes(x='n', y='probability', group ='generator', colour='generator_class', linetype='targeted'))
    # Continue plot building
    plot = plot +\
        geom_line() +\
        geom_hline(yintercept=0.05) +\
        scale_x_continuous(breaks=list(NHST_data_plot['n'].unique()), labels=list(NHST_data_plot['n'].unique()), trans='log') +\
        scale_y_continuous(limits=(0,1), labels=percent_format()) +\
        scale_colour_manual({'Original_data': '#808080', 'custom_ctgan': '#71A860', 'ctgan': '#1E64C8', 'custom_tvae': '#DC4E28', 'tvae': '#F1A42B'},
                            labels=labels_plot)+\
        scale_linetype_manual(values={'default': 'solid', 'targeted': 'dashed'},
                             labels=['No', 'Yes']) +\
        labs(x='n (log scale)',
             colour='Generator', linetype='Debiased', y='Type 1 error') +\
        guides(colour=guide_legend(nrow=3),
               linetype=guide_legend(nrow=2)) +\
        theme_bw() +\
        theme(aspect_ratio=1.1,
              axis_title=element_text(size=10), # axis title size
              strip_text=element_text(size=8), # facet_grid title size
              axis_text=element_text(size=8), # axis labels size
              figure_size=figure_size,
              legend_position='right',
              legend_title=element_text(size=10), # legend title size
              legend_title_align='left',
              legend_text=element_text(size=7), 
              legend_key=element_blank()) 

    return plot

def calculate_conv_rate(data: pd.DataFrame, generator: str=None, estimator: str=None,
                        intercept: bool=False, unit_rescale: dict={}, metric: str='se', round_decimals: int=2, show_ci: bool=False, quantile: int=0.975):
    """
    Calculate convergence rate a in c/n**a
    """         
    # SE: log[sd(estimate)] = log(c) - a log(n), or equivalently log[sd(estimate-groundtruth)] = log(c) - a log(n)
    if metric == 'se':
        empirical_se = data.query(f'generator==\'{generator}\' & estimator==\'{estimator}\'').groupby('n')['bias'].apply(lambda x: np.std(x, ddof=1)).reset_index()
        if intercept:
            log_lr = OLS(np.log(empirical_se['bias']), sm.add_constant(np.log(empirical_se['n']))).fit() # add intercept
        else:
            unit_sd = unit_rescale[estimator[:-len('_bias')]] # rescale estimate with asymptotic variance
            log_lr = OLS(np.log(empirical_se['bias']/unit_sd), np.log(empirical_se['n'])).fit() # an intercept is not included by default
        
    # bias: log[mean(estimate-groundtruth)**2] = log(c) - a log(n**2)
    elif metric == 'bias':
        empirical_bias = data.query(f'generator==\'{generator}\' & estimator==\'{estimator}\'').groupby('n')['bias'].mean().reset_index()
        if intercept:
            log_lr = OLS(np.log(empirical_bias['bias']**2), sm.add_constant(np.log(empirical_bias['n']**2))).fit() # add intercept
        else:
            unit_sd = unit_rescale[estimator[:-len('_bias')]] # rescale estimate with asymptotic variance
            log_lr = OLS(np.log((empirical_bias['bias']/unit_sd)**2), np.log(empirical_bias['n']**2)).fit() # an intercept is not included by default
    
    # MSE: log[mean((estimate-groundtruth)**2)] = log(c) - a log(n**2)
    elif metric == 'MSE':
        empirical_mse = data.query(f'generator==\'{generator}\' & estimator==\'{estimator}\'').groupby('n')['bias'].apply(lambda x: np.mean(x**2)).reset_index()
        if intercept:
            log_lr = OLS(np.log(empirical_mse['bias']), sm.add_constant(np.log(empirical_mse['n']**2))).fit() # add intercept
        else:
            unit_sd = unit_rescale[estimator[:-len('_bias')]] # rescale estimate with asymptotic variance
            log_lr = OLS(np.log((empirical_mse['bias']/unit_sd)), np.log(empirical_mse['n']**2)).fit() # an intercept is not included
       
    # extract rate and rate_se
    coeff_index = 1 if intercept else 0
    rate = -log_lr.params[coeff_index]
    rate_se = np.sqrt(log_lr.cov_params().to_numpy().diagonal()[coeff_index])
    rate_q = ss.t.ppf(quantile, df=(log_lr.nobs-1))
    rate_ll = rate-rate_q*rate_se
    rate_ul = rate+rate_q*rate_se
        
    # round decimals
    rate = f'{rate:.{round_decimals}f}'
    rate_ll = f'{rate_ll:.{round_decimals}f}'
    rate_ul = f'{rate_ul:.{round_decimals}f}'
        
    output = f'{rate} [{rate_ll}; {rate_ul}]' if show_ci else rate
        
    return output

def table_convergence_rate(meta_data: pd.DataFrame, select_estimators: list=[], 
                           intercept: bool=False, unit_rescale: dict={}, metric: str='se', round_decimals: int=2, show_ci: bool=False, quantile: int=0.975):
    """
    Table with convergence rates
    """ 
    
    # Average bias of estimator for population parameter per generator (over sets) for each n and run
    bias_data = meta_data.groupby(['n', 'run', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'run', 'generator'],
        var_name='estimator',
        value_name='bias')
    
    # Calculate convergence rate for every estimator
    output_data = expand_grid({'generator': bias_data['generator'].unique(),
                               'estimator': bias_data['estimator'].unique()})
    output_data['convergence rate'] = output_data.apply(
        lambda x: calculate_conv_rate(data=bias_data, generator=x['generator'], estimator=x['estimator'], 
                                      intercept=intercept, unit_rescale=unit_rescale, metric=metric, round_decimals=round_decimals, show_ci=show_ci, quantile=quantile), axis=1)
    
    return output_data

#### Additional
# to show why there are NaN values for the convergence rate for untargated synthcity generator ('ctgan_untargeted', 'tvae_untargeted')
def calculate_conv_rate_additional(data: pd.DataFrame, generator: str=None, estimator: str=None,
                                   intercept: bool=False, unit_rescale: dict={}, metric: str='se', round_decimals: int=2, show_ci: bool=False, quantile: int=0.975):
    """
    Calculate convergence rate a in c/n**a
    """         
    # SE: log[sd(estimate)] = log(c) - a log(n), or equivalently log[sd(estimate-groundtruth)] = log(c) - a log(n)
    if metric == 'se':
        empirical_se = data.query(f'generator==\'{generator}\' & estimator==\'{estimator}\'').groupby('n')['bias'].apply(lambda x: np.std(x, ddof=1)).reset_index()
        print(empirical_se)
        if intercept:
            log_lr = OLS(np.log(empirical_se['bias']), sm.add_constant(np.log(empirical_se['n']))).fit() # add intercept
        else:
            unit_sd = unit_rescale[estimator[:-len('_bias')]] # rescale estimate with asymptotic variance
            log_lr = OLS(np.log(empirical_se['bias']/unit_sd), np.log(empirical_se['n'])).fit() # an intercept is not included by default
        
    # bias: log[mean(estimate-groundtruth)**2] = log(c) - a log(n**2)
    elif metric == 'bias':
        empirical_bias = data.query(f'generator==\'{generator}\' & estimator==\'{estimator}\'').groupby('n')['bias'].mean().reset_index()
        if intercept:
            log_lr = OLS(np.log(empirical_bias['bias']**2), sm.add_constant(np.log(empirical_bias['n']**2))).fit() # add intercept
        else:
            unit_sd = unit_rescale[estimator[:-len('_bias')]] # rescale estimate with asymptotic variance
            log_lr = OLS(np.log((empirical_bias['bias']/unit_sd)**2), np.log(empirical_bias['n']**2)).fit() # an intercept is not included by default
    
    # MSE: log[mean((estimate-groundtruth)**2)] = log(c) - a log(n**2)
    elif metric == 'MSE':
        empirical_mse = data.query(f'generator==\'{generator}\' & estimator==\'{estimator}\'').groupby('n')['bias'].apply(lambda x: np.mean(x**2)).reset_index()
        if intercept:
            log_lr = OLS(np.log(empirical_mse['bias']), sm.add_constant(np.log(empirical_mse['n']**2))).fit() # add intercept
        else:
            unit_sd = unit_rescale[estimator[:-len('_bias')]] # rescale estimate with asymptotic variance
            log_lr = OLS(np.log((empirical_mse['bias']/unit_sd)), np.log(empirical_mse['n']**2)).fit() # an intercept is not included
       
    # extract rate and rate_se
    coeff_index = 1 if intercept else 0
    rate = -log_lr.params[coeff_index]
    rate_se = np.sqrt(log_lr.cov_params().to_numpy().diagonal()[coeff_index])
    rate_q = ss.t.ppf(quantile, df=(log_lr.nobs-1))
    rate_ll = rate-rate_q*rate_se
    rate_ul = rate+rate_q*rate_se
        
    # round decimals
    rate = f'{rate:.{round_decimals}f}'
    rate_ll = f'{rate_ll:.{round_decimals}f}'
    rate_ul = f'{rate_ul:.{round_decimals}f}'
        
    output = f'{rate} [{rate_ll}; {rate_ul}]' if show_ci else rate
        
    return output

def table_convergence_rate_additional(meta_data: pd.DataFrame, select_estimators: list=[], 
                                      intercept: bool=False, unit_rescale: dict={}, metric: str='se', round_decimals: int=2, show_ci: bool=False, quantile: int=0.975):
    """
    Table with convergence rates
    """ 
    
    # Average bias of estimator for population parameter per generator (over sets) for each n and run
    bias_data = meta_data.groupby(['n', 'run', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'run', 'generator'],
        var_name='estimator',
        value_name='bias')
    
    # Calculate convergence rate for every estimator
    # but now only for ctgan_untargeted and tvae_untargeted
    output_data = expand_grid({'generator': ['ctgan_untargeted', 'tvae_untargeted'],
                               'estimator': bias_data['estimator'].unique()})
    output_data['convergence rate'] = output_data.apply(
        lambda x: calculate_conv_rate_additional(data=bias_data, generator=x['generator'], estimator=x['estimator'], 
                                                 intercept=intercept, unit_rescale=unit_rescale, metric=metric, round_decimals=round_decimals, show_ci=show_ci, quantile=quantile),
        axis=1)
    
    return output_data