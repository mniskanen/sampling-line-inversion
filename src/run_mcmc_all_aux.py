# -*- coding: utf-8 -*-

import os
import dill as pickle
import numpy as np

from MCMC_methods import MCMCSampler
from InversionModel import InversionModel
from invert_synthetic import generate_synthetic_measurement


results_folder = '../results/MCMC/'
mcmc_length = 2e5

# List combinations of model unknowns to sample
aux_to_sample = ((),  # don't sample any aux parameters
                 (0),  # only sample temperature
                 (1),  # only sample fractal dimension
                 (2),  # only sample primary particle diameter
                 (3),  # only sample Hamaker constant
                 (4),  # only sample flow velocity
                 (0,1,2,3,4,)  # sample all aux parameters simultaneously
                 )


def save_run(obj, results_folder_path, fname):
    with open(results_folder_path + fname, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_run(results_folder_path, fname):
    with open(results_folder_path + fname, 'rb') as inp:
        data = pickle.load(inp)
    return data


# Run ALL the MCMC we have in the paper
if __name__ == '__main__':
    
    if not os.path.exists(results_folder):
        raise FileNotFoundError('Results folder does not exist')
    
    # First, do synthetic data --------------------------------------------------------------------
    
    # Generate the synthetic measurement
    fwd_discr = 30
    dilution_ratio = 100
    true_N0, measurement, fwd_bins = generate_synthetic_measurement(fwd_discr, dilution_ratio)
    
    # Define configuration for the inverse problem
    inv_model = InversionModel(bins_per_decade=16)
    inv_model.load_synthetic_measurement(measurement, true_N0, fwd_bins, dilution_ratio)
    
    
    print('Run MCMC with synthetic data, known aux. params:')
    
    # Give model parameter values (these will be used during MAP estimation for the
    # initialization of MCMC and during MCMC sampling for all aux parameters that are
    # kept constant (i.e. not sampled)).
    init_aux_params = np.array([250+273, 1.7, 27e-9, 2e-19, 3.5])
    
    sampler = MCMCSampler(inv_model, mcmc_length, init_aux_params, aux_to_sample[0])
    sampler.run_mcmc()
    
    save_run(sampler, results_folder, 'true_aux_simulated')
    
    
    print('Run MCMC with synthetic data, marginalize combinations of aux. params, init. Df = 2.1:')
    
    init_aux_params = np.array([250+273, 2.1, 27e-9, 2e-19, 3.5])
    
    # Sample these aux combinations
    do_these = [0, 2, 4, 5, 6]
    for i in do_these:
        print(f'i: {i}, aux to sample: {aux_to_sample[i]}')
        
        sampler = MCMCSampler(inv_model, mcmc_length, init_aux_params, aux_to_sample[i])
        sampler.run_mcmc()
        
        save_run(sampler, results_folder, f'aux{i}_simulated')
    
    
    # Next, do real measurement data --------------------------------------------------------------
        
    inv_model = InversionModel(bins_per_decade=16)
    init_aux_params = np.array([250+273, 1.7, 27e-9, 2e-19, 3.5])
    
    # Inversion for Line 1
    print('Run MCMC with real data (Line 1), all aux. params marginalized:')
    inv_model.load_EEPS_measurement(time_instant=722)
    
    sampler = MCMCSampler(inv_model, mcmc_length, init_aux_params, aux_to_sample[6])
    sampler.run_mcmc()
    
    save_run(sampler, results_folder, f'aux{i}_line1')
    
    
    
    # Inversion for Line 2
    print('Run MCMC with real data (Line 2), all aux. params marginalized:')
    inv_model.load_EEPS_measurement(time_instant=1872)
    
    sampler = MCMCSampler(inv_model, mcmc_length, init_aux_params, aux_to_sample[6])
    sampler.run_mcmc()
    
    save_run(sampler, results_folder, f'aux{i}_line2')
    
    print('All done.')
