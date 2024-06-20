# -*- coding: utf-8 -*-

import time
import numpy as np

from tqdm import tqdm

from InversionModel import InversionModel
from Gauss_Newton_methods import minimize


def invert_AH_GN(time_indexes=None, show_output=True):
    
    inv_model = InversionModel(bins_per_decade=16)

    # Can change model parameters here
    inv_model.meas_system.sampling_line.update_aerosol_model(
        temperature=250+273,
        fractal_dim=1.7,
        primary_particle_diam=27e-9,
        hamaker_constant=2e-19,
        flow_velocity=3.5
        )
    
    if (isinstance(time_indexes, int) or isinstance(time_indexes, np.int64)
        or time_indexes is None):
        # Make into an array
        time_indexes = np.array([time_indexes])
    
    range_i = time_indexes.shape[0]
    N0_vector = np.zeros((range_i, inv_model.bins.n))
    
    tic = time.perf_counter()
    
    inv_model.load_EEPS_measurement(time_indexes[0])
    N0_vector[0], post_cov, modelled_measurement = minimize(inv_model, show_output=show_output)
    
    for i in tqdm(range(1, range_i)):
        inv_model.load_EEPS_measurement(time_indexes[i])
        inv_model.initial_guess = N0_vector[i - 1]
        N0_vector[i], _, _ = minimize(inv_model, show_output=False)
    
    toc = time.perf_counter()
    print(f"\nIn total, the script took {toc - tic : .2f} seconds.")
    
    if range_i > 1:
        # For a time series only return the MAP estimates
        return N0_vector
    
    else:
        return inv_model, N0_vector.squeeze(), post_cov, modelled_measurement



if __name__ == '__main__':
    
    # invert_AH_GN()  # Invert a random time index
    
    invert_AH_GN(time_indexes=722)
