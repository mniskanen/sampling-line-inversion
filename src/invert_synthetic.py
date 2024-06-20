# -*- coding: utf-8 -*-

import numpy as np

from InversionModel import InversionModel, discretize_particle_size
from MeasurementSystem import MeasurementSystem
from Gauss_Newton_methods import minimize


def generate_synthetic_measurement(fwd_discr, dilution_ratio):
    """
    Input:
        fwd_discr = Number of bins per decade for the fwd model
        dilution_ratio = Dilution ratio
    """
    
    # Create synthetic data
    rng = np.random.default_rng(1)  # Set seed for reproducible results
    
    added_noise_level = 10 # Percent
    
    bins = discretize_particle_size(fwd_discr)
    meas_system = MeasurementSystem(bins)
    meas_system.dilution_ratio = dilution_ratio
    
    # Set AerosolModel parameters (these are used when generating the data)
    meas_system.sampling_line.update_aerosol_model(
        temperature=250+273,
        fractal_dim=1.7,
        primary_particle_diam=27e-9,
        hamaker_constant=2e-19,
        flow_velocity=3.5
        )
    
    # Density (independent of discretisation, i.e. considered already normalized by bin width)
    # These are the values used in the paper
    true_N0_density = 4 * np.exp(
        -0.5 * ((np.log10(bins.centers) - np.log10(7e-9)) / 0.33)**2
        )
    true_N0_density += 8 * np.exp(
        -0.5 * ((np.log10(bins.centers) - np.log10(5e-8)) / 0.67)**2
        )
    
    true_N0_density = 10**true_N0_density
    true_N0 = true_N0_density * bins.width  # "Undo" normalization by bin width, these are now
                                            # absolute particle numbers per bin
    
    measurement_true = meas_system.forward_model(np.log10(true_N0))
    
    # Add noise
    EEPS_noise_std = np.load('../measurement_data/noise_std.npy')
    noise_std = np.max((EEPS_noise_std,
                        added_noise_level / 100 * measurement_true),
                       axis=0)
    measurement_noisy = measurement_true + noise_std * rng.normal(size=measurement_true.shape)
    
    return true_N0, measurement_noisy, bins



def invert_synthetic_GN(inversion_Df, show_output=True):
    
    fwd_discr = 30
    dilution_ratio = 100
    
    true_N0, measurement, fwd_bins = generate_synthetic_measurement(fwd_discr, dilution_ratio)
    
    inv_discr = 16
    inv_model = InversionModel(bins_per_decade=inv_discr)
    
    # Update AerosolModel parameters
    inv_model.meas_system.sampling_line.update_aerosol_model(
        temperature=250+273,
        fractal_dim=inversion_Df,
        primary_particle_diam=27e-9,
        hamaker_constant=2e-19,
        flow_velocity=3.5
        )
    
    inv_model.load_synthetic_measurement(measurement, true_N0, fwd_bins, dilution_ratio)
    
    return inv_model, minimize(inv_model, show_output)


if __name__ == '__main__':
    
    invert_synthetic_GN(1.7)
