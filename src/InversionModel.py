# -*- coding: utf-8 -*-

import numpy as np

from scipy.linalg import toeplitz
from collections import namedtuple

from MeasurementSystem import MeasurementSystem


def discretize_particle_size(bins_per_decade, min_d_m=5.62e-9, max_d_m=562e-9):
    ''' Create a bin representation of the particle size range. The bins will be logarithmically
    spaced, with bins_per_decade number of bins for every order of magnitude covered. The maximum
    and minimum limits refer to the maximum and minimum edges of the largest and smallest bins,
    respectively. '''
    
    # First set up the particle size discretization
    Bins = namedtuple('Bins', ['centers', 'edges', 'width', 'n'])
    
    # User selectable number of bins
    binwidth = 1 / bins_per_decade
    
    # Smallest and largest bin edges
    log_min_d_m = np.log10(min_d_m)
    log_max_d_m = np.log10(max_d_m)
    
    # Number of bins
    nN = int(np.round(bins_per_decade * (log_max_d_m - log_min_d_m)))
    
    d_m_edges = np.logspace(log_min_d_m, log_max_d_m, nN + 1)
    
    # Bin centers
    d_m = 10**(np.log10(d_m_edges[1:]) - 0.5 * binwidth)
    
    # Size bins modelled as mobility diameters
    return Bins(d_m, d_m_edges, binwidth, nN)


class InversionModel():
    
    def __init__(self, bins_per_decade):
        
        self.bins = discretize_particle_size(bins_per_decade)
        
        self.meas_system = MeasurementSystem(self.bins)
        
        self.load_prior()
        
        # Specify the initial guess
        self.initial_guess = np.ones(self.bins.n) * 3
        
        # Denote that the measurements haven't been loaded yet
        self.EEPS_current = None
    
    
    def to_linear(self, x):
        """ Transform logarithmic values back to the absolute (linear) scale (when using the
        log-transform to enforce positivity). """
        return 10**x
    
    
    def log_post_and_gradient(self, N0, return_grad=True):
    
        modelled_measurement, J = self.meas_system.forward_model(N0, return_Jacobian=True)
        
        likelihood = -0.5 * np.linalg.norm(
            self.L_noise @ (self.measurement - modelled_measurement)
            )**2
        prior = -0.5 * np.linalg.norm(
            self.D @ (N0 - self.N_0_prior)
            )**2
        
        if return_grad:
            grad = (J.T @ self.noise_Icov) @ (self.measurement - modelled_measurement) \
                - self.W @ (N0 - self.N_0_prior)
            return likelihood + prior, grad
        else:
            return likelihood + prior
    
    
    def outside_aux_param_bounds(self, aux_sample):
        """ Check if any of the parameters in the supplied sample are outside the prior bounds. """
        
        for i in range(self.aux_bounds.shape[0]):
            if aux_sample[i] < self.aux_bounds[i, 0] or aux_sample[i] > self.aux_bounds[i, 1]:
                return True
        
        return False
    
    
    def neg_log_posterior(self, N_0):
        """ Compute the negative logarithm of the posterior at N_0.
        N_0 : logarithm of particle size distribution
        """
        
        likelihood = 0.5 * np.linalg.norm(
            self.L_noise @ (self.measurement - self.meas_system.forward_model(N_0))
            )**2
        
        prior = 0.5 * np.linalg.norm(
            self.D @ (N_0 - self.N_0_prior)
            )**2
        
        return likelihood + prior
    
    
    def load_prior(self, corr_length=None):
        
        # Prior for the aux parameters
        self.aux_bounds = np.array([
            [230 + 273, 270 + 273],  # temperature
            [1.5, 2.2],              # fractal dimension
            [20e-9, 34e-9],          # primary particle diameter
            [1e-19, 5e-19],          # Hamaker constant
            [3.2, 3.8]               # flow velocity
            ])
        
        
        # Prior for the initial PSD
        
        # Correlation length == 1 corresponds to here to one order of magnitude
        if corr_length is None:
            corr_length = 12 / 16
        
        # Standard deviation of the size distribution values (log transformed)
        std = 2.0
        
        # Prior mean (log transformed)
        mean = 3.0
        
        a = std**2
        distance_matrix = np.zeros((self.bins.n, self.bins.n))
        for i in range(self.bins.n):
            for j in range(self.bins.n):
                distance_matrix[i, j] = np.linalg.norm(
                    np.log10(self.bins.centers[i]) - np.log10(self.bins.centers[j])
                    )**2
        
        b = corr_length / np.sqrt(2 * np.log(100))
        
        self.pr_cov = a * np.exp(-0.5 * distance_matrix / b**2) + 1e-9 * np.eye(self.bins.n)
        
        # Direct inverse
        # self.W = np.linalg.inv(pr_cov)
        # self.D = np.linalg.cholesky(self.W)
        
        # Inverse using Gohberg & Semencul formula for Toeplitz matrices (this is a bit
        # better numerically)
        rhs = np.zeros(self.pr_cov.shape[0])
        rhs[0] = 1
        x = np.linalg.solve(self.pr_cov, rhs)
        B = toeplitz(x, np.zeros(x.shape[0]))
        C = toeplitz(np.concatenate(([0], np.flipud(x[1:]))), np.zeros(x.shape[0]))
        self.W = 1 / x[0] * (B @ B.T - C @ C.T)
        self.D = np.linalg.cholesky(self.W).T
        
        # Prior mean
        self.N_0_prior = mean * np.ones(self.bins.n)
        
        # Multiply by bin width to make independent of discretisation
        self.N_0_prior = np.log10(10**self.N_0_prior * self.bins.width)
            
    
    def preload_EEPS_measurements(self):
        """ Load the whole measurement set at once so it's faster to use when inverting it all.
        """
        self.EEPS_dNdlogDp = np.load('../measurement_data/EEPS_dNdlogDp.npy')
        self.EEPS_current = np.load('../measurement_data/EEPS_current.npy')
        self.DRs = np.load('../measurement_data/DR.npy')
        
    
    def load_EEPS_measurement(self, time_instant=None):
        
        measurement_noise_level = 0.10  # Percentage of the amplitude
        self.meas_type = 'EEPS'
        
        if self.EEPS_current is None:
            self.preload_EEPS_measurements()
        
        if time_instant is None:
            time_instant = np.random.randint(0, self.EEPS_current.shape[0])
            print(f"Time instant: {time_instant}")
        
        EEPS_dNdlogDp_sample = self.EEPS_dNdlogDp[time_instant]
        EEPS_current_sample = self.EEPS_current[time_instant]
        # Scale the EEPS-returned PN back to absolute scale because they are
        # by default normalised by the log width (1/16)
        self.EEPS_N_sample = EEPS_dNdlogDp_sample * (1/16)
        
        self.measurement = EEPS_current_sample
        self.meas_system.dilution_ratio = self.DRs[time_instant]
        
        # Noise model
        self.empty_noise_std = np.load('../measurement_data/noise_std.npy')
        meas_noise_std = measurement_noise_level * np.abs(EEPS_current_sample)
        
        # Choose the larger value, i.e. noise is always at least the empty noise
        self.noise_std = np.max((self.empty_noise_std, meas_noise_std), axis=0)
        
        self.noise_cov = np.diag(self.noise_std**2)
        self.noise_Icov = np.diag(self.noise_std**-2)
        self.L_noise = np.diag(self.noise_std**-1)
    
    
    def load_synthetic_measurement(self, measurement, true_N0, truth_bins, dilution_ratio):
        
        measurement_noise_level = 0.10  # Percentage of the amplitude
        self.meas_type = 'Synthetic'
        
        self.measurement = measurement
        self.true_N_0 = true_N0
        self.truth_bins = truth_bins
        self.meas_system.dilution_ratio = dilution_ratio
        
        # Noise model
        self.empty_noise_std = np.load('../measurement_data/noise_std.npy')
        
        meas_noise_std = measurement_noise_level * np.abs(self.measurement)
        
        # Choose the larger value, i.e. noise is always at least the empty noise
        self.noise_std = np.max((self.empty_noise_std, meas_noise_std), axis=0)
        
        self.noise_cov = np.diag(self.noise_std**2)
        self.noise_Icov = np.diag(self.noise_std**-2)
        self.L_noise = np.diag(self.noise_std**-1)
