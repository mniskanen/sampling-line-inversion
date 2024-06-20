# -*- coding: utf-8 -*-

import numpy as np


class ProposalDensity():
    """ A multivariate Gaussian proposal density of one model, i.e. has fixed dimensions.
    The unit-lag proposal may work better for trans-dimensional parameters that can have lots of
    jumps. By default, this class implements the 'normal' covariance, i.e., centered around the
    mean. """
    
    def __init__(self,
                 initial_covariance,
                 max_mcmclen,
                 target_acceptance_rate=0.234,
                 unit_lag=False
                 ):
        
        self.n_AM_updt = 0  # Number of times the global scale factor has been updated
        self.n_proposed = 0  # Number of times this proposal has been used
        self.n_accepted = 0  # Number of times the proposal has been accepted
        self.N = 0  # Number of times the sample covariance has been updated
        self.unit_lag = unit_lag  # 1: use the unit-lag covariance, 0: don't use
        
        if initial_covariance.ndim == 0:
            self.n_params = 1
            self.covariance = np.array([[initial_covariance]])
        
        elif initial_covariance.ndim == 1:
            self.n_params = len(initial_covariance)
            self.covariance = np.diag(initial_covariance)
            
        elif initial_covariance.ndim == 2:
            assert(initial_covariance.shape[0] == initial_covariance.shape[1])
            self.n_params = initial_covariance.shape[0]
            self.covariance = initial_covariance.copy()
        
        self.update_cholesky()
        
        # Forget the initial covariance
        # self.covariance = np.zeros((self.n_params, self.n_params))
        
        self.max_mcmclen = max_mcmclen
        self.target_acceptance_rate = target_acceptance_rate
        
        self.allocation_chunk = int(1e4)  # Chunksize for allocating a longer vector
        
        self.AM_factor = 1  # Current global proposal scale factor
        self.AM_factors = np.zeros(self.allocation_chunk)  # Store global scale factors
        self.AM_factors[self.n_AM_updt] = self.AM_factor
        
        # Another global scale factor. Constant over the run but makes the
        # variable global scale factor get closer to one.
        self.pcoeff = 2.4 / np.sqrt(np.max((1, self.n_params)))
        
        self.mean  = np.zeros(self.n_params)  # Sample mean, used for the normal covariance
        
        # self.inv_cholesky = None  # To calculate proposal probability
    
    
    def draw_proposal(self):
        
        self.n_proposed += 1
        return self.pcoeff * self.AM_factor * self.cholesky @ np.random.randn(self.n_params)
    
    
    def proposal_probability(self, loc1, loc2):
        """ Evaluate the probability of making a proposal from loc1 to loc2 (or vice versa since
        the proposal is symmetric.
        Only evaluate the non-constant (over a single proposal) part.
        Return the logarithm of the probability.
        """
        return -0.5 * np.linalg.norm(
            1 / (self.pcoeff + self.AM_factor) * self.inv_cholesky @ (loc1 - loc2)
            )**2
    
    
    def update_AM_factor(self, alpha):
        
        self.n_AM_updt += 1
        safe_alpha = np.max([alpha, -1e2]) # to prevent underflow warnings
        self.AM_factor *= np.exp(
                2 * self.n_AM_updt**(-2/3) * ( np.exp(safe_alpha) - self.target_acceptance_rate )
                )
        self.AM_factor = np.max([1e-6, self.AM_factor])
        
        # If vector full, pad with zeros
        if self.n_AM_updt == self.AM_factors.shape[0]:
            self.AM_factors = np.pad(
                self.AM_factors, (0, self.allocation_chunk), mode='constant', constant_values=0
                )
        
        # Store
        self.AM_factors[self.n_AM_updt] = self.AM_factor
    
    
    def remove_leftover_zeros(self):
        self.AM_factors = self.AM_factors[:self.n_AM_updt]
    
    
    def update_cholesky(self):
        self.cholesky = np.linalg.cholesky(self.covariance)
        # self.inv_cholesky = np.linalg.cholesky(np.linalg.inv(self.covariance))
    
    
    def update_covariance(self, new_values):
        
        if self.unit_lag:
            """ Updates the unit-lag proposal covariance matrix, which includes
            only the magnitude and direction of the new sample, not estimating
            the covariance matrix about the sample mean (the traditional 
            covariance may not work very well with the trans-D sampler).
            The unit-lag covariance matrix is updated simply by adding the
            "covariance" of the difference. """
            
            self.N += 1
            self.covariance = self.N / (self.N + 1) * self.covariance + \
                self.N / (self.N + 1)**2 * np.outer(new_values, new_values)
        
        else:
            """ Updates the model covariance matrix iteratively by adding the
            latest sample. """
            
            if np.sum(self.mean) == 0:
                # Probably the first iteration --> only set the mean
                self.mean = new_values
            
            else:
                self.N += 1
                coeff = 1 / (self.N + 1)
                
                sample_diff = new_values - self.mean
                new_mean = self.mean + coeff * sample_diff
                
                self.covariance += coeff * (np.outer(sample_diff, sample_diff)
                                            - self.N * coeff * self.covariance)
                
                self.mean = new_mean