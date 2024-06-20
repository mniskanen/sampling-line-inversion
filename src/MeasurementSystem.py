# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:20:27 2024

@author: mpniskan
"""

import numpy as np

from EEPS_Device import EEPS_Device
from SamplingLine import SamplingLine


class MeasurementSystem():
    ''' A model for the whole measurement system, including the sampling line, dilution, and
    the measurement device (EEPS in this case). '''
    
    def __init__(self, bins):
        
        # A model for the measurement device
        self.EEPS = EEPS_Device(bins)
        
        # Initialise model of the sampling line
        sampling_line_length = 3.2  # Meters
        sampling_line_diameter = 0.005  # Meters
        flow_velocity = 3.5  # Meters per second
        
        self.sampling_line = SamplingLine(bins.centers,
                                          sampling_line_length,
                                          sampling_line_diameter,
                                          flow_velocity
                                          )
        
        self.dilution_ratio = None
    
    
    def dilute(self, undiluted):
        
        if self.dilution_ratio is None:
            raise ValueError('Dilution ratio has not been specified')
        
        return undiluted / self.dilution_ratio
    
    
    def forward_model(self, N_0, return_Jacobian=False, return_current=True):
        ''' Run the forward model, i.e. "take the measurement": send the aerosol through the
        sampling line, dilute, and then measure with EEPS. Can also optionally output particle
        number instead of the electric currents.
        '''
        
        # Check that the input N_0 is on a log scale
        if np.max(N_0) > 15.0:
            raise ValueError('Input concentration to the forward model is too high (> 10^15). '
                              + 'Make sure N_0 is given on a log10 scale.')
        
        # Transform back to absolute particle numbers
        N_0 = 10**N_0
        
        if return_Jacobian:
            N_1, J = self.sampling_line.run(N_0, return_Jacobian)
            
            # Jacbobian of the posivity transform
            J_transform = np.log(10) * np.diag(N_0)
            J = J @ J_transform
            
            N_1 = self.dilute(N_1)
            J = self.dilute(J)
            
            if return_current:
                return self.EEPS.apply_meas_op(N_1), self.EEPS.apply_meas_op(J)
            
            return N_1, J
        
        else:
            N_1 = self.sampling_line.run(N_0)
            
            N_1 = self.dilute(N_1)
            
            if return_current:
                return self.EEPS.apply_meas_op(N_1)
            
            return N_1
