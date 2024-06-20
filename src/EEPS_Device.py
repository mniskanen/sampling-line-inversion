# -*- coding: utf-8 -*-

import numpy as np


class EEPS_Device():
    """ This is a class that contains information on the measurement system, namely the measurement
    operator, which maps the particle numbers in size bins to the currents the EEPS measures.
    The particle size considered by this class is the mobility diameter, because it is what we
    can measure in practice.
    """
    
    def __init__(self, bins):
        
        # The standard EEPS output bins
        EEPS_binwidth = 1 / 16
        self.d_m_EEPS_edges = np.logspace(np.log10(5.62e-9), np.log10(562e-9), 33)
        self.d_m_EEPS = 10**(np.log10(self.d_m_EEPS_edges[1:]) - 0.5 * EEPS_binwidth)
        
        # The 'inversion' matrix
        self.soot_matrix = np.load('../measurement_data/soot_matrix.npy')
        
        # Measurement operator for the specified size discretization
        self.meas_op = self.specify_measurement_operator(bins.edges, bins.width)
    
    
    def specify_measurement_operator(self, d_m_edges, binwidth):
        
        self.interp_bins_to_basis = self.calculate_bins_to_basis_interpolator(d_m_edges, binwidth)
        
        meas_op = self.soot_matrix @ self.interp_bins_to_basis
        
        # # Minimum and maximum measurable particle concentrations (found on page 140 (B-4) of
        # # TSI 3090 EEPS manual). Approx limits (with 1 second averaging):
        # #   [1.4e4, 4.8e8] - smallest particle size (5.6 nm)
        # #   [1.4e2, 4.8e6] - largest particle size (560 nm)
        # self.minimum_N_undiluted = np.logspace(np.log10(1.4e4), np.log10(1.4e2), 32)
        # self.maximum_N_undiluted = np.logspace(np.log10(4.8e8), np.log10(4.8e6), 32)
        
        return meas_op
    
    
    def calculate_bins_to_basis_interpolator(self, d_m_edges, binwidth):
        """This returns a matrix connecting the modelled bins (the representation of the PSD we
        use to denote results and what the coagulation model uses as well) to the EEPS
        representation, which seems to use _some_ basis functions centered at the EEPS primary
        channel diameters (nodes).
        """
        
        # These are diameters given in Wang et al. (2016)
        EEPS_primary_channels = np.logspace(np.log10(5.62e-9), np.log10(562e-9), 17)
        primary_channel_binwidth = 1 / 8
        
        basis_type = 2  # 0 : constant
                        # 1 : Linear
                        # 2 : cos^2
        
        n_bins = d_m_edges.shape[0] - 1
        bins_to_basis = np.zeros((17, n_bins))
        
        # 'scale' is used here to control width of the basis functions.
        # scale == 2 with cos^2 basis (type 2) replicates the basis in (Mirme 2013)
        if basis_type == 1 or basis_type == 2:
            scale = 0.5
        elif basis_type == 0:
            scale = 1
        
        for i in range(17):
            # Construct each row of the connecting matrix
            # Assumption: each primary channel point is the middle point of a basis function
            basis_midp = EEPS_primary_channels[i]
            if basis_type == 0:
                basis_startp = 10**(
                    np.log10(basis_midp) - scale * primary_channel_binwidth * 0.5
                    )
                basis_endp = 10**(
                    np.log10(basis_midp) + scale * primary_channel_binwidth * 0.5
                    )
            else:
                basis_startp = 10**(
                    np.log10(basis_midp) - scale * primary_channel_binwidth
                    )
                basis_endp = 10**(
                    np.log10(basis_midp) + scale * primary_channel_binwidth
                    )
            
            # Brute force it: loop through all bins for every basis function
            for j in range(n_bins):
                if basis_type == 1:
                    bins_to_basis[i, j] += self._integrate_left(basis_startp,
                                                                basis_midp,
                                                                d_m_edges[j],
                                                                d_m_edges[j + 1])
                    bins_to_basis[i, j] += self._integrate_right(basis_midp,
                                                                  basis_endp,
                                                                  d_m_edges[j],
                                                                  d_m_edges[j + 1])
                elif basis_type == 2:
                    bins_to_basis[i, j] += self._integrate_cos2(basis_startp,
                                                                basis_endp,
                                                                d_m_edges[j],
                                                                d_m_edges[j + 1])
                elif basis_type == 3:
                    bins_to_basis[i, j] += self._integrate_cos2_modified(basis_startp,
                                                                          basis_endp,
                                                                          d_m_edges[j],
                                                                          d_m_edges[j + 1])
                
                elif basis_type == 0:
                    bins_to_basis[i, j] += self._integrate_constant(basis_startp,
                                                                    basis_endp,
                                                                    d_m_edges[j],
                                                                    d_m_edges[j + 1])
        
        if basis_type != 0:
            bins_to_basis /= binwidth  # Convert bin values to density (from absolute numbers)
        
        bins_to_basis /= scale  # To keep values consistent if 'scale' is changed
        
        bins_to_basis *= 4  # Fitting coefficient
            
        return bins_to_basis
    
    
    def _integrate_constant(self, EEPS_basis_startp, EEPS_basis_endp,
                            inversion_bin_startp, inversion_bin_endp):
        
        a = np.log10(EEPS_basis_startp)
        c = np.log10(EEPS_basis_endp)
        bin_a = np.log10(inversion_bin_startp)
        bin_b = np.log10(inversion_bin_endp)
        
        inversion_bin_width = bin_b - bin_a
        
        overlap = np.min((c, bin_b)) - np.max((a, bin_a))
        
        return np.max((0, overlap)) / inversion_bin_width
        
    
    def _integrate_cos2(self, a, c, bin_a, bin_b):
        """Integrate over basis functions of type:
            cos^2(pi * (x - a)/(c - a) - pi/2),     x = [a, c].
        """
        assert c > a
        assert bin_b > bin_a
        
        a = np.log10(a)
        c = np.log10(c)
        bin_a = np.log10(bin_a)
        bin_b = np.log10(bin_b)
        
        if bin_b >= a and bin_a <= c:
            if bin_a < a:
                bin_a = a
            if bin_b > c:
                bin_b = c
            
            return 0.5 * (bin_b - bin_a) + (a - c) / (4 * np.pi) * (
                np.sin(2 * np.pi * (a - bin_b) / (a - c))
                - np.sin(2 * np.pi * (a - bin_a) / (a - c))
                )
        else:
            return 0
    
    
    def _integrate_left(self, a, b, bin_a, bin_b):
        """Compute the contribution of a single bin to the integral of the left side basis
        function."""
        assert b > a
        assert bin_b > bin_a
        
        a = np.log10(a)
        b = np.log10(b)
        bin_a = np.log10(bin_a)
        bin_b = np.log10(bin_b)
        
        if bin_b >= a and bin_a <= b:
            if bin_a < a:
                bin_a = a
            if bin_b > b:
                bin_b = b
            # breakpoint()
            return (bin_b - bin_a) / (b - a) * ((bin_b + bin_a) / 2 - a)
        else:
            return 0
    
    
    def _integrate_right(self, b, c, bin_a, bin_b):
        """Compute the contribution of a single bin to the integral of the left side basis
        function."""
        assert c > b
        assert bin_b > bin_a
        
        c = np.log10(c)
        b = np.log10(b)
        bin_a = np.log10(bin_a)
        bin_b = np.log10(bin_b)
        
        if bin_b >= b and bin_a <= c:
            if bin_a < b:
                bin_a = b
            if bin_b > c:
                bin_b = c
            return (bin_b - bin_a) / (c - b) * (c - (bin_b + bin_a) / 2)
        else:
            return 0
    
    
    def apply_meas_op(self, N):
        return self.meas_op @ N
