# -*- coding: utf-8 -*-

import numpy as np

from scipy.optimize import minimize_scalar

from SamplingLine import SamplingLine


def accurate_rv(r_m, r_1, D_f, T):
    # A function to compute the difference between a desired r_m and r_m computed from r_v
    def f(r_v, r_m, r_1, D_f, T):
        r_m_model = SamplingLine.spherical_to_mobility_radius(r_v, r_1, D_f, T)
        return np.abs(r_m - r_m_model.flatten())
    
    options = {'xatol' : 1e-12}
    max_bound = np.max(r_m)  # r_v can't be larger than r_m
    min_bound = 0.1 * np.min(r_m)
    
    r_v = np.zeros(len(r_m))
    for i in range(len(r_m)):
        res = minimize_scalar(f,
                              bounds=(min_bound, max_bound),
                              args=(r_m[i], r_1, D_f, T),
                              method='bounded',
                              options=options
                              )
        r_v[i] = res.x
    
    return r_v


def test_rm_to_rv_interpolation():
        
    d_m = np.logspace(np.log10(1e-9), np.log10(600e-9), 40)
    r_m = d_m / 2
    
    s_line = SamplingLine(d_m, sampling_line_length=0, sampling_line_diameter=0.1, flow_velocity=1)
    s_line.prepare_for_mcmc()
    
    # Specify the ranges of r_1 and D_f over which we test
    lims_r_1 = np.array([5.5e-9, 19.5e-9])
    lims_D_f = np.array([1.05, 2.95])
    
    N_r_1 = 12  # Number of primary particle sizes
    N_D_f = 17  # Number of fractal dimensions
    
    r_1 = np.linspace(lims_r_1[0], lims_r_1[1], N_r_1)
    D_f = np.linspace(lims_D_f[0], lims_D_f[1], N_D_f)
    T = 273 + 250
    
    error = np.zeros((N_r_1, N_D_f))
    
    for i in range(N_r_1):
        for j in range(N_D_f):
            r_v_interp = s_line.compute_rv_interpolate(r_1[i], D_f[j])
            
            # Compute true r_v
            r_v = accurate_rv(r_m, r_1[i], D_f[j], T)
            
            error[i, j] = np.linalg.norm(r_v - r_v_interp.flatten()) \
                 / np.linalg.norm(r_v)
    
    error *= 100  # To percents
    
    # import matplotlib.pyplot as plt
    # plt.figure(num=8975)
    # plt.clf()
    # plt.imshow(error)
    # plt.colorbar()
    # plt.title('Relative error as a function of r_1 and D_f (%)')
    # plt.xlabel('D_f')
    # plt.ylabel('r_1')
    # plt.show()
    
    avg_error = error.mean()
    max_error = error.max()
    
    # print(f'Average error: {avg_error :.3f} %')
    # print(f'Maximum error: {max_error :.3f} %')
    
    assert avg_error < 0.2
    assert max_error < 1


if __name__ == '__main__':
    test_rm_to_rv_interpolation()