# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from InversionModel import InversionModel


def Jacob_perturb(N, inv_model):
    """ Solve the Jacobian using perturbation.
    """
    g0 = inv_model.meas_system.forward_model(N, return_current=False)
    
    J = np.zeros((inv_model.bins.n, inv_model.bins.n))
    g_perturbed = np.zeros_like(J)
    eps = 1e-6 * np.max(N)
    
    for i in range(inv_model.bins.n):
        N_pb = N.copy()
        N_pb[i] += eps
        g_perturbed[:, i] = inv_model.meas_system.forward_model(N_pb, return_current=False)
    
    for k in range(inv_model.bins.n):
        for i in range(inv_model.bins.n):
            J[k, i] = (g_perturbed[k, i] - g0[k]) / eps
    
    return J


def test_jacobian():
    
    """ Compare Jacobians given by perturbation and an analytical (backpropagation) solution.
    """
    
    # Create synthetic data
    rng = np.random.default_rng(1)  # Set seed for reproducible results
    
    discr = 26  # Bins per decade (log spaced)
    
    bw = 1 / discr  # Log bin width
    
    dilution_ratio = 50
    
    inv_model = InversionModel(bins_per_decade=discr)
    inv_model.meas_system.dilution_ratio = dilution_ratio
    
    N0_density = 8 * np.exp(
        -0.5 * ((np.log10(inv_model.bins.centers) - np.log10(7e-9)) / (4 * (1 / 16)))**2
        )
    N0_density += 7 * np.exp(
        -0.5 * ((np.log10(inv_model.bins.centers) - np.log10(4e-8)) / (6 * (1 / 16)))**2
        )
    N0_density += 6 * np.exp(
        -0.5 * ((np.log10(inv_model.bins.centers) - np.log10(3e-7)) / (4 * (1 / 16)))**2
        )
    
    # Draw from the prior
    N0_density = 2 + (inv_model.N_0_prior
                      + np.linalg.cholesky(inv_model.pr_cov)
                      @ rng.normal(size=inv_model.pr_cov.shape[0])
                      )
    
    N0_density = 10**N0_density
    N0 = N0_density * bw
    N0 = np.log10(N0)
    
    _, J_analytical = inv_model.meas_system.forward_model(
        N0, return_Jacobian=True, return_current=False
        )
    J_perturbed = Jacob_perturb(N0, inv_model)
    
    smallest_value = 1e-12
    J_an_mask = np.ma.masked_array(J_analytical, np.abs(J_analytical) < smallest_value)
    J_pb_mask = np.ma.masked_array(J_perturbed, np.abs(J_perturbed) < smallest_value)
    rel_error_componentwise = (J_an_mask - J_pb_mask) / J_an_mask  * 100
    # mean_error = np.ma.mean(np.ma.abs(rel_error))
    rel_error = np.linalg.norm(J_an_mask - J_pb_mask) / np.linalg.norm(J_an_mask) * 100
    
    print(f'Relative difference between analytical and perturbed: {rel_error : .3g} %')
    
    plt.figure(num=7437), plt.clf()
    
    plt.subplot(131)
    plt.imshow(np.ma.log10(np.ma.abs(J_an_mask)))
    plt.colorbar()
    plt.title('log10(abs(Analytical Jacobian))')
    
    plt.subplot(132)
    plt.imshow(np.ma.log10(np.ma.abs(J_pb_mask)))
    plt.colorbar()
    plt.title('log10(abs(Perturbed Jacobian))')
    
    plt.subplot(133)
    # Center colormap around zero (easy to see deviations)
    c_range = 10#np.max(np.abs(rel_error_componentwise))
    plt.imshow(rel_error_componentwise, cmap='RdBu', vmin=-c_range, vmax=c_range)
    plt.colorbar()
    plt.title('Relative error %')
    
    plt.draw()
    
    assert rel_error < 1


if __name__ == '__main__':
    test_jacobian()