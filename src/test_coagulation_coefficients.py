# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt

from InversionModel import discretize_particle_size
from MeasurementSystem import MeasurementSystem



def plot_coag_coeffs(ax1=None, ax2=None):
    '''Compare different coagulation coefficients. The optional axs inputs allow to plot in the
    same subplot. '''
    
    bins = discretize_particle_size(bins_per_decade=16)
    meas_system = MeasurementSystem(bins)
    
    dm_nm = bins.centers * 1e9  # Mobility diameter in nanometers
    
    if ax1 is None:
        fig, ax1 = plt.subplots(num=62, clear=True)
    
    t1 = time.perf_counter()
    meas_system.sampling_line.update_aerosol_model(
        temperature=250+273,
        fractal_dim=1.7,
        primary_particle_diam=30e-9,
        hamaker_constant=2e-19,
        compute_vdw=False
        )
    K1 = meas_system.sampling_line.K
    t2 = time.perf_counter()
    K2 = meas_system.sampling_line.compute_coagulation_coefficient_spherical(bins.centers)
    t3 = time.perf_counter()
    
    mid = int(bins.centers.shape[0] / 2)  # Middle index
    
    ax1.loglog(dm_nm, np.diag(K2), 'C0-', label='Spherical ($d_{m,1}$ = $d_{m,2}$)')
    ax1.loglog(dm_nm, np.diag(K1), 'C0--', label='Fractal ($d_{m,1}$ = $d_{m,2}$)')
    
    ax1.loglog(dm_nm, K2[0, :],
                'C1-', label=f'Spherical ($d_{{m,1}}$ = {dm_nm[0] :.1f} nm)')
    ax1.loglog(dm_nm, K1[0, :],
                'C1--', label=f'Fractal ($d_{{m,1}}$ = {dm_nm[0] :.1f} nm)')
    
    ax1.loglog(dm_nm, K2[mid, :],
                'C2-', label=f'Spherical ($d_{{m,1}}$ = {dm_nm[mid] :.1f} nm)')
    ax1.loglog(dm_nm, K1[mid, :],
                'C2--', label=f'Fractal ($d_{{m,1}}$ = {dm_nm[mid] :.1f} nm)')
    
    ax1.set_xlabel('Particle mobility diameter $d_{m,2}$ (nm) ')
    ax1.set_ylabel(r'Coagulation coefficient $(cm^3 s^{-1})$')
    ax1.grid(which='both')
    ax1.legend()
    
    
    # Compare coagulation coefficients with and without van der Waals + viscous factors -----------
    if ax2 is None:
        fig, ax2 = plt.subplots(num=63, clear=True)
        
    meas_system.sampling_line.update_aerosol_model(
        temperature=250+273,
        fractal_dim=3.0,
        primary_particle_diam=30e-9,
        hamaker_constant=2e-19,
        compute_vdw=True
        )
    K1 = meas_system.sampling_line.K
    
    ax2.loglog(dm_nm, np.diag(K2), 'C0-', label='No vdW ($d_{m,1}$ = $d_{m,2}$)')
    ax2.loglog(dm_nm, np.diag(K1), 'C0--', label='With vdW ($d_{m,1}$ = $d_{m,2}$)')

    ax2.loglog(dm_nm, K2[0, :],
                'C1-', label=f'No vdW ($d_{{m,1}}$ = {dm_nm[0] :.1f} nm)')
    ax2.loglog(dm_nm, K1[0, :],
                'C1--', label=f'With vdW ($d_{{m,1}}$ = {dm_nm[0] :.1f} nm)')
    
    ax2.loglog(dm_nm, K2[mid, :],
                'C2-', label=f'No vdW ($d_{{m,1}}$ = {dm_nm[mid] :.1f} nm)')
    ax2.loglog(dm_nm, K1[mid, :],
                'C2--', label=f'With vdW ($d_{{m,1}}$ = {dm_nm[mid] :.1f} nm)')
    
    ax2.set_xlabel('Particle mobility diameter $d_{m,2}$ (nm) ')
    ax2.set_ylabel(r'Coagulation coefficient $(cm^3 s^{-1})$')
    ax2.grid(which='both')
    ax2.legend()
    
    print(f'Time for agglomerate kernel: {1e3 * (t2 - t1): .3g} milliseconds')
    print(f'Time for spherical kernel: {1e3 * (t3 - t2): .3g} milliseconds')

def test_coagulation_coefficients():
    
    bins = discretize_particle_size(bins_per_decade=16)
    meas_system = MeasurementSystem(bins)
    
    # Test that we get the correct spherical kernel when we don't include van der Waals forces and
    # make fractional dimension 3
    meas_system.sampling_line.update_aerosol_model(
        temperature=250+273,
        fractal_dim=3.0,
        primary_particle_diam=27e-9,
        hamaker_constant=2e-19,
        compute_vdw=False
        )
    K1 = meas_system.sampling_line.K
    K2 = meas_system.sampling_line.compute_coagulation_coefficient_spherical(bins.centers)
    
    # Test to make sure that changing fractional dimension does not affect calculation of the
    # spherical coefficient
    meas_system.sampling_line.update_aerosol_model(
        temperature=250+273,
        fractal_dim=1.7,
        primary_particle_diam=27e-9,
        hamaker_constant=2e-19,
        compute_vdw=False
        )
    K3 = meas_system.sampling_line.compute_coagulation_coefficient_spherical(bins.centers)
    
    
    difference1 = np.linalg.norm(K1 - K2) / np.linalg.norm(K1) * 100
    difference2 = np.linalg.norm(K2 - K3) / np.linalg.norm(K2) * 100
    
    print(f'Difference between kernels: {difference1 : .3f} %')
    
    assert difference1 < 0.1
    assert difference2 < 0.1

if __name__ == '__main__':
    
    plot_coag_coeffs()
