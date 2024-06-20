# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class EdgeFiller():
    '''Fill first and last bin to their edges (for visualisation purposes).'''
    
    def __init__(self, dp, dp_edges):
        
        # Use this class also to transform to nanometers
        self.dp = dp * 1e9
        self.dp_edges = dp_edges * 1e9
        
        self.dp_full = np.concatenate([[self.dp_edges[0]], self.dp, [self.dp_edges[-1]]])
        self.d_dp_start = np.log10(self.dp_full[0]) - np.log10(self.dp_full[1])
        self.d_dp_end = np.log10(self.dp_full[-1]) - np.log10(self.dp_full[-2])
        
    def fill_edges(self, N):
        grad_start = (N[1] - N[0]) / (np.log10(self.dp[1]) - np.log10(self.dp[0]))
        grad_end = (N[-1] - N[-2]) / (np.log10(self.dp[-1]) - np.log10(self.dp[-2]))
        
        N_full = np.concatenate([
            [N[0] + grad_start * self.d_dp_start], N, [N[-1] + grad_end * self.d_dp_end]
            ])
        
        return N_full


def plot_system(
        inv_model,
        N_0,  # Estimate for the initial PSD
        posterior_cov=None,  # Posterior covariance matrix
        predicted_measurement=None,  # Measurement corresponding to the input N_0
        posterior_CIs=None,  # Credible intervals from the MCMC samples
        estimate_name='MAP estimate',
        N_1=None,  # PSD after the sampling line
        data_stds=None,  # Standard deviations in the data space (from MCMC)
        ax1=None,  # Axis where to plot the estimate
        ax2=None   # Axis where to plot the data fit
        ):
    
    
    if ax1 is None:
        if estimate_name == 'CM estimate':
            fig1, ax1 = plt.subplots(num=10, clear=True)
        else:
            fig1, ax1 = plt.subplots(num=1, clear=True)
        
    
    if posterior_cov is None and posterior_CIs is None:
        raise ValueError('No input for posterior uncertainty')
    
    uncertainty_alpha = 0.15  # For the plots
    
    # Just add a tiny bit to both ends to fill up the plot from bin center to edge
    ef = EdgeFiller(inv_model.bins.centers, inv_model.bins.edges)
    plt_dp = ef.dp_full
    
    N0_full = ef.fill_edges(N_0)
    plt_N0 = inv_model.to_linear(N0_full) / inv_model.bins.width
    ax1.semilogx(plt_dp, plt_N0,
                 label=f'{estimate_name}'
                 )
    
    if N_1 is not None:
        N1_full = ef.fill_edges(N_1)
        plt_N1 = inv_model.to_linear(N1_full) / inv_model.bins.width
        ax1.semilogx(plt_dp, plt_N1,
                     label='PSD after sampling line'
                     )
    
    # Uncertainty of N0
    if posterior_CIs is None:
        try:
            posterior_std = np.sqrt(np.diag(posterior_cov))
        except:
            raise ValueError('Negative posterior variance values')

        post_std_full = ef.fill_edges(posterior_std)
        fill_top1 = inv_model.to_linear(N0_full + 1.96 * post_std_full) / inv_model.bins.width
        fill_bottom1 = inv_model.to_linear(N0_full - 1.96 * post_std_full) / inv_model.bins.width
        label = "95 % posterior credible interval"
        
    else:
        fill_top1 = inv_model.to_linear(ef.fill_edges(posterior_CIs[0])) / inv_model.bins.width
        fill_bottom1 = inv_model.to_linear(ef.fill_edges(posterior_CIs[1])) / inv_model.bins.width
        label = "95 % posterior credible interval"
    
    ax1.fill_between(plt_dp, fill_top1, fill_bottom1, alpha=uncertainty_alpha,
                      facecolor='b',
                      label=label
                      )
    
    # Plot the EEPS-inverted PSD, or the truth
    if inv_model.meas_type == 'EEPS':
        ef_EEPS = EdgeFiller(
            inv_model.meas_system.EEPS.d_m_EEPS, inv_model.meas_system.EEPS.d_m_EEPS_edges
            )
        plt_dp_EEPS = ef_EEPS.dp_full
        
        plt_EEPS_N1 = inv_model.to_linear(
            ef_EEPS.fill_edges(np.log10(inv_model.EEPS_N_sample))
            ) / (1/16)
        
        ax1.semilogx(
            plt_dp_EEPS, plt_EEPS_N1, 'g--', label='PSD estimate from EEPS'
            )
        
    if inv_model.meas_type == 'Synthetic':
        ef_truth = EdgeFiller(inv_model.truth_bins.centers, inv_model.truth_bins.edges)
        plt_dp_truth = ef_truth.dp_full
        
        plt_truth_N0 = inv_model.to_linear(
            ef_truth.fill_edges(np.log10(inv_model.true_N_0))
            ) / inv_model.truth_bins.width
        
        ax1.semilogx(plt_dp_truth, plt_truth_N0,
            color='k', linestyle='--', linewidth=1, label='True initial PSD'
            )
    
    # Plot the size range EEPS can measure
    # plt.loglog(np.array([5.62e-9, 562e-9]), np.array([1, 1]), 'k--', linewidth=2)
    # anno_args = {
    #     'ha': 'center',
    #     'va': 'center',
    #     'size': 12
    # }
    # plt.annotate('Measurable size range',
    #              xy=(10**(0.5 * (np.log10(5.62e-9) + np.log10(562e-9))), 2),
    #              **anno_args)
    # anno_args['size'] = 20
    # plt.annotate('[', xy=(5.62e-9, 1), **anno_args)
    # plt.annotate(']', xy=(562e-9, 1), **anno_args)
    
    
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylim([1e-2, 1e11])
    ax1.set_title('Particle size distribution')
    ax1.set_xlabel('Particle mobility diameter (nm)')
    ax1.set_ylabel(r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$')
    ax1.legend(loc='best')
    ax1.grid(which='both')
    plt.pause(0.01)
    
    
    
    if predicted_measurement is not None:
        if ax2 is None:
            if estimate_name == 'CM estimate':
                fig2, ax2 = plt.subplots(num=20, clear=True)
            else:
                fig2, ax2 = plt.subplots(num=2, clear=True)
        
        electrodes = np.arange(1, 23)
        ax2.semilogy(electrodes, inv_model.measurement, 'kx', label='Measurement')
        ax2.semilogy(electrodes, predicted_measurement, label='Model fit')
        
        if data_stds is None:
            fill_top1 = predicted_measurement + 1 * inv_model.noise_std
            fill_bottom1 = predicted_measurement - 1 * inv_model.noise_std
        else:
            fill_top1 = data_stds[0]
            fill_bottom1 = data_stds[1]
        
        ax2.fill_between(electrodes, fill_top1, fill_bottom1, alpha=0.2, facecolor='b',
                         label='Noise standard deviation'
                         )
        
        if np.max(predicted_measurement) > 1e4:
            max_y = 1e5
        else:
            max_y = 1e4
        ax2.set_ylim([1e0, max_y])
        ax2.set_title('Measurement and model fit')
        ax2.set_ylabel('Current (fA)')
        ax2.set_xlabel('Electrode number')
        ax2.legend()
        ax2.grid()
        plt.pause(0.01)


def plot_marginal_posterior(ax, sampler, slice_index, color, name, **kwargs):
    """ Plot a histogram of a marginal posterior using MCMC samples. Samples are assumed
    to be normalised by the bin width. Specify the histogram bins you want. """
    
    hist_bins = np.logspace(7.9, 8.7, 50)
    
    burnin = int(sampler.mcmclen * 0.25)
    samples = 10**sampler.N0_samples[burnin:, slice_index] / sampler.inv_model.bins.width
    
    density, bin_edges = np.histogram(samples, bins=hist_bins, density=True)
    
    ax.plot(bin_edges[1:], density, color=color, label=name, **kwargs)
    ax.set_xscale('log')
    ax.legend()
