# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from ProposalDensity import ProposalDensity
from Gauss_Newton_methods import minimize
from visualization import plot_system
from autocorrelation import autocorr_time, autocorr_func_1d



class ProposedState():
    """ Stores all variables related to a MCMC proposal."""
    
    def __init__(self):
        self.reset()
    
    
    def reset(self):
        self.alpha = float('-inf')
        self.N = None
        self.aux_params = None
        self.log_posterior = None
        self.gradient = None
        self.proposal_type = None
    
    
    def set_state(self, alpha, N, aux_params, log_posterior, gradient, proposal_type):
        self.alpha = alpha
        self.N = N
        self.aux_params = aux_params
        self.log_posterior = log_posterior
        self.gradient = gradient
        self.proposal_type = proposal_type


class MCMCSampler():
    
    def __init__(self, inv_model, length, init_aux_params, aux_to_sample):
        
        ''' Input:
            inv_model = Initialized InversionModel object with data loaded
            length = number of MCMC samples
            init_aux_params = starting values for the aux parameters
            aux_to_sample = indexes of auxiliary (model) variables that are to be sampled
                            0 - temperature
                            1 - fractal dimension
                            2 - primary particle diameter
                            3 - Hamaker constant
                            4 - flow velocity
        '''
        
        self.mcmclen = int(length)
        self.aux_idx = np.array(aux_to_sample, dtype=int)
        
        self.inv_model = inv_model
        
        # Preallocate vectors to store MCMC samples
        self.N0_samples = np.zeros((self.mcmclen, self.inv_model.bins.n))
        self.aux_samples = np.zeros((self.mcmclen, 5))
        self.log_posts = np.zeros(self.mcmclen)
        
        # Initial values for the parameters of the aerosol model
        self.aux_samples[0, :] = init_aux_params
        
        # Update AerosolModel with the aux parameters
        self.inv_model.meas_system.sampling_line.update_aerosol_model(
            temperature=self.aux_samples[0, 0],
            fractal_dim=self.aux_samples[0, 1],
            primary_particle_diam=self.aux_samples[0, 2],
            hamaker_constant=self.aux_samples[0, 3],
            flow_velocity=self.aux_samples[0, 4]
            )
                
        # Compute MAP and posterior covariance estimate as a starting point for MCMC
        self.MAP, self.cov_MAP, _ = minimize(self.inv_model, show_output=False)
        
        # If the returned matrix is not positive definite, add something small to the diagonal
        # to hopefully coerce it pd.
        if np.any(np.linalg.eig(self.cov_MAP)[0] < 0):
            self.cov_MAP += 1e-9 * np.eye(self.cov_MAP.shape[0])
        
        self.iter = 0
        self.proposed_state = ProposedState()
        
        # Start MCMC at the MAP estimate
        self.N0_samples[0, :] = self.MAP
        
        # Evaluate posterior at starting location
        self.log_posts[0], self.grad_old = self.inv_model.log_post_and_gradient(self.N0_samples[0])
        
        # Specify proposal density for N
        try:
            self.prop_N0 = ProposalDensity(self.cov_MAP, self.mcmclen, target_acceptance_rate=0.57)
        except np.linalg.LinAlgError:
            # The covariance matrix was probably not positive definite, try to fix it by adding
            # something on the diagonal
            self.cov_MAP += 1e-9 * np.eye(self.cov_MAP.shape[0])
            self.prop_N0 = ProposalDensity(self.cov_MAP, self.mcmclen, target_acceptance_rate=0.57)
        self.prop_N0.update_cholesky()
        self.inv_pchol = np.linalg.inv(self.prop_N0.cholesky)
        
        # Specity proposal density for AerosolModel parameters
        init_prop_covariance_aux_full = np.array([10, 0.1, 1e-9, 1e-20, 0.1])**2
        self.init_prop_covariance_aux = init_prop_covariance_aux_full[self.aux_idx]
        self.prop_aux = ProposalDensity(self.init_prop_covariance_aux, self.mcmclen,
                                        target_acceptance_rate=0.34)
        
        # Do some precomputations to speed up MCMC evaluation
        self.inv_model.meas_system.sampling_line.prepare_for_mcmc()
    
    
    def update_N(self):
        """ Propose an (adaptive) MALA update to the PSD values. """
        
        if self.grad_old is None:
            # There has been an update in the aux parameters --> need to recalculate gradient
            _, self.grad_old = self.inv_model.log_post_and_gradient(self.N0_samples[self.iter])
        
        # Draw a MALA update for N (drift towards the gradient)
        tau = self.prop_N0.AM_factor * self.prop_N0.pcoeff
        step =  tau * self.prop_N0.covariance @ self.grad_old \
            + np.sqrt(2 * tau) * self.prop_N0.cholesky @ np.random.normal(
                size=self.inv_model.bins.n)
        
        new_N0 = self.N0_samples[self.iter] + step
        
        self.prop_N0.n_proposed += 1
        
        if np.any(new_N0 > 12):
            new_post = -1e100
            grad_new = None
            ratio = 0
            
        else:
            new_post, grad_new = self.inv_model.log_post_and_gradient(new_N0)
            
            ratio = - 1 / (4 * tau) * (
                np.linalg.norm(
                    self.inv_pchol @ (-step - tau * self.prop_N0.covariance @ grad_new)
                    )**2
                - np.linalg.norm(
                    self.inv_pchol @ (step - tau * self.prop_N0.covariance @ self.grad_old)
                    )**2
                )
            
        alpha = np.min((0., new_post - self.log_posts[self.iter] + ratio))
        
        self.proposed_state.set_state(
            alpha, new_N0, self.aux_samples[self.iter], new_post, grad_new, 'N'
            )
        
        self.prop_N0.update_AM_factor(alpha)
    
    
    def update_auxiliary_params(self):
        # Update aux parameters (using adaptive Metropolis)
        
        step = np.zeros(5)
        step[self.aux_idx] = self.prop_aux.draw_proposal()  # Keep correct ordering of aux params
        new_aux = self.aux_samples[self.iter] + step
        
        self.inv_model.meas_system.sampling_line.store_current_parameters()
        
        if self.inv_model.outside_aux_param_bounds(new_aux):
            new_post = -1e100
            
        else:
            self.inv_model.meas_system.sampling_line.update_aerosol_model(
                temperature=new_aux[0],
                fractal_dim=new_aux[1],
                primary_particle_diam=new_aux[2],
                hamaker_constant=new_aux[3],
                flow_velocity=new_aux[4]
                )
            
            new_post = self.inv_model.log_post_and_gradient(
                self.N0_samples[self.iter], return_grad=False
                )
            
        alpha = np.min((0., new_post - self.log_posts[self.iter]))
        
        self.proposed_state.set_state(
            alpha, self.N0_samples[self.iter], new_aux, new_post, None, 'aux'
            )
        
        self.prop_aux.update_AM_factor(alpha)
    
    
    def compute_one_iteration(self):
        """ Compute one iteration of the MCMC sampler. """
        
        self.proposed_state.reset()
        
        # Propose to update either N or AerosolModel parameters
        if self.aux_idx.size == 0 or np.random.random(1) < 0.5:
            self.update_N()
        else:
            self.update_auxiliary_params()
        
        # Accept / reject step
        if np.log(np.random.rand(1)) < self.proposed_state.alpha:
            
            self.N0_samples[self.iter + 1, :] = self.proposed_state.N
            self.aux_samples[self.iter + 1, :] = self.proposed_state.aux_params
            self.log_posts[self.iter + 1] = self.proposed_state.log_posterior
            
            if self.proposed_state.proposal_type == 'N':
                self.prop_N0.n_accepted += 1
                self.grad_old = self.proposed_state.gradient
            
            elif self.proposed_state.proposal_type == 'aux':
                self.prop_aux.n_accepted += 1
                self.grad_old = None  # Make sure we don't use the old gradient anymore
        
        else:
            self.N0_samples[self.iter + 1, :] = self.N0_samples[self.iter]
            self.aux_samples[self.iter + 1, :] = self.aux_samples[self.iter]
            self.log_posts[self.iter + 1] = self.log_posts[self.iter]
            
            if self.proposed_state.proposal_type == 'aux':
                # Return to the old parameters
                self.inv_model.meas_system.sampling_line.restore_stored_parameters()
        
        # Update proposal covariance iteratively with the new sample
        if self.proposed_state.proposal_type == 'N':
            self.prop_N0.update_covariance(self.N0_samples[self.iter + 1])
        
        elif self.proposed_state.proposal_type == 'aux':
            # Use only self.aux_idx parameters
            self.prop_aux.update_covariance(self.aux_samples[self.iter + 1, self.aux_idx])
        
        # Compute the Cholesky decomposition of the proposal covariance
        if self.iter > 1000 and self.iter % 100 == 0:
            self.prop_N0.update_cholesky()
            self.prop_aux.update_cholesky()
            self.inv_pchol = np.linalg.inv(self.prop_N0.cholesky)
        
        self.iter += 1
    
    
    def run_mcmc(self):
        
        for ii in tqdm(range(self.mcmclen - 1)):
            self.compute_one_iteration()
        
        self.prop_N0.remove_leftover_zeros()
        self.prop_aux.remove_leftover_zeros()
        
        if self.prop_N0.n_proposed > 0:
            print(f'N_0 acceptance rate: {self.prop_N0.n_accepted / self.prop_N0.n_proposed :.2f}')
        if self.prop_aux.n_proposed > 0:
            print(f'Aux acceptance rate: {self.prop_aux.n_accepted / self.prop_aux.n_proposed :.2f}')
    
    
    def summarise_posterior(self, ax=None):
        """ Compute and plot conditional mean and credible interval estimates using the collected
        MCMC samples. """
        
        self.burnin = int(self.mcmclen * 0.25)
        self.CM_est = np.mean(self.N0_samples[self.burnin:self.iter, :], axis=0)
        modelled_measurement = self.inv_model.meas_system.forward_model(self.CM_est)
        
        self.N1_CM = np.log10(self.inv_model.meas_system.sampling_line.run(10**self.CM_est))
        
        n_samples = np.min((1000, self.N0_samples.shape[0] - self.burnin))
        
        N1_samples = np.zeros((n_samples, self.N0_samples.shape[1]))
        data_samples = np.zeros((n_samples, modelled_measurement.shape[0]))
        
        idxs = np.arange(self.burnin, self.N0_samples.shape[0],
                          int((self.N0_samples.shape[0] - self.burnin) / n_samples))
        for i in range(n_samples):
            N1_abs_scale = self.inv_model.meas_system.sampling_line.run(10**self.N0_samples[idxs[i]])
            N1_samples[i] = np.log10(N1_abs_scale)
            data_samples[i] = self.inv_model.meas_system.EEPS.apply_meas_op(
                self.inv_model.meas_system.dilute(N1_abs_scale))
            data_samples[i] += self.inv_model.noise_std * np.random.normal(size=data_samples[i].shape)
        
        self.post_cov = np.cov(self.N0_samples[self.burnin:self.iter, :].T)
        
        # Credible intervals: find tails of the distribution using percentiles so that the wanted
        # percentage of posterior mass (68 %, 95 %, ...) lies in between.
        # pr1 = 50 + 68.27/2;
        # pr2 = 50 + 95.45/2;
        # pr3 = 50 + 99.73/2;
        pr1 = 50 + 95/2;
        
        pr_plot = np.percentile(
            self.N0_samples[self.burnin:self.iter, :], (100 - pr1, pr1), axis=0
            )
        
        pr_plot_data = np.percentile(
            data_samples, (100 - pr1, pr1), axis=0
            )
        
        if ax is None:
            plot_system(
                self.inv_model, self.CM_est, N_1=self.N1_CM,
                predicted_measurement=modelled_measurement, posterior_CIs=pr_plot,
                estimate_name='CM estimate', data_stds=pr_plot_data
                )
        else:
            plot_system(
                self.inv_model, self.CM_est, N_1=self.N1_CM, posterior_CIs=pr_plot, 
                estimate_name='CM estimate', ax1=ax
                )
        
        print(f'PN N_0 over PN N_1: {np.sum(10**self.CM_est) / np.sum(10**self.N1_CM): .2f}')
        print('--')
        print('CM (N0 - N1) / N0:')
        print((10**self.CM_est - 10**self.N1_CM) / 10**self.CM_est)
        print('--')
        if self.inv_model.meas_type == 'Synthetic':
            
            # Interpolate truth to the inverse grid
            truth_interp = np.interp(
                self.inv_model.bins.centers, self.inv_model.truth_bins.centers,
                self.inv_model.true_N_0
                ) / self.inv_model.truth_bins.width
            
            rel_error = 100 * np.linalg.norm(
                truth_interp - 10**self.CM_est / self.inv_model.bins.width
                ) / np.linalg.norm(truth_interp)
            
            print(f'Relative error between truth and CM: {rel_error : .1f} %')
    
    
    def plot_chains(self):
    
        markersize = 0.5
        linewidth = 0.5
        pltstyle = '-'
        start_plot = 0  # Index from which we start plotting
        
        plt.figure(num=50), plt.clf()
        plt.subplot(511)
        pltidx = int(np.floor(0 * self.inv_model.bins.n / 5))
        plt.plot(self.N0_samples[start_plot:self.iter, pltidx], pltstyle, markersize=markersize,
                 linewidth=linewidth,
                  label=f'IACT: {autocorr_time(self.N0_samples[self.burnin:self.iter, pltidx]):.0f}')
        plt.legend(loc='lower left')
        plt.subplot(512)
        pltidx = int(np.floor(1 * self.inv_model.bins.n / 5))
        plt.plot(self.N0_samples[start_plot:self.iter, pltidx], pltstyle, markersize=markersize,
                 linewidth=linewidth,
                  label=f'IACT: {autocorr_time(self.N0_samples[self.burnin:self.iter, pltidx]):.0f}')
        plt.legend(loc='lower left')
        plt.subplot(513)
        pltidx = int(np.floor(2 * self.inv_model.bins.n / 5))
        plt.plot(self.N0_samples[start_plot:self.iter, pltidx], pltstyle, markersize=markersize,
                 linewidth=linewidth,
                  label=f'IACT: {autocorr_time(self.N0_samples[self.burnin:self.iter, pltidx]):.0f}')
        plt.legend(loc='lower left')
        plt.subplot(514)
        pltidx = int(np.floor(3 * self.inv_model.bins.n / 5))
        plt.plot(self.N0_samples[start_plot:self.iter, pltidx], pltstyle, markersize=markersize,
                 linewidth=linewidth,
                  label=f'IACT: {autocorr_time(self.N0_samples[self.burnin:self.iter, pltidx]):.0f}')
        plt.legend(loc='lower left')
        plt.subplot(515)
        pltidx = int(np.floor(4 * self.inv_model.bins.n / 5))
        plt.plot(self.N0_samples[start_plot:self.iter, pltidx], pltstyle, markersize=markersize,
                 linewidth=linewidth,
                  label=f'IACT: {autocorr_time(self.N0_samples[self.burnin:self.iter, pltidx]):.0f}')
        plt.legend(loc='lower left')
        plt.draw()
        
        plt.figure(num=51), plt.clf()
        plt.subplot(511)
        plt.plot(self.aux_samples[start_plot:self.iter, 0], pltstyle, markersize=markersize,
                 linewidth=linewidth,
                  label=f'IACT: {autocorr_time(self.aux_samples[self.burnin:self.iter, 0]):.0f}')
        plt.legend(loc='lower left')
        plt.subplot(512)
        plt.plot(self.aux_samples[start_plot:self.iter, 1], pltstyle, markersize=markersize,
                 linewidth=linewidth,
                  label=f'IACT: {autocorr_time(self.aux_samples[self.burnin:self.iter, 1]):.0f}')
        plt.legend(loc='lower left')
        plt.subplot(513)
        plt.plot(self.aux_samples[start_plot:self.iter, 2], pltstyle, markersize=markersize,
                 linewidth=linewidth,
                  label=f'IACT: {autocorr_time(self.aux_samples[self.burnin:self.iter, 2]):.0f}')
        plt.legend(loc='lower left')
        plt.subplot(514)
        plt.plot(self.aux_samples[start_plot:self.iter, 3], pltstyle, markersize=markersize,
                 linewidth=linewidth,
                  label=f'IACT: {autocorr_time(self.aux_samples[self.burnin:self.iter, 3]):.0f}')
        plt.legend(loc='lower left')
        plt.subplot(515)
        plt.plot(self.aux_samples[start_plot:self.iter, 4], pltstyle, markersize=markersize,
                 linewidth=linewidth,
                  label=f'IACT: {autocorr_time(self.aux_samples[self.burnin:self.iter, 4]):.0f}')
        plt.legend(loc='lower left')
        plt.draw()
        
        plt.figure(num=52), plt.clf()
        plt.subplot(211)
        plt.plot(self.log_posts, label='log posterior')
        plt.xlabel('Iteration')
        plt.legend()
        plt.subplot(212)
        plt.semilogy(self.prop_N0.AM_factors, label='N0 Proposal scale factor')
        plt.semilogy(self.prop_aux.AM_factors, label='Aux Proposal scale factor')
        plt.legend()
        plt.draw()
        
        acflen = np.min([1000, self.burnin])
        thin = 1
        acflen = int(acflen / thin)
        total_iact = 0
        plt.figure(num=12), plt.clf()
        for i in range(self.inv_model.bins.n):
            plt.plot(
                thin * np.arange(acflen),
                autocorr_func_1d(self.N0_samples[self.burnin::thin, i])[:acflen],
                'k', alpha=0.5
                )
            total_iact += autocorr_time(self.N0_samples[self.burnin:, i])
        avg_iact = total_iact / self.inv_model.bins.n
        plt.plot(np.array([0, thin * acflen]), np.array([0, 0]), 'k', linewidth=2)
        plt.title(
            f'Autocorrelation functions {"":<10} Avg. IACT: {avg_iact:.1f}'
            )
        plt.grid()
        plt.draw()
