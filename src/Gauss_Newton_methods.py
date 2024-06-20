# -*- coding: utf-8 -*-

import time
import numpy as np

from visualization import plot_system


def minimize(inv_model, show_output=True):
    # Compute Gauss-Newton minimization
    
    max_iter = int(50)
    
    N_0_est = np.zeros((max_iter + 1, inv_model.bins.n))
    N_0_est[0, :] = inv_model.initial_guess
    
    f_values = np.zeros(max_iter + 1)
    f_values[0] = inv_model.neg_log_posterior(N_0_est[0, :])
    
    # Zeroth iteration
    modelled_measurement, J = inv_model.meas_system.forward_model(N_0_est[0], return_Jacobian=True)
    inv_H = np.linalg.inv(J.T @ inv_model.noise_Icov @ J + inv_model.W)
    post_cov = inv_H
    
    # # Plot the starting guess
    # if show_output:
    #     plot_system(ex, N_0_est[0], post_cov, predicted_measurement=modelled_measurement,
    #                 jacobian=J, iteration_no=0)
    #     plt.pause(1)
    
    # Setting up for the first iteration
    ii = 0
    min_step_reached = False
    enough_improvement = True
    required_improvement = 1e-3  # Minimum relative change in functional to keep iterating
    
    start_time = time.perf_counter()
    while (ii < max_iter) and not min_step_reached and enough_improvement:
        
        grad = (J.T @ inv_model.noise_Icov) @ (inv_model.measurement - modelled_measurement) \
            - inv_model.W @ (N_0_est[ii] - inv_model.N_0_prior)
        
        GN_dir =  inv_H @ grad
        
        # Line search
        N_0_est[ii + 1], f_values[ii + 1], min_step_reached = linesearch(
            inv_model.neg_log_posterior, GN_dir, N_0_est[ii], f_values[ii]
            )
        
        if (f_values[ii] - f_values[ii + 1]) / f_values[ii] < required_improvement:
            enough_improvement = False
        
        # Evaluate model and linearise at the new MAP
        modelled_measurement, J = inv_model.meas_system.forward_model(N_0_est[ii + 1],
                                                                      return_Jacobian=True)
        inv_H = np.linalg.inv(J.T @ inv_model.noise_Icov @ J + inv_model.W)
        post_cov = inv_H  # Laplace approximation
        
        # if show_output:
        #     # Calculate PSD after the sampling line
        #     N_1 = np.log10(inv_model.meas_system.sampling_line.run(10**N_0_est[ii + 1]))
        #     plot_system(inv_model, N_0_est[ii + 1], post_cov, N_1=N_1,
        #                 predicted_measurement=modelled_measurement)
        
        ii += 1
    end_time = time.perf_counter()
    
    
    MAP_est = N_0_est[ii]
    
    if show_output:
        
        # Calculate PSD after the sampling line
        N_1 = np.log10(inv_model.meas_system.sampling_line.run(10**MAP_est))
        
        if ii == max_iter:
            print(f'Converged - Maximum iteration count reached ({max_iter})')
        if not enough_improvement:
            print(
                f'Converged - Improvement in functional less than {required_improvement * 100} %'
                )
        if min_step_reached:
            print('Converged - Minimum step length reached')
        print(f' ------ Inversion took {end_time - start_time :.2g} seconds')
        PN_tot_init = np.sum(10**MAP_est)  # Total PN before the sampling line
        PN_tot_after = np.sum(10**N_1)  # Total PN after the sampling line
        print(f'Total PN N_MAP: {PN_tot_init : .3g}')
        print(f'Total PN after sampling line: {PN_tot_after : .3g}')
        print('Rel. increase in total PN w.r.t. PN after sampling line:'
              + f' {100 * (PN_tot_init - PN_tot_after) / PN_tot_after : .1f} %')
        print('--')
        if inv_model.meas_type == 'Synthetic':
            
            # Interpolate truth to inverse grid
            truth_interp = np.interp(
                inv_model.bins.centers, inv_model.truth_bins.centers, inv_model.true_N_0
                )
            
            # Keep particle number the same
            truth_interp *= inv_model.bins.width / inv_model.truth_bins.width
            
            rel_error = 100 * np.linalg.norm(truth_interp - 10**MAP_est
                                        ) / np.linalg.norm(truth_interp)
            print(f'Relative error between truth and MAP: {rel_error : .1f} %')
    
        plot_system(inv_model, MAP_est, post_cov, N_1=N_1,
                    predicted_measurement=modelled_measurement)
    
    return MAP_est, post_cov, modelled_measurement


def linesearch(fn, direction, N_0, previous_best_f_value):
    ''' Do simple linesearch.
    fn : function to be minimized (in our case the negative log posterior)
    '''
    
    min_stepl = 1e-3
    
    # Brute force, backtrack until the functional value increases, then choose previous step
    stepl = 1
    reduce = 0.7
    dN_old = stepl * direction
    while any((N_0 + dN_old) > 10):
        stepl *= reduce
        dN_old = stepl * direction
    post_old = fn(N_0 + dN_old)
    
    found_best_value = False
    while not found_best_value:
        stepl *= reduce
        dN_new = stepl * direction
        post_new = fn(N_0 + dN_new)
        
        if((post_new < previous_best_f_value and post_new > post_old)
            or stepl < (min_stepl * reduce)
            ):
            # Output the second to last iteration values (which was the best value)
            found_best_value = True
            post_new = post_old
            dN_new = dN_old
            stepl /= reduce  # undo the last reduction in step length
        
        else:
            # carry forward the previous iteration values
            dN_old = dN_new
            post_old = post_new
    
    
    # # For debug
    # print(f'Step length: {stepl:.2g}')
    
    if stepl < min_stepl:
        min_step_reached = True
    else:
        min_step_reached = False
    
    return N_0 + dN_new, post_new, min_step_reached
