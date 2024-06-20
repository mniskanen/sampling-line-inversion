# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as tck

from matplotlib.transforms import (Bbox, TransformedBbox,
                                   ScaledTranslation)
from mpl_toolkits.axes_grid1.inset_locator import (BboxConnector,
                                                   BboxConnectorPatch,
                                                   BboxPatch)

from test_coagulation_coefficients import plot_coag_coeffs
from InversionModel import InversionModel
from invert_synthetic import invert_synthetic_GN
from run_mcmc_all_aux import load_run
from invert_AHmeasurements import invert_AH_GN
from visualization import plot_system, plot_marginal_posterior, EdgeFiller
from Gauss_Newton_methods import minimize


"""
A script to create and save all the PSD estimation figures in the paper.

NOTE:
The Laplace approximation results are computed on the fly because they're fast enough, but the
MCMC results are loaded from file. Because of this, before running this script, run
'run_mcmc_all_aux.py' which does the MCMC and saves the results.
The MCMC script will take some hours in total with the default (as in, used in the paper)
chain length of 200,000 samples. Shorten that to reduce the running time (with the obvious
trade off of increased Monte Carlo error). Or, then just don't run the MCMC parts of this script.
"""

SAVE_FIGURES = False

# If you set this to True, calculate MCMC results first by running 'run_mcmc_all_aux.py'
PLOT_MCMC_RESULTS = False

FIGURES_FOLDER = '../results/figures/'  # Save the figures to this folder
RESULTS_FOLDER = '../results/MCMC/'  # Where to load the saved MCMC runs

FIG_WIDTH = 7.2
FIG_HEIGHT = 4.8
DPI = 300
plt.rcParams['axes.xmargin'] = 0.02

plt.close('all')


def format_PSD_axis(ax):
    ''' Apply common formatting for all PSD plots we'll show. '''
    
    # Do some manual configuration for the x-axis:
    
    # Use scalar format, not powers of ten
    ax.xaxis.set_major_formatter(tck.ScalarFormatter())
    
    # Don't print any decimals
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%.0f'))
    
    # Set xticks we want to show (now the first one will show as 6 but placed at 5.6)
    ax.set_xticks([5.6, 10, 20, 50, 100, 250, 562])
    
    # Modify first tick to show the correct value
    ticklabels = ax.get_xticklabels()
    ticklabels[0].set_text('5.6')
    ax.set_xticklabels(ticklabels)
    
    # Other stuff -----------------
    ax.grid(visible=False, which='both')
    ax.grid('on', which='major')



#%% Coagulation coefficient examples

fig, axs = plt.subplots(1, 2, num=62, clear=True)
fig.set_figwidth(2 * FIG_WIDTH)
fig.set_figheight(FIG_HEIGHT)

plot_coag_coeffs(axs[0], axs[1])

axs[0].set_title('Spherical vs. fractal')
axs[0].set_title('a)', loc='left')
format_PSD_axis(axs[0])
axs[1].set_title('Spherical vs. van der Waals/viscous')
axs[1].set_title('b)', loc='left')
format_PSD_axis(axs[1])

if SAVE_FIGURES:
    plt.savefig(FIGURES_FOLDER + 'f01.png', dpi=DPI, bbox_inches='tight')


#%% Prior covariance matrix

fig, ax = plt.subplots(num=63, clear=True)
fig.set_figwidth(FIG_WIDTH)
fig.set_figheight(FIG_HEIGHT)

inv_model = InversionModel(bins_per_decade=16)

X, Y = np.meshgrid(np.arange(1, 33), np.arange(1, 33))
Z = inv_model.pr_cov
im = ax.pcolormesh(X, Y, Z)
ax.invert_yaxis()
ax.set_aspect('equal')
ax.set_xlabel('Size bin no.')
ax.set_ylabel('Size bin no.')
ax.set_title('Prior covariance')
fig.colorbar(im, label=f'(log N)$^{2}$')

if SAVE_FIGURES:
    plt.savefig(FIGURES_FOLDER + 'f02.png', dpi=DPI, bbox_inches='tight')


#%% Simulated data, known aux parameters, Laplace approximation

fig, axs = plt.subplots(1, 2, num=64, clear=True)
fig.set_figwidth(2 * FIG_WIDTH)
fig.set_figheight(FIG_HEIGHT)

inv_model, result = invert_synthetic_GN(1.7, show_output=False)
MAP_est = result[0]
post_cov = result[1]
modelled_measurement = result[2]

N_1 = np.log10(inv_model.meas_system.sampling_line.run(10**MAP_est))

plot_system(inv_model, MAP_est, post_cov, N_1=N_1, predicted_measurement=modelled_measurement,
            ax1=axs[1], ax2=axs[0])

axs[0].xaxis.set_major_locator(tck.IndexLocator(base=3, offset=0))
axs[0].set_title('a)', loc='left')
axs[0].set_ylim((1e-1, 1e4))
axs[0].grid(visible=False, which='both')
axs[0].grid('on', which='major')

format_PSD_axis(axs[1])
axs[1].set_ylim((1e1, 1e9))
axs[1].set_title('b)', loc='left')
axs[1].set_title('Estimated particle size distributions (Laplace approx.)')

if SAVE_FIGURES:
    plt.savefig(FIGURES_FOLDER + 'f04.png', dpi=DPI, bbox_inches='tight')



#%% Simulated data, known aux parameters, MCMC

if PLOT_MCMC_RESULTS:
    
    fig, axs = plt.subplot_mosaic([
        ["Zoom 1", "Zoom 2", "PSD", "PSD"],
        ["Full chain", "Full chain", "PSD", "PSD"],
    ], num=65, clear=True)
    fig.set_figwidth(2 * FIG_WIDTH)
    fig.set_figheight(FIG_HEIGHT)
    
    
    def connect_bbox(bbox1, bbox2,
                     loc1a, loc2a, loc1b, loc2b,
                     prop_lines, prop_patches=None):
        if prop_patches is None:
            prop_patches = {
                **prop_lines,
                "alpha": prop_lines.get("alpha", 1) * 0.2,
                "clip_on": False,
            }
    
        c1 = BboxConnector(
            bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
        c2 = BboxConnector(
            bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)
    
        bbox_patch1 = BboxPatch(bbox1, **prop_patches)
        bbox_patch2 = BboxPatch(bbox2, **prop_patches)
    
        p = BboxConnectorPatch(bbox1, bbox2,
                               loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                               clip_on=False,
                               **prop_patches)
    
        return c1, c2, bbox_patch1, bbox_patch2, p
    
    
    def zoom_effect01(ax1, ax2, xmin, xmax, **kwargs):
        """
        Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will
        be marked.
    
        Parameters
        ----------
        ax1
            The main axes.
        ax2
            The zoomed axes.
        xmin, xmax
            The limits of the colored area in both plot axes.
        **kwargs
            Arguments passed to the patch constructor.
        """
        
        bbox = Bbox.from_extents(xmin, 0, xmax, 1)
    
        mybbox1 = TransformedBbox(bbox, ax1.get_xaxis_transform())
        mybbox2 = TransformedBbox(bbox, ax2.get_xaxis_transform())
    
        prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}
    
        c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
            mybbox1, mybbox2,
            loc1a=3, loc2a=2, loc1b=4, loc2b=1,
            prop_lines=kwargs, prop_patches=prop_patches)
    
        ax1.add_patch(bbox_patch1)
        ax2.add_patch(bbox_patch2)
        ax2.add_patch(c1)
        ax2.add_patch(c2)
        ax2.add_patch(p)
    
        return c1, c2, bbox_patch1, bbox_patch2, p
    
    
    resuts_folder = '../results/Jun11/'
    sampler = load_run(resuts_folder, 'aux0_simulated')
    # sampler = load_run(RESULTS_FOLDER, 'true_aux_simulated')
    sampler.summarise_posterior(ax=axs["PSD"])
    
    sizebin = 0
    markersize = 0.5
    linewidth = 0.5
    pltstyle = '-'
    
    axs["Full chain"].plot(sampler.N0_samples[0:sampler.iter, sizebin],
                           pltstyle, markersize=markersize, linewidth=linewidth
                           )
    axs["Full chain"].set_xlabel('Iteration number')
    axs["Full chain"].set_ylabel(r'$\tilde{\mathbf{N}}$')
    
    axs["Zoom 1"].plot(sampler.N0_samples[0:10000, sizebin],
                       pltstyle, markersize=markersize, linewidth=linewidth
                       )
    axs["Zoom 1"].autoscale(enable=True, axis='x', tight=True)
    axs["Zoom 1"].set_xticklabels([])
    zoom_effect01(axs["Zoom 1"], axs["Full chain"], 0, 10000)
    
    # x-limits for the second zoom
    lim_a = 150000
    lim_b = 160000
    
    if sampler.mcmclen < lim_b:
        print('Cannot draw the second zoom because MCMC length was too short')
        
    else:
        axs["Zoom 2"].plot(np.arange(lim_a, lim_b), sampler.N0_samples[lim_a:lim_b, sizebin],
                           pltstyle, markersize=markersize, linewidth=linewidth)
        axs["Zoom 2"].autoscale(enable=True, axis='x', tight=True)
        axs["Zoom 2"].set_xticklabels([])
        zoom_effect01(axs["Zoom 2"], axs["Full chain"], lim_a, lim_b)
        
        for label, ax in axs.items():
            # label physical distance to the left and up:
            trans = ScaledTranslation(72/72, 7/72, plt.gcf().dpi_scale_trans)
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', va='bottom')#, fontfamily='serif')
            if label == 'Zoom 2':
                # a way to not print the full chain label...
                break
    
    axs['Zoom 1'].set_title('a)', loc='left')
    axs['Zoom 2'].set_title('b)', loc='left')
    axs['Full chain'].set_title('c)', loc='center')
    
    axs['PSD'].set_title('d)', loc='left')
    axs['PSD'].set_title('Estimated particle size distributions (MCMC)')
    format_PSD_axis(axs['PSD'])
    axs['PSD'].set_ylim((1e1, 1e9))
    
    plt.tight_layout()
    
    if SAVE_FIGURES:
        plt.savefig(FIGURES_FOLDER + 'f05.png', dpi=DPI, bbox_inches='tight')


#%% Simulated data, wrong Df, MCMC

if PLOT_MCMC_RESULTS:
    
    fig, axs = plt.subplots(1, 2, num=66, clear=True)
    fig.set_figwidth(2 * FIG_WIDTH)
    fig.set_figheight(FIG_HEIGHT)
    
    sampler = load_run(RESULTS_FOLDER, 'aux0_simulated')
    sampler.summarise_posterior(axs[0])
    
    axs[0].set_title('a)', loc='left')
    axs[0].set_title('Estimated particle size distributions (MCMC), wrong $D_f$')
    format_PSD_axis(axs[0])
    axs[0].set_ylim((1e1, 1e9))
    
    # Simulated data, Df marginalized, MCMC
    
    sampler = load_run(RESULTS_FOLDER, 'aux2_simulated')
    sampler.summarise_posterior(axs[1])
    
    axs[1].set_title(r'b) Estimated particle size distributions (MCMC), $D_f$ marginalized')
    format_PSD_axis(axs[1])
    axs[1].set_ylim((1e1, 1e9))
    
    if SAVE_FIGURES:
        plt.savefig(FIGURES_FOLDER + 'f06.png', dpi=DPI, bbox_inches='tight')



#%% Double plot:
    # 1) Simulated data, marginalized over all aux parameters, MCMC
    # 2) Simulated data, marginalized aux parameters, MCMC

if PLOT_MCMC_RESULTS:
    
    fig, axs = plt.subplots(nrows=1, ncols=2, num=67, clear=True)
    fig.set_figwidth(2 * FIG_WIDTH)
    fig.set_figheight(FIG_HEIGHT)
    fig.subplots_adjust(wspace=0.1)
    
    # Figure 1 ------------------------------------------------------------
    
    sampler = load_run(RESULTS_FOLDER, 'aux6_simulated')
    sampler.summarise_posterior(axs[0])
    
    axs[0].set_title('a)', loc='left')
    axs[0].set_title('Estimated particle size distributions (MCMC)')
    format_PSD_axis(axs[0])
    axs[0].set_ylim((1e1, 1e9))
    
    # Figure 2 ------------------------------------------------------------
    
    sampler_noaux = load_run(RESULTS_FOLDER, 'aux0_simulated')
    sampler_aux1 = load_run(RESULTS_FOLDER, 'aux2_simulated')
    sampler_aux3 = load_run(RESULTS_FOLDER, 'aux4_simulated')
    sampler_aux4 = load_run(RESULTS_FOLDER, 'aux5_simulated')
    sampler_aux5 = load_run(RESULTS_FOLDER, 'aux6_simulated')
    
    slice_index = 5
    detail_diameter = sampler_noaux.inv_model.bins.centers[slice_index] * 1e9
    
    # Add an annotation to left plot
    axs[0].annotate(f'{detail_diameter :.2f} nm bin', xy=(detail_diameter, 2e7),
                    xytext=(detail_diameter, 5e5), ha='center',
                arrowprops=dict(arrowstyle="-|>", facecolor='black'))
    axs[0].annotate('', xy=(detail_diameter, 2e4),
                    xytext=(detail_diameter, 4e5), ha='center',
                arrowprops=dict(arrowstyle="-|>", facecolor='black'))
    
    plot_marginal_posterior(axs[1], sampler_noaux, slice_index, 'k', 'All aux. parameters fixed')
    plot_marginal_posterior(axs[1], sampler_aux4, slice_index, 'C3', 'Marginalized flow velocity')
    plot_marginal_posterior(axs[1], sampler_aux3, slice_index, 'C2', 'Marginalized Hamaker constant')
    plot_marginal_posterior(axs[1], sampler_aux1, slice_index, 'C1', 'Marginalized fractal dimension')
    plot_marginal_posterior(axs[1], sampler_aux5, slice_index, 'C0', 'Marginalized all aux. parameters')
    
    axs[1].set_title('b)', loc='left')
    axs[1].set_title(f'Slice through the posterior at $d_m$ = {detail_diameter : .2f} nm')
    axs[1].set_xlabel('dN/dlogdp $(cm^{-3})$')
    axs[1].set_ylabel('Posterior density')
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[1].grid('on', which='both')
    
    # Plot the true value
    truth = np.interp(
        detail_diameter,
        sampler_noaux.inv_model.truth_bins.centers * 1e9,
        sampler_noaux.inv_model.true_N_0
        )
    truth /= sampler_noaux.inv_model.truth_bins.width
    axs[1].plot([truth, truth], [0, plt.ylim()[1] * 0.10], 'k--', linewidth=2, label='True value')
    axs[1].annotate('True initial PSD', xy=(truth*1.03, plt.ylim()[1] * 0.05),
                    xytext=(3.1e8, 1.3e-8), arrowprops=dict(arrowstyle="-|>", facecolor='black')
                    )
    axs[1].set_ylim(0, None)
    
    # Draw connecting lines between subplots
    from matplotlib.patches import ConnectionPatch
    
    detail_xlim = axs[1].get_xlim()
    detail_ylim = axs[1].get_ylim()
    detail_d_m = detail_diameter
    axs[0].plot([detail_d_m, detail_d_m], [detail_xlim[0], detail_xlim[1]], 'k-', linewidth=1,
                alpha=0.5)
    
    con1 = ConnectionPatch(xyA=(detail_d_m, detail_xlim[0]), coordsA=axs[0].transData,
                           xyB=(detail_xlim[0], detail_ylim[0]), coordsB=axs[1].transData)
    con2 = ConnectionPatch(xyA=(detail_d_m, detail_xlim[1]), coordsA=axs[0].transData,
                           xyB=(detail_xlim[1], detail_ylim[0]), coordsB=axs[1].transData)
    con1.set_alpha(0.2)
    con2.set_alpha(0.2)
    
    fig.add_artist(con1)
    fig.add_artist(con2)
    
    
    if SAVE_FIGURES:
        plt.savefig(FIGURES_FOLDER + 'f07.png', dpi=DPI, bbox_inches='tight')


#%% Real measurements (an inverted full 30 min measurement)

fig, ax = plt.subplots(num=68, clear=True)
fig.set_figwidth(2 * FIG_WIDTH)
fig.set_figheight(FIG_HEIGHT)

EEPS_current = np.load('../measurement_data/EEPS_current.npy')
d_m_EEPS = np.logspace(np.log10(5.62e-9), np.log10(562e-9), 33)

# Indexes which are not nan
idx_measurement = np.argwhere(~np.isnan(EEPS_current[:, 0])).squeeze()

# Indexes from the original measurement matrix (now skips nan parts)
n_measurements = idx_measurement.shape[0]

timevec = np.arange(0, n_measurements + 1) / 60  # In minutes

# Do the inversion
result = invert_AH_GN(idx_measurement, show_output=True)
result = 10**result  # To absolute scale

binwidth = 1 / 16
electrodes = np.arange(23)
X, Y = np.meshgrid(timevec, d_m_EEPS * 1e9)
Z = result.T / binwidth
im = ax.pcolormesh(X, Y, Z, norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()))

ax.set_yscale('log')

ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.0f'))
ax.set_yticks([5.6, 10, 20, 50, 100, 250, 562])
ticklabels = ax.get_yticklabels()
ticklabels[0].set_text('5.6')
ax.set_yticklabels(ticklabels)

ax.set_ylabel('Particle diameter (nm)')
ax.set_xlabel('Time (min)')

im.set_clim([10, 6e9])
cbar = fig.colorbar(im, ax=ax, label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$')


# Plot red lines
line1 = 722 / 60
line2 = 1838 / 60
ax.plot(line1 * np.array([1, 1]), np.array([5.6, 560]), 'r--')
ax.plot(line2 * np.array([1, 1]), np.array([5.6, 560]), 'r--')
ax.set_ylim([5.6, 560])

ax.text(line1, 590, 'Line 1', color='r')
ax.text(line2, 590, 'Line 2', color='r')


if SAVE_FIGURES:
    plt.savefig(FIGURES_FOLDER + 'f08.png', dpi=DPI, bbox_inches='tight')


#%% Compute the relative difference between EEPS and MAP estimates
# (have to run above section first)

fig, ax = plt.subplots(num=69, clear=True)
fig.set_figwidth(FIG_WIDTH)
fig.set_figheight(FIG_HEIGHT)

EEPS_dNdlogDp = np.load('../measurement_data/EEPS_dNdlogDp.npy')
EEPS_N = EEPS_dNdlogDp[idx_measurement] * binwidth  # Undo normalization by bin width

# First, do a diffusion correction to the EEPS values (so modelling the sampling line as just
# causing diffusion losses, which is usually done)

from MeasurementSystem import MeasurementSystem
from InversionModel import discretize_particle_size

bins = discretize_particle_size(bins_per_decade=16)
meas_system = MeasurementSystem(bins)

# Use spherical particles to compute the diffusion coefficient
meas_system.sampling_line.update_aerosol_model(
    temperature=250+273,
    fractal_dim=3.0,
    primary_particle_diam=27e-9,
    hamaker_constant=2e-19,
    flow_velocity=3.5,
    compute_vdw=False
    )
particle_penetration = meas_system.sampling_line.diffusion_P
EEPS_N_diffcorrected = (EEPS_N / particle_penetration).T

MAP = result.T

rel_difference = (MAP - EEPS_N_diffcorrected) / EEPS_N_diffcorrected

sm_idx = 9
lg_idx = 20
particlenum_ratio_total = np.sum(MAP, axis=0) / np.sum(EEPS_N_diffcorrected, axis=0)
particlenum_ratio_small = (np.sum(MAP[:sm_idx, :], axis=0)
                           / np.sum(EEPS_N_diffcorrected[:sm_idx, :], axis=0)
                           )
particlenum_ratio_medium = (np.sum(MAP[sm_idx:lg_idx, :], axis=0)
                           / np.sum(EEPS_N_diffcorrected[sm_idx:lg_idx, :], axis=0)
                           )
particlenum_ratio_large = (np.sum(MAP[lg_idx:, :], axis=0)
                           / np.sum(EEPS_N_diffcorrected[lg_idx:, :], axis=0)
                           )
ax.plot(timevec[:-1], particlenum_ratio_small,
         label=fr'Small ({bins.edges[0] * 1e9 : .1f} nm ${{\leq}}$ d${{_m}}$ $<$ {bins.edges[sm_idx] * 1e9 : .1f} nm)')
ax.plot(timevec[:-1], particlenum_ratio_medium,
         label=fr'Medium ({bins.edges[sm_idx] * 1e9 : .1f} nm ${{\leq}}$ d${{_m}}$ $<$ {bins.edges[lg_idx] * 1e9 : .1f} nm)')
ax.plot(timevec[:-1], particlenum_ratio_large,
         label=fr'Large ({bins.edges[lg_idx] * 1e9 : .1f} nm ${{\leq}}$ d${{_m}}$ ${{\leq}}$ {bins.edges[-1] * 1e9 : .1f} nm)')
ax.plot(timevec[:-1], particlenum_ratio_total, 'k-',
         label=fr'All ({bins.edges[0] * 1e9 : .1f} nm ${{\leq}}$ d${{_m}}$ ${{\leq}}$ {bins.edges[-1] * 1e9 : .1f} nm)')
ax.set_title('Ratio of particle numbers: MAP / EEPS')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Ratio of particle numbers')
ax.legend()
ax.grid()
ax.axis([0, timevec[-1], 0, 5])

if SAVE_FIGURES:
    plt.savefig(FIGURES_FOLDER + 'f09.png', dpi=DPI, bbox_inches='tight')


#%% Invert an AH measurement (G-N) (line 1)


fig, axs = plt.subplots(1, 2, num=70, clear=True)
fig.set_figwidth(2 * FIG_WIDTH)
fig.set_figheight(FIG_HEIGHT)

inv_model, MAP_est, post_cov, modelled_measurement = invert_AH_GN(
    time_indexes=idx_measurement[722], show_output=False)

N_1 = np.log10(inv_model.meas_system.sampling_line.run(10**MAP_est))

plot_system(inv_model, MAP_est, post_cov, N_1=N_1, predicted_measurement=modelled_measurement,
            ax1=axs[1], ax2=axs[0])

axs[0].xaxis.set_major_locator(tck.IndexLocator(base=3, offset=0))
axs[0].set_title('a)', loc='left')
axs[0].set_ylim((1e-1, 1e4))
axs[0].grid(visible=False, which='both')
axs[0].grid('on', which='major')

axs[1].set_title('b)', loc='left')
axs[1].set_title('Estimated particle size distributions, line 1')
format_PSD_axis(axs[1])
axs[1].set_ylim((1e1, 1e9))

if SAVE_FIGURES:
    plt.savefig(FIGURES_FOLDER + 'f10.png', dpi=DPI, bbox_inches='tight')


#%% Invert an AH measurement (G-N) (line 2)

fig, axs = plt.subplots(1, 2, num=71, clear=True)
fig.set_figwidth(2 * FIG_WIDTH)
fig.set_figheight(FIG_HEIGHT)

inv_model, MAP_est, post_cov, modelled_measurement = invert_AH_GN(
    time_indexes=idx_measurement[1838], show_output=False)

N_1 = np.log10(inv_model.meas_system.sampling_line.run(10**MAP_est))

plot_system(inv_model, MAP_est, post_cov, N_1=N_1, predicted_measurement=modelled_measurement,
            ax1=axs[1], ax2=axs[0])

axs[0].xaxis.set_major_locator(tck.IndexLocator(base=3, offset=0))
axs[0].set_title('a)', loc='left')
axs[0].set_ylim((1e0, 1e5))
axs[0].grid(visible=False, which='both')
axs[0].grid('on', which='major')

axs[1].set_title('b)', loc='left')
axs[1].set_title('Estimated particle size distributions, line 2')
format_PSD_axis(axs[1])
axs[1].set_ylim((1e1, 1e10))

if SAVE_FIGURES:
    plt.savefig(FIGURES_FOLDER + 'f11.png', dpi=DPI, bbox_inches='tight')


#%% AH measurement (MCMC, all aux parameters marginalized)

if PLOT_MCMC_RESULTS:
    
    fig, axs = plt.subplots(1, 2, num=72, clear=True)
    fig.set_figwidth(2 * FIG_WIDTH)
    fig.set_figheight(FIG_HEIGHT)
    
    # Line 1
    sampler = load_run(RESULTS_FOLDER, 'aux6_line1')
    sampler.summarise_posterior(ax=axs[0])
    
    axs[0].set_title('a)', loc='left')
    axs[0].set_title('')
    format_PSD_axis(axs[0])
    axs[0].set_ylim((1e1, 1e9))
    
    
    # Line 2
    sampler = load_run(RESULTS_FOLDER, 'aux6_line2')
    sampler.summarise_posterior(ax=axs[1])
    
    axs[1].set_title('b)', loc='left')
    axs[1].set_title('')
    format_PSD_axis(axs[1])
    axs[1].set_ylim((1e1, 1e10))
    
    fig.suptitle('Estimated particle size distributions (MCMC), aux. parameters marginalized')
    
    if SAVE_FIGURES:
        plt.savefig(FIGURES_FOLDER + 'f12.png', dpi=DPI, bbox_inches='tight')


#%% Test the influence of different prior correlation lengths

fig, axs = plt.subplots(1, 2, num=73, clear=True, layout='constrained')
fig.set_figwidth(2 * FIG_WIDTH)
fig.set_figheight(FIG_HEIGHT)

inv_model = InversionModel(bins_per_decade=16)
inv_model.meas_system.sampling_line.update_aerosol_model(
    temperature=250+273,
    fractal_dim=1.7,
    primary_particle_diam=27e-9,
    hamaker_constant=2e-19,
    flow_velocity=3.5
    )

correlation_lengths = np.array([8, 12]) / 16
N0_vector = np.zeros((correlation_lengths.shape[0], inv_model.bins.n))
cov_vector = np.zeros((correlation_lengths.shape[0], inv_model.bins.n, inv_model.bins.n))
inv_model.load_EEPS_measurement(248)

ef = EdgeFiller(inv_model.bins.centers, inv_model.bins.edges)
plt_dp = ef.dp_full

for i in range(correlation_lengths.shape[0]):
    inv_model.load_prior(corr_length=correlation_lengths[i])
    N0_vector[i], cov_vector[i], _ = minimize(inv_model, show_output=False)
    
    N0_full = ef.fill_edges(N0_vector[i])
    plt_N0 = 10**(N0_full) / inv_model.bins.width
    
    axs[i].semilogx(plt_dp, plt_N0, 'C0-', alpha=1.0, label='MAP estimate')
    
    post_std_full = ef.fill_edges(np.sqrt(np.diag(cov_vector[i])))
    fill_top1 = 10**(N0_full + 1.96 * post_std_full) / inv_model.bins.width
    fill_bottom1 = 10**(N0_full - 1.96 * post_std_full) / inv_model.bins.width
    axs[i].fill_between(plt_dp, fill_top1, fill_bottom1, alpha=0.15,
                      facecolor='b', label='95 % posterior credible interval')
    
    axs[i].set_yscale('log')
    axs[i].set_xscale('log')
    axs[i].set_ylim([1e1, 1e9])
    axs[i].set_title(
        fr'MAP estimate, corr. length $\tilde{{l}} = {correlation_lengths[i] * 16 : .0f} / 16$'
        )
    axs[i].set_xlabel('Particle mobility diameter (nm)')
    axs[0].set_ylabel('dN / dlogdp $(cm^{-3})$')
    axs[i].grid('on', which='major')
    plt.pause(0.01)

# Plot the other MAP estimate
N0_full = ef.fill_edges(N0_vector[0])
plt_N0 = 10**(N0_full) / inv_model.bins.width
axs[1].semilogx(plt_dp, plt_N0, 'k--', alpha=0.5, linewidth=1,
                label=fr'MAP estimate, $\tilde{{l}} = {correlation_lengths[0] * 16 : .0f} / 16$')

N0_full = ef.fill_edges(N0_vector[1])
plt_N0 = 10**(N0_full) / inv_model.bins.width
axs[0].semilogx(plt_dp, plt_N0, 'k--', alpha=0.5, linewidth=1,
                label=fr'MAP estimate, $\tilde{{l}} = {correlation_lengths[1] * 16 : .0f} / 16$')

for i in range(correlation_lengths.shape[0]):
    axs[i].legend(loc='best')
    format_PSD_axis(axs[i])

if SAVE_FIGURES:
    plt.savefig(FIGURES_FOLDER + 'f13.png', dpi=DPI, bbox_inches='tight')


print('Figures done.')