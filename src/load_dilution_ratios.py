# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat


# Idea from https://stackoverflow.com/a/14314054
def moving_average(a, n=3, pad=False):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    if pad:
        ma = ret[n - 1:] / n
        return np.concatenate([ma[0] * np.ones(int(np.ceil(n / 2) - 1)),
                               ma,
                               ma[-1] * np.ones(int(np.floor(n / 2)))])
    else:
        return ret[n - 1:] / n


def load_dilution_ratios():
    DR = loadmat('../measurement_data/dilution_ratios_VW.mat')['DR_20210211_VW'].T
    
    reg_param = 0.01
    ma_window = 60
    min_cutoff = 1
    max_cutoff = 120
    
    ''' DR is ordered so that:
        DR[0] = raw exhaust co2,
        DR[1] = exhaust co2 after porous tube diluter
        DR[2] = exhaust co2 after ejector diluter
        I.e., the total dilution ratio is DR[1] / DR[3]
    '''
    
    # Simplified wet correction: the wet co2 amount is on average 10 % lower than the dry one
    DR_raw = DR[0] * 0.9
    
    DR_tot = DR_raw / (DR[2] + reg_param * np.mean(DR[2]))
    
    DR_tot[DR_tot < min_cutoff] = min_cutoff
    DR_tot[DR_tot > max_cutoff] = max_cutoff
    
    DR_tot = moving_average(DR_tot, n=ma_window, pad=True)
    
    return DR_tot


def plot_dilution_ratios(DR):
    
    plt.figure(num=50), plt.clf()
    ylims = [-20, 120]
    
    plt.plot(DR)
    plt.ylim(ylims)
    plt.grid('on')
    plt.title('Dilution ratios')
    plt.ylabel('Total DR')
    plt.xlabel('Time (s)')
    plt.draw()


if __name__ == '__main__':
    
    DR = load_dilution_ratios()
    plot_dilution_ratios(DR)