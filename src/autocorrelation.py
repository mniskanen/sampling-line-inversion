# -*- coding: utf-8 -*-

import numpy as np


def acf(x, length=20):
    """ Compute autocorrelation. """
    
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])


# From https://dfm.io/posts/autocorr/ -------------------------------------------------------------

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real + 1e-50
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_time(y, c=5.0):
    # f = np.zeros(y.shape[1])
    # for yy in y:
    f = autocorr_func_1d(y)
    # f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]