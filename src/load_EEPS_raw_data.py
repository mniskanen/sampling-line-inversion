# -*- coding: utf-8 -*-

import numpy as np

from pandas import read_csv

from load_dilution_ratios import load_dilution_ratios


''' This script loads the raw data and saves them in a numpy readable file. '''

    
# Load the inversion matrix (SOOT)
soot_matrix = read_csv(
    '../measurement_data/SOOT.matrix',
    delimiter=',',
    header=None,
    dtype=float,
    skiprows=1
    )

soot_matrix = soot_matrix.to_numpy()
soot_matrix = soot_matrix[:, :17]



# Load electrometer data
# This loads the data and saves it in a format that is easier to load and use with numpy

# The records have periodic 10 second 'pauses', where 34 non-empty rows of info is printed
# instead. We should skip over these. Read_csv seems to automatically skip over empty lines.

EEPS_record = read_csv(
    '../measurement_data/20210211EepsRaw_VW_measurement.txt',
    encoding='ANSI',
    delimiter='\t',
    header=None,
    # dtype=float,
    skiprows=16,
    # usecols=[35, 36, 37, 38, 39],
    # skipfooter=26,
    on_bad_lines='skip'
    ).to_numpy()

# Find row(s) that have the printed info (first row starts with 'Channel')
commentrow_start = np.where(EEPS_record == 'Channel')[0]

# Replace the comment rows with nans
for i in range(len(commentrow_start)):
    EEPS_record[commentrow_start[i] : commentrow_start[i] + 34] = 'nan'

# Collect the inverted particle numbers
dNdlogDp = EEPS_record[:, 1:1+32].astype(float)

# Compensate for dilution ratio, and truncate to times we have dilution ratios for
DR = load_dilution_ratios()

# Synchronize the calculated DR to the EEPS measurement and clip a correct length
sync_idx = 17202
DR = DR[sync_idx : sync_idx + EEPS_record.shape[0]]
dNdlogDp *= DR[:, np.newaxis]

# Ensure positivity (when plotting)
dNdlogDp_positive = dNdlogDp.copy()
dNdlogDp_positive[np.where(dNdlogDp_positive < 1)] = 1

# Collect electrometer data (current)
EEPS_current = EEPS_record[:, 38:38+22].astype(float)



# Load a measurement with no exhaust gas (to estimate noise floor)

EEPS_record = read_csv(
    '../measurement_data/20210211EepsRaw_empty.txt',
    encoding='ANSI',
    delimiter='\t',
    header=None,
    skiprows=16,
    on_bad_lines='skip'
    ).to_numpy()

# Collect electrometer data (current)
empty_measurement = EEPS_record[:, 38:38+22].astype(float)

# Compute noise statistics from an empy part of the measurement
noise_std = np.std(empty_measurement, axis=0)


#%% Save the measurements to be loaded by numpy

savefolder = '../measurement_data/'


EEPS_dNdlogDp = dNdlogDp_positive

np.save(savefolder + 'soot_matrix', soot_matrix)
np.save(savefolder + 'noise_std', noise_std)
np.save(savefolder + 'EEPS_current', EEPS_current)
np.save(savefolder + 'EEPS_dNdlogDp', EEPS_dNdlogDp)
np.save(savefolder + 'DR', DR)
