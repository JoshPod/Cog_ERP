import mne
import matplotlib.pyplot as plt
import numpy as np
from alphawaves.dataset import AlphaWaves
from scipy.signal import welch

"""
================================
Spectral analysis of the trials
================================

This example shows how to extract the epochs from the dataset of a given
subject and then do a spectral analysis of the signals. The expected behavior
is that there should be a peak around 10 Hz for the 'closed' epochs, due to the
Alpha rhythm that appears when a person closes here eyes.

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
#
# License: BSD (3-clause)

import warnings
warnings.filterwarnings("ignore")

# define the dataset instance
direc = '../Data/AlpData/'
subject = 'subject_001.mat'
filepath = direc + subject
dataset = AlphaWaves()

# get the data from subject of interest
raw = dataset._get_single_subject_data(filepath)

# filter data and resample
fmin = 3
fmax = 40
raw.filter(fmin, fmax, verbose=False)
raw.resample(sfreq=128, verbose=False)

# detect the events and cut the signal into epochs
events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
event_id = {'closed': 1, 'open': 2}
epochs = mne.Epochs(raw, events, event_id, tmin=2.0, tmax=8.0, baseline=None,
                    verbose=False)
epochs.load_data().pick_channels(['Oz'])

# estimate the averaged spectra for each condition
X_closed = epochs['closed'].get_data()
f, S_closed = welch(X_closed, fs=epochs.info['sfreq'], axis=2)
S_closed_mean = np.mean(S_closed, axis=0).squeeze()
X_opened = epochs['open'].get_data()
f, S_opened = welch(X_opened, fs=epochs.info['sfreq'], axis=2)
S_opened_mean = np.mean(S_opened, axis=0).squeeze()

# plot the results
fig = plt.figure(facecolor='white', figsize=(8, 6))
plt.plot(f, S_closed_mean, c='k', lw=4.0, label='closed')
plt.plot(f, S_opened_mean, c='r', lw=4.0, label='open')
plt.xlim(0, 40)
plt.xlabel('frequency', fontsize=14)
plt.title('PSD on both conditions (averaged over 5 trials)', fontsize=16)
plt.legend()
plt.show()

# Single Trial Plots
print('S_closed pre-mean: ', S_closed.shape)
plt.plot(f, S_closed[0, 0, :], label='Trial 1')
plt.plot(f, S_closed[1, 0, :], label='Trial 2')
plt.plot(f, S_closed[2, 0, :], label='Trial 3')
plt.plot(f, S_closed[3, 0, :], label='Trial 4')
plt.plot(f, S_closed[4, 0, :], label='Trial 5')
plt.plot(f, S_closed_mean, color='k', linewidth=2.0, label='Mean')
plt.legend()
plt.xlabel('frequency', fontsize=14)
plt.title('Eyes Closed PSD on Single Trials', fontsize=16)
plt.xlim(0, 40)
plt.show()

print('S_opened pre-mean: ', S_opened.shape)
plt.plot(f, S_opened[0, 0, :], label='Trial 1')
plt.plot(f, S_opened[1, 0, :], label='Trial 2')
plt.plot(f, S_opened[2, 0, :], label='Trial 3')
plt.plot(f, S_opened[3, 0, :], label='Trial 4')
plt.plot(f, S_opened[4, 0, :], label='Trial 5')
plt.plot(f, S_opened_mean, color='k', linewidth=2.0, label='Mean')
plt.legend()
plt.xlabel('frequency', fontsize=14)
plt.title('Eyes Opened PSD on Single Trials', fontsize=16)
plt.xlim(0, 40)
plt.show()
