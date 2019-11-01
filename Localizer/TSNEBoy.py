import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# Data Analysis Imports
sys.path.insert(0, '../../Data_Analysis')
sys.path.insert(0, '../../Data_Analysis/tsne')
import ProBoy as pb

'''
# IMPORTANT REF: https://distill.pub/2016/misread-tsne/
'''

'Data Loading, Cleaning and Plotting of Localization Experiment Data'
# Data Pathway Extraction.
dat_direc = './Data/BackUps/J4/'
lab_direc = dat_direc + 'Labels/'
dat_files = pb.pathway_extract(dat_direc, '.npy', 'Volt', full_ext=1)
imp_files = pb.pathway_extract(dat_direc, '.npy', 'Impedances', full_ext=1)
lab_files = pb.pathway_extract(lab_direc, '.npz', 'Labels', full_ext=1)

# Data Properties.
print(dat_files[0])
eeg = np.load(dat_files[0])
print('EEG DIMS: ', eeg.shape)

# Labels Checking.
labels = np.load(lab_files[0])['arr_0']
print(labels)

'_________P300 Averaging_________'
np3 = []
p3 = []
p3_elec = []
np3_elec = []
eeg_band = []
eeg_sub = []

for i in range(len(dat_files)):
    'Pre-Process Data Chunk'
    eeg = np.load(dat_files[i])
    # For Sub-Band Analysis Breakdown.
    'Electrode Index:  0) Fz, 1) Cz, 2) Pz, 3) P4, 4) P3, 5) O1, 6) O2, 7) A2.'
    'Sub-Band Electrodes: 0) Fz, 1) Cz, 3) P4, 4) P3.  |  Ref: https://ieeexplore.ieee.org/abstract/document/1556780'
    # The cortex produces low amplitude and fast oscillations during waking, and generates high-amplitude,
    # slow cortical oscillations during the onset of sleep  |  Ref : https://ieeexplore.ieee.org/abstract/document/1556780'
    # For Cz / Target electrode breakdown.
    eeg = pb.prepro(eeg, samp_ratekHz=0.5, zero='ON', ext='INC-Ref', elec=[1],
                    filtH='ON', hlevel=1, filtL='ON', llevel=10,
                    notc='LOW', notfq=50, ref_ind=-7, ref='A2', avg='OFF')
    # NP300 Trials.
    if labels[i] == 0:
        if np3 == []:
            np3 = eeg
            np3 = np.expand_dims(np3, axis=2)
        else:
            eeg = np.expand_dims(eeg, axis=2)
            np3 = np.append(np3, eeg, axis=2)
    # P300 Trials.
    if labels[i] == 1:
        if p3 == []:
            p3 = eeg
            p3 = np.expand_dims(p3, axis=2)
        else:
            eeg = np.expand_dims(eeg, axis=2)
            p3 = np.append(p3, eeg, axis=2)

# Post Extraction DIMS.
print('NP3 DIMS: ', np.shape(np3))
print('P3 DIMS: ', np.shape(p3))

'---Alt TSNE---'
tsneRZ = 1
if tsneRZ == 1:
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold
    from time import time

    # Parameters.
    n_components = 2
    # Typically between 5-50.
    perplexities = [5, 15, 25, 45, 65, 85, 100]  # 5, 15,
    # Typically between 2500 - 7500
    learning_rates = [2, 5, 10, 50, 250]  # , 12500, 12500, 15000
    (fig, subplots) = plt.subplots(len(learning_rates), len(perplexities) + 1, figsize=(15, 8))
    # Label Mapping | RED == NP3 | GREEN == P3.
    red = labels == 0
    green = labels == 1

    dataswitch = 0
    if dataswitch == 0:
        # EEG Data.
        all3 = np.append(np3, p3, axis=2)
        all3 = np.squeeze(all3)
        print('all3 DIMS: ', all3.shape)
        print('all3 DIMS: ', all3.shape)
        X = all3
        # Feature Scaling
        X = pb.min_max_scaler(X)  # pb.stand_scaler(X)  #
        # Swap Axes
        X = np.swapaxes(X, 0, 1)
        # Cut down to 1st 500ms of the trial period, meaning 1st 250 samples.
        # 1st is 125ms aug_wait, then 2000ms presentation duration, then a inter-trial interval of 1000ms.
        X = X[:, 0:250]
        print('FINAL EEG DIMS: ', X.shape)
    elif dataswitch == 1:
        X = np.loadtxt('../../Data_Analysis/TSNE/mnist2500_X.txt')
        labels = np.loadtxt('../../Data_Analysis/TSNE/mnist2500_labels.txt')

    # Plotting Np3 vs P3.
    ax = subplots[0][0]
    ax.scatter(X[red, 0], X[red, 1], c="r")
    ax.scatter(X[green, 0], X[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    for j, learning_rate in enumerate(learning_rates):
        for i, perplexity in enumerate(perplexities):
            print('LOC DATA: Perplexity={0} | Learning Rate={1}'.format(
                perplexity, learning_rate))
            ax = subplots[j][i + 1]
            t0 = time()
            tsne = manifold.TSNE(n_components=n_components, init='random',
                                 random_state=0, perplexity=perplexity, learning_rate=learning_rate, n_iter=10000, n_iter_without_progress=300, verbose=1)
            Y = tsne.fit_transform(X)
            t1 = time()
            print('-------Duration: {0} sec'.format(np.round(t1 - t0), decimals=2))

            ax.set_title('Perplexity={0} | \n Learning Rate={1}'.format(perplexity, learning_rate))
            'Plotting'
            ax.scatter(Y[red, 0], Y[red, 1], c="r")
            ax.scatter(Y[green, 0], Y[green, 1], c="g")
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')
    plt.tight_layout()
    plt.show()
