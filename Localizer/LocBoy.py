import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# Data Analysis Imports
sys.path.insert(0, '../../Data_Analysis/')
import ProBoy as pb

'Data Loading, Cleaning and Plotting of Localization Experiment Data'
# Data Pathway Extraction.
dat_direc = './Data/BackUps/0003_L1/L2/'
lab_direc = dat_direc + 'Labels/'
dat_files = pb.pathway_extract(dat_direc, '.npy', 'Volt', full_ext=1)
imp_files = pb.pathway_extract(dat_direc, '.npy', 'Impedances', full_ext=1)
lab_files = pb.pathway_extract(lab_direc, '.npz', 'Labels', full_ext=1)

# Data Properties.
print(dat_files[0])
eeg = np.load(dat_files[0])
print('EEG DIMS: ', eeg.shape)

'Plots Initialization'
# Creates four polar axes, and accesses them through the returned array
fig, axes = plt.subplots(5, 2)
'Impedance Analysis'
# Load / Impedances Exrtaction / Zero Mean.
'ONLY 1st TRIAL'
imp = pb.zero_mean(np.load(imp_files[0])[:, 0:7])
print('Imp DIMS: ', imp.shape)
imp_chans = imp.shape[1]
# Smoothing for plots: Savitsky-Golay | https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.savgol_filter.html
for i in range(imp_chans):
    smo_imp = savgol_filter(imp[:, i], 53, 6)
    smo_imp = np.expand_dims(smo_imp, axis=1)
    if i == 0:
        imps = smo_imp
    else:
        imps = np.append(imps, smo_imp, axis=1)
# Assinging legends to plots.
Fz, = axes[0, 0].plot(imps[:, 0])
Cz, = axes[0, 0].plot(imps[:, 1])
Pz, = axes[0, 0].plot(imps[:, 2])
P4, = axes[0, 0].plot(imps[:, 3])
P3, = axes[0, 0].plot(imps[:, 4])
O1, = axes[0, 0].plot(imps[:, 5])
O2, = axes[0, 0].plot(imps[:, 6])
fig.legend((Fz, Cz, Pz, P4, P3, O1, O2), ('Fz', 'Cz', 'Pz', 'P4', 'P3', 'O1', 'O2'), 'upper left')

axes[0, 0].set_title('Impedances')
# Average
imp_avg = np.average(imp, axis=1)
axes[0, 1].plot(imp_avg)
axes[0, 1].set_title('Impedances Average')

'Time-Stamp Analysis'
time_stamps = pb.time_stamper(eeg, show_stamps=1)

'Impedances Time Stanp vs EEG Time-Stamp Coherence'
imp_stamps = pb.time_stamper(np.load(imp_files[0]), show_stamps=1)
axes[1, 0].plot(time_stamps, imp_stamps)
axes[1, 0].set_title('Time_Stamps Imp vs EEG')

# Cross Trial Time Checking.
timer_ = []
for i in range(len(dat_files)):
    times_ = np.load(dat_files[i])
    times_ = np.squeeze(times_[:, -1])
    timer_ = np.append(timer_, times_)
    # print(np.shape(timer_))
timer_ = timer_[:] - timer_[0]
x_axis = np.arange(len(timer_))
axes[1, 1].plot(x_axis, timer_)
axes[1, 1].set_title('Cross Trial Time-Stamps')

# Labels Checking.
labels = np.load(lab_files[0])['arr_0']
# print(labels)

'_________P300 Averaging_________'
np3 = []
p3 = []
p3_elec = []
np3_elec = []
eeg_band = []
eeg_sub = []
for i in range(len(dat_files)):  # len(dat_files)
    'Pre-Process Data Chunk'
    eeg = np.load(dat_files[i])
    # For Sub-Band Analysis Breakdown.
    'Electrode Index:  0) Fz, 1) Cz, 2) Pz, 3) P4, 4) P3, 5) O1, 6) O2, 7) A2.'
    'Sub-Band Electrodes: 0) Fz, 1) Cz, 3) P4, 4) P3.  |  Ref: https://ieeexplore.ieee.org/abstract/document/1556780'
    # The cortex produces low amplitude and fast oscillations during waking, and generates high-amplitude,
    # slow cortical oscillations during the onset of sleep  |  Ref : https://ieeexplore.ieee.org/abstract/document/1556780'
    eeg_band = pb.prepro(eeg, samp_ratekHz=0.5, zero='ON', ext='NO-Ref', elec=[5, 6],
                         filtH='ON', hlevel=0.5, filtL='ON', llevel=25,
                         notc='NIK', notfq=50, ref_ind='None', ref='OFF', avg='ON')  # 0, 1, 3, 4, 5, 6
    if i == 0:
        eeg_sub = eeg_band
    else:
        eeg_sub = np.append(eeg_sub, eeg_band, axis=1)
    # For Single Electrode Analysis Breakdown.
    eeg_elec = pb.prepro(eeg, samp_ratekHz=0.5, zero='ON', ext='INC-Ref', elec=[0, 1, 2, 3, 4, 5, 6],
                         filtH='ON', hlevel=1, filtL='ON', llevel=10,
                         notc='LOW', notfq=50, ref_ind=-7, ref='A2', avg='OFF')
    # For Cz / Target electrode breakdown.
    eeg = pb.prepro(eeg, samp_ratekHz=0.5, zero='ON', ext='INC-Ref', elec=[1],
                    filtH='ON', hlevel=1, filtL='ON', llevel=15,
                    notc='LOW', notfq=50, ref_ind=-7, ref='A2', avg='ON')
    # NP300 Trials.
    if labels[i] == 0:
        if np3 == []:
            np3 = np.copy(eeg)
            np3 = np.expand_dims(np3, axis=2)
            np3_elec = eeg_elec
            np3_elec = np.expand_dims(np3_elec, axis=2)
        else:
            eeg = np.expand_dims(eeg, axis=2)
            np3 = np.append(np3, np.copy(eeg), axis=2)
            eeg_elec = np.expand_dims(eeg_elec, axis=2)
            np3_elec = np.append(np3_elec, eeg_elec, axis=2)
    # P300 Trials.
    if labels[i] == 1:
        if p3 == []:
            p3 = np.copy(eeg)
            p3 = np.expand_dims(p3, axis=2)
            p3_elec = eeg_elec
            p3_elec = np.expand_dims(p3_elec, axis=2)
        else:
            eeg = np.expand_dims(eeg, axis=2)
            p3 = np.append(p3, np.copy(eeg), axis=2)
            eeg_elec = np.expand_dims(eeg_elec, axis=2)
            p3_elec = np.append(p3_elec, eeg_elec, axis=2)

# Post Extraction DIMS.
print('EEG SUB DIMS: ', eeg_sub.shape)
print('NP3 DIMS: ', np.shape(np3))
print('NP3 ELEC DIMS: ', np.shape(np3_elec))
print('P3 DIMS: ', np.shape(p3))
print('P3 ELEC DIMS: ', np.shape(p3_elec))

'---LDA Analysis---'
ldaR = 1
if ldaR == 1:
    '---Data Prep---'
    aug_data = np.append(p3, np3, axis=2)
    aug_data = np.squeeze(aug_data)
    aug_data = aug_data[0:200, :]
    print('aug_data dims: ', aug_data.shape)
    '---Labels Prep---'
    'Labels need to be re-written as '
    num_trials = np.int(np.shape(aug_data)[1] / 2)
    p3_lab = np.ones(num_trials)
    np3_lab = np.zeros(num_trials)
    print('Labels 1: ', labels)
    labels = np.append(p3_lab, np3_lab)
    labels = labels.astype(int)
    print('Labels 2: ', labels)
    print('labels dims: ', labels.shape)
    '---LDA ANALYSIS---'
    print('aug_data dims: ', aug_data.shape)
    pb.lda_(aug_data, labels, split=None, div=0.75,
            num_comp=None, meth='eigen', scaler='min', covmat=1, verbose=1)

'----------Plots----------'
plotter = 0
if plotter == 1:
    # Time Axis Generator
    samp_rateKHz = 0.5
    t_x, time_idx = pb.temp_axis(eeg, samp_rateKHz, plt_secs=600)
    'P300'
    # Cross Channel Averages.
    p3_elec = np.average(p3_elec, axis=2)
    p3_elec = p3_elec[0:time_idx, :]
    Fz, = axes[2, 0].plot(t_x, p3_elec[:, 0])
    Cz, = axes[2, 0].plot(t_x, p3_elec[:, 1])
    Pz, = axes[2, 0].plot(t_x, p3_elec[:, 2])
    P4, = axes[2, 0].plot(t_x, p3_elec[:, 3])
    P3, = axes[2, 0].plot(t_x, p3_elec[:, 4])
    O1, = axes[2, 0].plot(t_x, p3_elec[:, 5])
    O2, = axes[2, 0].plot(t_x, p3_elec[:, 6])
    axes[2, 0].set_title('P300 Elec Averages')
    fig.legend((Fz, Cz, Pz, P4, P3, O1, O2), ('Fz', 'Cz',
                                              'Pz', 'P4', 'P3', 'O1', 'O2'), 'center left')
    # Trial Averages.
    p3 = np.squeeze(p3)
    p3 = p3[0:time_idx]
    axes[3, 0].plot(t_x, p3)
    axes[3, 0].set_title('P300 Trial Averages')
    # Cross Trial Average.
    p3_avg = np.average(p3, axis=1)
    axes[4, 0].plot(t_x, p3_avg)
    axes[4, 0].set_title('P300 Cross Average')
    'NP300'
    # Cross Channel Averages.
    np3_elec = np.average(np3_elec, axis=2)
    np3_elec = np3_elec[0:time_idx, :]
    Fz, = axes[2, 1].plot(t_x, np3_elec[:, 0])
    Cz, = axes[2, 1].plot(t_x, np3_elec[:, 1])
    Pz, = axes[2, 1].plot(t_x, np3_elec[:, 2])
    P4, = axes[2, 1].plot(t_x, np3_elec[:, 3])
    P3, = axes[2, 1].plot(t_x, np3_elec[:, 4])
    O1, = axes[2, 1].plot(t_x, np3_elec[:, 5])
    O2, = axes[2, 1].plot(t_x, np3_elec[:, 6])
    axes[2, 1].set_title('NP300 Elec Averages')
    fig.legend((Fz, Cz, Pz, P4, P3, O1, O2), ('Fz', 'Cz',
                                              'Pz', 'P4', 'P3', 'O1', 'O2'), 'center right')
    # Trial Averages.
    np3 = np.squeeze(np3)
    np3 = np3[0:time_idx]
    axes[3, 1].plot(t_x, np3)
    axes[3, 1].set_title('NP300 Trial Averages')
    # Cross Trial Average.
    np3_avg = np.average(np3, axis=1)
    axes[4, 1].plot(t_x, np3_avg)
    axes[4, 1].set_title('NP300 Cross Average')
    'Plots show'
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    plt.show()
    # Stats
    print('P300 Average Range: ', pb.ranger(p3_avg))
    print('NP300 Average Range: ', pb.ranger(np3_avg))
    '----Sub-Band Analysis----'
    eeg_sub_avg = np.average(eeg_sub, axis=1)
    eeg_sub_avg = np.expand_dims(eeg_sub_avg, axis=1)
    'Band Power Reporting'
    # Option to report absolute band power, with each frequency band as a percentage of the total power of the signal.
    # This is called the relative band power. | Ref : https://raphaelvallat.com/bandpower.html'
    values, rel_pow = pb.band_power_plots(eeg_sub_avg, sing_plt='ON', plotter='REL')
    pb.sub_band_chunk_plot(eeg_sub, divs=5, pow_disp='REL', plotter='ON')
    '----AVERAGE Plots ----'
    plt.plot(t_x, p3_avg)
    plt.title('P300 vs NP300')
    plt.plot(t_x, np3_avg)
    plt.show()
    '----Save Averages for Random Noise Augmentation----'
    # Save description of data in root data for augmentation.
    description = dat_direc + ' :' + str(len(dat_files))
    np.savez('./Data/BackUps/RandRoot/rand_root_data.npz', p3_avg, np3_avg, description)
