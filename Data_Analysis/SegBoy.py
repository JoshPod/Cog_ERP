# Segment, Label, Interpolate and Clean data using marker labels generated via the marker stream.
# This is performed across all trials of one subject's session.
import numpy as np
import ProBoy as pb
import matplotlib.pyplot as plt
'''
    SegBoy - Method of data preparation at single subject level.

    Initially define file locations for data (imp/ eeg), labels and markers, as well
    as some basic experimental parameters.

    'EEG Data Management.'
    1) Slice and extract eeg data using 'pb.slice_ext' function.
    2) Reshape this for 'pb.prepro' signal processing function.
    3) Plot some data.
    4) Save these data to './SegData' as 'eeg_data.npy'
'''
'-----------Experimental Parameters'
num_chan = 8  # 8 Chan in order to include A2 ref.
num_emoji = 7
num_seq = 5

data_file = '..//Data_Aquisition/Data/P_3Data/Em_{0}/'.format(num_emoji)
labels_file = data_file + 'Labels'
print(labels_file)
markers_file = data_file + 'marker_data.npz'

# PyTorch (1) or LDABoy (0) Formatting.
format_sw = 0

# General Plotting Switch
plot_sw = 0

# Save Switch
save_sw = 1

# Output size of segmented emoji event matrices.
out_size = 200

'-----------EEG Data Prep'
# Slice Extraction and Interpolation Resampling.
starts, ends, seq_chk, r_data, r_times_zer, r_times, num_trials = pb.slice_ext(
    data_file, 'volt', labels_file, markers_file, num_chan, num_emoji, num_seq, out_size=out_size, plotter=0, verbose=0)
print('EEG R_Data Dims: ', r_data.shape)
# Plots
if plot_sw == 1:
    plt.subplot(211)
    plt.plot(r_times_zer[:, 0, 0, 0], r_data[:, 0, 0, 0, 0])
    plt.title('1st EEG R_Data Resampled Event.')
    plt.subplot(212)
    plt.plot(r_times_zer[:, -1, -1, -1], r_data[:, -1, -1, -1, -1])
    plt.title('Last EEG R_Data Resampled Event.')
    plt.show()

# Signal Processing.
# A2 ref always -7, as there are 6 variables at end: ACC8, ACC9, ACC10, Packet, Trigger & Time-Stamps.
# 0) Fz, 1) Cz, 2) Pz, 3) P4, 4) P3, 5) O1, 6) O2, 7) A2.'
elec_chans = [1]  # [0, 1, 2, 3, 4, 5, 6]
averager = 'ON'
if averager == 'OFF':
    num_chans = len(elec_chans)
elif averager == 'ON':
    num_chans = 1

# Refernce index is -1 as the unnecessary channels have been parsed out by slice_ext
for t in range(num_trials):
    for s in range(num_seq):
        for e in range(num_emoji):
            plus_dat = pb.prepro(r_data[:, :, e, s, t], samp_ratekHz=0.5, zero='ON',
                                 ext='INC-Ref', elec=elec_chans,
                                 filtH='ON', hlevel=1, filtL='ON', llevel=20,
                                 notc='LOW', notfq=50, ref_ind=-1, ref='A2', avg=averager)
            plus_dat = np.expand_dims(plus_dat, axis=1)
            if t == 0 and s == 0 and e == 0:
                p_data = plus_dat
            else:
                p_data = np.append(p_data, plus_dat, axis=2)
print('EEG Post Pre_Pro R_Data Dims: ', p_data.shape)
# Plots
if plot_sw == 1:
    plt.subplot(211)
    plt.plot(r_times_zer[:, 0, 0, 0], r_data[:, 0, 0, 0, 0])
    plt.title('EEG Resampled R_Data - Before Preprocessing.')
    plt.subplot(212)
    plt.plot(r_times_zer[:, 0, 0, 0], p_data[:, 0, 0])
    plt.title('EEG Resampled P_Data - After Preprocessing.')
    plt.show()

# Reshaping Resampled and Pre-Processed Data for LDABoy
# Samples x Channels x Singleton x Events (emoji*seqs*trials).
pr_data = np.reshape(p_data, (out_size, num_chans, (num_emoji * num_seq * num_trials)))
if format_sw == 1:
    pr_data = np.expand_dims(pr_data, axis=2)
print('EEG FINAL | PR_Data Post Resampling / Pre-Proecssing and Reshaping Dims: ', pr_data.shape)

'-----------Imp Data Prep'
# Slice Extraction and Interpolation Resampling.
starts, ends, seq_chk, r_data, r_times_zer, r_times, num_trials = pb.slice_ext(
    data_file, 'imp', labels_file, markers_file, num_chan, num_emoji, num_seq, out_size=out_size, plotter=0, verbose=0)
print('Imp R_Data Dims: ', r_data.shape)
# Plots
if plot_sw == 1:
    plt.subplot(211)
    plt.plot(r_times_zer[:, 0, 0, 0], r_data[:, 0, 0, 0, 0])
    plt.title('1st Imp R_Data Resampled Event.')
    plt.subplot(212)
    plt.plot(r_times_zer[:, -1, -1, -1], r_data[:, -1, -1, -1, -1])
    plt.title('Last Imp R_Data Resampled Event.')
    plt.show()
# Reshaping: Samples x Channels x Singleton x Events (emoji*seqs*trials).
ir_data = np.reshape(r_data, (out_size, num_chan, (num_emoji * num_seq * num_trials)))
if format_sw == 1:
    ir_data = np.expand_dims(ir_data, axis=2)
print('Imp FINAL |  R_Data Post Reshape Dims: ', ir_data.shape)

'------------Labels Prep'
sp_labels = pb.spatial_labeller(labels_file, num_emoji, num_seq, verbose=0)
tp_labels = pb.temporal_labeller(labels_file, num_emoji, num_seq, verbose=0)
bn_labels = pb.binary_labeller(np.copy(sp_labels), verbose=1)

print(sp_labels.shape, np.unique(sp_labels))
print(tp_labels.shape)
print(bn_labels.shape)

'Save EEG Data, Impedances Data and Labels.'
if save_sw == 1:
    np.savez('./SegData/database0005', pr_data, ir_data, sp_labels, tp_labels, bn_labels)
