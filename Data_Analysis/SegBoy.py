# Segment, Label, Interpolate and Clean data using marker labels generated via the marker stream.
# This is performed on a subject level basis.
import numpy as np
import ProBoy as pb
from scipy import signal
import matplotlib.pyplot as plt


def time_check(data_file, labels_file, markers_file):
    data = np.load(data_file)
    labels = np.load(labels_file)
    markers = np.load(markers_file)
    # EEG Data.
    sequence = 'arr_0'
    seq1_data = data[sequence][:, 0:7]
    print('Seq Data DIMS: ', seq1_data.shape)
    # EEG Timestamps.
    seq1_time = data[sequence][:, -1]
    print('Seq Time DIMS: ', seq1_time.shape)
    print('First Data Time Stamp: ', seq1_time[0], ': ', 0)
    print('Last Data Time Stamp: ', seq1_time[-1], ': ', seq1_time[-1] - seq1_time[0])
    # Marker Data.
    seq_mark_end = markers['arr_3']
    seq_mark_str = markers['arr_1']
    print('Seq1 Mark DIMS: ', seq_mark_str.shape)
    print('1st Mark Stamp: ', seq_mark_str[0])
    # Diff between 1st EEG Timestamp and 1st Marker Timestamp.
    print('Data Marker Offset: ', seq_mark_str[0] - seq1_time[0])

    for i in range(len(seq_mark_str)):
        print('Length Mark Collection Emoji {0}: '.format(
            i + 1), seq_mark_end[i] - seq_mark_str[i], 'Start: ', seq_mark_str[i], 'End: ', seq_mark_end[i])

    'Plots'
    # Plot EEG Data Timestamps.
    plt.plot(seq1_time)
    num_emojis = 7
    print('1st Sequence Start Times: ', seq_mark_str[0:6])
    mark1 = np.zeros(len(seq1_time))
    mark2 = np.zeros(len(seq1_time))

    for i in range(num_emojis):
        # Plot Marker Start Times.
        mark1[:] = seq_mark_str[i]
        print('Start Time: ', seq_mark_str[i])
        plt.plot(mark1)
        # Plot Marker End Times.
        mark2[:] = seq_mark_end[i]
        print('End Time: ', seq_mark_end[i])
        plt.plot(mark2)
    plt.title('Marker Start and End Points Overlaid on EEG OR IMP Data Timestamps.')
    plt.show()

    # Data with Data Timestamp Axis.
    plt.plot(seq1_time, seq1_data[:, 0])
    plt.title('Data with Data Timestamp Axis')
    plt.show()
    # Data with Pre-Gen Timestamp Axis.
    gen_time = np.arange(len(seq1_data[:, 0]))
    plt.plot(gen_time, seq1_data[:, 0])
    plt.title('Data with Pre-Gen Timestamp Axis')
    plt.show()

    # Find nearest value of the marker start timestamps in the corresponding data timestamp array.
    arr = seq1_time
    v = seq_mark_str[0]
    idx = (np.abs(arr - v)).argmin()
    print('Start Idx: ', idx, 'Idx of Seq Time: ',
          seq1_time[idx], 'Idx of Seq Data: ', seq1_data[idx, 0])
    # Find nearest value of the marker end timestamps in the corresponding data timestamp array.
    arr = seq1_time
    v = seq_mark_end[6]
    idx = (np.abs(arr - v)).argmin()
    print('End Idx: ', idx, 'Idx of Seq Time: ',
          seq1_time[idx], 'Idx of Seq Data: ', seq1_data[idx, 0])


def slice_ext(data_file, labels_file, markers_file, num_emoji, num_seq, num_trials):

    # Get Trial Data file locations.
    dat_files = pb.pathway_extract(data_file, '.npz', 'volt', full_ext=0)
    eeg_files = pb.path_subplant(data_file, np.copy(dat_files), 0)
    print('EEG Files DIMS: ', np.shape(eeg_files), 'EEG Files: ', eeg_files)
    # Get Labels file locations.
    grn_files = pb.pathway_extract(labels_file, '.npz', 'trial', full_ext=0)
    lab_files = pb.path_subplant(labels_file, np.copy(grn_files), 0)
    print('Lab Files DIMS: ', np.shape(lab_files), 'Lab Files: ', lab_files)
    # Marker Data.
    markers = np.load(markers_file)
    starts = markers['arr_1']
    ends = markers['arr_3']
    # Markers reshape by trials and seqs.
    starts = np.reshape(starts, (7, 5, 7))
    ends = np.reshape(ends, (7, 5, 7))
    # Aggregate arrays.
    r_data = np.zeros((250, num_emoji, num_seq, num_trials))  # Samples x Channels x Seqs x Trials
    r_times = np.zeros((250, num_seq, num_trials))

    for t in range(num_trials):
        # Loading Data.
        data = np.load(eeg_files[t])
        print('EEG File_Name', eeg_files[t])
        # Loading Labels.
        labels = np.load(lab_files[t])
        print('LABS File_Name', lab_files[t])
        for i in range(num_seq):
            # EEG Data.
            sequence = 'arr_{0}'.format(i)
            seq_data = data[sequence][:, 0:7]
            # print('Seq Data DIMS: ', seq1_data.shape)
            seq_time = data[sequence][:, -1]
            for j in range(num_emoji):
                # START: Find nearest value of the marker timestamps in the corresponding data timestamp array.
                v_s = starts[t, i, j]
                str_idx = (np.abs(seq_time - v_s)).argmin()
                # END: Find nearest value of the marker timestamps in the corresponding data timestamp array.
                # Pad to ensure all P3 wave form extracted, time in seconds so 0.3 = 300ms.
                if ends[t, i, j] < starts[t, i, j] + 0.3:  # v_e = ends[t, i, j] + p3_buff
                    v_e = starts[t, i, j] + 0.5
                else:
                    v_e = 'End Marker Positioned Before P3 Propogation.'
                end_idx = (np.abs(seq_time - v_e)).argmin()
                print('V_s: ', v_s, 'Start IDX: ', str_idx, 'V_e: ', v_e, 'End IDX: ', end_idx)
                print('Diff in time between Start and End: ', v_e - v_s)
                # Index into data array to extract currect P300 chunk.
                seq_chk = seq_data[str_idx:end_idx, :]
                seq_temp = seq_time[str_idx:end_idx]
                print('Emoji: {0} | Seq: {1}'.format(j + 1, i + 1), 'Seq_Chk Dims: ', seq_chk.shape)
                # Zeroing.
                seq_temp = seq_temp - seq_temp[0]
                seq_chk = pb.zero_mean(seq_chk)
                # Resampling Interpolation Method @ Channel Level.
                r_data[:, :, i, t], r_times[:, i, t] = pb.interp2D(
                    seq_chk, seq_temp, output_size=250, plotter=0, verbose=0)
                print('r_data DIMS: ', r_data.shape, 'r_times DIMS: ', r_times.shape)
    return starts, ends, seq_chk, r_data, r_times


data_file = '..//Data_Aquisition/Data/P_3Data/'
labels_file = '..//Data_Aquisition/Data/P_3Data/Labels/'
markers_file = '..//Data_Aquisition/Data/P_3Data/marker_data.npz'

timer = 0
if timer == 1:
    time_check(data_file, labels_file, markers_file)

slicer = 1
if slicer == 1:
    # Experimental Parameters.
    num_emoji = 7
    num_seq = 5
    num_trials = 7
    # Slice Straction.
    starts, ends, seq_chk, r_data, r_times = slice_ext(
        data_file, labels_file, markers_file, num_emoji, num_seq, num_trials)
    print('r_data DIMS: ', r_data.shape, 'r_times DIMS: ', r_times.shape)
    plt.show(plt.plot(r_times[:, -1, -1], r_data[:, :, -1, -1]))

# FFT Resample Method.
# seq_res, seq_pin = signal.resample(np.squeeze(seq_chk[:, j]), 250, seq_temp)
# print('seq_res DIMS: ', seq_res.shape, 'seq_pin DIMS: ', seq_pin.shape)
# plt.plot(seq_pin, seq_res)
# print('seq_pin: ', seq_pin)
