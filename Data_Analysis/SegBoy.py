# Segment, Label, Interpolate and Clean data using marker labels generated via the marker stream.
# This is performed on a subject level basis.
import numpy as np
import matplotlib.pyplot as plt

data = np.load(
    '..//Data_Aquisition/Data/P_3Data/voltages_t0001_s1_190905_201639677837.npz')
labels = np.load('..//Data_Aquisition/Data/P_3Data/Labels/0001_trial_labels.npz')
markers = np.load('..//Data_Aquisition/Data/P_3Data/marker_data.npz')

print('Data: ', data)
print('Label: ', labels)
print('Markers: ', markers)

seq1_data = data['arr_1'][:, 0:7]
print('Seq1 Data DIMS: ', seq1_data.shape)
# plt.show(plt.plot(seq1_data))

seq1_time = data['arr_1'][:, -1]
print('Seq1 Time DIMS: ', seq1_time.shape)
# plt.show(plt.plot(seq1_time))
print('1st Data Time Stamp: ', seq1_time[0])

emoji = 7
seq = 5
trials = 49
mark_ind = emoji * seq * trials

seq1_mark = markers['arr_1']
print('Seq1 Mark DIMS: ', seq1_mark.shape)
# plt.show(plt.plot(seq1_mark))
print('1st Mark Stamp: ', seq1_mark[0])

print('Data Marker Offset: ', seq1_mark[0] - seq1_time[0])
# print(np.linspace(0, len(seq1_mark), num=trials))


# Plot Markers and Timestamps from the sequence data file together to see if there is an alignment.
