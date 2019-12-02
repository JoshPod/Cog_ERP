def zero_mean(data):
    'Zeros Data, accepts orientation Samples x Channels.'
    a, b = np.shape(data)
    # Preform zero array.
    zero_data = np.zeros((a, b))
    for i in range(b):
        zero_data[:, i] = data[:, i] - np.mean(data[:, i])
    return zero_data

import numpy as np
from pylsl import StreamInlet, resolve_stream
# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'Impeadance')  # type='Impeadance')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

# Keyword Paramters.
imps = []
channels = ['Fz', 'Cz', 'Pz', 'P4', 'P3', 'O1', 'O2']
controller = 0
num_chans = 7
imp_limit = 15

while controller == 0:
    # Grab Impedances Data Chunks.
    chunk, timestamps = inlet.pull_chunk(timeout=2, max_samples=500)
    if timestamps:
        imps = np.asarray(chunk)
        imps = imps[:, 0:num_chans]
        print(imps.shape)
        imps = zero_mean(imps)
        for j in range(1):
            con_list = np.zeros(num_chans)  # , dtype=int
            for i in range(num_chans):
                # Range value of channel durng pause.
                r_val = np.amax(imps[:, i]) - np.amin(imps[:, i])
                if r_val > imp_limit:
                    print('----Channel Impedances Awaiting Stabilization: {0}  |  Range Value: {1}'.format(channels[i], r_val))
                elif r_val < imp_limit:
                    con_list[i] = 1
                    print('----Channel Impedances Stabilised: {0}  |  Range Value: {1}'.format(channels[i], r_val))
                if np.sum(con_list) == num_chans:
                    controller = 1
