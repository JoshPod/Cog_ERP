'''Example program to demonstrate how to read string-valued markers from LSL.'''
from classes_main import LslStream  # , LslBuffer, EmojiStimulus


# Generate marker stream.
marker_stream = LslStream(type='Markers')
from pylsl import StreamInlet, resolve_streams, resolve_byprop
import numpy as np

# list all streams on network
net_streams = resolve_streams(wait_time=1.0)
print('Streams on Network: ', net_streams)

# first resolve a marker stream on the lab network
print('looking for a marker stream...')
# streams = resolve_stream('type', 'Markers')
streams = resolve_byprop('name', 'Marker_Stream')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
stamps = []

import time
timeout = time.time() + 15   # 60 * 5 = 5 minutes from now

while True:
    # Time Controller.
    test = 0
    if time.time() > timeout:
        break
    test = test - 1

    # Get a new sample.
    sample, timestamp = inlet.pull_sample()
    print('Marker: {0} | Timestamp: {1}'.format(sample[0], timestamp))
    stamps = np.append(stamps, timestamp)

# Normalize Time-Stamps from System > Seconds.
stamps = stamps - stamps[0]
stamp_diff = np.diff(stamps)

print('STAMPS DIMS: ', np.shape(stamps))
print('STAMPS: ', stamps)
print('STAMP DIFFS: ', stamp_diff)
