'''Example program to demonstrate how to read string-valued markers from LSL.'''
# import time
# import numpy as np
from pylsl import resolve_streams, resolve_byprop, StreamInlet, StreamInfo, StreamOutlet  # ContinuousResolver
from multiprocessing import Process

# Create Marker Stream.
marker_stream_info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
marker_outlet = StreamOutlet(marker_stream_info)
print('Marker Gen: ', marker_stream_info)
print('Marker Stream Generated.')

# List all streams on network.
net_streams = resolve_streams(wait_time=1.0)
print('Streams on Network: ', net_streams)

# Connect to Marker Stream.
marker_stream = resolve_byprop('source_id', 'myuidw43536')
marker_inlet = StreamInlet(marker_stream[0])
print('Marker Stream: ', marker_stream)
print('Marker Stream Connected.')


def pusher():
    while True:
        # Push Marker.
        marker_outlet.push_sample(['Inside'])
        print('...pushed')


def puller():
    while True:
        # Pull Marker.
        sample, timestamp = marker_inlet.pull_sample(timeout=1)
        print('Marker: {0} | Timestamp: {1}'.format(sample[0], timestamp))

        # Available Markers.
        # ava_markers = marker_inlet.samples_available()
        # print('Available Markers: ', ava_markers)


if __name__ == '__main__':
    Process(target=pusher).start()
    Process(target=puller).start()
