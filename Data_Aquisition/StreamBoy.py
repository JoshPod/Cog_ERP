# herp
from pylsl import resolve_streams


def net_streams():
    # List all streams on network.
    net_streams = resolve_streams(wait_time=1.0)
    print('Streams on Network: ', net_streams)


net_streams()

# '''
#
# Resolve Impedances stream, collecting 200 chunks at a time, check these against the limit unitl the impedances drop substantially.
#
# '''
# """Example program to show how to read a multi-channel time series from LSL."""
#
# from pylsl import StreamInlet, resolve_stream
#
# # first resolve an EEG stream on the lab network
# print("looking for an EEG stream...")
# streams = resolve_stream('type', 'EEG')
#
# # create a new inlet to read from the stream
# inlet = StreamInlet(streams[0])
#
# while True:
#     # get a new sample (you can also omit the timestamp part if you're not
#     # interested in it)
#     sample, timestamp = inlet.pull_sample()
#     print(timestamp, sample)
