# herp
from pylsl import resolve_streams


def net_streams():
    # List all streams on network.
    net_streams = resolve_streams(wait_time=1.0)
    print('Streams on Network: ', net_streams)


net_streams()
