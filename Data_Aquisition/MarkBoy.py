
combo = 0
if combo == 1:
    import numpy as np
    from pylsl import StreamInlet, resolve_stream, local_clock
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui

    plot_duration = 2.0

    # first resolve an EEG stream on the lab network
    print('Looking for an EEG stream...')
    streams = resolve_stream('type', 'EEG')
    print('...Found an EEG stream.')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    markers = StreamInlet(resolve_stream('type', 'Markers')[0])

    # Create the pyqtgraph window
    win = pg.GraphicsWindow()
    win.setWindowTitle('LSL Plot ' + inlet.info().name())
    plt = win.addPlot()
    plt.setLimits(xMin=0.0, xMax=plot_duration, yMin=-1.0 * (inlet.channel_count - 1), yMax=1.0)

    t0 = [local_clock()] * inlet.channel_count
    curves = []
    for ch_ix in range(inlet.channel_count):
        curves += [plt.plot()]

    def update():
        global inlet, curves, t0
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=32)
        # print('CHUNK DIMS + TSTAMP: ', np.shape(chunk), timestamps)
        if timestamps:
            timestamps = np.asarray(timestamps)
            y = np.asarray(chunk)

            for ch_ix in range(inlet.channel_count):
                old_x, old_y = curves[ch_ix].getData()
                if old_x is not None:
                    old_x += t0[ch_ix]  # Undo t0 subtraction
                    this_x = np.hstack((old_x, timestamps))
                    this_y = np.hstack((old_y, y[:, ch_ix] - ch_ix))
                else:
                    this_x = timestamps
                    this_y = y[:, ch_ix] - ch_ix
                t0[ch_ix] = this_x[-1] - plot_duration
                this_x -= t0[ch_ix]
                b_keep = this_x >= 0
                curves[ch_ix].setData(this_x[b_keep], this_y[b_keep])
        strings, timestamps = markers.pull_chunk(0)
        print('STRINGS MARKERS: ', strings, timestamps)
        if timestamps:
            for string, ts in zip(strings, timestamps):
                plt.addItem(pg.InfiniteLine(ts - t0[0], angle=90, movable=False, label=string[0]))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(2)

    # Start Qt event loop unless running in interactive mode or using pyside.
    if __name__ == '__main__':
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

'''Example program to demonstrate how to read a multi-channel time-series
from LSL in a chunk-by-chunk manner (which is more efficient).'''

chunker = 0
if chunker == 1:
    from pylsl import StreamInlet, resolve_stream

    # first resolve an EEG stream on the lab network
    print('looking for an EEG stream...')
    streams = resolve_stream('type', 'EEG')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        chunk, timestamps = inlet.pull_chunk()
        if timestamps:
            print(timestamps, chunk)

samps = 1
if samps == 1:
    """Example program to show how to read a multi-channel time series from LSL."""

    from pylsl import StreamInlet, resolve_stream

    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    print('Stream: ', streams)

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        # print(timestamp, sample)
