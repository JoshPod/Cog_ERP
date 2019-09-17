import os
import numpy as np
import matplotlib.pyplot as plt


def interp2D(data, timestamps, output_size, plotter, verbose):
    # Resamples 2D data matrices of Samples x Channels via interpolation to produce uniform output matrices of output size x channels.

    # Calcualte number of chans.
    a, b = np.shape(data)
    num_chans = np.minimum(a, b)

    # Gen place-holder for resampled data.
    r_data = np.zeros(output_size, num_chans)
    r_time = np.linspace(0, 0.5, output_size)

    for k in range(num_chans):
        # Interpolate Data and Sub-Plot.
        yinterp = np.interp(r_time, timestamps, data[:, k])
        # Aggregate Resampled Channel Data and Timestamps.
        r_data[:, k] = yinterp

        # Plots
        if plotter == 1:
            # Sub-Plot Non-Resampled Channel Chk
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, data[:, k])
            # Sub-Plot Resampled Channel Chk
            plt.subplot(2, 1, 2)
            plt.plot(r_time, yinterp)
            plt.show()
        if verbose == 1:
            print('Original Chk DIMS: ', data[:, k].shape,
                  'Resampled Chk Dims: ', yinterp.shape)

    return r_data, r_time


data = np.random.rand(250, 7)
a, b = np.shape(data)
num_chans = np.minimum(a, b)

print(num_chans)
