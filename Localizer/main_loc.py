# BCI Speller using EEG (Quick-20 Dry Headset from Cognionics Inc.)
# General imports
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
# Stimuli Imports
import psychopy as pp
# Custom imports
from func_loc import save_labels, stamp_check, loc_shuffle, folder_gen, live_plotter
from classes_loc import LslStream, LslBuffer, EmojiStimulus
# Data Analysis Imports
sys.path.insert(0, 'C:\P300_Project\Data_Analysis')
import ProBoy as pb


'''
    Main Experiment.

        + Launch with Ctrl+Shift+B

        10 Trials per session.
        Data dimenions = ??? Samples per channel.
        Sequence duration = ~????ms
        + Kill with Ctrl+Q

'''


# Main #
if __name__ == '__main__':
    # CONNECTION TO STREAM #
    print('-- STREAM CONNECTION --')
    # Connect to the stream and create the stream handle
    print('Connecting to data stream...')
    data_stream = LslStream(type='EEG')
    # Connect to the impedances stream, whoever wrote the tags in the CogDAQ software wrote this one wrong
    impedances_stream = LslStream(type='Impeadance')
    # Get the number of channels from the inlet to use later
    channelsn = data_stream.inlet.channel_count
    print('Number of channels on the stream: {0}'.format(channelsn))
    # Get the nominal sampling rate (rate at which the server sends information)
    srate = data_stream.inlet.info().nominal_srate()
    print('Sample Rate: {0}Hz \n'.format(np.int(srate)))

    # STIMULUS INITIALISATION #
    print('-- STIMULUS SETUP -- ')
    emoji_list = glob.glob('SVGs\Loc_Stim\\*.png')
    num_emoji = np.int(len(emoji_list))
    num_trials = 200
    pres_duration = 5
    aug_duration = 0.125
    inter_trial_int = 3
    # Invert = Use Inversion Stimuli | Flash = Use Rectangle Stimuli.
    aug = 'Flash'
    init = 'Exp'
    info = 'Details'
    window_scaling = 'Full'
    stimulus_scaling = 'Small'

    'Aug List'
    aug_list = loc_shuffle(num_emoji, num_trials)

    'Initializing Stimuli Class Object'
    estimulus = EmojiStimulus()
    estimulus.__init__(aug, init, info, window_scaling, stimulus_scaling)
    estimulus.experiment_setup(pres_duration, aug_duration, num_trials,
                               inter_trial_int, aug_list)
    'Print Useful Info'
    print('Loc Shuffle Order: ', estimulus.loc_shuffle)
    print('Duration of each sequence: {0}'.format(estimulus.trial_duration * 1000))
    ammount = int(np.ceil(estimulus.trial_duration * srate))
    print('Ammount of samples per sequence: {0}'.format(ammount))

    'Create Data and Labels Directories.'
    folder_gen('./Data/LocData/', './Data/LocData/Labels')
    data_direc = './Data/LocData/'
    lab_direc = './Data/LocData/Labels/'

    'BUFFER INITIALIZATION'
    'DATA BUFFER: Samples holding.'
    buffer = LslBuffer()
    # 1st array now full due to these additions.
    data_stream.chunk()
    pp.clock.wait(1)
    data_stream.chunk()
    data_stream.connect
    'IMP BUFFER: Impedances holding.'
    buffer = LslBuffer()
    # 1st array now full due to these additions.
    impedances_stream.chunk()
    pp.clock.wait(1)
    impedances_stream.chunk()
    impedances_stream.connect
    imp_buffer = LslBuffer()

    # Initializing Data Naming Formatter.
    namer = []

    'START THE EXPERIMENT'
    print('\n -- EXPERIMENT STARTING --')
    prediction_list = []
    # Tell the stream to start
    for t in range(estimulus.num_trials):
        print('Trial:', str(t + 1))
        # Establish data naming index.
        if t < 9:
            namer = '000'
        elif t >= 9 and t < 99:
            namer = '00'
        elif t >= 99 and t < 999:
            namer = '0'
        elif t >= 999:
            namer = ''

        # Play sequence number s according to aug_shuffle.
        estimulus.play_emoji(t)
        # Read the data during the sequence (giving some room for error).
        buffer.add(data_stream.chunk(max_samples=ammount))
        imp_buffer.add(impedances_stream.chunk(max_samples=ammount))

        if t == 0:
            data = np.asarray(buffer.take_old(
                ammount, delete=False, filename='Voltages_t{0}_{1}'.format(namer, t + 1)))
            imp = np.asarray(imp_buffer.take_old(
                ammount, delete=False, filename='Impedances_t{0}_{1}'.format(namer, t + 1)))
            print('Trial {0} / Duration: {1} | Data DIMS: {2}'.format(
                t + 1, np.round_(stamp_check(data), decimals=2), np.shape(data)))
        # Save just the last part of the data (the one that has to belong to the trial).
        elif t != 0:
            data = np.asarray(buffer.take_new(
                ammount, delete=True, filename='Voltages_t{0}_{1}'.format(namer, t + 1)))
            imp = np.asarray(imp_buffer.take_new(
                ammount, delete=True, filename='Impedances_t{0}_{1}'.format(namer, t + 1)))
            print('Trial {0} / Duration: {1} | Data DIMS: {2}'.format(
                t + 1, np.round_(stamp_check(data), decimals=2), np.shape(data)))

        'Plots.'
        plotter = 1
        if plotter == 1:
            # plt.plot(pre_data)
            # plt.show()
            # time_stamps = pb.time_stamper(data, show_stamps=1)
            # plt.plot(time_stamps)
            # plt.show()
            # plt.plot(imp[:, -1])
            # plt.show()
            times_ = np.arange(ammount)
            pre_data = pb.prepro(data, samp_ratekHz=0.5, zero='ON', ext='INC-Ref', elec=[1],
                                 filtH='ON', hlevel=1, filtL='ON', llevel=10,
                                 notc='LOW', notfq=50, ref_ind=-7, ref='A2', avg='OFF')

        '-------------------------------------------'
        'Sequential Prediction'
        prediction_list.append(4)
        '-------------------------------------------'
        'Trial Prediction'
        # Here we would collapse to one of the many predictions.
        final_prediction = prediction_list[0]
        '-------------------------------------------'
        'Label Saving: .npy'
        save_file_name = '{0}_Trial_Labels'.format(t + 1)
        save_labels(t + 1, lab_direc, save_file_name,
                    estimulus.loc_shuffle[t], estimulus.loc_shuffle)
        '-------------------------------------------'
        # Clear buffers
        # buffer.clear(names=True)
        # imp_buffer.clear(names=True)

    # Close everything
    estimulus.quit()

'Zipping: VOID.'
# 'Data Saving: .npz'
# # Zip the EEG data files
# buffer.zip()
# imp_buffer.zip()
