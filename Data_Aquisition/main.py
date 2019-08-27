# BCI Speller using EEG (Quick-20 Dry Headset from Cognionics Inc.)
# General imports
import os
import glob
import numpy as np
# Stimuli Imports
import psychopy as pp
# Custom imports
from functions import list_gen, save_labels, stamp_check, folder_gen
from classes_main import LslStream, LslBuffer, EmojiStimulus, LslMarkers

'''
    Main Experiment.

        + Launch with Ctrl+Shift+B

        49 Trials per session consisting of 5 cycles each.
        Data dimenions = 938 Samples per channel.
        Sequence duration = ~1875ms (give or take 5ms)

        -For Invert use: from classes_invert import LslStream, LslBuffer, EmojiStimulus
        -For Flash use: from classes_flash import LslStream, LslBuffer, EmojiStimulus

        -Ensure application of pseudo-randomised order.
        -Ensure Cz and associated electrodes sub-200kΩ.
        -Ensure participant familiar with task.
        -Ensure is provided breaks when necessary.
        -Ensure Lab Streaming Layer has been started in Cognionics DAQ.
        -Ensure battery level > than half.

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
    print('The sampling rate is: {0} \n'.format(srate))

    # GENERATE MARKER STREAM #
    LslMarkers()

    # STIMULUS INITIALISATION #
    print('-- STIMULUS SETUP -- ')
    emoji_list = glob.glob('SVGs\Exp_Stim\All\\*.png')
    num_emoji = np.int(len(emoji_list) / 2)
    # Experimental Durations
    pres_duration = 1
    aug_duration = 0.5  # Duration of the augmentation on screen
    aug_wait = 0.5  # Temporal distance between augmentations
    inter_trial_int = 0.5
    inter_seq_interval = 0.5
    cue_interval = 0.5
    # Experimental Sequences and Trials
    seq_number = 5
    num_trials = 1
    num_iter = 50
    # Dynamic Variables
    aug = 'Invert'
    init = 'Exp'
    info = 'Details'
    window_scaling = 0.5
    stimulus_scaling = 'Medium'

    print('----Generating non-consecutive randomised augmentation lists... \n')
    aug_list = list_gen(num_emoji, seq_number, num_trials, num_iter)
    print('----Completed non-consecutive randomised augmentation lists.')
    print('----Aug List DIMS: ', np.shape(aug_list))

    'Initializing Stimuli Class Object'
    estimulus = EmojiStimulus()
    estimulus.__init__(aug, init, info, window_scaling, stimulus_scaling)
    estimulus.experiment_setup(pres_duration, aug_duration, aug_wait,
                               inter_seq_interval, seq_number, num_trials,
                               inter_trial_int, cue_interval, aug_list)
    # Generate randomised order for presentation of fixation cue.
    print('Initializing fixation shuffling sequence...')
    estimulus.cue_shuffle()

    'Print Useful Info'
    print('Duration of each sequence: {0}ms'.format(estimulus.sequence_duration * 1000))
    ammount = int(np.ceil(estimulus.sequence_duration * srate))
    print('Ammount of samples per sequence: {0}'.format(ammount))

    'Create Data and Labels Directories.'
    folder_gen('./Data/P_3Data/', './Data/P_3Data/Labels')
    data_direc = './Data/P_3Data/'
    lab_direc = './Data/P_3Data/Labels/'

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

    # START THE EXPERIMENT #
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
        for s in range(estimulus.num_seq):
            # Play sequence number s according to aug_shuffle
            estimulus.play_seq(s, t, aug)
            # Read the data during the sequence (giving some room for error)
            buffer.add(data_stream.chunk(max_samples=ammount))
            imp_buffer.add(impedances_stream.chunk(max_samples=ammount))
            # if s == 0:
            #     data = np.asarray(buffer.take_old(
            #         ammount, delete=False, filename='voltages_t{2}{0}_s{1}_'.format(t + 1, s + 1, namer)))
            #     imp_buffer.take_old(
            #         ammount, delete=False, filename='impedances_t{2}{0}_s{1}_'.format(t + 1, s + 1, namer))
            #     print('Sequence {0} / Duration: {1} | Data DIMS: {2}'.format(
            #         s + 1, np.round_(stamp_check(data)), np.shape(data)))
            # Save just the last part of the data (the one that has to belong to the trial)
            if s == s:
                data = np.asarray(buffer.take_new(
                    ammount, delete=True, filename='voltages_t{2}{0}_s{1}_'.format(t + 1, s + 1, namer)))
                imp_buffer.take_new(
                    ammount, delete=True, filename='impedances_t{2}{0}_s{1}_'.format(t + 1, s + 1, namer))
                print('Sequence {0} / Duration: {1} | Data DIMS: {2}'.format(
                    s + 1, np.round_(stamp_check(data)), np.shape(data)))
            '-------------------------------------------'
            'Sequential Prediction'
            prediction_list.append(4)
        '-------------------------------------------'
        'Trial Prediction'
        # Here we would cramp all the single choices into a final one
        final_prediction = prediction_list[0]
        # Shuffle again the augmentations
        estimulus.shuffle()
        '-------------------------------------------'
        'Label Saving: .npy'
        save_file_name = '{0}_trial_labels'.format(t + 1)
        save_labels(t + 1, lab_direc, save_file_name, estimulus.fix_shuffle[t], estimulus.fix_shuffle,
                    estimulus.num_trials, estimulus.aug_non_con[:, :, t], estimulus.num_seq, estimulus.num_emoji)
        '-------------------------------------------'
        'Data Saving: .npz'
        # Zip the EEG data files
        buffer.zip()
        imp_buffer.zip()
        # Clear buffers
        buffer.clear(names=True)
        imp_buffer.clear(names=True)
    # Close everything
    estimulus.quit()
