# BCI Speller using EEG (Quick-20 Dry Headset from Cognionics Inc.)
# General imports
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
# Stimuli Imports
import psychopy as pp
# Custom imports
from functions import list_gen, save_labels, folder_gen, name_gen, net_streams, saver, zipr
from classes_main import LslStream, EmojiStimulus, LslMarker

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


if __name__ == '__main__':
    '=====================================================================================EEG + IMP STREAM'
    # CONNECTION TO STREAM #
    print('-- STREAM CONNECTION --')
    # Connect to the stream and create the stream handle
    print('Connecting to data stream...')
    eeg_stream = LslStream(type='EEG')
    # Connect to the impedances stream, whoever wrote the tags in the CogDAQ software wrote this one wrong
    imp_stream = LslStream(type='Impeadance')
    # Get the number of channels from the inlet to use later
    channelsn = eeg_stream.inlet.channel_count
    print('Number of channels on the stream: {0}'.format(channelsn))
    # Get the nominal sampling rate (rate at which the server sends information)
    srate = eeg_stream.inlet.info().nominal_srate()
    print('The sampling rate is: {0} \n'.format(srate))
    '===============================================================================GENERATE MARKER STREAM'
    # List all streams on network.
    net_streams()
    'Pushing.'
    # Create Marker Outlet.
    stream_name = name_gen(10)
    source_name = name_gen(10)
    marker_outlet = LslMarker(name=stream_name, type='Markers', channel_count=1, nominal_srate=0,
                              channel_format='string', source_id=source_name)
    # Pause for Marker Outlet Initialization.
    time.sleep(1)
    'Pulling.'
    # Connect to Marker Stream.
    marker_inlet = LslStream(type='Markers')
    print('Marker Inlet Generated: ', marker_inlet)
    print('Puller Active...')
    'Initialize'
    marker_outlet.push(marker='Init')
    init_mark, init_time = marker_inlet.pull(timeout=1)
    print('Init Mark 1: ', init_mark, 'Init Time 1: ', init_time)
    'Marker Array Aggregates.'
    tr_sample1 = []  # aggregate variable for 1st marker in event pairs, '0':'6'.
    tr_timestamp1 = []  # aggregate variable for 1st marker event timestamps.
    tr_sample2 = []
    tr_timestamp2 = []
    '============================================================================STIMULUS INITIALISATION'
    print('-- STIMULUS SETUP -- ')
    emoji_list = glob.glob('SVGs\Exp_Stim\All\\*.png')
    num_emoji = np.int(len(emoji_list) / 2)
    # Experimental Durations
    pres_duration = 1  # 1  # Duration of very 1st initial cue presenetation
    aug_duration = 0.125  # 0.5  # Duration of the augmentation on screen
    aug_wait = 0.250  # 0.5  # Temporal distance between augmentations
    inter_trial_int = 0.125  # 0.5
    inter_seq_interval = 0.5  # 0.5
    cue_interval = 1  # 0.5
    # Experimental Sequences and Trials
    seq_number = 5
    num_trials = 21
    num_iter = 1000
    # Dynamic Variables
    aug = 'Invert'
    init = 'Exp'
    info = 'Details'
    window_scaling = 'Full'  # 'Full' or '0.5'.
    stimulus_scaling = 'Small'
    '===============================================================================RANDOMISATION LISTS'
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
    '==========================================================================BUFFER INITIALIZATION'
    'DATA BUFFER: Samples holding.'
    eeg_stream.inlet.open_stream()
    imp_stream.inlet.open_stream()
    # Initialize data file naming formatter variable.
    namer = []
    '==============================================================================START EXPERIMENT'
    print('\n -- EXPERIMENT STARTING --')
    prediction_list = []
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
            seq_sample1, seq_timestamp1, seq_sample2, seq_timestamp2, eeg, eeg_time, imp, imp_time = estimulus.play_seq(
                s, t, aug, marker_outlet, marker_inlet, eeg_stream, imp_stream, ammount)
            # Add Marker Data to aggregate array.
            tr_sample1 = np.append(tr_sample1, seq_sample1)
            tr_timestamp1 = np.append(tr_timestamp1, seq_timestamp1)
            tr_sample2 = np.append(tr_sample2, seq_sample2)
            tr_timestamp2 = np.append(tr_timestamp2, seq_timestamp2)
            print('Tr Sample 1 DIMS: ', np.shape(tr_sample1))
            '--------------------------------------------------------'
            'Sequential Prediction'
            prediction_list.append(4)
            'Save Sequence.'
            filename = 'voltages_t{2}{0}_s{1}_'.format(t + 1, s + 1, namer)
            saver(filename, eeg, eeg_time)
            filename = 'impedanc_t{2}{0}_s{1}_'.format(t + 1, s + 1, namer)
            saver(filename, imp, imp_time)
        'Trial Prediction'
        # Here we would cramp all the single choices into a final one
        final_prediction = prediction_list[0]
        # Shuffle again the augmentations
        estimulus.shuffle()
        'Label Saving: .npy'
        save_file_name = '{0}_trial_labels'.format(t + 1)
        save_labels(t + 1, lab_direc, save_file_name, estimulus.fix_shuffle[t], estimulus.fix_shuffle,
                    estimulus.num_trials, estimulus.aug_non_con[:, :, t], estimulus.num_seq, estimulus.num_emoji)
        'Zip Sequence.'
        zipr(data_direc, ext='.npy', keyword='volt', full_ext=1, compress=False)
        zipr(data_direc, ext='.npy', keyword='imp', full_ext=1, compress=False)
    # Prints
    print('Aug List: ', aug_list.shape, aug_list)
    print('Aug List Slice: ', aug_list[:, 0, 0])
    print('Aug_Non_Con: ', estimulus.aug_non_con.shape, estimulus.aug_non_con)
    # Marker Check
    marker_file = './Data/P_3Data/marker_data.npz'
    np.savez(marker_file, tr_sample1, tr_timestamp1, tr_sample2, tr_timestamp2)
    print(tr_sample1.shape)
    print(tr_timestamp1.shape)
    print(tr_sample2.shape)
    print(tr_timestamp2.shape)
    # Delete Outlet Stream.
    marker_outlet.outlet_del()
    pp.clock.wait(2)
    # Delete Inlet Stream.
    marker_inlet.inlet_del()
    # Close everything
    estimulus.quit()
