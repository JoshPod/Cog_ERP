# Function Bank
import os.path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Unused Modules
# import matplotlib
# import pandas as pd
# import seaborn as sns
# from scipy import signal
# from sklearn.svm import SVC
# from collections import OrderedDict
# from mne.decoding import Vectorizer
# from pyriemann.spatialfilters import CSP
# from pyriemann.classification import MDM
# from sklearn.metrics import confusion_matrix
# from sklearn.grid_search import GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
# from sklearn.model_selection import train_test_split, ShuffleSplit  # cross_val_predict,

# Shortcuts
'Ctrl + Alt + Shift + [ = Collapse all functions'


def int_labels(y):
    # Generate and output integer labels.
    y2 = []
    for p in range(len(y)):
        if p == 0:
            y2 = int(y[p])
        else:
            y2 = np.append(y2, int(y[p]))
    return y2


def lda_(data, labels, split, div, num_comp, meth, scaler, covmat, verbose):
    '''
    Application of LDA for data discrimation analysis.

    Assumes Trials x Samples.

    Inputs:
        data = data matrix of EEG.
        labels = ground truth labels.
        split = num splits in the Stratified Shuffle Splits for train and test sets.
        div = division in the split between train and test e.g 0.85 == 85% train weighted.
        n_components = dimensions of the embedding space.
        meth = LDA method e.g. 'eigen'
        scaler = 'min' for min_max scaler, 'standard' for standard scikit learn scaler.
        covmat = compute and print covariance matrix of results, if 1 perform.
        verbose = 1 : print out info.

    Output:
        Plots of the TSNE, this analysis is only for visualization.

    Example:

    pb.lda_(data, labels, split, div, num_comp, meth, scaler, verbose)

    '''
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedShuffleSplit

    '---Parameters---'
    if split is None:
        split = 2
    if div is None:
        div = 0.85
    if num_comp is None:
        num_comp = 2
    if meth is None:
        meth = 'eigen'

    '---Data Prep---'
    # SwapAxes.
    data = np.swapaxes(data, 0, 1)
    # Feature Scaling.
    if scaler == 'min':
        data = min_max_scaler(data)
    elif scaler == 'standard':
        data = stand_scaler(data)
    # Data splitting for test and train.
    sss = StratifiedShuffleSplit(n_splits=split, train_size=div, random_state=2)
    sss.get_n_splits(data, labels)
    for train_index, test_index in sss.split(data, labels):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    '---LDA---'
    clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto',
                                     n_components=num_comp).fit(X_train, y_train)
    # Plots Visulizing Groups in LDA: X_r fitted comparison plots.
    X_r2 = clf.fit(X_train, y_train).transform(X_train)
    # Performance.
    if meth == 'eigen':
        print('Explained Covariance Ratio of Components \n: ', clf.explained_variance_ratio_)
    print('Classes: ', clf.classes_)
    predictions = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)
    if verbose == 1:
        print('X_r2 DIMS: ', X_r2.shape)
        print('Size of Test Sample: ', len(y_test))
        print('Actual Labels: ', y_test)
        print('Predictions:   ', predictions)
        print('Predict Probabilities: \n', np.round(pred_proba, decimals=2))
        print('Score: ', score)
    if covmat == 1:
        num_classes = len(np.unique(y_test))
        num_samps = len(predictions)
        cov_mat = np.zeros((num_classes, num_classes))
        count = 0
        for i in range(num_classes):
            for j in range(num_classes):
                # if y_test[count] == predictions[count]:
                cov_mat[y_test[count], predictions[count]
                        ] = cov_mat[y_test[count], predictions[count]] + 1
                count = count + 1
        print(cov_mat)


def tSNE_2D(X, labels, n_components, perplexities, learning_rates, scaler):
    '''
    Application of 2D TSNE for visualization of high dimensional data.

    Assumes Trials x Samples.

    Inputs:
        X = data matrix of EEG.
        labels = ground truth labels.
        n_components = dimensions of the embedding space.
        perplexities = akin to complexity of the data and following computations.
        learning_rates = degree at which computations will attempt to converge.
        scaler = 'min' for min_max scaler, 'standard' for standard scikit learn scaler.

    Output:
        Plots of the TSNE, this analysis is only for visualization.

    Example:

    pb.tSNE_2D(aug_data, labels, n_components=None, perplexities=None, learning_rates=None, scaler='min')

    '''
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold
    from time import time

    '---Parameters---'
    if n_components is None:
        n_components = 2  # Typically between 5-50.
    if perplexities is None:
        perplexities = [15, 30, 45, 60]  # Typically around 30.
    if learning_rates is None:
        learning_rates = [5, 10, 500, 1000]  # Typically between 10 - 10000
    '---Subplot Prep---'
    (fig, subplots) = plt.subplots(len(learning_rates), len(perplexities) + 1, figsize=(15, 8))
    '---Data Prep---'
    # Feature Scaling.
    if scaler == 'min':
        X = min_max_scaler(X)
    elif scaler == 'standard':
        X = stand_scaler(X)
    # Swap Axes.
    X = np.swapaxes(X, 0, 1)
    print('FINAL EEG DIMS: ', X.shape)
    '---Label Mapping | RED == P3 | GREEN == NP3---'
    red = labels == 0
    green = labels == 1
    '---Plotting P3 vs NP3---'
    ax = subplots[0][0]
    ax.scatter(X[red, 0], X[red, 1], c="r")
    ax.scatter(X[green, 0], X[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    '---TSNE---'
    for j, learning_rate in enumerate(learning_rates):
        for i, perplexity in enumerate(perplexities):
            print('LOC DATA: Perplexity={0} | Learning Rate={1}'.format(
                perplexity, learning_rate))
            ax = subplots[j][i + 1]
            t0 = time()
            tsne = manifold.TSNE(n_components=n_components, init='random',
                                 random_state=0, perplexity=perplexity, learning_rate=learning_rate,
                                 n_iter=10000, n_iter_without_progress=300, verbose=1)
            Y = tsne.fit_transform(X)
            t1 = time()
            print('-------Duration: {0} sec'.format(np.round(t1 - t0), decimals=2))
            ax.set_title('Perplexity={0} | \n Learning Rate={1}'.format(perplexity, learning_rate))
            'Plotting'
            ax.scatter(Y[red, 0], Y[red, 1], c="r")
            ax.scatter(Y[green, 0], Y[green, 1], c="g")
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')
    plt.tight_layout()
    plt.show()


def stand_scaler(X):
    '''
    Standard Scaler from SciKit Learn package.

    Assumes Trials x Samples.

    Input: Data Matrix.

    Output: Standardized Data Matrix.

    Example:

    stand_X = pb.stand_scaler(X)

    '''
    from sklearn.preprocessing import StandardScaler
    # https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X


def min_max_scaler(X):
    '''
    Min Max Scaler from SciKit Learn package.

    Assumes Trials x Samples.

    Imput: Data Matrix.

    Output: Standardized Data Matrix.

    Example:

    mms_X = pb.min_max_scaler(X)

    '''
    from sklearn.preprocessing import MinMaxScaler
    # https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/
    mss = MinMaxScaler()
    X = mss.fit_transform(X)
    return X


def slice_ext(data_file, data_type, labels_file, markers_file, num_chan, num_emoji, num_seq, out_size, plotter, verbose):
    '''
    Method for extracting emoji level / event data chunks based on the on-set / offset
    of marker stream pushed label timestamps. This means we extract data only during the
    times at which the stimuli has begun and ended, yielding more rigourous time-corrective
    data values. These emoji-level chunks are interpolated to ensure consistency in
    the temporal separation of data time-points.

    ASSUMES 8 Channels: Fz, Cz, Pz, P4, P3, O1, O2, A2. / [0:7] , important for seq_data parsing.

    # Inputs:

    data_file = the file location containing either the eeg / imp .npz's (Trial-Level).
    data_type = either 'volt' or 'imp' for voltages or impedances data file extraction and slicing.
    labels_file = the file location containing the labels files e.g. 0001_trial_labels.npz (Trial-Level).
    marker_file = the file location containing the marker file (all pushed markers and timestamps) (Session-Level).
    num_chan = number of channels for extraction.
    num_emoji = number of emojis in the stimulus array.
    num_seq = number of sequences in each trial.
    out_size = size of the channel level signal chunk to want returned from the interpolation function.
    plotter = plot showing the extraction and resampling of one emoji event across all channels using zeroed
              data. The data is not zeroed, only the timestamps, data zeroing is done by prepro function,
              see note below. 1 == plot, 0 == no plot.
    verbose = details of the function operating, 1 == print progress, 0 == do not print.

    # Outputs:

    starts = marker timestamps for all pushed emoji labels occuring at the START of the event (pre-augmenttion).
    ends = marker timestamps for all pushed emoji labels occuring at the END of the event (post-augmentation).
    seq_chk = Extracted data array using marker start and end pushed timestamps, yet to be re-sampled.
    r_data = Aggregate arrays of the extracted and resampled event data, dims = Samples x Channels x Seqs x Trials
    r_times_zer = Aggregate arrays of the ZEROED extracted and resampled event timestamps, dims = Samples x Seqs x Trials
    r_times = 1D array of the extracted non-re-sampled event timestamps for temporal linear session time checking.
    num_trials = number of trials across entire session.


    Example:

    NOTE: the timestamps ARE zeroed, the data is NOT zeroed. The interp function requires the x-axis to be increasing.
    The timestamps from LSL are output in such large system time numbers it cannot reliably detect the increasing,
    or some strange rounding is occuring.

    -Ensure non-zeroed time-stamps are stored, reshaped and plotted to ensure there is cross-session temporal consistency.
    '''

    # Get Trial Data file locations.
    dat_files = pathway_extract(data_file, '.npz', data_type, full_ext=0)
    eeg_files = path_subplant(data_file, np.copy(dat_files), 0)
    if verbose == 1:
        print('EEG Files DIMS: ', np.shape(eeg_files), 'EEG Files: ', eeg_files)
    # Experimental Parameters.
    num_trials = len(eeg_files)
    # Get Labels file locations.
    grn_files = pathway_extract(labels_file, '.npz', 'trial', full_ext=0)
    lab_files = path_subplant(labels_file, np.copy(grn_files), 0)
    if verbose == 1:
        print('Lab Files DIMS: ', np.shape(lab_files), 'Lab Files: ', lab_files)
    # Marker Data.
    markers = np.load(markers_file)
    # Marker Timestamps.
    starts = markers['arr_1']
    ends = markers['arr_3']
    # Marker Labels e.g. '6', or '0', or '1', a string of the actual emoji location augmented..
    mark_start = markers['arr_0']
    mark_end = markers['arr_2']
    # Markers reshape by trials and seqs.
    starts = np.reshape(starts, (num_trials, num_seq, num_emoji))
    ends = np.reshape(ends, (num_trials, num_seq, num_emoji))
    # Aggregate arrays.
    # Samples x Channels x Seq x Trials
    r_data = np.zeros((out_size, num_chan, num_emoji, num_seq, num_trials))
    # Samples x Sequences x Trials : Non-Zeroed.
    r_times = []
    # Aggregate arrays for zeroed timestamps plotting.
    r_times_zer = np.zeros((out_size, num_emoji, num_seq, num_trials))

    for t in range(num_trials):
        # Loading Data.
        data = np.load(eeg_files[t])
        # Loading Labels.
        labels = np.load(lab_files[t])  # .npz containg both labels related files (see below).
        # Matrix containing the order of augmentations for all emoji locations across the trial.
        order = labels['arr_0']
        # Extract Targ Cued for each seqence.
        targs = labels['arr_1']  # List of target cues across the entire trial.
        targ_cue = targs[t]  # List of the nth target cue for the nth trial.
        if verbose == 1:
            print('EEG File_Name', eeg_files[t])
            print('Labels: ', labels)
            print('LABS File_Name', lab_files[t])
            print('Order', order)
            print('Targs', targs)
            print('Targ Cue', targ_cue)
            print('Marker Start Labels', mark_start)
            print('Marker End Labels', mark_end)

        for i in range(num_seq):
            pres_ord = order[i, :]  # List of the nth Sequence's augmentation order from 1 trial.
            # temporal position of target cue augmented during the trial.
            f_index = pres_ord[targ_cue]
            if verbose == 1:
                print('Pres Ord: ', pres_ord)
                print('F Index: ', f_index)
            # EEG Data.
            sequence = 'arr_{0}'.format(i)  # list key for extraction of 1 sequence worth of data.
            # Sequence-Level data parsing only relevent electrodes
            seq_data = data[sequence][:, 0:num_chan]
            # Sequence-level timestamps from main data array.
            seq_time = data[sequence][:, -1]
            # print('seq_time: ', seq_time)
            # plt.show(plt.plot(seq_time))
            if verbose == 1:
                print('Seq Data DIMS: ', seq_data.shape)
                print('Seq Time DIMS: ', seq_time.shape)
            for j in range(num_emoji):
                # START: Find nearest value of the marker timestamps in the corresponding data timestamp array.
                v_s = starts[t, i, j]
                # Index in timestamp array closest to onset of marker indcating the start of the emoji event.
                str_idx = (np.abs(seq_time - v_s)).argmin()
                # END: Find nearest value of the marker timestamps in the corresponding data timestamp array.
                # Pad to ensure all P3 wave form extracted, taking marker start point and adding 0.5s, indexing to that location in the data array.
                # Just a check to ensure the end marker is not below 0.3s (past peak of the P3 waveform).
                if ends[t, i, j] < starts[t, i, j] + 0.3:
                    v_e = starts[t, i, j] + 0.5
                else:
                    print('Crash Code: End Marker Positioned Before P3 Propogation.')
                    v_e = starts[t, i, j] + 0.5
                # Index in timestamp array closest to onset of marker indcating the end of the emoji event.
                # print('Seq Time: ', seq_time)
                print('V_s: ', v_s, 'V_E: ', v_e)
                end_idx = (np.abs(seq_time - v_e)).argmin()
                print('str_idx : ', str_idx, 'end_idx: ', end_idx)
                # Indexing into data array to extract currect P300 chunk.
                seq_chk = seq_data[str_idx: end_idx, :]
                # Indexing into timestamp array to extract currect P300 chunk timestamps.
                if verbose == 1:
                    print('Str Idx: ', str_idx, 'End Idx: ', end_idx)
                seq_temp = seq_time[str_idx: end_idx]  # Non-Zeroed Timestamps @ Sequence Level.
                r_times = np.append(r_times, seq_temp)  # Non-Zeroed Timestamps @ Trial Level.
                # Zeroed Timestamps @ Sequence Level.
                seq_temp_zer = seq_temp - seq_temp[0]
                # Resampling Interpolation Method @ Channel Level, using zeroed timestamp values.
                r_data[:, :, j, i, t], r_times_zer[:, j, i, t] = interp2D(
                    seq_chk, seq_temp_zer, output_size=out_size, plotter=0, verbose=0)
                'Verbose Details of operation.'
                if verbose == 1:
                    print('V_s: ', v_s, 'Start IDX: ', str_idx, 'V_e: ', v_e, 'End IDX: ', end_idx)
                    print('Diff in time between Start and End: ', v_e - v_s)
                    print('Emoji: {0} | Seq: {1}'.format(j + 1, i + 1),
                          'Seq_Chk Dims: ', seq_chk.shape)
                    print('r_data DIMS: ', r_data.shape, 'r_times DIMS: ', r_times.shape)
                    'Zeroed Data Section for Plotting.'
    if plotter == 1:
        plt.plot(r_times_zer[:, 0, 0, 0], r_data[:, 0, 0, 0, 0])
        plt.title(
            'Resampled Timestamps (X Axis) and Data (Y Axis) for 1st Channel in 1st Sequence in 1st Trial')
        plt.show()
        plt.plot(r_times)
        plt.title(
            'Non-Resampled Timestamps to check ascending and consistent session progression in temporal terms.')
        plt.show()
    return starts, ends, seq_chk, r_data, r_times_zer, r_times, num_trials


def time_check(data_file, markers_file):
    '''
    Compares timestapms collected by the eeg / imp and marker streams to ensure maxinmal alignment.
    Plots the onset and offset of pushed marker stream samples against the timestamp eeg / imp stream values.
    Also, plots data using data stream timstamp axis vs pre-gen perfect axis to illustrate temporal acquisition inconsistency.

    # Inputs;

    data_file = specific data .npz file for a single trial, assumes 14 channels, 7 actual electrodes.

    makers_file = marker_data.npz containing the pushed marker labels for the entire session period.

    # NO OUTPUTS.

    # Example:

    data_file = '..//Data_Aquisition/Data/P_3Data/voltages_t0001_s1_190917_102243806411.npz'
    markers_file = '..//Data_Aquisition/Data/P_3Data/marker_data.npz'

    time_check(data_file, markers_file)

    '''

    data = np.load(data_file)
    markers = np.load(markers_file)
    # EEG Data.
    sequence = 'arr_0'
    seq1_data = data[sequence][:, 0:7]
    print('Seq Data DIMS: ', seq1_data.shape)
    # EEG Timestamps.
    seq1_time = data[sequence][:, -1]
    print('Seq Time DIMS: ', seq1_time.shape)
    print('First Data Time Stamp: ', seq1_time[0], ': ', 0)
    print('Last Data Time Stamp: ', seq1_time[-1], ': ', seq1_time[-1] - seq1_time[0])
    # Marker Data.
    seq_mark_end = markers['arr_3']
    seq_mark_str = markers['arr_1']
    print('Seq1 Mark DIMS: ', seq_mark_str.shape)
    print('1st Mark Stamp: ', seq_mark_str[0])
    # Diff between 1st EEG Timestamp and 1st Marker Timestamp.
    print('Data Marker Offset: ', seq_mark_str[0] - seq1_time[0])

    for i in range(len(seq_mark_str)):
        print('Length Mark Collection Emoji {0}: '.format(
            i + 1), seq_mark_end[i] - seq_mark_str[i], 'Start: ', seq_mark_str[i], 'End: ', seq_mark_end[i])

    'Plots'
    # Plot EEG Data Timestamps.
    plt.plot(seq1_time)
    num_emojis = 7
    print('1st Sequence Start Times: ', seq_mark_str[0:6])
    mark1 = np.zeros(len(seq1_time))
    mark2 = np.zeros(len(seq1_time))

    for i in range(num_emojis):
        # Plot Marker Start Times.
        mark1[:] = seq_mark_str[i]
        print('Start Time: ', seq_mark_str[i])
        plt.plot(mark1)
        # Plot Marker End Times.
        mark2[:] = seq_mark_end[i]
        print('End Time: ', seq_mark_end[i])
        plt.plot(mark2)
    plt.title('Marker Start and End Points Overlaid on EEG OR IMP Data Timestamps.')
    plt.show()

    # Data with Data Timestamp Axis.
    plt.plot(seq1_time, seq1_data[:, 0])
    plt.title('Data with Data Timestamp Axis')
    plt.show()
    # Data with Pre-Gen Timestamp Axis.
    gen_time = np.arange(len(seq1_data[:, 0]))
    plt.plot(gen_time, seq1_data[:, 0])
    plt.title('Data with Pre-Gen Timestamp Axis')
    plt.show()

    # Find nearest value of the marker start timestamps in the corresponding data timestamp array.
    arr = seq1_time
    v = seq_mark_str[0]
    idx = (np.abs(arr - v)).argmin()
    print('Start Idx: ', idx, 'Idx of Seq Time: ',
          seq1_time[idx], 'Idx of Seq Data: ', seq1_data[idx, 0])
    # Find nearest value of the marker end timestamps in the corresponding data timestamp array.
    arr = seq1_time
    v = seq_mark_end[6]
    idx = (np.abs(arr - v)).argmin()
    print('End Idx: ', idx, 'Idx of Seq Time: ',
          seq1_time[idx], 'Idx of Seq Data: ', seq1_data[idx, 0])


def binary_labeller(labels, verbose):
    '''
    Option for binary labelling of te data as either containing P3 ('0') or containing NP3 ('1').

    Assumes 1D array of integers computed via the spatial labeller output as specified in the script.

    Verbose: if 1 prints some examples from the ouput, if 0, no printing.

    '''
    y = labels
    for i in range(len(y)):
        if int(y[i]) != 0:
            y[i] = '0'
        elif int(y[i]) == 0:
            y[i] = '1'
    if verbose == 1:
        print('Base Normalized Y Labels: ', y[0:10])
    return y


def spatial_labeller(labels_file, num_emoji, num_seq, verbose):
    '''

    Method of extracting spatial labels from the flash order of emoji augmentations.
    e.g. Sequence = [3, 5, 1, 6, 0, 2, 4] | Target Cue = 3, meaning that the 4th emoji
    underwent an augmentation, as from this sequence above you can see that the 4th
    emoji was augmented 2nd (1).

    Labelling the augmentation events spatially involves describing each emoji in terms
    of distance from the target emoji which is being attended/ fixated.

    Sequence = [3, 5, 1, 6, 0, 2, 4] , would become [2, 1, 0, 1, 2, 3, 4], with zero indicating
    the emoji location cued and value labels emanating from this location increasing
    as a function of distance from this spatial location.

    # Inputs:

    labels_file = Simply specify the file location of the trial labels.
                  e.g. '..//Data_Aquisition/Data/P_3Data/Labels/'
    num_emoji =  number of emoji in the stimulus array.

    num_seq = number of sequences in each trial.

    verbose = specify if you want to print details of labelling (1 == Yes, 0 == No).

    # Outputs:

    sp_labels = a 1D label array for all event chunks segmented by SegBoy (around 500ms each).

    # Example:

    sp_labels = pb.spatial_labeller(labels_file, num_emoji, num_seq, verbose=0)

    OR

    sp_labels = spatial_labeller('..//Data_Aquisition/Data/P_3Data/Labels/', 7, 5, 0)

    '''

    grn_files = pathway_extract(labels_file, '.npz', 'trial', full_ext=0)
    lab_files = path_subplant(labels_file, np.copy(grn_files), 0)
    if verbose == 1:
        print('Lab Files DIMS: ', np.shape(lab_files), 'Lab Files: ', lab_files)

    # Experimental Parameters.
    num_trials = len(lab_files)
    if verbose == 1:
        print('Num Trials: ', num_trials)

    # Aggregate PreFrom Array @ Trial Level.
    sp_labels = []

    for t in range(num_trials):
        # Loading Labels.
        labels = np.load(lab_files[t])
        order = labels['arr_0']
        # Extract Targ Cued for each seqence.
        targs = labels['arr_1']
        targ_cue = targs[t]
        if verbose == 1:
            print('Labels: ', labels)
            print('LABS File_Name', lab_files[t])
            print('Order', order)
            print('Targs', targs)
            print('Targ Cue', targ_cue)
            # Spatial Labelling.
            print('------Spatial Labeller')
        # Aggregate Preform Array.
        fin_sp = []
        for j in range(num_seq):
            sp_pres = []
            targ_cue = targs[j]
            if verbose == 1:
                print('Targ Cue: ', targ_cue)
            for i in range(num_emoji):
                pin = np.array2string(np.abs(i - targ_cue))
                if i == 0:
                    sp_pres = pin
                else:
                    sp_pres = np.append(sp_pres, pin)
                if i and j and t == [0, 0, 0]:
                    sp_labels = pin
                else:
                    sp_labels = np.append(sp_labels, pin)
            sp_pres = np.expand_dims(sp_pres, axis=1)
            if j == 0:
                fin_sp = sp_pres
            else:
                fin_sp = np.append(fin_sp, sp_pres, axis=1)
            if verbose == 1:
                print('Sequence {}: '.format(j + 1), sp_pres.shape, ' \n', sp_pres)
        if verbose == 1:
            print('Trial {}: '.format(t + 1), fin_sp.shape, ' \n', fin_sp)
            # Aggreaate into 1D tp_labels array across the Seesion.
            print('Spatial Labels 1D Array: ', sp_labels.shape, ' \n', sp_labels)
    return sp_labels


def temporal_labeller(labels_file, num_emoji, num_seq, verbose):
    '''
    Method of extracting temporal labels from the flash order of emoji augmentations.
    e.g. Sequence = [3, 5, 1, 6, 0, 2, 4] | Target Cue = 3, meaning that the 4th emoji
    underwent an augmentation, as from this sequence above you can see that the 4th
    emoji was augmented 2nd (1).

    Labelling the augmentation events temporal involves describing each emoji in terms
    of distance in TIME from the target emoji which is being attended/ fixated.

    Sequence = [3, 5, 1, 6, 0, 2, 4] , would become [+2, +4, 0, +5, -1, +1, +3], with zero
    indicating the emoji location cued and value labels assigned to other locations
    differing as a function of temporal distance from this timed event.

    # Inputs:

    labels_file = Simply specify the file location of the trial labels.
                  e.g. '..//Data_Aquisition/Data/P_3Data/Labels/'

    num_emoji =  number of emoji in the stimulus array.

    num_seq = number of sequences in each trial.

    verbose = specify if you want to print details of labelling (1 == Yes, 0 == No).

    # Outputs:

    tp_labels = a 1D label array for all event chunks segmented by SegBoy (around 500ms each).

    # Example:

    tp_labels = pb.temporal_labeller(labels_file, num_emoji, num_seq, verbose=0)

    OR

    tp_labels = temporal_labeller('..//Data_Aquisition/Data/P_3Data/Labels/', 7, 5, verbose=0)

    '''
    # Get Labels file locations.
    labels_file = labels_file
    grn_files = pathway_extract(labels_file, '.npz', 'trial', full_ext=0)
    lab_files = path_subplant(labels_file, np.copy(grn_files), 0)
    if verbose == 1:
        print('Lab Files DIMS: ', np.shape(lab_files), 'Lab Files: ', lab_files)

    # Experimental Parameters.
    num_trials = len(lab_files)
    if verbose == 1:
        print('Num Trials: ', num_trials)

    # Aggregate PreFrom Array @ Trial Level.
    tp_labels = []
    for t in range(num_trials):
        # Loading Labels.
        labels = np.load(lab_files[t])
        order = labels['arr_0']
        # Extract Targ Cued for each seqence.
        targs = labels['arr_1']
        targ_cue = targs[t]
        if verbose == 1:
            print('Labels: ', labels)
            print('LABS File_Name', lab_files[t])
            print('Order', order)
            print('Targs', targs)
            print('Targ Cue', targ_cue)
        for j in range(num_seq):
            tp_pres = []
            pres_ord = order[j, :]
            targ_cue = targs[j]
            f_index = pres_ord[targ_cue]
            if verbose == 1:
                print('Pres Order: ', pres_ord)
                print('Targ Cue: ', targ_cue)
            for i in range(num_emoji):
                pin = np.array2string(pres_ord[i] - f_index)
                # Sequence Level Aggregation.
                if i == 0:
                    tp_pres = pin
                else:
                    tp_pres = np.append(tp_pres, pin)
                # Cross Trial Aggregation.
                if i and j and t == [0, 0, 0]:
                    tp_labels = pin
                else:
                    tp_labels = np.append(tp_labels, pin)
            tp_pres = np.expand_dims(tp_pres, axis=1)
            if j == 0:
                # Aggregate Sequence Labels to Trial Labels.
                fin_tp = tp_pres
            else:
                fin_tp = np.append(fin_tp, tp_pres, axis=1)
            if verbose == 1:
                print('Sequence {}: '.format(j + 1), tp_pres.shape, ' \n', tp_pres)
    if verbose == 1:
        print('Trial {}: '.format(t + 1), fin_tp.shape, ' \n', fin_tp)
        # Aggreaate into 1D tp_labels array across the Seesion.
        print('Temporal `Labels 1D Array: ', tp_labels.shape, ' \n', tp_labels)
    return tp_labels


def interp2D(data, timestamps, output_size, plotter, verbose):
    # Resamples 2D data matrices of Samples x Channels via interpolation to produce uniform output matrices of output size x channels.

    # Calcualte number of chans.
    a, b = np.shape(data)
    num_chans = np.minimum(a, b)
    # Gen place-holder for resampled data.
    r_data = np.zeros((output_size, num_chans))
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
            plt.title('Orignal Signal With Inconsistent Timestamps.')
            # Sub-Plot Resampled Channel Chk
            plt.subplot(2, 1, 2)
            plt.plot(r_time, yinterp)
            plt.title('Signal Post-Interpolation Method Resampling.')
            plt.show()
        if verbose == 1:
            print('Original Chk DIMS: ', data[:, k].shape,
                  'Resampled Chk Dims: ', yinterp.shape)

    return r_data, r_time


def is_odd(num):
    return num % 2


def ranger(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)


def sess_plot(data, label, ses_tag, num_trl_per_sess):
    # Plot all P3s and NP3s per session.
    # Assumes Sampes, Channels, Trials.
    time = np.arange(0, 500, 2)
    u, p3_ind = np.unique(ses_tag, return_index=True)
    num_trials = data.shape[0]
    p3_ind = np.append(p3_ind, [p3_ind[-1] + num_trl_per_sess])
    np3_ind = p3_ind + np.int(num_trials / 2)
    print('UNQ: ', u.shape, u)
    print('P3 Index: ', p3_ind.shape, p3_ind)
    print('NP3 Index: ', np3_ind.shape, np3_ind)
    for i in range(len(u)):
        plt_p3 = np.average(np.squeeze(data[p3_ind[i]:p3_ind[i + 1], :, :]), axis=0)
        plt_np3 = np.average(np.squeeze(data[np3_ind[i]:np3_ind[i + 1], :, :]), axis=0)
        print('Session {0} | P3 mV Range: {1} / NP3 mV Range: {2}'.format(i +
                                                                          1, ranger(plt_p3), ranger(plt_np3)))
        # Plot Legends.
        p3_p, = plt.plot(time, plt_p3, label='P3')
        np3_p, = plt.plot(time, plt_np3, label='NP3')
        plt.title('Session: {} Signal Averages '.format(i + 1))
        plt.legend([p3_p, np3_p], ['P3', 'NP3'])
        plt.show()
    return u, p3_ind, np3_ind


def rand_data(data, num_trls, num_samps, num_chans):
    # Generate purely randomised data.
    data_ = np.random.rand(num_trls, num_samps, num_chans)
    return data_


def uni_100(data, label, num_trls, num_samps, num_chans, n_con, zer_m, plotter):
    data_ = np.zeros((num_trls, num_samps, num_chans))
    print('Num Samps: ', num_samps, 'Noise: ', n_con, 'Data DIMS: ', data.shape)
    # Generate purely uniform data.
    for i in range(num_trls):
        if label[i] == 0:
            # Create sine wave + add noise.
            window = signal.cosine(num_samps)
            noise = np.random.uniform(0, n_con, num_samps)
            if n_con > 0:
                waveform = window+noise
                waveform = np.expand_dims(waveform, axis=-1)
            elif n_con == 0:
                waveform = window
                waveform = np.expand_dims(waveform, axis=-1)
            data_[i, :, :] = waveform
            if plotter == 1:
                # Plot differences between original signal ('window'), generated noise and the combined waveform.
                win_p, = plt.plot(window, label='Window')
                noise_p, = plt.plot(noise, label='Noise')
                wave_p, = plt.plot(waveform, label='Waveform')
                plt.legend([win_p, noise_p, wave_p], ['Window', 'Noise', 'Waveform'])
                plt.title('Comparing Raw Signal, Noise and Waveform - A Curve')
                plt.show()
        elif label[i] == 1:
            # Create flat signal at zero + add noise.
            window = np.ones(num_samps)
            noise = np.random.uniform(0, n_con, num_samps)
            if n_con > 0:
                waveform = window+noise
                waveform = np.expand_dims(waveform, axis=-1)
            elif n_con == 0:
                waveform = window
                waveform = np.expand_dims(waveform, axis=-1)
            data_[i, :, :] = waveform
            if plotter == 2:
                # Plot differences between original signal ('window'), generated noise and the combined waveform.
                win_p, = plt.plot(window, label='Window')
                noise_p, = plt.plot(noise, label='Noise')
                wave_p, = plt.plot(waveform, label='Waveform')
                plt.legend([win_p, noise_p, wave_p], ['Window', 'Noise', 'Waveform'])
                plt.title('Comparing Raw Signal, Noise and Waveform - B Flat')
                plt.show()
    if zer_m == 1:
        data_ = zero_mean(np.squeeze(data_))
        data_ = np.expand_dims(data_, axis=2)
    if plotter == 1:
        # eeg_series = temp_axis(data, 500)
        raw, = plt.plot(data[400, :, :], label='Raw')
        noised, = plt.plot(data_[400, :, :], label='Noised')
        plt.legend([raw, noised], ['Raw', 'Noised'])
        plt.title('Comparing Raw Signal and Noised Waveform')
        plt.show()
    return data_


def net_sets_parser(data, label, train_per, val_per, test_per):

    # Add new singleton dimesion.
    # Input Dims: Trials, Samples, Channels.
    # Expects: Trials, Singleton, Channels, Samples.
    data = np.swapaxes(np.copy(data), 1, 2)
    data = np.expand_dims(np.copy(data), axis=1)
    print('HERE DATA DIMS: ', data.shape)

    total = np.shape(data)[0]
    tr_dv = np.int(total*train_per)
    vl_dv = np.int(tr_dv + (total*val_per))
    te_dv = np.int(vl_dv + (total*test_per))

    'Train'
    X_train = data[0:tr_dv, :, :, :]
    X_train = X_train.astype('float32')
    print('X_train Dims: ', np.shape(X_train))
    y_train = label[0:tr_dv]
    y_train = y_train.astype('float32')
    print('y_train Dims: ', np.shape(y_train))

    'Val'
    X_val = data[tr_dv:vl_dv, :, :, :]
    X_val = X_val.astype('float32')
    print('X_val Dims: ', np.shape(X_val))
    y_val = label[tr_dv:vl_dv]
    y_val = y_val.astype('float32')
    print('y_val Dims: ', np.shape(y_val))

    'Test'
    X_test = data[vl_dv:te_dv, :, :, :]
    X_test = X_test.astype('float32')
    print('X_test Dims: ', np.shape(X_test))
    y_test = label[vl_dv:te_dv]
    y_test = y_test.astype('float32')
    print('y_test Dims: ', np.shape(y_test))

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepro(eeg, samp_ratekHz, zero, ext, elec, ref_ind, ref, filtH, hlevel, filtL, llevel, notc, notfq, avg):
    '---------------------------------------------------'
    'ZERO: mornalize data with zero meaning.'
    if zero == 'ON':
        eeg = zero_mean(np.copy(eeg))
    '---------------------------------------------------'
    'EXTRACT: relevant electrodes:  0) Fz, 1) Cz, 2) Pz, 3) P4, 4) P3, 5) O1, 6) O2, 7) A2.'
    'Reference: Guger (2012): Dry vs Wet Electrodes | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3345570/'
    # Ensure reference electrode A2 is extracted by calculating the position in the array.
    # A2 ref_ind always -7, as there are 6 variables at end: ACC8, ACC9, ACC10, Packet, Trigger & Time-Stamps.
    if ext == 'INC-Ref':
        all_eeg = eeg[:, [0, 1, 2, 3, 4, 5, 6]]
        grab = np.append(elec, ref_ind)
        eeg = np.squeeze(eeg[:, [grab]])
    if ext == 'NO-Ref':
        all_eeg = eeg[:, [0, 1, 2, 3, 4, 5, 6]]
        eeg = np.squeeze(eeg[:, [elec]])
    '------------------------------------------------------'
    'REFERENCING: using REF electrode.'
    if ref == 'A2':
        eeg = referencer(np.copy(eeg), -1)
    elif ref == 'AVG':
        eeg = avg_referencer(np.copy(eeg), all_eeg)
    '---------------------------------------------------'
    'FILTERING: highpass filter.'
    if filtH == 'ON':
        eeg = butter_highpass_filter(np.copy(eeg), hlevel, 500, order=5)
    '---------------------------------------------------'
    'FILTERING: lowpass filter.'
    if filtL == 'ON':
        eeg = butter_lowpass_filter(np.copy(eeg), llevel, 500, order=5)
    '----------------------------------------------------'
    'FILTERING: 50Hz notch filter.'
    if notc == 'NIK':
        samp_rateHz = samp_ratekHz * 1000
        eeg = notchyNik(np.copy(eeg), Fs=samp_rateHz, freq=notfq)
    elif notc == 'NOC':
        eeg = notchy(eeg, 500, notfq)
    elif notc == 'LOW':
        eeg = butter_lowpass_filter(np.copy(eeg), notfq, 500, order=5)
    '---------------------------------------------------'
    'AVERAGING: cross-channels.'
    if avg == 'ON':
        eeg = np.average(np.copy(eeg), axis=1)
        eeg = np.expand_dims(eeg, axis=1)
    '---------------------------------------------------'
    return eeg


def notchy(data, Fs, freq):
    'Notch Filter at 50Hz using the IIR: forward-backward filtering (via filtfilt)'
    'Requies Channel x Samples orientation.'

    'Example:'
    # grnd_data = notchyNik(a2_data, Fs=250)

    import mne
    data = np.swapaxes(data, 0, 1)
    filt_data = mne.filter.notch_filter(
        data, Fs=Fs, freqs=freq, method='iir', verbose=False, picks=None)
    print('1st Data Value: ', data[0, 0], '1st Grounded Value: ', filt_data[0, 0])
    filt_data = np.swapaxes(filt_data, 0, 1)
    return filt_data


def notchyNik(data, Fs, freq):
    from scipy import signal
    fs = Fs
    Q = 30.0  # Quality factor
    w0 = freq/(fs/2)  # Normalized Frequency
    # Design notch filter
    b, a = signal.iirnotch(w0, Q)

    chans = np.amin(np.shape(data))
    for i in range(chans):
        input_data = signal.filtfilt(b, a, np.copy(data[:, i]))
        data[:, i] = input_data
    return data


def freqy(data, fs):
    'FFT of 1st channel in eeg_data.'
    from scipy import fftpack
    x = data[:, 0]
    fft_data = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x)) * fs
    return fft_data, freqs


def sess_tagger(sub_path, i):
    sess_ = sub_path[i]
    sess_ = np.copy(sess_[-3])
    sess_tag = sess_.astype(int)
    return sess_tag


def sub_tagger(sub_path, i):
    sub_ = sub_path[i]
    sub_ = np.copy(sub_[25:30])
    # sub_tag = sub_.astype(np.int)
    sub_tag = sub_
    return sub_tag


def sub_tag_2(sub_path, i):
    # Provides subject tagging.
    sub_tag = sub_tagger(sub_path, i)
    sub = np.array2string(sub_tag)
    # Isolate numerical elements.
    sub = sub[1:6]
    # Preform of subject indicator result.
    res = []
    for i in range(len(sub)):
        # If the LEADING element is '0' we want to skip that.
        if sub[i] == '0':
            # If result is empty, keep it empty.
            if res == []:
                res = []
        else:
            # Values < 10.
            # Once element changes from '0' we grab that as the sub number.
            if i == len(sub)-1:
                if res == []:
                    # If the leading values is NOT '0' AND our res preform has NOT been filled, then append.
                    res = sub[i]
                    res = np.copy(res)
                    print(res, type(res))
            # Vales => 10.
            # Once element changes from '0' we grab that and the rest of the values.
            if i == len(sub)-2:
                if res == []:
                    res = sub[i:]
                    res = np.copy(res)
                    print(res, type(res))
            # Vales => 100.
            # Once element changes from '0' we grab that and the rest of the values.
            if i == len(sub)-3:
                if res == []:
                    res = sub[i:]
                    res = np.copy(res)
                    print(res, type(res))
            # Vales => 1000.
            # Once element changes from '0' we grab that and the rest of the values.
            if i == len(sub)-4:
                if res == []:
                    res = sub[i:]
                    res = np.copy(res)
                    print(res, type(res))
    return res


def expSubSessParser(data, label, all, exp, sub, seord, num_trls, exp_q, sub_q, seord_q):

    data_ = []
    label_ = []

    if all == 1:
        exp_q = np.unique(exp)
        sub_q = np.unique(sub)
        seord_q = np.unique(seord)
        data_ = data
        label_ = label
        exp_ = exp
        seord_ = seord
        sub_ = sub
    elif all == 0:
        for p in range(num_trls):
            if exp[p] in exp_q and seord[p] in seord_q and sub[p] in sub_q:
                if p == 0:
                    data_ = data[p, :, :]
                    data_ = np.expand_dims(data_, axis=0)
                    label_ = label[p]
                    exp_ = exp[p]
                    seord_ = seord[p]
                    sub_ = sub[p]
                elif p != 0:
                    if data_ == []:
                        data_ = data[p, :, :]
                        data_ = np.expand_dims(data_, axis=0)
                        label_ = label[p]
                        exp_ = exp[p]
                        seord_ = seord[p]
                        sub_ = sub[p]
                    else:
                        data_ = np.append(data_, np.expand_dims(data[p, :, :], axis=0), axis=0)
                        label_ = np.append(label_, label[p])
                        exp_ = np.append(exp_, exp[p])
                        seord_ = np.append(seord_, seord[p])
                        sub_ = np.append(sub_, sub[p])
    return data_, label_, exp_, seord_, sub_


def basics(time, signal):
    maxi = np.max(signal)
    mini = np.min(signal)
    return[maxi, mini]


def butter_bandpass(lowcut, highcut, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import lfilter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def passer(signal, low_cut, high_cut, fs):
    'Finally, what is the best filter for P300 detection? Laurent Bougrain, Carolina Saavedra, Radu Ranta'
    'https://hal.inria.fr/hal-00756669/document'

    'Example: '
    # x = pb.temp_axis(eeg_avg, 0.5)
    # band_data = np.squeeze(passer(x, grnd_data, 1, 40))
    num_samps, num_chans = np.shape(signal)
    band_data = np.zeros((num_samps, num_chans))

    for i in range(num_chans):
        band_data[:, i] = butter_bandpass_filter(signal[:, i], low_cut, high_cut, fs, order=5)
    return band_data


def butter_lowpass(cutoff, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    'Example: y = butter_lowpass_filter(data, cutoff, fs, order)'
    'Band-pass filter assumes sharper transitions from the pass band to the stop band as the order of the filter increases'
    from scipy.signal import lfilter

    # print('Iterator: ', (np.amin(np.shape(data))))

    for i in range(np.amin(np.shape(data))):
        low_data = data[:, i]
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, low_data)
        data[:, i] = y
    return data


def butter_highpass(cutoff, fs, order=5):
    from scipy.signal import butter
    'Band-pass filter assumes sharper transitions from the pass band to the stop band as the order of the filter increases.'
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    from scipy.signal import lfilter

    # print('Iterator: ', (np.amin(np.shape(data))))

    if data.ndim == 1:
        b, a = butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        data = y
    elif data.ndim == 2:
        for i in range(np.amin(np.shape(data))):
            # print('---------------Iterator: ', (np.amin(np.shape(data))))
            # print('---------------Data DIMS: ', np.shape(data))
            high_data = data[:, i]
            b, a = butter_highpass(cutoff, fs, order=order)
            y = lfilter(b, a, high_data)
            data[:, i] = y
    return data


def low_pass_grnd(data, Fs):
    from scipy.signal import freqs

    'Example: '
    # grnd_data = low_pass_grnd(a2_data, 250)

    # Filter requirements.
    order = 8
    fs = Fs      # sample rate, Hz
    cutoff = 50  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.
    w, h = freqs(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    # Generate time-stamp values.
    T = 5.0        # seconds
    n = int(T * fs)  # total number of samples
    t = np.linspace(0, T, n, endpoint=False)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cutoff, fs, order)
    print('1st Data Value: ', data[0, 0], '1st Grounded Value: ', y[0, 0])
    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    plt.subplots_adjust(hspace=0.35)
    # plt.show()
    return y


def referencer(data, grndIdx):
    'Substraction referencing method, commonly use A2 for P300 pre-processing.'
    'Data = Raw EEG time-series.'
    'grndIdx = Channel Index in data tensor of ground electrode.'
    'Requires a 2D tensor for use.'
    'Assumes data in Samples x Channels orientation.'

    'Example:'
    # 'Referencing to A2.'  # Try referncing at A2 aswell.
    # ref_data = referencer(data, 9)

    # print(grndIdx)
    samp, chan = np.shape(data)
    # print(samp, chan)
    # Isolate Ground Channel
    grnd = data[:, grndIdx]
    # Remove Ground Channel from main data.
    data = np.delete(data, grndIdx, axis=1)
    # print('Data Post Delete Dims: ', np.shape(data))
    # Pre-assign Ground Data Array.
    grnd_data = np.zeros((samp, chan - 1))
    for i in range(chan - 1):
        grnd_data[:, i] = data[:, i] - grnd
    return grnd_data


def avg_referencer(eeg, all_eeg):
    'Average referencing across all electrodes.'
    'ext = Relevant EEG electrodes.'
    'all_eeg = All electrodes in relevant montage sampled from: 0) Fz, 1) Cz, 2) Pz, 3) P4, 4) P3, 5) O1, 6) O2, 7) A2.'
    'Requires a 2D tensor for use.'
    'Assumes data in Samples x Channels orientation.'

    'Example:'
    #  eeg = avg_referencer(np.copy(eeg), all_eeg)

    print('EEG Pre AVG REF DIMS: ', eeg.shape)

    # 1st Remove the A2 electrode by deleting the final column of data.
    eeg = np.delete(eeg, (-1), axis=1)
    dims = eeg.shape[1]

    # Generate average reference from eeg montage.
    avg_ref = np.average(all_eeg, axis=1)
    print('AVG REF DIMS: ', avg_ref.shape)

    # Subtract average reference from each eeg channel.
    if dims >= 1:
        for i in range(dims):
            chan_data = eeg[:, i]
            chan_data = chan_data - avg_ref
            # chan_data = np.expand_dims(chan_data, axis=1)
            eeg[:, i] = chan_data
    else:
        eeg = eeg - avg_ref

    # Data Dimesions Check:
    print('Post AVG REF DIMS: ', eeg.shape)
    return eeg


def subplot_dual(dataA, dataB, type, title):
    'Subplotting of 2 data arrays.'
    'Both must use same orientation.'
    'Type: Referenced data plot = 0, 7 Chan Plot plot = 1, 2 Chan Occ Plot = 2.'
    'Title: Name OG Data vs Transformed data.'

    'Example:'
    # subplot_dual(data, ref_data, 0, 'Raw vs Referenced Data')

    f, axarr = plt.subplots(2, 1)
    if type == 0:
        labelsA = ('Fp1', 'F3', 'Fz', 'C3', 'Cz', 'Pz', 'P4', 'P3', 'C4', 'A2')
    elif type == 1:
        labelsA = ('F3', 'Fz', 'C3', 'Cz', 'Pz', 'P4', 'P3', 'C4', 'A2')
    linesA = axarr[0].plot(dataA)
    axarr[0].legend(linesA, labelsA)
    labelsB = ('F3', 'Fz', 'C3', 'Cz', 'Pz', 'P4', 'P3', 'C4', 'A2')
    linesB = axarr[1].plot(dataB)
    axarr[1].legend(linesB, labelsB)
    f.suptitle(title, fontsize=16)


def subplotter(dataA, dataB, time_axis, title):
    'Subplotting of 2 data arrays.'
    'Both must use same orientation.'
    'Type: Referenced data plot = 0, 7 Chan Plot plot = 1, 2 Chan Occ Plot = 2.'
    'Title: Name OG Data vs Transformed data.'

    'Example:'
    # subplotterl(data, ref_data, 'Raw vs Referenced Data')

    f, axarr = plt.subplots(2, 1)
    axarr[0].plot(time_axis, dataA)
    axarr[1].plot(time_axis, dataB)
    f.suptitle(title, fontsize=16)
    plt.show()


def zero_mean(data):
    'Zeros Data, accepts orientation Samples x Channels.'
    a, b = np.shape(data)
    # Preform zero array.
    zero_data = np.zeros((a, b))
    for i in range(b):
        zero_data[:, i] = data[:, i] - np.mean(data[:, i])
    return zero_data


def zerodat(data):
    'Zeros Data, accepts orientation Samples x Channels.'
    'Zero data using 1st value channel sample subtraction'
    'Therefore, signal begins at zero baseline.'

    'Exmaple: '
    # zero_data = zerodat(band_data)

    a, b = np.shape(data)
    # Preform zero array.
    zero_data = np.zeros((a, b))
    for i in range(b):
        zero_data[:, i] = data[:, i] - data[0, i]
    return zero_data


def zero_std(data):
    'Divide by Standard Deviation from each channel.'

    a, b = np.shape(data)
    # Preform zero array.
    zero_std = np.zeros((a, b))
    for i in range(b):
        # Get Channel std.
        std_x = data[:, i].std()
        sub_std = np.zeros(a)+std_x
        zero_std[:, i] = data[:, i]/sub_std

    return zero_std


def scale(x, out_range=(0, 30)):
    # Conversion to mV range.
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def scaleR(X, x_min, x_max):
    # https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return x_min + nom/denom


def scale_range(input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


def scaler2D(data):
    from sklearn.preprocessing import MinMaxScaler
    # Grab data dims: Trials x Samples x Channels
    tr, sm, ch = np.shape(data)

    for j in range(tr):
        for i in range(ch):
            # Extract channel Data.
            chan_data = data[j, :, i]
            # Format with additional singleton dimension.
            chan_data = np.expand_dims(chan_data, axis=1)
            # Initialize the scaler function.
            scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
            # Fit the data.
            scaler.fit(chan_data)
            # Transform the data.
            clean_data = scaler.transform(chan_data)
            # Re-format for appending.
            clean_data = np.squeeze(clean_data)
            data[j, :, i] = clean_data
    return data


def scaler1D(data, low, high):
    from sklearn.preprocessing import MinMaxScaler
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(copy=True, feature_range=(low, high))
    scaler.fit(data)
    data = scaler.transform(data)
    return data


def power(stop_sig, non_stop_sig, plot, title):
    'Example: '
    # freqs, Pxx_spec = power(avg_data, 0, 'Sing Trial')
    # print(freqs[17:24])  # For relevant frequenies extraction.
    # print(Pxx_spec[17:24])  # For relevant psd value extraction.

    from scipy import signal
    # import matplotlib.mlab as mlab
    'Stop Sig'
    freqs, psd = signal.welch(stop_sig)
    f, Pxx_spec = signal.periodogram(stop_sig, 250, 'flattop', scaling='spectrum')
    dt = 0.004  # Because 1 / 0.004 = 250
    Pxx, freqs = plt.psd(stop_sig, 256, 1 / dt, label='Stop Signal PSD')
    'Non-Stop Sig'
    freqs, psd = signal.welch(non_stop_sig)
    f, Pxx_spec = signal.periodogram(non_stop_sig, 250, 'flattop', scaling='spectrum')
    dt = 0.004  # Because 1 / 0.004 = 250
    Pxx, freqs = plt.psd(non_stop_sig, 256, 1 / dt, label='Non-Stop Signal PSD')
    'Plot Formatting'
    plt.legend()
    plt.xlim([12, 16])
    plt.ylim([-40, -5])
    plt.yticks(np.arange(-45, -5, step=10))
    plt.title(title)
    plt.grid(b=None)
    plt.show()
    return f, Pxx_spec


def sing_power(sig, plot, title):
    from scipy import signal
    # import matplotlib.mlab as mlab
    freqs, psd = signal.welch(sig)
    f, Pxx_spec = signal.periodogram(sig, 250, 'flattop', scaling='spectrum')
    if plot == 1:
        'Method 2'
        dt = 0.004  # Because 1 / 0.004 = 250
        Pxx, freqs = plt.psd(sig, 256, 1 / dt)
        plt.xlim([12, 16])
        title2 = title
        plt.title(title2)
        # plt.show()
        # Pxx, freqs = plt.psd(s, 512, 1 / dt)
    return f, Pxx_spec


def nancheck(data):
    'Check all channels for nan values.'
    'If even 1 nan found in an array, change all other values to nan.'
    'Assumes orientation Samples x Channels.'
    a, b = np.shape(data)
    # Preform zero array.
    nan_data = np.zeros((a, b))
    for i in range(b):
        if np.isnan(data[:, i]).any() is True:
            nan_data[:, i] = np.nan
            print('-------------------------NAN CHAN')
        if np.isnan(data[:, i]).any() is False:
            nan_data[:, i] = data[:, i]
            print('-------------------------NORM CHAN')
    return nan_data


def sing_data_extract(direc, ext, keyword, arr):
    'direc = get data directory.'
    'ext = select your file delimiter.'
    'keyword = unique filenaming word/phrase.'
    'arr = cycle in trial you want to extract.'

    'Example: '
    # eeg = pb.sing_data_extract('C:\P300_Project\Data_Aquisition\Data\\', '.npz', 'volt', 'arr_0')

    eegfiles = [i for i in os.listdir(direc) if os.path.splitext(i)[1] == ext]
    eeg_files = []
    for j in range(len(eegfiles)):
        if eegfiles[j].find(keyword) != -1:
            eeg_files.append(eegfiles[j])

    file_name = eeg_files[0]
    data = np.load(direc + file_name)
    data = data[arr]
    print('Extracted File: ', file_name, 'EEG Dims: ', np.shape(data))
    return data


def labels_inf_extract(direc, file_name):
    'direc - directory where label data is stored.'
    'file_name = name of labels file you want to extract info from.'

    'Example: '
    # matSize, num_seq, num_emojis, num_trials = pb.labels_inf_extract(
    #     'C:\P300_Project\Data_Aquisition\Data\Labels\\', '0001_trial_labels.npz')

    xA = np.load(direc + file_name)
    x1 = xA['arr_0']
    matSize = np.shape(x1)
    num_seq = matSize[0]
    num_emojis = matSize[1]
    num_trials = len(xA['arr_1'])
    return xA, matSize, num_seq, num_emojis, num_trials


def pathway_extract(direc, ext, keyword, full_ext):
    'direc = get data directory.'
    'ext = select your file delimiter.'
    'keyword = unique filenaming word/phrase.'

    'Example: '
    # eeg = pb.sing_data_extract('C:\P300_Project\Data_Aquisition\Data\\', '.npz', 'volt')

    files = [i for i in os.listdir(direc) if os.path.splitext(i)[1] == ext]
    _files = []
    for j in range(len(files)):
        if files[j].find(keyword) != -1:
            if full_ext != 1:
                _files.append(files[j])
            if full_ext == 1:
                _files.append(direc + files[j])
    return _files


def subject_extract(direc):
    'direc = get data directory with all subject data.'
    # sub_files = os.listdir(direc)
    sub_files = [i for i in os.listdir(direc) if i.find('0') == 0]
    eeg_files = []
    lab_files = []
    for i in range(len(sub_files)):
        eeg_files.append(direc + sub_files[i])
        lab_files.append(direc + sub_files[i] + '/Labels/')
    return eeg_files, lab_files


def path_subplant(main_direc, paths, lab):
    _files = []
    for i in range(len(paths)):
        if lab == 0:
            _files.append(main_direc + '/' + paths[i])
        elif lab == 1:
            _files.append(main_direc + '/Labels/' + paths[i])

    return _files


def time_stamper(data, show_stamps):
    # Isolate final channel from Cognionics array containing time-stamps.
    # ALso subtract 1st value in this array from all subsequent values.
    # This transforms data from system time to seconds units.
    'Example: '
    # eeg_time = pb.time_stamper(eeg, 1)

    eeg_time = np.abs(data[0, -1] - data[:, -1]) * 1000
    if show_stamps == 1:
        print('1st Time-Stamp: ', eeg_time[0], 'Last Time-Stamp: ', eeg_time[-1])
    return eeg_time


def simp_temp_axis(data, samp_ratekHz):
    'Generate an x axis for plotting time-series at diff Hz.'
    # data is just your eeg array.
    # samprate needs to be given in KHz e.g. 0.5 = 500Hz.
    # plot_secs is the number of seconds you want plotting per trial.
    'Example: '
    # x = pb.temp_axis(eeg_avg, 0.5)
    f = data.shape
    f = np.amax(f)

    constant = 1 / samp_ratekHz  # Temporal Constamt
    x = np.arange(0, (f * constant), constant)  # Temporal Axis

    return x


def temp_axis(data, samp_ratekHz, plt_secs):
    'Generate an x axis for plotting time-series at diff Hz.'
    # data is just your eeg array.
    # samprate needs to be given in KHz e.g. 0.5 = 500Hz.
    # plot_secs is the number of seconds you want plotting per trial.
    'Example: '
    # x = pb.temp_axis(eeg_avg, 0.5)
    f = data.shape
    f = np.amax(f)

    constant = 1 / samp_ratekHz  # Temporal Constamt
    x = np.arange(0, (f * constant), constant)  # Temporal Axis
    # Grab time-series to length of plot_secs.
    time_idx = [i for i, e in enumerate(x) if e == plt_secs]
    time_idx = time_idx[0]
    # Index array to size useing plot_secs.
    x = x[0:time_idx]

    return x, time_idx


def fiir_bandpass_link_250Hz(data):
    'https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter'
    'https://fiiir.com/'

    h = [
        0.000000000000000001,
        -0.000000000000000010,
        -0.000000000000007656,
        -0.000000000000052146,
        -0.000000000000123279,
        -0.000000000000048186,
        0.000000000000435947,
        0.000000000001263637,
        0.000000000001677191,
        0.000000000000602304,
        -0.000000000002196291,
        -0.000000000005280268,
        -0.000000000006356883,
        -0.000000000004611331,
        -0.000000000002397930,
        -0.000000000003951451,
        -0.000000000011400561,
        -0.000000000021663591,
        -0.000000000028076781,
        -0.000000000026514510,
        -0.000000000020632329,
        -0.000000000020218850,
        -0.000000000032492012,
        -0.000000000053814676,
        -0.000000000070801610,
        -0.000000000071967215,
        -0.000000000060110840,
        -0.000000000052585370,
        -0.000000000066482939,
        -0.000000000100942579,
        -0.000000000134344492,
        -0.000000000142291661,
        -0.000000000121912991,
        -0.000000000098718168,
        -0.000000000105739799,
        -0.000000000151261369,
        -0.000000000206013201,
        -0.000000000226224262,
        -0.000000000195192254,
        -0.000000000144329202,
        -0.000000000129078878,
        -0.000000000176840602,
        -0.000000000255001405,
        -0.000000000294308009,
        -0.000000000252419729,
        -0.000000000159841211,
        -0.000000000099679995,
        -0.000000000132032318,
        -0.000000000230751678,
        -0.000000000297396138,
        -0.000000000248960668,
        -0.000000000101961407,
        0.000000000031214409,
        0.000000000041488897,
        -0.000000000068348734,
        -0.000000000172142092,
        -0.000000000129445886,
        0.000000000077883550,
        0.000000000313283118,
        0.000000000401800900,
        0.000000000297980022,
        0.000000000145764690,
        0.000000000158857805,
        0.000000000418775842,
        0.000000000780025543,
        0.000000000987679169,
        0.000000000915350528,
        0.000000000703028872,
        0.000000000648568746,
        0.000000000932773396,
        0.000000001429901898,
        0.000000001798338457,
        0.000000001791954678,
        0.000000001510244573,
        0.000000001336151035,
        0.000000001590484558,
        0.000000002212065824,
        0.000000002778774155,
        0.000000002883035085,
        0.000000002530297335,
        0.000000002173790868,
        0.000000002315922046,
        0.000000003022810446,
        0.000000003816968868,
        0.000000004089400390,
        0.000000003679564723,
        0.000000003074177863,
        0.000000002994556403,
        0.000000003715915192,
        0.000000004755993782,
        0.000000005271603429,
        0.000000004844667625,
        0.000000003929634405,
        0.000000003494092808,
        0.000000004124923617,
        0.000000005418825818,
        0.000000006278387735,
        0.000000005913677151,
        0.000000004642903058,
        0.000000003692071981,
        0.000000004088878860,
        0.000000005637821329,
        0.000000006984502186,
        0.000000006819227767,
        0.000000005164643729,
        0.000000003498482640,
        0.000000126442836441,
        0.000000468242731209,
        0.000000007335172522,
        -0.000001879037080235,
        -0.000003185425486147,
        -0.000000640363278417,
        0.000005443871650191,
        0.000008878569011673,
        0.000002979650238087,
        -0.000010218051708852,
        -0.000017980047837430,
        -0.000008127521218717,
        0.000015443251898907,
        0.000030847514296493,
        0.000017350541320345,
        -0.000020018577791358,
        -0.000047559137090656,
        -0.000031904933064986,
        0.000022616087840912,
        0.000067991273029736,
        0.000053134020479693,
        -0.000021522706726257,
        -0.000091625607738989,
        -0.000082336469062071,
        0.000014593175799959,
        0.000117245076469952,
        0.000120199731678298,
        -0.000000000324076394,
        -0.000143649372003191,
        -0.000166940991539296,
        -0.000022131356571775,
        0.000175367817116744,
        0.000235960745058857,
        0.000077126068110387,
        -0.000173430921496987,
        -0.000272094946308719,
        -0.000087808105773726,
        0.000246577295959476,
        0.000426803789581711,
        0.000258173037228463,
        -0.000130874242031506,
        -0.000369236566089373,
        -0.000180469132587659,
        0.000325663686349011,
        0.000707875474170479,
        0.000588686011130428,
        0.000041038975337928,
        -0.000422905553372027,
        -0.000303773000936786,
        0.000385369997613234,
        0.001064766003776141,
        0.001091353342746040,
        0.000389579317676190,
        -0.000402468620153644,
        -0.000477665681491034,
        0.000360752571336783,
        0.001430217730293392,
        0.001738559066962817,
        0.000927602457911652,
        -0.000299699883893189,
        -0.000751155368315161,
        0.000134308048820305,
        0.001662024516121907,
        0.002424805627174197,
        0.001608102725161517,
        -0.000145152122717095,
        -0.001209526520347722,
        -0.000468929194003873,
        0.001533075855638499,
        0.002950973692505942,
        0.002309797547369736,
        -0.000012412063567699,
        -0.001965818714381166,
        -0.001666036793773981,
        0.000744672926751270,
        0.003032327847613439,
        0.002845467233111482,
        -0.000000000354906137,
        -0.003128601999524876,
        -0.003674632323672197,
        -0.001031774325169514,
        0.002335625146261115,
        0.002998671270252978,
        -0.000186498604855188,
        -0.004744970795803299,
        -0.006648460353736270,
        -0.004091336186048318,
        0.000538324655611015,
        0.002583801992120114,
        -0.000562909836432822,
        -0.006725990255440449,
        -0.010604816512212550,
        -0.008632202653646162,
        -0.002612788114599488,
        0.001513224603704024,
        -0.000951083864311337,
        -0.008764631047720775,
        -0.015366927689306980,
        -0.014731228533316893,
        -0.007284754739715795,
        -0.000151513983114585,
        -0.000906293386456860,
        -0.010239882028025005,
        -0.020545498369504013,
        -0.022426452112676666,
        -0.013660298187797012,
        -0.002185972935542158,
        0.000458513505119576,
        -0.010022079484244590,
        -0.025575339222655168,
        -0.032085490118841251,
        -0.022382510359233283,
        -0.004237830542351018,
        0.005077254666946797,
        -0.005707532253295625,
        -0.029807019478879530,
        -0.045931608234930192,
        -0.036553813006946378,
        -0.005905714733400702,
        0.019470530220708510,
        0.011456112030613240,
        -0.032634996341546735,
        -0.079954810231259027,
        -0.081039147573575582,
        -0.006840855726484700,
        0.123652513967809186,
        0.248416362646564082,
        0.299704039459438742,
        0.248416362646564914,
        0.123652513967808797,
        -0.006840855726484785,
        -0.081039147573575554,
        -0.079954810231259388,
        -0.032634996341546776,
        0.011456112030613114,
        0.019470530220708489,
        -0.005905714733400656,
        -0.036553813006946316,
        -0.045931608234929963,
        -0.029807019478879523,
        -0.005707532253295548,
        0.005077254666946800,
        -0.004237830542351079,
        -0.022382510359233297,
        -0.032085490118841320,
        -0.025575339222655147,
        -0.010022079484244601,
        0.000458513505119575,
        -0.002185972935542093,
        -0.013660298187797005,
        -0.022426452112676694,
        -0.020545498369504055,
        -0.010239882028025012,
        -0.000906293386456849,
        -0.000151513983114627,
        -0.007284754739715812,
        -0.014731228533317002,
        -0.015366927689307058,
        -0.008764631047720788,
        -0.000951083864311317,
        0.001513224603704072,
        -0.002612788114599489,
        -0.008632202653646141,
        -0.010604816512212571,
        -0.006725990255440488,
        -0.000562909836432827,
        0.002583801992120089,
        0.000538324655611013,
        -0.004091336186048329,
        -0.006648460353736270,
        -0.004744970795803280,
        -0.000186498604855191,
        0.002998671270253006,
        0.002335625146261119,
        -0.001031774325169512,
        -0.003674632323672200,
        -0.003128601999524877,
        -0.000000000354906118,
        0.002845467233111471,
        0.003032327847613439,
        0.000744672926751266,
        -0.001666036793773990,
        -0.001965818714381154,
        -0.000012412063567690,
        0.002309797547369753,
        0.002950973692505948,
        0.001533075855638491,
        -0.000468929194003883,
        -0.001209526520347749,
        -0.000145152122717108,
        0.001608102725161536,
        0.002424805627174214,
        0.001662024516121932,
        0.000134308048820313,
        -0.000751155368315145,
        -0.000299699883893195,
        0.000927602457911646,
        0.001738559066962815,
        0.001430217730293389,
        0.000360752571336784,
        -0.000477665681491039,
        -0.000402468620153641,
        0.000389579317676192,
        0.001091353342746036,
        0.001064766003776136,
        0.000385369997613233,
        -0.000303773000936788,
        -0.000422905553372031,
        0.000041038975337921,
        0.000588686011130421,
        0.000707875474170473,
        0.000325663686349006,
        -0.000180469132587667,
        -0.000369236566089369,
        -0.000130874242031495,
        0.000258173037228473,
        0.000426803789581715,
        0.000246577295959485,
        -0.000087808105773717,
        -0.000272094946308712,
        -0.000173430921496995,
        0.000077126068110384,
        0.000235960745058846,
        0.000175367817116743,
        -0.000022131356571755,
        -0.000166940991539283,
        -0.000143649372003177,
        -0.000000000324076395,
        0.000120199731678290,
        0.000117245076469949,
        0.000014593175799965,
        -0.000082336469062060,
        -0.000091625607738980,
        -0.000021522706726263,
        0.000053134020479699,
        0.000067991273029734,
        0.000022616087840911,
        -0.000031904933064978,
        -0.000047559137090657,
        -0.000020018577791359,
        0.000017350541320346,
        0.000030847514296490,
        0.000015443251898909,
        -0.000008127521218713,
        -0.000017980047837418,
        -0.000010218051708841,
        0.000002979650238091,
        0.000008878569011670,
        0.000005443871650188,
        -0.000000640363278406,
        -0.000003185425486112,
        -0.000001879037080202,
        0.000000007335172519,
        0.000000468242731160,
        0.000000126442836345,
        0.000000003498482565,
        0.000000005164643672,
        0.000000006819227735,
        0.000000006984502188,
        0.000000005637821360,
        0.000000004088878879,
        0.000000003692071992,
        0.000000004642903059,
        0.000000005913677135,
        0.000000006278387737,
        0.000000005418825832,
        0.000000004124923633,
        0.000000003494092819,
        0.000000003929634402,
        0.000000004844667619,
        0.000000005271603429,
        0.000000004755993793,
        0.000000003715915203,
        0.000000002994556411,
        0.000000003074177872,
        0.000000003679564720,
        0.000000004089400384,
        0.000000003816968872,
        0.000000003022810446,
        0.000000002315922048,
        0.000000002173790863,
        0.000000002530297335,
        0.000000002883035092,
        0.000000002778774156,
        0.000000002212065817,
        0.000000001590484557,
        0.000000001336151034,
        0.000000001510244579,
        0.000000001791954693,
        0.000000001798338459,
        0.000000001429901901,
        0.000000000932773397,
        0.000000000648568739,
        0.000000000703028869,
        0.000000000915350538,
        0.000000000987679172,
        0.000000000780025548,
        0.000000000418775851,
        0.000000000158857816,
        0.000000000145764692,
        0.000000000297980021,
        0.000000000401800894,
        0.000000000313283104,
        0.000000000077883543,
        -0.000000000129445883,
        -0.000000000172142095,
        -0.000000000068348732,
        0.000000000041488893,
        0.000000000031214404,
        -0.000000000101961407,
        -0.000000000248960666,
        -0.000000000297396142,
        -0.000000000230751681,
        -0.000000000132032310,
        -0.000000000099679987,
        -0.000000000159841211,
        -0.000000000252419732,
        -0.000000000294308021,
        -0.000000000255001416,
        -0.000000000176840601,
        -0.000000000129078874,
        -0.000000000144329198,
        -0.000000000195192251,
        -0.000000000226224262,
        -0.000000000206013203,
        -0.000000000151261366,
        -0.000000000105739794,
        -0.000000000098718164,
        -0.000000000121913000,
        -0.000000000142291665,
        -0.000000000134344487,
        -0.000000000100942576,
        -0.000000000066482941,
        -0.000000000052585366,
        -0.000000000060110840,
        -0.000000000071967220,
        -0.000000000070801617,
        -0.000000000053814669,
        -0.000000000032491999,
        -0.000000000020218851,
        -0.000000000020632348,
        -0.000000000026514522,
        -0.000000000028076787,
        -0.000000000021663593,
        -0.000000000011400565,
        -0.000000000003951452,
        -0.000000000002397943,
        -0.000000000004611353,
        -0.000000000006356911,
        -0.000000000005280280,
        -0.000000000002196281,
        0.000000000000602299,
        0.000000000001677172,
        0.000000000001263602,
        0.000000000000435917,
        -0.000000000000048205,
        -0.000000000000123282,
        -0.000000000000052144,
        -0.000000000000007677,
        -0.000000000000000054,
        -0.000000000000000047,
    ]
    a, b = np.shape(data)
    x = np.zeros(a)
    x[0: len(h)] = h

    filt_sigs = np.zeros((a, b))
    for i in range(b):
        filt_chan = np.convolve(data[:, i], x, mode='same')
        filt_sigs[:, i] = filt_chan
    return(filt_sigs)


def fiir_bandpass_link_500Hz_2D(data):
    'https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter'
    'https://fiiir.com/'

    # 5 - 40 Hz.
    # Cut off Low = 5, Transition = 5.
    # Cut off High = 40, Transition = 5.

    h = [
        -0.000000000000000005,
        0.000000000000000004,
        -0.000000000000000489,
        -0.000000000000004065,
        -0.000000000000017120,
        -0.000000000000048299,
        -0.000000000000102445,
        -0.000000000000170542,
        -0.000000000000219925,
        -0.000000000000190101,
        -0.000000000000000002,
        0.000000000000431318,
        0.000000000001152905,
        0.000000000002147840,
        0.000000000003309031,
        0.000000000004437630,
        0.000000000005271984,
        0.000000000005547255,
        0.000000000005074080,
        0.000000000003816250,
        0.000000000001941506,
        -0.000000000000176784,
        -0.000000000002019903,
        -0.000000000003041201,
        -0.000000000002818024,
        -0.000000000001199444,
        0.000000000001594376,
        0.000000000004960260,
        0.000000000008006730,
        0.000000000009748046,
        0.000000000009358691,
        0.000000000006426739,
        0.000000000001133523,
        -0.000000000005701905,
        -0.000000000012747647,
        -0.000000000018443291,
        -0.000000000021388875,
        -0.000000000020745302,
        -0.000000000016538179,
        -0.000000000009767904,
        -0.000000000002270585,
        0.000000000003665553,
        0.000000000005854688,
        0.000000000002810537,
        -0.000000000005770822,
        -0.000000000018785833,
        -0.000000000033847847,
        -0.000000000047767674,
        -0.000000000057302812,
        -0.000000000060004342,
        -0.000000000054946620,
        -0.000000000043131802,
        -0.000000000027427289,
        -0.000000000012002720,
        -0.000000000001364686,
        0.000000000000796697,
        -0.000000000007337767,
        -0.000000000025048969,
        -0.000000000049014236,
        -0.000000000073919229,
        -0.000000000093651218,
        -0.000000000102813980,
        -0.000000000098196351,
        -0.000000000079803956,
        -0.000000000051143783,
        -0.000000000018619220,
        0.000000000009884697,
        0.000000000026927117,
        0.000000000027488394,
        0.000000000010486325,
        -0.000000000020604405,
        -0.000000000058266660,
        -0.000000000092550207,
        -0.000000000113446650,
        -0.000000000113474373,
        -0.000000000089817182,
        -0.000000000045424641,
        0.000000000011294948,
        0.000000000068219970,
        0.000000000112358411,
        0.000000000133135054,
        0.000000000125304248,
        0.000000000090704092,
        0.000000000038310138,
        -0.000000000017553045,
        -0.000000000060515377,
        -0.000000000076236942,
        -0.000000000056257048,
        -0.000000000000605539,
        0.000000000081588226,
        0.000000000173788243,
        0.000000000255772552,
        0.000000000308563012,
        0.000000000319376607,
        0.000000000285322045,
        0.000000000214802643,
        0.000000000126107332,
        0.000000000043363162,
        -0.000000000009285556,
        -0.000000000013890322,
        0.000000000036269383,
        0.000000000134207890,
        0.000000000260113205,
        0.000000000385576930,
        0.000000000480302332,
        0.000000000519725760,
        0.000000000491650991,
        0.000000000400130526,
        0.000000000265423924,
        0.000000000119799718,
        -0.000000000000000001,
        -0.000000000061904170,
        -0.000000000047019849,
        0.000000000044719059,
        0.000000000193382419,
        0.000000000362923957,
        0.000000000509418635,
        0.000000000591807385,
        0.000000000582545503,
        0.000000000475423166,
        0.000000000288390016,
        0.000000000060370339,
        -0.000000000157465207,
        -0.000000000314057722,
        -0.000000000371496333,
        -0.000000000315437099,
        -0.000000000160092714,
        0.000000000053790430,
        0.000000000267845340,
        0.000000000419997786,
        0.000000000460104378,
        0.000000000363418118,
        0.000000000138321956,
        -0.000000000173959391,
        -0.000000000508037236,
        -0.000000000789670166,
        -0.000000000954164223,
        -0.000000000963334831,
        -0.000000000816626673,
        -0.000000000553239702,
        -0.000000000244236157,
        0.000000000023896775,
        0.000000000170494580,
        0.000000000141614390,
        -0.000000000074759829,
        -0.000000000443381995,
        -0.000000000888160556,
        -0.000000001309312636,
        -0.000000001607452826,
        -0.000000001708845929,
        -0.000000001585455236,
        -0.000000001264408744,
        -0.000000000823920402,
        -0.000000000375996400,
        -0.000000000039638961,
        0.000000000089128802,
        -0.000000000037046010,
        -0.000000000402198873,
        -0.000000000927747464,
        -0.000000001489159498,
        -0.000000001945182016,
        -0.000000002172852123,
        -0.000000002099687412,
        -0.000000001724721891,
        -0.000000001122455769,
        -0.000000000427819414,
        0.000000000195020528,
        0.000000000592931773,
        0.000000000663403213,
        0.000000000383348396,
        -0.000000000180768435,
        -0.000000000885050579,
        -0.000000001540638734,
        -0.000000001959369482,
        -0.000000002001727725,
        -0.000000001615029280,
        -0.000000000851598979,
        0.000000000138769093,
        0.000000001141566119,
        0.000000001928608626,
        0.000000002315410592,
        0.000000002210662605,
        0.000000001644796564,
        0.000000000769184322,
        -0.000000000175515443,
        -0.000000000915607931,
        -0.000000001212356536,
        -0.000000000925134130,
        -0.000000000053390051,
        0.000000001254519819,
        0.000000002729918304,
        0.000000004044869323,
        0.000000004892242650,
        0.000000005065643591,
        0.000000004518877393,
        0.000000003388478105,
        0.000000001971090736,
        0.000000000658368511,
        -0.000000000157024060,
        -0.000000000182455473,
        0.000000000691764586,
        0.000000002350046555,
        0.000000004463606788,
        0.000000006558151781,
        0.000000008124401926,
        0.000000008746878978,
        0.000000008219760564,
        0.000000006619220935,
        0.000000004309943055,
        0.000000001877997392,
        -0.000000000000000037,
        -0.000000000724653091,
        0.000000000060208323,
        0.000000002345545786,
        0.000000005715080127,
        0.000000009403466382,
        0.000000012452729410,
        0.000000013940815615,
        0.000000013235277461,
        0.000000010212660532,
        0.000000005383656433,
        -0.000000000123281411,
        -0.000000004742965072,
        -0.000000006810645152,
        -0.000000005013831096,
        0.000000001110478489,
        0.000000010718911004,
        0.000000021332271817,
        0.000000028705690594,
        0.000000026941448600,
        0.000000008849356401,
        -0.000000012360763486,
        -0.000000014356226043,
        -0.000000042737486884,
        -0.000000198097768297,
        -0.000000588103458679,
        -0.000001260439328236,
        -0.000002146796703728,
        -0.000003047902131954,
        -0.000003676767442306,
        -0.000003755562154053,
        -0.000003138149589365,
        -0.000001914136494153,
        -0.000000448374840811,
        0.000000674981116457,
        0.000000805480645216,
        -0.000000556998756941,
        -0.000003544834962441,
        -0.000007779911735860,
        -0.000012369956958126,
        -0.000016073732205241,
        -0.000017614431357957,
        -0.000016068603891601,
        -0.000011220836255305,
        -0.000003766399101170,
        0.000004728075549763,
        0.000012135584992185,
        0.000016302524484133,
        0.000015680958717747,
        0.000009886349403680,
        0.000000003925916432,
        -0.000011484659442780,
        -0.000021189545218164,
        -0.000025634633911248,
        -0.000022224074836402,
        -0.000010096606723309,
        0.000009390873182328,
        0.000032737065393786,
        0.000054979463750557,
        0.000070873288085734,
        0.000076298762817461,
        0.000069532771525926,
        0.000052015273666842,
        0.000028338040501229,
        0.000005367498620546,
        -0.000009357104111413,
        -0.000009593328634651,
        0.000007764492828018,
        0.000041491076437899,
        0.000085876732213263,
        0.000131767497141473,
        0.000168642843286594,
        0.000187277126391620,
        0.000182326628685562,
        0.000154138134366355,
        0.000109215344971165,
        0.000059085895346754,
        0.000017717026535068,
        -0.000001968203571069,
        0.000008370913380602,
        0.000049838671658323,
        0.000115321739722915,
        0.000190595031051896,
        0.000257409996529763,
        0.000297923664611423,
        0.000299413678129213,
        0.000258056497595307,
        0.000180688685679935,
        0.000083913089146670,
        -0.000009439675744743,
        -0.000075783961378672,
        -0.000096905683079531,
        -0.000065184487486010,
        0.000013535430323051,
        0.000120370386572638,
        0.000227270977517335,
        0.000303602758444394,
        0.000323922575839815,
        0.000275007655274860,
        0.000160240446869099,
        -0.000000000000000011,
        -0.000172345874617771,
        -0.000318132534085260,
        -0.000403042394189642,
        -0.000406330120699630,
        -0.000327193614468050,
        -0.000186422817242246,
        -0.000022555226620225,
        0.000116856035155856,
        0.000186939009936746,
        0.000157333908326052,
        0.000021094171285294,
        -0.000201857718153825,
        -0.000467993024698925,
        -0.000719772891556236,
        -0.000899521280968133,
        -0.000964268794421358,
        -0.000897787503650017,
        -0.000716513408195216,
        -0.000467418958231290,
        -0.000217856589139019,
        -0.000039475188103017,
        0.000010017650619659,
        -0.000097326032373829,
        -0.000350536391126778,
        -0.000700016367125557,
        -0.001067803870182446,
        -0.001365785018311804,
        -0.001517658346416734,
        -0.001479220901518072,
        -0.001251645293619038,
        -0.000883897945231860,
        -0.000463014215415626,
        -0.000093998373363316,
        0.000126103655137308,
        0.000133637401088409,
        -0.000083291845411060,
        -0.000478656909570901,
        -0.000956856161710507,
        -0.001394276525547099,
        -0.001669304335046687,
        -0.001693568274147883,
        -0.001436465832146052,
        -0.000936294031076656,
        -0.000294364046467596,
        0.000347357721327949,
        0.000840280859210688,
        0.001066828143309338,
        0.000972235697442047,
        0.000582043327918912,
        0.000000000256197594,
        -0.000614161018448369,
        -0.001082502322666877,
        -0.001253394307549025,
        -0.001041747144061263,
        -0.000454434119287908,
        0.000406395838856528,
        0.001364157979259522,
        0.002208797613472180,
        0.002748494050791707,
        0.002859468540049823,
        0.002521074517937662,
        0.001826428600047300,
        0.000964572961222963,
        0.000177217785610635,
        -0.000300238026360743,
        -0.000299188741477655,
        0.000235540083397924,
        0.001225513295359685,
        0.002471493246958141,
        0.003697687773538245,
        0.004617801096565148,
        0.005007195312673036,
        0.004763102427051846,
        0.003936937808799607,
        0.002729046628746594,
        0.001445278819189449,
        0.000424470106087888,
        -0.000046225347619837,
        0.000192755366799392,
        0.001125985498229177,
        0.002557683117501900,
        0.004151914139698490,
        0.005510505445075422,
        0.006270828730028147,
        0.006199652631870028,
        0.005259032051396969,
        0.003626058613668432,
        0.001659059949801610,
        -0.000183969473410420,
        -0.001456514742437560,
        -0.001837632092428063,
        -0.001220219086348891,
        0.000250268973124557,
        0.002199090275873235,
        0.004104878861439933,
        0.005423942124853760,
        0.005726968150848706,
        0.004814222783768612,
        0.002778932513581960,
        -0.000000000000000020,
        -0.002937987069949102,
        -0.005381263417256724,
        -0.006768582170842054,
        -0.006778694264747967,
        -0.005425594270321380,
        -0.003074531132209382,
        -0.000370186028274846,
        0.001909993305564314,
        0.003044766738538663,
        0.002555341657243929,
        0.000341879705640572,
        -0.003267053905793534,
        -0.007569868812597692,
        -0.011644791877693197,
        -0.014568015609065142,
        -0.015646706574997726,
        -0.014609552004962152,
        -0.011704430712754850,
        -0.007672602784070903,
        -0.003597421688745940,
        -0.000656492530552524,
        0.000167999289080560,
        -0.001648000019050808,
        -0.006001624145195696,
        -0.012136918858699807,
        -0.018778665073791771,
        -0.024405895544277353,
        -0.027609434218948924,
        -0.027453303581532980,
        -0.023752768086720624,
        -0.017194896353878192,
        -0.009259031526475751,
        -0.001938254586103821,
        0.002690579750486043,
        0.002961919311579279,
        -0.001926238912041079,
        -0.011609454911608268,
        -0.024484355880443606,
        -0.037902449269599597,
        -0.048610301704033271,
        -0.053361419980958454,
        -0.049583612279410536,
        -0.035966872677339834,
        -0.012845947965868895,
        0.017712143650462100,
        0.052148995309944081,
        0.085952096779490728,
        0.114395707552124318,
        0.133349078989260017,
        0.139999169036633453,
        0.133349078989259184,
        0.114395707552124110,
        0.085952096779491310,
        0.052148995309944060,
        0.017712143650461882,
        -0.012845947965868907,
        -0.035966872677339917,
        -0.049583612279410508,
        -0.053361419980958197,
        -0.048610301704033174,
        -0.037902449269599681,
        -0.024484355880443589,
        -0.011609454911608280,
        -0.001926238912041048,
        0.002961919311579376,
        0.002690579750486018,
        -0.001938254586103910,
        -0.009259031526475770,
        -0.017194896353878220,
        -0.023752768086720617,
        -0.027453303581532876,
        -0.027609434218948910,
        -0.024405895544277384,
        -0.018778665073791743,
        -0.012136918858699778,
        -0.006001624145195689,
        -0.001648000019050759,
        0.000167999289080549,
        -0.000656492530552579,
        -0.003597421688745943,
        -0.007672602784070908,
        -0.011704430712754853,
        -0.014609552004962100,
        -0.015646706574997695,
        -0.014568015609065170,
        -0.011644791877693184,
        -0.007569868812597661,
        -0.003267053905793516,
        0.000341879705640599,
        0.002555341657243917,
        0.003044766738538625,
        0.001909993305564300,
        -0.000370186028274838,
        -0.003074531132209387,
        -0.005425594270321346,
        -0.006778694264747967,
        -0.006768582170842072,
        -0.005381263417256712,
        -0.002937987069949071,
        -0.000000000000000011,
        0.002778932513581968,
        0.004814222783768656,
        0.005726968150848740,
        0.005423942124853821,
        0.004104878861439981,
        0.002199090275873249,
        0.000250268973124572,
        -0.001220219086348899,
        -0.001837632092428107,
        -0.001456514742437579,
        -0.000183969473410391,
        0.001659059949801633,
        0.003626058613668475,
        0.005259032051397018,
        0.006199652631870087,
        0.006270828730028216,
        0.005510505445075484,
        0.004151914139698528,
        0.002557683117501931,
        0.001125985498229185,
        0.000192755366799377,
        -0.000046225347619840,
        0.000424470106087916,
        0.001445278819189465,
        0.002729046628746613,
        0.003936937808799641,
        0.004763102427051872,
        0.005007195312673072,
        0.004617801096565181,
        0.003697687773538268,
        0.002471493246958157,
        0.001225513295359686,
        0.000235540083397904,
        -0.000299188741477662,
        -0.000300238026360730,
        0.000177217785610633,
        0.000964572961222963,
        0.001826428600047311,
        0.002521074517937677,
        0.002859468540049836,
        0.002748494050791726,
        0.002208797613472188,
        0.001364157979259529,
        0.000406395838856531,
        -0.000454434119287924,
        -0.001041747144061274,
        -0.001253394307549016,
        -0.001082502322666882,
        -0.000614161018448379,
        0.000000000256197596,
        0.000582043327918921,
        0.000972235697442058,
        0.001066828143309351,
        0.000840280859210693,
        0.000347357721327954,
        -0.000294364046467589,
        -0.000936294031076667,
        -0.001436465832146061,
        -0.001693568274147880,
        -0.001669304335046701,
        -0.001394276525547121,
        -0.000956856161710518,
        -0.000478656909570902,
        -0.000083291845411064,
        0.000133637401088410,
        0.000126103655137310,
        -0.000093998373363315,
        -0.000463014215415622,
        -0.000883897945231856,
        -0.001251645293619037,
        -0.001479220901518064,
        -0.001517658346416736,
        -0.001365785018311829,
        -0.001067803870182473,
        -0.000700016367125582,
        -0.000350536391126813,
        -0.000097326032373874,
        0.000010017650619611,
        -0.000039475188103051,
        -0.000217856589139056,
        -0.000467418958231330,
        -0.000716513408195233,
        -0.000897787503650019,
        -0.000964268794421359,
        -0.000899521280968128,
        -0.000719772891556215,
        -0.000467993024698899,
        -0.000201857718153800,
        0.000021094171285306,
        0.000157333908326059,
        0.000186939009936750,
        0.000116856035155850,
        -0.000022555226620235,
        -0.000186422817242255,
        -0.000327193614468052,
        -0.000406330120699628,
        -0.000403042394189645,
        -0.000318132534085254,
        -0.000172345874617753,
        0.000000000000000001,
        0.000160240446869110,
        0.000275007655274871,
        0.000323922575839828,
        0.000303602758444402,
        0.000227270977517331,
        0.000120370386572638,
        0.000013535430323056,
        -0.000065184487486013,
        -0.000096905683079539,
        -0.000075783961378671,
        -0.000009439675744731,
        0.000083913089146680,
        0.000180688685679938,
        0.000258056497595318,
        0.000299413678129227,
        0.000297923664611431,
        0.000257409996529767,
        0.000190595031051903,
        0.000115321739722922,
        0.000049838671658327,
        0.000008370913380598,
        -0.000001968203571066,
        0.000017717026535077,
        0.000059085895346759,
        0.000109215344971164,
        0.000154138134366363,
        0.000182326628685567,
        0.000187277126391622,
        0.000168642843286591,
        0.000131767497141474,
        0.000085876732213267,
        0.000041491076437898,
        0.000007764492828016,
        -0.000009593328634649,
        -0.000009357104111403,
        0.000005367498620548,
        0.000028338040501232,
        0.000052015273666847,
        0.000069532771525943,
        0.000076298762817468,
        0.000070873288085739,
        0.000054979463750556,
        0.000032737065393792,
        0.000009390873182329,
        -0.000010096606723321,
        -0.000022224074836405,
        -0.000025634633911247,
        -0.000021189545218171,
        -0.000011484659442782,
        0.000000003925916434,
        0.000009886349403695,
        0.000015680958717755,
        0.000016302524484137,
        0.000012135584992189,
        0.000004728075549764,
        -0.000003766399101176,
        -0.000011220836255315,
        -0.000016068603891600,
        -0.000017614431357946,
        -0.000016073732205238,
        -0.000012369956958129,
        -0.000007779911735859,
        -0.000003544834962436,
        -0.000000556998756943,
        0.000000805480645211,
        0.000000674981116458,
        -0.000000448374840804,
        -0.000001914136494152,
        -0.000003138149589368,
        -0.000003755562154048,
        -0.000003676767442290,
        -0.000003047902131950,
        -0.000002146796703729,
        -0.000001260439328231,
        -0.000000588103458674,
        -0.000000198097768294,
        -0.000000042737486889,
        -0.000000014356226045,
        -0.000000012360763477,
        0.000000008849356400,
        0.000000026941448597,
        0.000000028705690596,
        0.000000021332271823,
        0.000000010718911008,
        0.000000001110478487,
        -0.000000005013831093,
        -0.000000006810645144,
        -0.000000004742965068,
        -0.000000000123281411,
        0.000000005383656436,
        0.000000010212660542,
        0.000000013235277467,
        0.000000013940815613,
        0.000000012452729409,
        0.000000009403466388,
        0.000000005715080122,
        0.000000002345545776,
        0.000000000060208323,
        -0.000000000724653090,
        -0.000000000000000035,
        0.000000001877997393,
        0.000000004309943074,
        0.000000006619220968,
        0.000000008219760596,
        0.000000008746879017,
        0.000000008124401987,
        0.000000006558151834,
        0.000000004463606820,
        0.000000002350046588,
        0.000000000691764604,
        -0.000000000182455461,
        -0.000000000157024066,
        0.000000000658368495,
        0.000000001971090716,
        0.000000003388478088,
        0.000000004518877370,
        0.000000005065643569,
        0.000000004892242639,
        0.000000004044869323,
        0.000000002729918300,
        0.000000001254519816,
        -0.000000000053390046,
        -0.000000000925134128,
        -0.000000001212356541,
        -0.000000000915607950,
        -0.000000000175515464,
        0.000000000769184304,
        0.000000001644796543,
        0.000000002210662580,
        0.000000002315410577,
        0.000000001928608621,
        0.000000001141566115,
        0.000000000138769086,
        -0.000000000851598982,
        -0.000000001615029276,
        -0.000000002001727733,
        -0.000000001959369498,
        -0.000000001540638747,
        -0.000000000885050586,
        -0.000000000180768455,
        0.000000000383348374,
        0.000000000663403196,
        0.000000000592931769,
        0.000000000195020521,
        -0.000000000427819421,
        -0.000000001122455767,
        -0.000000001724721882,
        -0.000000002099687405,
        -0.000000002172852126,
        -0.000000001945182017,
        -0.000000001489159499,
        -0.000000000927747471,
        -0.000000000402198885,
        -0.000000000037046015,
        0.000000000089128806,
        -0.000000000039638957,
        -0.000000000375996402,
        -0.000000000823920407,
        -0.000000001264408746,
        -0.000000001585455244,
        -0.000000001708845941,
        -0.000000001607452828,
        -0.000000001309312634,
        -0.000000000888160555,
        -0.000000000443382003,
        -0.000000000074759832,
        0.000000000141614391,
        0.000000000170494573,
        0.000000000023896763,
        -0.000000000244236161,
        -0.000000000553239700,
        -0.000000000816626677,
        -0.000000000963334849,
        -0.000000000954164229,
        -0.000000000789670165,
        -0.000000000508037237,
        -0.000000000173959401,
        0.000000000138321952,
        0.000000000363418121,
        0.000000000460104376,
        0.000000000419997775,
        0.000000000267845336,
        0.000000000053790430,
        -0.000000000160092717,
        -0.000000000315437109,
        -0.000000000371496333,
        -0.000000000314057717,
        -0.000000000157465205,
        0.000000000060370331,
        0.000000000288390020,
        0.000000000475423175,
        0.000000000582545506,
        0.000000000591807382,
        0.000000000509418634,
        0.000000000362923958,
        0.000000000193382416,
        0.000000000044719052,
        -0.000000000047019855,
        -0.000000000061904174,
        -0.000000000000000010,
        0.000000000119799710,
        0.000000000265423920,
        0.000000000400130526,
        0.000000000491650991,
        0.000000000519725757,
        0.000000000480302330,
        0.000000000385576943,
        0.000000000260113210,
        0.000000000134207889,
        0.000000000036269386,
        -0.000000000013890313,
        -0.000000000009285559,
        0.000000000043363156,
        0.000000000126107327,
        0.000000000214802644,
        0.000000000285322037,
        0.000000000319376599,
        0.000000000308563006,
        0.000000000255772556,
        0.000000000173788244,
        0.000000000081588224,
        -0.000000000000605539,
        -0.000000000056257043,
        -0.000000000076236942,
        -0.000000000060515380,
        -0.000000000017553040,
        0.000000000038310142,
        0.000000000090704093,
        0.000000000125304242,
        0.000000000133135045,
        0.000000000112358406,
        0.000000000068219953,
        0.000000000011294937,
        -0.000000000045424658,
        -0.000000000089817191,
        -0.000000000113474382,
        -0.000000000113446668,
        -0.000000000092550220,
        -0.000000000058266666,
        -0.000000000020604416,
        0.000000000010486316,
        0.000000000027488390,
        0.000000000026927123,
        0.000000000009884704,
        -0.000000000018619220,
        -0.000000000051143779,
        -0.000000000079803950,
        -0.000000000098196353,
        -0.000000000102813987,
        -0.000000000093651223,
        -0.000000000073919227,
        -0.000000000049014239,
        -0.000000000025048977,
        -0.000000000007337768,
        0.000000000000796702,
        -0.000000000001364685,
        -0.000000000012002721,
        -0.000000000027427282,
        -0.000000000043131787,
        -0.000000000054946615,
        -0.000000000060004350,
        -0.000000000057302816,
        -0.000000000047767668,
        -0.000000000033847849,
        -0.000000000018785845,
        -0.000000000005770824,
        0.000000000002810548,
        0.000000000005854699,
        0.000000000003665559,
        -0.000000000002270565,
        -0.000000000009767879,
        -0.000000000016538160,
        -0.000000000020745294,
        -0.000000000021388865,
        -0.000000000018443281,
        -0.000000000012747647,
        -0.000000000005701913,
        0.000000000001133523,
        0.000000000006426746,
        0.000000000009358698,
        0.000000000009748056,
        0.000000000008006757,
        0.000000000004960303,
        0.000000000001594412,
        -0.000000000001199418,
        -0.000000000002817996,
        -0.000000000003041173,
        -0.000000000002019892,
        -0.000000000000176778,
        0.000000000001941511,
        0.000000000003816257,
        0.000000000005074087,
        0.000000000005547267,
        0.000000000005272014,
        0.000000000004437669,
        0.000000000003309068,
        0.000000000002147863,
        0.000000000001152931,
        0.000000000000431345,
        0.000000000000000002,
        -0.000000000000190117,
        -0.000000000000219941,
        -0.000000000000170546,
        -0.000000000000102452,
        -0.000000000000048297,
        -0.000000000000017107,
        -0.000000000000004029,
        -0.000000000000000444,
        0.000000000000000044,
        0.000000000000000033,
    ]
    a, b = np.shape(data)
    x = np.zeros(a)
    x[0: len(h)] = h

    filt_sigs = np.zeros((a, b))
    for i in range(b):
        filt_chan = np.convolve(data[:, i], x, mode='same')
        filt_sigs[:, i] = filt_chan
    return(filt_sigs)


def fiir_bandpass_link_500Hz_1D(data):
    'https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter'
    'https://fiiir.com/'

    # 5 - 40 Hz.
    # Cut off Low = 5, Transition = 5.
    # Cut off High = 40, Transition = 5.

    h = [
        -0.000000000000000005,
        0.000000000000000004,
        -0.000000000000000489,
        -0.000000000000004065,
        -0.000000000000017120,
        -0.000000000000048299,
        -0.000000000000102445,
        -0.000000000000170542,
        -0.000000000000219925,
        -0.000000000000190101,
        -0.000000000000000002,
        0.000000000000431318,
        0.000000000001152905,
        0.000000000002147840,
        0.000000000003309031,
        0.000000000004437630,
        0.000000000005271984,
        0.000000000005547255,
        0.000000000005074080,
        0.000000000003816250,
        0.000000000001941506,
        -0.000000000000176784,
        -0.000000000002019903,
        -0.000000000003041201,
        -0.000000000002818024,
        -0.000000000001199444,
        0.000000000001594376,
        0.000000000004960260,
        0.000000000008006730,
        0.000000000009748046,
        0.000000000009358691,
        0.000000000006426739,
        0.000000000001133523,
        -0.000000000005701905,
        -0.000000000012747647,
        -0.000000000018443291,
        -0.000000000021388875,
        -0.000000000020745302,
        -0.000000000016538179,
        -0.000000000009767904,
        -0.000000000002270585,
        0.000000000003665553,
        0.000000000005854688,
        0.000000000002810537,
        -0.000000000005770822,
        -0.000000000018785833,
        -0.000000000033847847,
        -0.000000000047767674,
        -0.000000000057302812,
        -0.000000000060004342,
        -0.000000000054946620,
        -0.000000000043131802,
        -0.000000000027427289,
        -0.000000000012002720,
        -0.000000000001364686,
        0.000000000000796697,
        -0.000000000007337767,
        -0.000000000025048969,
        -0.000000000049014236,
        -0.000000000073919229,
        -0.000000000093651218,
        -0.000000000102813980,
        -0.000000000098196351,
        -0.000000000079803956,
        -0.000000000051143783,
        -0.000000000018619220,
        0.000000000009884697,
        0.000000000026927117,
        0.000000000027488394,
        0.000000000010486325,
        -0.000000000020604405,
        -0.000000000058266660,
        -0.000000000092550207,
        -0.000000000113446650,
        -0.000000000113474373,
        -0.000000000089817182,
        -0.000000000045424641,
        0.000000000011294948,
        0.000000000068219970,
        0.000000000112358411,
        0.000000000133135054,
        0.000000000125304248,
        0.000000000090704092,
        0.000000000038310138,
        -0.000000000017553045,
        -0.000000000060515377,
        -0.000000000076236942,
        -0.000000000056257048,
        -0.000000000000605539,
        0.000000000081588226,
        0.000000000173788243,
        0.000000000255772552,
        0.000000000308563012,
        0.000000000319376607,
        0.000000000285322045,
        0.000000000214802643,
        0.000000000126107332,
        0.000000000043363162,
        -0.000000000009285556,
        -0.000000000013890322,
        0.000000000036269383,
        0.000000000134207890,
        0.000000000260113205,
        0.000000000385576930,
        0.000000000480302332,
        0.000000000519725760,
        0.000000000491650991,
        0.000000000400130526,
        0.000000000265423924,
        0.000000000119799718,
        -0.000000000000000001,
        -0.000000000061904170,
        -0.000000000047019849,
        0.000000000044719059,
        0.000000000193382419,
        0.000000000362923957,
        0.000000000509418635,
        0.000000000591807385,
        0.000000000582545503,
        0.000000000475423166,
        0.000000000288390016,
        0.000000000060370339,
        -0.000000000157465207,
        -0.000000000314057722,
        -0.000000000371496333,
        -0.000000000315437099,
        -0.000000000160092714,
        0.000000000053790430,
        0.000000000267845340,
        0.000000000419997786,
        0.000000000460104378,
        0.000000000363418118,
        0.000000000138321956,
        -0.000000000173959391,
        -0.000000000508037236,
        -0.000000000789670166,
        -0.000000000954164223,
        -0.000000000963334831,
        -0.000000000816626673,
        -0.000000000553239702,
        -0.000000000244236157,
        0.000000000023896775,
        0.000000000170494580,
        0.000000000141614390,
        -0.000000000074759829,
        -0.000000000443381995,
        -0.000000000888160556,
        -0.000000001309312636,
        -0.000000001607452826,
        -0.000000001708845929,
        -0.000000001585455236,
        -0.000000001264408744,
        -0.000000000823920402,
        -0.000000000375996400,
        -0.000000000039638961,
        0.000000000089128802,
        -0.000000000037046010,
        -0.000000000402198873,
        -0.000000000927747464,
        -0.000000001489159498,
        -0.000000001945182016,
        -0.000000002172852123,
        -0.000000002099687412,
        -0.000000001724721891,
        -0.000000001122455769,
        -0.000000000427819414,
        0.000000000195020528,
        0.000000000592931773,
        0.000000000663403213,
        0.000000000383348396,
        -0.000000000180768435,
        -0.000000000885050579,
        -0.000000001540638734,
        -0.000000001959369482,
        -0.000000002001727725,
        -0.000000001615029280,
        -0.000000000851598979,
        0.000000000138769093,
        0.000000001141566119,
        0.000000001928608626,
        0.000000002315410592,
        0.000000002210662605,
        0.000000001644796564,
        0.000000000769184322,
        -0.000000000175515443,
        -0.000000000915607931,
        -0.000000001212356536,
        -0.000000000925134130,
        -0.000000000053390051,
        0.000000001254519819,
        0.000000002729918304,
        0.000000004044869323,
        0.000000004892242650,
        0.000000005065643591,
        0.000000004518877393,
        0.000000003388478105,
        0.000000001971090736,
        0.000000000658368511,
        -0.000000000157024060,
        -0.000000000182455473,
        0.000000000691764586,
        0.000000002350046555,
        0.000000004463606788,
        0.000000006558151781,
        0.000000008124401926,
        0.000000008746878978,
        0.000000008219760564,
        0.000000006619220935,
        0.000000004309943055,
        0.000000001877997392,
        -0.000000000000000037,
        -0.000000000724653091,
        0.000000000060208323,
        0.000000002345545786,
        0.000000005715080127,
        0.000000009403466382,
        0.000000012452729410,
        0.000000013940815615,
        0.000000013235277461,
        0.000000010212660532,
        0.000000005383656433,
        -0.000000000123281411,
        -0.000000004742965072,
        -0.000000006810645152,
        -0.000000005013831096,
        0.000000001110478489,
        0.000000010718911004,
        0.000000021332271817,
        0.000000028705690594,
        0.000000026941448600,
        0.000000008849356401,
        -0.000000012360763486,
        -0.000000014356226043,
        -0.000000042737486884,
        -0.000000198097768297,
        -0.000000588103458679,
        -0.000001260439328236,
        -0.000002146796703728,
        -0.000003047902131954,
        -0.000003676767442306,
        -0.000003755562154053,
        -0.000003138149589365,
        -0.000001914136494153,
        -0.000000448374840811,
        0.000000674981116457,
        0.000000805480645216,
        -0.000000556998756941,
        -0.000003544834962441,
        -0.000007779911735860,
        -0.000012369956958126,
        -0.000016073732205241,
        -0.000017614431357957,
        -0.000016068603891601,
        -0.000011220836255305,
        -0.000003766399101170,
        0.000004728075549763,
        0.000012135584992185,
        0.000016302524484133,
        0.000015680958717747,
        0.000009886349403680,
        0.000000003925916432,
        -0.000011484659442780,
        -0.000021189545218164,
        -0.000025634633911248,
        -0.000022224074836402,
        -0.000010096606723309,
        0.000009390873182328,
        0.000032737065393786,
        0.000054979463750557,
        0.000070873288085734,
        0.000076298762817461,
        0.000069532771525926,
        0.000052015273666842,
        0.000028338040501229,
        0.000005367498620546,
        -0.000009357104111413,
        -0.000009593328634651,
        0.000007764492828018,
        0.000041491076437899,
        0.000085876732213263,
        0.000131767497141473,
        0.000168642843286594,
        0.000187277126391620,
        0.000182326628685562,
        0.000154138134366355,
        0.000109215344971165,
        0.000059085895346754,
        0.000017717026535068,
        -0.000001968203571069,
        0.000008370913380602,
        0.000049838671658323,
        0.000115321739722915,
        0.000190595031051896,
        0.000257409996529763,
        0.000297923664611423,
        0.000299413678129213,
        0.000258056497595307,
        0.000180688685679935,
        0.000083913089146670,
        -0.000009439675744743,
        -0.000075783961378672,
        -0.000096905683079531,
        -0.000065184487486010,
        0.000013535430323051,
        0.000120370386572638,
        0.000227270977517335,
        0.000303602758444394,
        0.000323922575839815,
        0.000275007655274860,
        0.000160240446869099,
        -0.000000000000000011,
        -0.000172345874617771,
        -0.000318132534085260,
        -0.000403042394189642,
        -0.000406330120699630,
        -0.000327193614468050,
        -0.000186422817242246,
        -0.000022555226620225,
        0.000116856035155856,
        0.000186939009936746,
        0.000157333908326052,
        0.000021094171285294,
        -0.000201857718153825,
        -0.000467993024698925,
        -0.000719772891556236,
        -0.000899521280968133,
        -0.000964268794421358,
        -0.000897787503650017,
        -0.000716513408195216,
        -0.000467418958231290,
        -0.000217856589139019,
        -0.000039475188103017,
        0.000010017650619659,
        -0.000097326032373829,
        -0.000350536391126778,
        -0.000700016367125557,
        -0.001067803870182446,
        -0.001365785018311804,
        -0.001517658346416734,
        -0.001479220901518072,
        -0.001251645293619038,
        -0.000883897945231860,
        -0.000463014215415626,
        -0.000093998373363316,
        0.000126103655137308,
        0.000133637401088409,
        -0.000083291845411060,
        -0.000478656909570901,
        -0.000956856161710507,
        -0.001394276525547099,
        -0.001669304335046687,
        -0.001693568274147883,
        -0.001436465832146052,
        -0.000936294031076656,
        -0.000294364046467596,
        0.000347357721327949,
        0.000840280859210688,
        0.001066828143309338,
        0.000972235697442047,
        0.000582043327918912,
        0.000000000256197594,
        -0.000614161018448369,
        -0.001082502322666877,
        -0.001253394307549025,
        -0.001041747144061263,
        -0.000454434119287908,
        0.000406395838856528,
        0.001364157979259522,
        0.002208797613472180,
        0.002748494050791707,
        0.002859468540049823,
        0.002521074517937662,
        0.001826428600047300,
        0.000964572961222963,
        0.000177217785610635,
        -0.000300238026360743,
        -0.000299188741477655,
        0.000235540083397924,
        0.001225513295359685,
        0.002471493246958141,
        0.003697687773538245,
        0.004617801096565148,
        0.005007195312673036,
        0.004763102427051846,
        0.003936937808799607,
        0.002729046628746594,
        0.001445278819189449,
        0.000424470106087888,
        -0.000046225347619837,
        0.000192755366799392,
        0.001125985498229177,
        0.002557683117501900,
        0.004151914139698490,
        0.005510505445075422,
        0.006270828730028147,
        0.006199652631870028,
        0.005259032051396969,
        0.003626058613668432,
        0.001659059949801610,
        -0.000183969473410420,
        -0.001456514742437560,
        -0.001837632092428063,
        -0.001220219086348891,
        0.000250268973124557,
        0.002199090275873235,
        0.004104878861439933,
        0.005423942124853760,
        0.005726968150848706,
        0.004814222783768612,
        0.002778932513581960,
        -0.000000000000000020,
        -0.002937987069949102,
        -0.005381263417256724,
        -0.006768582170842054,
        -0.006778694264747967,
        -0.005425594270321380,
        -0.003074531132209382,
        -0.000370186028274846,
        0.001909993305564314,
        0.003044766738538663,
        0.002555341657243929,
        0.000341879705640572,
        -0.003267053905793534,
        -0.007569868812597692,
        -0.011644791877693197,
        -0.014568015609065142,
        -0.015646706574997726,
        -0.014609552004962152,
        -0.011704430712754850,
        -0.007672602784070903,
        -0.003597421688745940,
        -0.000656492530552524,
        0.000167999289080560,
        -0.001648000019050808,
        -0.006001624145195696,
        -0.012136918858699807,
        -0.018778665073791771,
        -0.024405895544277353,
        -0.027609434218948924,
        -0.027453303581532980,
        -0.023752768086720624,
        -0.017194896353878192,
        -0.009259031526475751,
        -0.001938254586103821,
        0.002690579750486043,
        0.002961919311579279,
        -0.001926238912041079,
        -0.011609454911608268,
        -0.024484355880443606,
        -0.037902449269599597,
        -0.048610301704033271,
        -0.053361419980958454,
        -0.049583612279410536,
        -0.035966872677339834,
        -0.012845947965868895,
        0.017712143650462100,
        0.052148995309944081,
        0.085952096779490728,
        0.114395707552124318,
        0.133349078989260017,
        0.139999169036633453,
        0.133349078989259184,
        0.114395707552124110,
        0.085952096779491310,
        0.052148995309944060,
        0.017712143650461882,
        -0.012845947965868907,
        -0.035966872677339917,
        -0.049583612279410508,
        -0.053361419980958197,
        -0.048610301704033174,
        -0.037902449269599681,
        -0.024484355880443589,
        -0.011609454911608280,
        -0.001926238912041048,
        0.002961919311579376,
        0.002690579750486018,
        -0.001938254586103910,
        -0.009259031526475770,
        -0.017194896353878220,
        -0.023752768086720617,
        -0.027453303581532876,
        -0.027609434218948910,
        -0.024405895544277384,
        -0.018778665073791743,
        -0.012136918858699778,
        -0.006001624145195689,
        -0.001648000019050759,
        0.000167999289080549,
        -0.000656492530552579,
        -0.003597421688745943,
        -0.007672602784070908,
        -0.011704430712754853,
        -0.014609552004962100,
        -0.015646706574997695,
        -0.014568015609065170,
        -0.011644791877693184,
        -0.007569868812597661,
        -0.003267053905793516,
        0.000341879705640599,
        0.002555341657243917,
        0.003044766738538625,
        0.001909993305564300,
        -0.000370186028274838,
        -0.003074531132209387,
        -0.005425594270321346,
        -0.006778694264747967,
        -0.006768582170842072,
        -0.005381263417256712,
        -0.002937987069949071,
        -0.000000000000000011,
        0.002778932513581968,
        0.004814222783768656,
        0.005726968150848740,
        0.005423942124853821,
        0.004104878861439981,
        0.002199090275873249,
        0.000250268973124572,
        -0.001220219086348899,
        -0.001837632092428107,
        -0.001456514742437579,
        -0.000183969473410391,
        0.001659059949801633,
        0.003626058613668475,
        0.005259032051397018,
        0.006199652631870087,
        0.006270828730028216,
        0.005510505445075484,
        0.004151914139698528,
        0.002557683117501931,
        0.001125985498229185,
        0.000192755366799377,
        -0.000046225347619840,
        0.000424470106087916,
        0.001445278819189465,
        0.002729046628746613,
        0.003936937808799641,
        0.004763102427051872,
        0.005007195312673072,
        0.004617801096565181,
        0.003697687773538268,
        0.002471493246958157,
        0.001225513295359686,
        0.000235540083397904,
        -0.000299188741477662,
        -0.000300238026360730,
        0.000177217785610633,
        0.000964572961222963,
        0.001826428600047311,
        0.002521074517937677,
        0.002859468540049836,
        0.002748494050791726,
        0.002208797613472188,
        0.001364157979259529,
        0.000406395838856531,
        -0.000454434119287924,
        -0.001041747144061274,
        -0.001253394307549016,
        -0.001082502322666882,
        -0.000614161018448379,
        0.000000000256197596,
        0.000582043327918921,
        0.000972235697442058,
        0.001066828143309351,
        0.000840280859210693,
        0.000347357721327954,
        -0.000294364046467589,
        -0.000936294031076667,
        -0.001436465832146061,
        -0.001693568274147880,
        -0.001669304335046701,
        -0.001394276525547121,
        -0.000956856161710518,
        -0.000478656909570902,
        -0.000083291845411064,
        0.000133637401088410,
        0.000126103655137310,
        -0.000093998373363315,
        -0.000463014215415622,
        -0.000883897945231856,
        -0.001251645293619037,
        -0.001479220901518064,
        -0.001517658346416736,
        -0.001365785018311829,
        -0.001067803870182473,
        -0.000700016367125582,
        -0.000350536391126813,
        -0.000097326032373874,
        0.000010017650619611,
        -0.000039475188103051,
        -0.000217856589139056,
        -0.000467418958231330,
        -0.000716513408195233,
        -0.000897787503650019,
        -0.000964268794421359,
        -0.000899521280968128,
        -0.000719772891556215,
        -0.000467993024698899,
        -0.000201857718153800,
        0.000021094171285306,
        0.000157333908326059,
        0.000186939009936750,
        0.000116856035155850,
        -0.000022555226620235,
        -0.000186422817242255,
        -0.000327193614468052,
        -0.000406330120699628,
        -0.000403042394189645,
        -0.000318132534085254,
        -0.000172345874617753,
        0.000000000000000001,
        0.000160240446869110,
        0.000275007655274871,
        0.000323922575839828,
        0.000303602758444402,
        0.000227270977517331,
        0.000120370386572638,
        0.000013535430323056,
        -0.000065184487486013,
        -0.000096905683079539,
        -0.000075783961378671,
        -0.000009439675744731,
        0.000083913089146680,
        0.000180688685679938,
        0.000258056497595318,
        0.000299413678129227,
        0.000297923664611431,
        0.000257409996529767,
        0.000190595031051903,
        0.000115321739722922,
        0.000049838671658327,
        0.000008370913380598,
        -0.000001968203571066,
        0.000017717026535077,
        0.000059085895346759,
        0.000109215344971164,
        0.000154138134366363,
        0.000182326628685567,
        0.000187277126391622,
        0.000168642843286591,
        0.000131767497141474,
        0.000085876732213267,
        0.000041491076437898,
        0.000007764492828016,
        -0.000009593328634649,
        -0.000009357104111403,
        0.000005367498620548,
        0.000028338040501232,
        0.000052015273666847,
        0.000069532771525943,
        0.000076298762817468,
        0.000070873288085739,
        0.000054979463750556,
        0.000032737065393792,
        0.000009390873182329,
        -0.000010096606723321,
        -0.000022224074836405,
        -0.000025634633911247,
        -0.000021189545218171,
        -0.000011484659442782,
        0.000000003925916434,
        0.000009886349403695,
        0.000015680958717755,
        0.000016302524484137,
        0.000012135584992189,
        0.000004728075549764,
        -0.000003766399101176,
        -0.000011220836255315,
        -0.000016068603891600,
        -0.000017614431357946,
        -0.000016073732205238,
        -0.000012369956958129,
        -0.000007779911735859,
        -0.000003544834962436,
        -0.000000556998756943,
        0.000000805480645211,
        0.000000674981116458,
        -0.000000448374840804,
        -0.000001914136494152,
        -0.000003138149589368,
        -0.000003755562154048,
        -0.000003676767442290,
        -0.000003047902131950,
        -0.000002146796703729,
        -0.000001260439328231,
        -0.000000588103458674,
        -0.000000198097768294,
        -0.000000042737486889,
        -0.000000014356226045,
        -0.000000012360763477,
        0.000000008849356400,
        0.000000026941448597,
        0.000000028705690596,
        0.000000021332271823,
        0.000000010718911008,
        0.000000001110478487,
        -0.000000005013831093,
        -0.000000006810645144,
        -0.000000004742965068,
        -0.000000000123281411,
        0.000000005383656436,
        0.000000010212660542,
        0.000000013235277467,
        0.000000013940815613,
        0.000000012452729409,
        0.000000009403466388,
        0.000000005715080122,
        0.000000002345545776,
        0.000000000060208323,
        -0.000000000724653090,
        -0.000000000000000035,
        0.000000001877997393,
        0.000000004309943074,
        0.000000006619220968,
        0.000000008219760596,
        0.000000008746879017,
        0.000000008124401987,
        0.000000006558151834,
        0.000000004463606820,
        0.000000002350046588,
        0.000000000691764604,
        -0.000000000182455461,
        -0.000000000157024066,
        0.000000000658368495,
        0.000000001971090716,
        0.000000003388478088,
        0.000000004518877370,
        0.000000005065643569,
        0.000000004892242639,
        0.000000004044869323,
        0.000000002729918300,
        0.000000001254519816,
        -0.000000000053390046,
        -0.000000000925134128,
        -0.000000001212356541,
        -0.000000000915607950,
        -0.000000000175515464,
        0.000000000769184304,
        0.000000001644796543,
        0.000000002210662580,
        0.000000002315410577,
        0.000000001928608621,
        0.000000001141566115,
        0.000000000138769086,
        -0.000000000851598982,
        -0.000000001615029276,
        -0.000000002001727733,
        -0.000000001959369498,
        -0.000000001540638747,
        -0.000000000885050586,
        -0.000000000180768455,
        0.000000000383348374,
        0.000000000663403196,
        0.000000000592931769,
        0.000000000195020521,
        -0.000000000427819421,
        -0.000000001122455767,
        -0.000000001724721882,
        -0.000000002099687405,
        -0.000000002172852126,
        -0.000000001945182017,
        -0.000000001489159499,
        -0.000000000927747471,
        -0.000000000402198885,
        -0.000000000037046015,
        0.000000000089128806,
        -0.000000000039638957,
        -0.000000000375996402,
        -0.000000000823920407,
        -0.000000001264408746,
        -0.000000001585455244,
        -0.000000001708845941,
        -0.000000001607452828,
        -0.000000001309312634,
        -0.000000000888160555,
        -0.000000000443382003,
        -0.000000000074759832,
        0.000000000141614391,
        0.000000000170494573,
        0.000000000023896763,
        -0.000000000244236161,
        -0.000000000553239700,
        -0.000000000816626677,
        -0.000000000963334849,
        -0.000000000954164229,
        -0.000000000789670165,
        -0.000000000508037237,
        -0.000000000173959401,
        0.000000000138321952,
        0.000000000363418121,
        0.000000000460104376,
        0.000000000419997775,
        0.000000000267845336,
        0.000000000053790430,
        -0.000000000160092717,
        -0.000000000315437109,
        -0.000000000371496333,
        -0.000000000314057717,
        -0.000000000157465205,
        0.000000000060370331,
        0.000000000288390020,
        0.000000000475423175,
        0.000000000582545506,
        0.000000000591807382,
        0.000000000509418634,
        0.000000000362923958,
        0.000000000193382416,
        0.000000000044719052,
        -0.000000000047019855,
        -0.000000000061904174,
        -0.000000000000000010,
        0.000000000119799710,
        0.000000000265423920,
        0.000000000400130526,
        0.000000000491650991,
        0.000000000519725757,
        0.000000000480302330,
        0.000000000385576943,
        0.000000000260113210,
        0.000000000134207889,
        0.000000000036269386,
        -0.000000000013890313,
        -0.000000000009285559,
        0.000000000043363156,
        0.000000000126107327,
        0.000000000214802644,
        0.000000000285322037,
        0.000000000319376599,
        0.000000000308563006,
        0.000000000255772556,
        0.000000000173788244,
        0.000000000081588224,
        -0.000000000000605539,
        -0.000000000056257043,
        -0.000000000076236942,
        -0.000000000060515380,
        -0.000000000017553040,
        0.000000000038310142,
        0.000000000090704093,
        0.000000000125304242,
        0.000000000133135045,
        0.000000000112358406,
        0.000000000068219953,
        0.000000000011294937,
        -0.000000000045424658,
        -0.000000000089817191,
        -0.000000000113474382,
        -0.000000000113446668,
        -0.000000000092550220,
        -0.000000000058266666,
        -0.000000000020604416,
        0.000000000010486316,
        0.000000000027488390,
        0.000000000026927123,
        0.000000000009884704,
        -0.000000000018619220,
        -0.000000000051143779,
        -0.000000000079803950,
        -0.000000000098196353,
        -0.000000000102813987,
        -0.000000000093651223,
        -0.000000000073919227,
        -0.000000000049014239,
        -0.000000000025048977,
        -0.000000000007337768,
        0.000000000000796702,
        -0.000000000001364685,
        -0.000000000012002721,
        -0.000000000027427282,
        -0.000000000043131787,
        -0.000000000054946615,
        -0.000000000060004350,
        -0.000000000057302816,
        -0.000000000047767668,
        -0.000000000033847849,
        -0.000000000018785845,
        -0.000000000005770824,
        0.000000000002810548,
        0.000000000005854699,
        0.000000000003665559,
        -0.000000000002270565,
        -0.000000000009767879,
        -0.000000000016538160,
        -0.000000000020745294,
        -0.000000000021388865,
        -0.000000000018443281,
        -0.000000000012747647,
        -0.000000000005701913,
        0.000000000001133523,
        0.000000000006426746,
        0.000000000009358698,
        0.000000000009748056,
        0.000000000008006757,
        0.000000000004960303,
        0.000000000001594412,
        -0.000000000001199418,
        -0.000000000002817996,
        -0.000000000003041173,
        -0.000000000002019892,
        -0.000000000000176778,
        0.000000000001941511,
        0.000000000003816257,
        0.000000000005074087,
        0.000000000005547267,
        0.000000000005272014,
        0.000000000004437669,
        0.000000000003309068,
        0.000000000002147863,
        0.000000000001152931,
        0.000000000000431345,
        0.000000000000000002,
        -0.000000000000190117,
        -0.000000000000219941,
        -0.000000000000170546,
        -0.000000000000102452,
        -0.000000000000048297,
        -0.000000000000017107,
        -0.000000000000004029,
        -0.000000000000000444,
        0.000000000000000044,
        0.000000000000000033,
    ]
    a = np.shape(data)
    x = np.zeros(a)
    x[0: len(h)] = h
    filt_chan = np.convolve(data, x, mode='same')

    return(filt_chan)


def random_index(targCue, num_emoji):
    import random
    z = np.arange(num_emoji)
    random.shuffle(z)
    fin = z[0]

    print(fin)
    if fin == targCue:
        fin = z[1]
    return fin


def sess_inc(num_sess, sub_path, lab_path):
    sub_path2 = []
    lab_path2 = []
    for i in range(len(sub_path)):
        y = sub_path[i]
        if num_sess == 1:
            if y[31:36] == '00001':
                sub_path2 = np.append(sub_path2, sub_path[i])
                lab_path2 = np.append(lab_path2, lab_path[i])
        if num_sess == 2:
            if y[31:36] == '00001' or y[31:36] == '00002':
                sub_path2 = np.append(sub_path2, sub_path[i])
                lab_path2 = np.append(lab_path2, lab_path[i])
        if num_sess == 3:
            if y[31:36] == '00001' or y[31:36] == '00002' or y[31:36] == '00003':
                sub_path2 = np.append(sub_path2, sub_path[i])
                lab_path2 = np.append(lab_path2, lab_path[i])
    if num_sess == 4:
        sub_path2 = sub_path
        lab_path2 = lab_path
    return sub_path2, lab_path2


def band_power_plots(data, sing_plt, plotter):
    'Calculates bandpower values for 5 major EEG sub-bands.'
    'sing_plt = ON means it generates a single plt.'
    'plotter = changes whether present in absolute or relative power (out of 100).'
    fs = 500                                # Sampling rate (500 Hz)
    # print('Data DIMS: ', data.shape)

    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(data))

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)

    # Define EEG bands
    'Delta adapted to 0.5 | Ref: https://ieeexplore.ieee.org/abstract/document/5626721'
    eeg_bands = {'Delta': (0.5, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}

    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                           (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

    # Plot.
    bands = np.arange(5)
    values = [eeg_band_fft['Delta'], eeg_band_fft['Theta'],
              eeg_band_fft['Alpha'], eeg_band_fft['Beta'], eeg_band_fft['Gamma']]
    sum_pow = np.sum(values)
    # print('SUM POWER: ', sum_pow)

    # Rel Power
    rel_pow = np.arange(len(values))
    for i in range(len(rel_pow)):
        rel_pow[i] = (values[i] / sum_pow) * 100
    # print('REL POWERS %: ', rel_pow)

    if sing_plt == 'ON':
        fig, ax = plt.subplots()
        if plotter == 'ABS':
            plt.bar(bands, values)
        if plotter == 'REL':
            plt.bar(bands, rel_pow)
        plt.xticks(bands, ('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'))
        plt.show()
    return values, rel_pow


def sub_band_chunk_plot(data, divs, pow_disp, plotter):
    'This creates a plot with 5 subplots, each providing a sub-band analysis of 1/nth of an experimental session.'
    'This should demonstrate the change in sub-band freqs, with more high amp low oscillations at the end.'
    'divs = number of divisions of data made across the experimental sessions.'
    'pow_dsip = either "ABS" for absolute values or "REL" for relative power.'
    'NOTE: No overlap is performed across chunks.'

    # Initializing data slicing for divisions.
    trials = data.shape[1]
    samp_ind = np.linspace(0, trials, divs + 1)
    samp_ind = np.round(samp_ind, decimals=0)
    samp_ind = samp_ind.astype('int')
    print('SAMP IND: ', samp_ind)
    # X axis plotting labels.
    x_axis = ('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')

    if divs > 1:
        if plotter == 'ON':
            fig, axes = plt.subplots(divs)
        for i in range(divs):
            eeg_chk = data[:, samp_ind[i]:samp_ind[i + 1]]
            # Avergae across division sub-samples (individual trials).
            eeg_chk = np.average(eeg_chk, axis=1)
            eeg_chk = np.expand_dims(eeg_chk, axis=1)
            values, rel_pow = band_power_plots(eeg_chk, sing_plt='OFF', plotter='OFF')
            if pow_disp == 'ABS':
                x_ax = autolabel(x_axis, values)
                if plotter == 'ON':
                    axes[i].bar(x_ax, values)
            elif pow_disp == 'REL':
                x_ax = autolabel(x_axis, rel_pow)
                if plotter == 'ON':
                    axes[i].bar(x_ax, rel_pow)
            # print('IND A: ', samp_ind[i], 'IND B: ', samp_ind[i + 1])
            # print('EEG CHK AVG VALS: ', eeg_chk.shape, eeg_chk[0:10])
            # print('EEG CHK POW CHK VAL: ', values, '\n CHK REL: ', rel_pow)
        if plotter == 'ON':
            fig.tight_layout()
            plt.show()
    else:
        eeg_chk = np.average(data, axis=1)
        eeg_chk = np.expand_dims(eeg_chk, axis=1)
        values, rel_pow = band_power_plots(eeg_chk, sing_plt='OFF', plotter='OFF')
        if pow_disp == 'ABS':
            x_ax = autolabel(x_axis, values)
            if plotter == 'ON':
                plt.bar(x_ax, values)
        elif pow_disp == 'REL':
            x_ax = autolabel(x_axis, values)
            if plotter == 'ON':
                plt.bar(x_ax, rel_pow)
        if plotter == 'ON':
            plt.show()
            plt.title('Sub-Band Plots')
        # Useful Info
        # print('EEG CHK AVG VALS: ', eeg_chk.shape, eeg_chk[0:10])
        # print('EEG CHK POW VALS: ', values)


def autolabel(x_axis, values):
    print(x_axis)
    x = []
    for i in range(len(values)):
        xA = x_axis[i] + (': ') + np.array2string(np.array(values[i])) + ('%')
        if i == 0:
            x = xA
        else:
            x = np.append(x, xA)
        print('X_AXIS LABEL: ', x)

    return x


def lda_loc_extract(dat_direc, verbose, norm, num_trials):
    'Extracts data from Localizer experiments for LDA analysis.'
    'dat_direc = location of data files.'
    'num_trials = number trials you want to extract from the experimental session, if [] it takes all trials.'
    '**kwargs = is used to detect if num_trials has been specifed. '
    'verbose = if 1 it prints dim info on returned variables.'
    'norm = add normalization if == 1.'
    # Data Pathway Extraction.
    lab_direc = dat_direc + 'Labels/'
    dat_files = pathway_extract(dat_direc, '.npy', 'Volt', full_ext=1)
    lab_files = pathway_extract(lab_direc, '.npz', 'Labels', full_ext=1)
    labels = np.load(lab_files[0])['arr_0']
    target_names = np.array(['P300', 'NP300'])
    '_________P300 Averaging_________'
    np3 = []
    p3 = []
    # Iterator Variables.
    iterator = num_trials
    if num_trials > len(labels):
        iterator = len(labels)
    labels = labels[0:iterator]
    # Extrsaction.
    for i in range(iterator):
        'Pre-Process Data Chunk'
        eeg = np.load(dat_files[i])
        # For Cz / Target electrodes breakdown.
        eeg = prepro(eeg, samp_ratekHz=0.5, zero='ON', ext='INC-Ref', elec=[0, 1, 3, 4],
                     filtH='ON', hlevel=1, filtL='ON', llevel=10,
                     notc='LOW', notfq=50, ref_ind=-7, ref='A2', avg='ON')
        # NP300 Trials.
        if labels[i] == 0:
            if np3 == []:
                np3 = eeg
                np3 = np.expand_dims(np3, axis=2)
            else:
                eeg = np.expand_dims(eeg, axis=2)
                np3 = np.append(np3, eeg, axis=2)
        # P300 Trials.
        if labels[i] == 1:
            if p3 == []:
                p3 = eeg
                p3 = np.expand_dims(p3, axis=2)
            else:
                eeg = np.expand_dims(eeg, axis=2)
                p3 = np.append(p3, eeg, axis=2)
    # Conjoin.
    p3 = np.squeeze(p3)
    np3 = np.squeeze(np3)
    eeg = np.append(p3, np3, axis=1)
    if norm == 1:
        eeg = scaler1D(eeg, -1, 1)
    eeg = np.swapaxes(eeg, 0, 1)
    # Info
    if verbose == 1:
        print('X DIMS: ', eeg.shape, '\n', eeg[50:54, 0:4])
        print('y DIMS: ', labels.shape, labels[0:4])
        print('Names DIMS: ', target_names.shape, target_names[0:4])
    return eeg, labels, target_names


'---------------Syntax Signal Processing Examples.'

'Cz Extraction'
# cz_data = data[:, [3, 4, 8]]

'Data Scaling'
# from sklearn.preprocessing import MinMaxScaler
# agg_data = agg_data.reshape(-1, 1)
# scaler = MinMaxScaler(copy=True, feature_range=(0, 5))
# scaler.fit(agg_data)
# clean_data = scaler.transform(agg_data)
# clean_data = np.reshape(clean_data, (num_samps, num_trials, num_subs))

'Bandpass Plotting'
'------------------------------Pre-Bandpass Plots'
'------------------------------------------------'
