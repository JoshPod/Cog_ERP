import sys
import numpy as np
import ProBoy as pb
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ----------------------------------------------------------------------------------
# MeanBoy data organisation script in which sequences are split into flash segments.
# ----------------------------------------------------------------------------------

# Hard Coded
'Only non-dynamic, hard-coded section of code is the sample rate of 500Hz.'

# Plot Dependency
sys.setrecursionlimit(2000)

# Data Pathway Extraction.
direc = '..//Data_Aquisition/Data/Exp_1/'
sub_path, lab_path = pb.subject_extract(direc)
'------------Session Mitigation Parameters------------'
num_sess = 4
sub_path, lab_path = pb.sess_inc(num_sess, sub_path, lab_path)


for k in range(len(sub_path)):
    eeg_sub_spec = sub_path[k]
    lab_sub_spec = lab_path[k]
    eeg_files = pb.pathway_extract(eeg_sub_spec, '.npz', 'volt', full_ext=0)
    print('EEG FIles: ', eeg_files)

    # Labels
    labels_files = pb.pathway_extract(lab_sub_spec, '.npz', 't', full_ext=0)
    print(labels_files)

    '------------Extraction Parameters------------'
    # Print Labels Info
    label_examp, matSize, num_seq, num_emoji, num_trials = pb.labels_inf_extract(
        lab_sub_spec, '0001_trial_labels.npz')
    print('Label Example: ', label_examp['arr_1'], 'Label Dims: ', matSize,
          'Num Seqs: ', num_seq, 'Num Emojis: ', num_emoji, 'Num Trials: ', num_trials)
    # Print Data Info
    data = np.load(eeg_sub_spec+'/'+eeg_files[0])
    print('File_Name', eeg_files[0])
    print('Data Shape: ', np.shape(data['arr_0']))

    '------------Sequence and Trial Mitigation Parameters------------'
    num_seq = 5
    num_trials = 49
    '---------------------------------------------'

    # Time-Stamp SWITCH | 0 == Perfect Generated Time Stamps, 1 == Actual Time Stamp data.
    stamper = 1
    # Sequence Plotter | 0 == Plot the sequences and pro-processing steps, 1 == No plots.
    seq_plotter = 1
    # Flash event segmenation parameters.
    samp_ratekHz = 0.5
    samp_rateHz = samp_ratekHz*1000
    temp_axis_multi = 1/samp_ratekHz
    p300 = 300
    pWave = p300+100  # Used in plotting to gather time seg.
    aug_dur = 125
    seq_dur = aug_dur * num_emoji
    p3_buff = aug_dur + 375
    # Index used to create segments in samples.
    p3samp = np.int(np.round(p3_buff/temp_axis_multi))
    print('p3samp / Samp Rate Hz: ', p3samp)
    # Pre-Form Storage of p3 postions in temporal space.
    p3store = np.zeros(len(eeg_files), dtype=int)*num_seq
    # Flash Event Indices.
    xcoords = np.linspace(0, seq_dur-aug_dur, num=num_emoji)/temp_axis_multi
    xcoords = xcoords.astype(int)
    print('xcoords / Flash Temp Marker: ', xcoords*temp_axis_multi, 'Sample Indices: ', xcoords)
    pcoords = (xcoords + p3samp)  # ADD buffer to collect full P300 later.
    print('pcoords / P3 Temp Marker: ', pcoords*temp_axis_multi, 'Sample Indcies: ', pcoords)
    # Plot legends
    flash_line = mlines.Line2D([], [], color='r', linestyle='dotted', label='Flash Events')
    prows = 5
    pcols = 2

    '=======================Extraction======================='
    # Extraction Loop Iterator.
    controller = int(len(eeg_files))

    for s in range(controller):
        print('Trial:{0}'.format(s+1), eeg_files[s])

        '--------------File Loading.'
        # Get EEG voltage data file.
        data = np.load(eeg_sub_spec+'/'+eeg_files[s])
        # Load labels file to allow printing of target emoji to title and highlighting temporal position of target augs in graph.
        labelsTot = np.load(lab_sub_spec+labels_files[s])

        '--------------Indexing Variables.'
        # TargOrder tells you which emoji is being cued in each of the 49 trials.
        targOrder = labelsTot['arr_1']
        # TargCue tells you the emoji currently being processed.
        targCue = labelsTot['arr_1'][s]
        # TotPres tells you when in the sequence the cued emoji is flashed.
        totPres = labelsTot['arr_0']
        print('Target Cued Emoji', targCue, 'Total Pres', totPres)
        for j in range(num_seq):
            print('--------------Sequence: {}'.format(j+1))
            # Extract actual order of randomised emoji flashes within the sequence.
            presOrder = labelsTot['arr_0'][j, :]
            '=======================Pre-Processing======================='
            # Extract one sequence of EEG data from the trial file you just opened.
            eeg = np.asarray(data['arr_{}'.format(j)])
            # Temporal Variables:
            # Temporal Axis Generator. print('EEG Series: ', eeg_series)
            eeg_series = pb.simp_temp_axis(eeg, samp_ratekHz)
            if stamper == 0:
                eeg_series = pb.time_stamper(eeg, 0)  # Time=Stamp Extraction.
            '---------------------------------------------------'
            '1) EXTRACT: relevant electrodes:  0) Fz, 1) Cz, 2) Pz, 3) P4, 4) P3, 5) O1, 6) O2, 7) A2.'
            'Reference: Guger (2012): Dry vs Wet Electrodes | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3345570/'
            eeg_ext = pb.zero_mean(eeg[:, [1, -1]])  # [5, 8, 11, 15, 16, 19]
            '----------------------------------------------------'
            '2) FILTERING: 50Hz notch filter.'
            eeg_lin = pb.notchyNik(np.copy(eeg_ext), Fs=samp_rateHz, freq=50)
            '------------------------------------------------------'
            '3) REFERENCING: using A2 electrode.'
            eeg_A2 = pb.referencer(np.copy(eeg_lin), -1)
            '---------------------------------------------------'
            '5) FILTERING: highpass filter.'
            eeg_high = pb.butter_highpass_filter(np.copy(eeg_A2), 1, 500, order=5)
            '---------------------------------------------------'
            '6) FILTERING: lowpass filter.'
            eeg_low = pb.butter_lowpass_filter(np.copy(eeg_high), 10, 500, order=5)
            '---------------------------------------------------'
            '7) AVERAGING: cross-channels.'
            eeg_avg = np.average(np.copy(eeg_low), axis=1)
            eeg_avg = np.expand_dims(eeg_avg, axis=1)
            print('EEG DATA DIMS: ', np.shape(eeg_avg))
            '---------------------------------------------------'
            '8) FINAL: pre-processed data.'
            eeg_fin = np.copy(eeg_avg)
            '=======================Segmentation======================='
            '9) SEGMENT: using flash events'
            num_chans = np.amin(np.shape(eeg_fin))
            p3segs = np.zeros((p3samp, num_chans))
            np3segs = np.zeros((p3samp, num_chans))
            # 1st get emoji index (target cue).
            # 2nd get flash order (presOrder).
            # 3rd find when emoji flash index.
            f_index = presOrder[targCue]
            p3segs = eeg_fin[xcoords[f_index]:pcoords[f_index], :]
            print('P3 Flash Co-ordinates: ', xcoords[f_index], pcoords[f_index])

            'Aggregate All Emoji Segs'
            # Selection of emoji based off position.
            # Grabs segments of EEG data based off flashes for emoji left - to - right.
            for i in range(num_emoji):
                if i == 0:
                    em = eeg_fin[xcoords[presOrder[i]]:pcoords[presOrder[i]], :]
                elif i != 0:
                    em = np.append(em, eeg_fin[xcoords[presOrder[i]]:pcoords[presOrder[i]]], axis=1)
                if i == num_emoji-1:
                    # Once on last emoji, expand dims for agg.
                    em = np.expand_dims(np.copy(em), axis=2)

            print('Pres Order: ', presOrder, 'Targ Cue: ', targCue, 'Emoji Index: ', f_index)

            '=======================Aggregate======================='
            # Aggregate segs over each sequence.
            if j == 0:
                p3aggs = p3segs
                emaggs = em
            elif j != 0:
                p3aggs = np.dstack((p3aggs, p3segs))
                emaggs = np.append(emaggs, em, axis=2)
            print('EM DIMS: ', np.shape(em), 'EMAGG DIMS: ', np.shape(emaggs))

        '=======================Plots======================='
        # Average for plots / 1st across trials / 2nd across channels.
        p3avg = np.average((np.average(p3aggs[:, :], axis=2)), axis=1)
        emavg = np.average(emaggs[:, :, :], axis=2)

        print('EMAVG DIMS: ', np.shape(emavg))

        # emavg = np.average(np.average(emaggs[:, :], axis=2), axis=1)

        # Set up subplots for final sequence in trial.
        if seq_plotter == 1:
            f, axarr = plt.subplots(prows, pcols, sharex=False, sharey=False)
            # Pre-processing Plots.
            axarr[0, 0].plot(eeg_series, eeg_ext)
            axarr[0, 0].set_title('Raw Data: Cz / A2 (Mean Zeroed)')
            axarr[1, 0].plot(eeg_series, eeg_lin)
            axarr[1, 0].set_title('50Hz Notch Filter')
            axarr[2, 0].plot(eeg_series, eeg_A2)
            axarr[2, 0].set_title('A2 Referencing')
            axarr[3, 0].plot(eeg_series, eeg_high)
            axarr[3, 0].set_title('Highpass Filtering')
            axarr[4, 0].plot(eeg_series, eeg_low)
            axarr[4, 0].set_title('Lowpass Filter')

            '-----------------Order Plots---------------------'
            axarr[0, 1].plot(eeg_series[xcoords[0]:pcoords[0]], emavg)
            axarr[0, 1].set_title('All Avg Emoji Segs')
            em0, = axarr[0, 1].plot(pb.simp_temp_axis(
                emavg[:, 0], samp_ratekHz), emavg[:, 0], label='0')
            em1, = axarr[0, 1].plot(pb.simp_temp_axis(
                emavg[:, 1], samp_ratekHz), emavg[:, 1], label='1')
            em2, = axarr[0, 1].plot(pb.simp_temp_axis(
                emavg[:, 2], samp_ratekHz), emavg[:, 2], label='2')
            em3, = axarr[0, 1].plot(pb.simp_temp_axis(
                emavg[:, 3], samp_ratekHz), emavg[:, 3], label='3')
            em4, = axarr[0, 1].plot(pb.simp_temp_axis(
                emavg[:, 4], samp_ratekHz), emavg[:, 4], label='4')
            em5, = axarr[0, 1].plot(pb.simp_temp_axis(
                emavg[:, 5], samp_ratekHz), emavg[:, 5], label='5')
            em6, = axarr[0, 1].plot(pb.simp_temp_axis(
                emavg[:, 6], samp_ratekHz), emavg[:, 6], label='6')
            axarr[0, 1].legend(handles=[em0, em1, em2, em3, em4, em5, em6])

            '-----------------Trial Subplots Plots---------------------'
            axarr[1, 1].plot(eeg_series[xcoords[presOrder[targCue]]                                        :pcoords[presOrder[targCue]]], np.squeeze(p3segs))
            axarr[1, 1].set_title('Sequence P3 Seg')
            axarr[2, 1].plot(eeg_series[xcoords[0]:pcoords[0]], np.squeeze(p3aggs))
            axarr[2, 1].set_title('All P3 Segs in Trial')

            '-----------------Plot Parameters---------------------'
            f.suptitle('Trial: {0} {1} \n Target Cued Emoji: {2} | Target Flash Time Marker: {3}'.format(
                s+1, eeg_files[s], targCue, f_index), fontsize=16)
            f.subplots_adjust(hspace=0.5)
            plt.xticks(np.arange(0, eeg_series[-1], step=100))
            plt.xlabel('Milliseconds (ms)')
            plt.ylabel('Micro-volts (μV)')
            '-----------------Agg Plots---------------------'
            axarr[3, 1].plot(eeg_series[0:len(p3avg)], p3avg)
            axarr[3, 1].set_title('P300 Average')
            '-----------------Flash Lines---------------------'
            for i in range(prows):
                for j in range(num_emoji):
                    axarr[i, 0].axvline(x=xcoords[j]*2, color='g', lw=1.5, ls='dotted')
                axarr[i, 0].axvline(x=xcoords[f_index]*2, color='r', lw=1.5, ls='dashed')
                axarr[i, 0].axvline(x=pcoords[f_index]*2, color='m', lw=1.5, ls='dashed')

        # Aggregate averaged data.
        if s == 0:
            p3fin = p3avg
        elif s != 0:
            p3fin = np.dstack((p3fin, p3avg))
    if seq_plotter == 1:
        plt.show()
    '-----------------Cross Chan / Trial PLots---------------------'
    ax = plt.subplot(1, 1, 1)
    p3fin = np.average(np.squeeze(p3fin), axis=1)
    p3, = ax.plot(pb.simp_temp_axis(p3fin, samp_ratekHz), p3fin)
    ax.legend(handles=[p3])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Microvolts (µV)')
    plt.title('Cross Trial Averages P3 vs Non-P3 \n Num Sess: {0} \n Num Seq: {1} \n Num Trials: {2}'.format(
        num_sess, num_seq, num_trials))
plt.show()


print(
    '\n\nVisual Inspection Questions per plot:\n\n'
    '-Does the Raw Data contain excessive drift or significant 50Hz noise?\n'
    '-Does the 50Hz notch filter clean this uniform spiking noise?\n'
    '-Does the A2 reference correct or enhance signal drift?\n'
    '-Does the highpass filter remove low frequency (sub 1Hz) noise?\n'
    '-Does the lowpass filter remove furry/ spiky high frequenies (>15Hz) and smooth the waveforms?\n'
    '-Can you see a peak around 300-400ms in a any of the averaged segments?\n'
    '-Can you see a P300 for just the final sequence of the trial?\n'
    '-Can you see P300s in any of the P300 sequence segments across the trial?\n'
    '-Can you see an averaged P300 output signal post-processing?\n'
    '\nHave you taken note of any experimental quirks which may explain your findings?\n\n'
    '-Are your impedances higher now than at the start of the trial?\n'
    '-Did your subject show discomfirt or boredom? -> If yes, take a break.\n'
    '-Is the participant doing the task correctly? -> If no, demo task again.\n'
    '-Is your batteries charge >50%? -> If no, change them.\n'
    '\nRecord anything of note in a text file with a corresponding subject, session and experiment identifier.\n'
    '\nIf all is good, save data and labels to a file with subject, session and experiment identifier.\n'
    '\nDelete labels and data from "Data_Acqusition\Data" folder.'

    # Example: 00001_00001_1
    # 1st value ???? = Sub Identifier.
    # 2nd value ???? = Session Identifier.
    # 3rd value ?    = Experiment Identifier, 0 = Flash, 1 = Invert.
    # 00001_00001_1 is Subject 1, Session 1, Invert Experiment.
    # 00005_00003_0 is Subject 5, Session 4, Invert Experiment.
)
