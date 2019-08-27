import sys
import numpy as np
import ProBoy as pb
# Unused Modules
# import os
# import random
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines

# ----------------------------------------------------------------------------------
# CrossBoy data organisation script collapses ALL data.
# ----------------------------------------------------------------------------------

# References
'Only non-dynamic, hard-coded section of code is the sample rate of 500Hz.'

# Plot Dependency.
sys.setrecursionlimit(2000)

# Data Pathway Extraction.
direc = '..//Data_Aquisition/Data/Exp_2/'
sub_path, lab_path = pb.subject_extract(direc)
print(np.transpose(sub_path))
print(np.transpose(lab_path))

'------------Extraction Parameters------------'
# Print Labels Info
label_examp, matSize, num_seq, num_emoji, num_trials = pb.labels_inf_extract(
    lab_path[0], '0001_trial_labels.npz')
num_sess = len(sub_path)
num_subs = len(np.unique([item[25:30] for item in sub_path]))
print('Label Example: ', label_examp['arr_1'], 'Label Dims: ', matSize,
      'Num Seqs: ', num_seq, 'Num Emojis: ', num_emoji, 'Num Trials: ', num_trials, 'Num Sess: ', num_sess, 'Num Subs: ', num_subs)

'-------------------------------------------------------------------------------------------------'
'-------------------------------------------------------------------------------------------------'
'CUSTOM EXTRACTION: Edit for different number of seqs or trials to check for participant fatigue.'
num_seq = 5
num_trials = 21
desc = ('Number of Sequences: {0} | Number of Trials: {1}'.format(num_seq, num_trials))
desc = np.array2string(np.copy(desc))
'-------------------------------------------------------------------------------------------------'
'-------------------------------------------------------------------------------------------------'

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
p3store = np.zeros(num_trials*num_sess, dtype=int)*num_seq
# Flash Event Indices.
xcoords = np.linspace(0, seq_dur-aug_dur, num=num_emoji)/temp_axis_multi
xcoords = xcoords.astype(int)
print('xcoords / Flash Temp Marker: ', xcoords*temp_axis_multi, 'Sample Indices: ', xcoords)
pcoords = (xcoords + p3samp)  # ADD buffer to collect full P300 later.
print('pcoords / P3 Temp Marker: ', pcoords*temp_axis_multi, 'Sample Indcies: ', pcoords)


# Flash vs Inverted.
for i in range(num_sess):  # num_sess
    print('+++++++Session Number: ', i+1)
    # Data Pathway
    dat_files = pb.pathway_extract(sub_path[i], '.npz', 'volt', full_ext=0)
    eeg_files = pb.path_subplant(sub_path[i], np.copy(dat_files), 0)
    print('EEG Files DIMS: ', np.shape(eeg_files), 'EEG Files: ', eeg_files)
    # Labels
    grn_files = pb.pathway_extract(lab_path[i], '.npz', 'trial', full_ext=0)
    lab_files = pb.path_subplant(lab_path[i], np.copy(grn_files), 0)
    print('Lab Files DIMS: ', np.shape(lab_files), 'Lab Files: ', lab_files)

    for j in range(num_trials):
        print('+++++++Trial Number: ', j+1)
        # Print Data Info
        _data = np.load(eeg_files[j])
        print('EEG File_Name', eeg_files[j])
        # Print Labels Info
        _labs = np.load(lab_files[j])
        print('LABS File_Name', lab_files[j])
        order = _labs['arr_0']
        # Extract Targ Cued for each seqence.
        targs = _labs['arr_1']
        targ_cue = targs[j]
        print('---------TARG CUE: ', targ_cue)

        for k in range(num_seq):
            idx = 'arr_' + str(k)
            data = _data[idx]
            pres_ord = order[k, :]
            print('Pres Ord: ', pres_ord)
            # Accumulate Targ Indices / Flash Position of P3 Cued Emoji/ Experiment Type (Flash vs Inverted)
            exp = sub_path[i]
            exp_type = np.int(exp[-1])
            f_index = pres_ord[targ_cue]
            sub_tag = pb.sub_tag_2(sub_path, i)
            sess_tag = i+1
            sess_ord = pb.sess_tagger(sub_path, i)
            if i+j+k == 0:
                # Experiment Tyoe (Flash vs Inverted)
                ex_list = exp_type
                # Actual Emoji Cued.
                gr_list = targ_cue
                # Cued Emoji Flash Index.
                p3_list = f_index
                # All emoji flash order.
                or_list = pres_ord
                # Subject Tag
                su_list = sub_tag
                # Session Number
                se_list = sess_tag
                # Session Order
                so_list = sess_ord
            else:
                ex_list = np.append(np.copy(ex_list), exp_type)
                gr_list = np.append(np.copy(gr_list), targ_cue)
                p3_list = np.append(np.copy(p3_list), f_index)
                or_list = np.vstack((np.copy(or_list), pres_ord))
                su_list = np.append(np.copy(su_list), sub_tag)
                se_list = np.append(np.copy(se_list), sess_tag)
                so_list = np.append(np.copy(so_list), sess_ord)
            print('+++++ Ex_List: ', np.shape(ex_list))
            print('///// Gr_List: ', np.shape(gr_list))
            print('***** P3_List: ', np.shape(p3_list))
            print('>>>>> Or_List: ', np.shape(or_list))
            print('}}}}} Su_List: ', np.shape(su_list))
            print('===== Se_List: ', np.shape(se_list))
            print('~~~~~ So_List: ', np.shape(so_list))

            'OPTIONS FOR PREPROCESSING'
            eeg_fin = data

            '=======================Segment======================='
            'SEGMENT: using flash events'
            num_chans = np.amin(np.shape(eeg_fin))
            segs = np.zeros((p3samp, num_chans, num_emoji))
            # Selection of emoji based off position.
            # Grabs segments of EEG data based off flashes for emoji left - to - right.
            for m in range(num_emoji):
                if m == 0:
                    segs = eeg_fin[xcoords[m]:pcoords[m], :]
                elif m != 0:
                    segs = np.dstack((segs, eeg_fin[xcoords[m]:pcoords[m]]))
                # print('M iteration cycle: ', m+1, '/ Segs DIMS: ', np.shape(segs))
            '=======================Aggregate======================='
            # Aggregate over sequences.
            segs = np.expand_dims(np.copy(segs), axis=3)
            if k == 0:
                sq_agg = segs
            elif k != 0:
                sq_agg = np.append(np.copy(sq_agg), segs, axis=3)
            print('=====Seg Dims: ', np.shape(segs), 'Seq Agg Dims: ', np.shape(sq_agg))
        # Aggregate over trials.
        sq_agg = np.expand_dims(np.copy(sq_agg), axis=4)
        if j == 0:
            tr_agg = sq_agg
        elif j != 0:
            tr_agg = np.append(np.copy(tr_agg), sq_agg, axis=4)
        print('Trial Agg DIMS: ', np.shape(tr_agg))
    # Aggregate over sessions.
    tr_agg = np.expand_dims(np.copy(tr_agg), axis=5)
    if i == 0:
        se_agg = tr_agg
    elif i != 0:
        se_agg = np.append(np.copy(se_agg), tr_agg, axis=5)
    print('Session Agg DIMS: ', np.shape(se_agg))

# Final Output
print('Num Sessions: ', num_sess)
print('Final Dims: ', np.shape(se_agg))

# Reshaping: Samples / Channels / Emoji / Sequence / Trials / Sessions
ax = np.shape(se_agg)
fn_agg = np.reshape(se_agg, [ax[0], ax[1], ax[2], np.prod(ax[3:])], order='F')
print('Fin Agg Reshape DIMS: ', np.shape(fn_agg))

# Print all P3 targ indices.
p3_list = np.transpose(p3_list)
print('P3 Flash Indices List: ', p3_list)

# Print Pres Order List.
print('Pres Order DIMS: ', np.shape(or_list), 'Pres Order List: ', or_list)

# Print Subject List.
print('Subject Order DIMS: ', np.shape(su_list), 'Subject Order List: ', su_list)

# Print Subject List.
print('Session Tag DIMS: ', np.shape(se_list), 'Session Tag List: ', se_list)

# Generate Session Order List 0:3.
print('Session Order DIMS: ', np.shape(so_list), 'Session Order List: ', so_list)

# Print all trial experiment types.
ex_list = np.transpose(ex_list)
print('Experiment Type List: ', ex_list)

# Print all emojis cued for each trial.
gr_list = np.transpose(gr_list)
print('Emoji Loactions Cued List: ', gr_list)

# Print description of data being analysed, num sequences / trials.
print('------------------------')
print(desc)
print('------------------------')

# Save
np.savez('./Agg_Data/agg_data.npz', fn_agg, p3_list, or_list,
         ex_list, gr_list, su_list, se_list, so_list, desc)
