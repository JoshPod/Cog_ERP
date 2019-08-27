import numpy as np
import ProBoy as pb
import matplotlib.pyplot as plt
# Unused Modules
# import sys
# import matplotlib.lines as mlines


# -----------------------------------------------------------------------------------
# 3 Boy extract P3 and Randomised Non-P3 data: selective of temp or spatial distance.
# -----------------------------------------------------------------------------------

database = np.load('./Agg_Data/agg_data.npz')

# Arr_0 = eeg data | Samples x Channels x Emoji x Trials
# Arr_1 = labels | Flash Index for Cued Target Emoji for each trial.
# Arr_2 = pres | Order of emoji flash events for each trial.
# Arr_3 = exp | Indexing for session experiment type Flash = 0 / Invert = 1.
# Arr_4 = gro | Actual spatial position of emoji.
# Arr_5 = sub | Subject tested.
# Arr_6 = sess | Session Number across all subjects, 0>40 something.
# Arr_7 = seord | Order of session i.e. 1st, 2nd, 3rd or 4th across entire testing period.
# Arr_8 = desc | description of num seqs and trials collated in Cross_Boy.

data = database['arr_0']
targs = database['arr_1']
pres = database['arr_2']
exp = database['arr_3']
gro = database['arr_4']
sub = database['arr_5']
sess = database['arr_6']
seord = database['arr_7']
desc2 = database['arr_8']

num_trials = np.shape(data)[-1]

print('Data Dims: ', data.shape, 'Targ Dims: ', targs.shape,
      'Pres Dims: ', pres.shape, 'Exp Dims: ', exp.shape, 'Grnd Dims: ', gro.shape)

# NP3 extraction.
# Generate non-cued list of differing spatial distances from the cued emoji.
# e.g. gro = 4, Level 1 could be ngro = 5 OR 3.
# Level 2 could be ngro = 6 OR 2.
# Level 3 could be ngro = 7 OR 1.
# if gro == randomly choose whether to go higher or lower.
# if gro == e.g. 7, you can't select 8, as only goes to 7, must therefore choose a value lower than 7.
levels = [1, 2, 3]

for j in range(len(levels)):  # len(levels)
    # Change level each iteration.
    level = levels[j]
    ngro = np.empty([0], dtype=int)
    ntargs = np.empty([0], dtype=int)
    for i in range(num_trials):
        # Middle of the 0-6 emoji location list = 3
        if gro[i] > 3:
            mes = gro[i]-level
            ngro = np.append(ngro, mes)
        elif gro[i] <= 3:
            mes = gro[i]+level
            ngro = np.append(ngro, mes)
        # NTarg Dev.
        ind = pres[i, ngro[i]]
        ntargs = np.append(ntargs, ind)
        # Print Check
        print('Gro: ', gro[i], '| Targ: ', targs[i], '| NGro: ',
              ngro[i], '| NTargs: ', ntargs[i], '| Pres: ', pres[i])

    # Print Check NP300 values working.
    print(gro[0:10], ngro[0:10])

    # Check that only values between 0-6 are in new ngro array to ensure indexing is viable.
    print('NGro Unique Values: ', np.unique(ngro))

    # Dataset analysis prep.
    # P3 extraction.
    for k in range(num_trials):  # (num_trials)
        p3_ex = data[:, :, targs[k], k]
        pre_p3_ex = np.copy(p3_ex)
        np3_ex = data[:, :, ntargs[k], k]
        pre_np3_ex = np.copy(np3_ex)
        print('Flash Order: ', pres[k], '\n Cued Emoji: ', gro[k],
              'Data Segment Extracted: ', targs[k], '\n Non-Cued Emoji: ', ngro[k], 'Non P3 Data Extracted: ', ntargs[k])
        '--------------------------------------------------------------------'
        '-------------OPTIONS FOR SEGMENT LEVEL PREPROCESSING----------------'
        preprocessor = 1
        plotter = 0
        if preprocessor == 1:
            # Apply all preprocessing steps.
            # A2 ref always -7, as there are 6 variables at end: ACC8, ACC9, ACC10, Packet, Trigger & Time-Stamps.
            'Electrode: 0) Fz, 1) Cz, 2) Pz, 3) P4, 4) P3, 5) O1, 6) O2, 7) A2.'
            print('Pre Processing Data Shape: ', np.shape(p3_ex))
            p3_ex = pb.prepro(p3_ex, samp_ratekHz=0.5, zero='ON', ext='INC-Ref', elec=[1],
                              filtH='ON', hlevel=1, filtL='ON', llevel=25,
                              notc='LOW', notfq=50, ref_ind=-7, ref='A2', avg='OFF')
            np3_ex = pb.prepro(np3_ex, samp_ratekHz=0.5, zero='ON', ext='INC-Ref', elec=[1],
                               filtH='ON', hlevel=1, filtL='ON', llevel=25,
                               notc='LOW', notfq=50, ref_ind=-7, ref='A2', avg='OFF')
            # p3_ex = pb.prepro(np.copy(p3_ex), samp_ratekHz=0.5, zero=1, ext=1, elec=[1],
            #                   ref=-7, notc=1, acref=1, filtH=1, hlevel=1, filtL=1, llevel=10, avg=0)
            # np3_ex = pb.prepro(np.copy(np3_ex), samp_ratekHz=0.5, zero=1, ext=1, elec=[1],
            #                    ref=-7, notc=1, acref=1, filtH=1, hlevel=1, filtL=1, llevel=10, avg=0)
            print('Post Processing Data Shape: ', np.shape(p3_ex))

        elif preprocessor == 0:
            # Apply ONLY: Zeroing / Electrode Extraction / 50Hz Notch.
            print('Pre Processing Data Shape: ', np.shape(p3_ex))
            p3_ex = pb.prepro(np.copy(p3_ex), samp_ratekHz=0.5, zero=1, ext=1, elec=[1],
                              ref=-7, notc=0, acref=1, filtH=0, hlevel=1, filtL=0, llevel=10, avg=0)
            np3_ex = pb.prepro(np.copy(np3_ex), samp_ratekHz=0.5, zero=1, ext=1, elec=[1],
                               ref=-7, notc=0, acref=1, filtH=0, hlevel=1, filtL=0, llevel=10, avg=0)
            print('Post Processing Data Shape: ', np.shape(p3_ex))

        if plotter == 1:
            # Plotting:
            samp_ratekHz = 0.5
            if k < 5:  # Only plot for 1st 5 trials of each level.
                # Only plots the 1st segment processed.
                eeg_series = pb.temp_axis(p3_ex, samp_ratekHz)
                raw_P3, = plt.plot(eeg_series, pb.zero_mean(pre_p3_ex[:, [0]]), label='Raw P3')
                post_P3, = plt.plot(eeg_series, p3_ex, label='Post P3')
                raw_NP3, = plt.plot(eeg_series, pb.zero_mean(
                    pre_np3_ex[:, [0]]), label='Raw NP3')
                post_NP3, = plt.plot(eeg_series, np3_ex, label='Post NP3')
                plt.legend([raw_P3, post_P3, raw_NP3, post_NP3], [
                           'Raw P3', 'Post P3', 'Raw NP3', 'Post NP3'])
                plt.title('Pre-Processing Checks \n Level: {0} \n Trial: {1}'.format(j+1, k+1))
                plt.show()
        '--------------------------------------------------------------------'
        # Dimension Expansion
        p3_ex = np.expand_dims(np.copy(p3_ex), axis=2)
        np3_ex = np.expand_dims(np.copy(np3_ex), axis=2)
        if k == 0:
            p3segs = p3_ex
            np3segs = np3_ex
        else:
            p3segs = np.append(np.copy(p3segs), p3_ex, axis=2)
            np3segs = np.append(np.copy(np3segs), np3_ex, axis=2)
        print('P3 Data Dims: ', p3segs.shape, 'NP3 Data Dims: ', np3segs.shape)
    # Dataset saving at different levels.
    seg_data = np.append(p3segs, np3segs, axis=2)
    # Collapse Reshape Checking.
    if np.array_equal(p3segs[:, 0, 0], seg_data[:, 0, 0]) is True:
        print('-----Formatting Correct.')
        # Label Gen.
        'ONES == P3  /  ZEROS == NP3.'
        labels = np.append(np.ones((np.shape(p3segs)[2])), np.zeros((np.shape(np3segs)[2])))
        labels = labels.astype(int)
        print(np.shape(labels), labels)
        # Additional Parameter Info.
        # Ground Truths and Non-Ground Truths - Emoji's Cued / Non-Cued.
        grtru = np.append(gro, ngro)
        # Targets and Non Targets - Emoji Flash Indices / Non Indices.
        tar = np.append(targs, ntargs)
        # Flash Presentation Order.
        pror = np.append(pres, pres)
        # Experiment Type.
        extp = np.append(exp, exp)
        # Subject Tag.
        stag = np.append(sub, sub)
        # Session Tag.
        sestag = np.append(sess, sess)
        sestag = sestag.astype(int)
        # Session Order.
        sessord = np.append(seord, seord)
        sessord = sessord.astype(int)
        # Saving.  'E://P300_Project/Seg_Data/seg_data{}.npz'
        if j == 0:
            desc1 = 'Level 1 Dataset: '
            desc2 = np.array2string(np.copy(desc2))
            desc = desc1+desc2
            # 'E://P300_Project/Seg_Data/seg_data{}.npz'
            np.savez('./Seg_Data/seg_data{}.npz'.format(str(level)), seg_data,
                     labels, grtru, tar, pror, extp, stag, sestag, sessord, desc)
        elif j == 1:
            desc1 = 'Level 2 Dataset: \n'
            desc = desc1+desc2
            np.savez('./Seg_Data/seg_data{}.npz'.format(str(level)), seg_data,
                     labels, grtru, tar, pror, extp, stag, sestag, sessord, desc)
        elif j == 2:
            desc1 = 'Level 3 Dataset: \n'
            desc = desc1+desc2
            np.savez('./Seg_Data/seg_data{}.npz'.format(str(level)), seg_data,
                     labels, grtru, tar, pror, extp, stag, sestag, sessord, desc)
    else:
        print('-----Formatting Incorrect.')
