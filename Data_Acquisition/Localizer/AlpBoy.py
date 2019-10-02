import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../Data_Analysis')
import ProBoy as pb
'AlpBoy loads Alpha Wave dataset and runs eeg band analysis on it.'
# For each subject we provide a single .mat file containing the complete recording of the session.
# The file is a 2D-matrix where the rows contain the observations at each time sample.
# Columns 2 to 17 contain the recordings on each of the 16 EEG electrodes.
# The first column of the matrix represents the timestamp of each observation.
# Column 18 and 19 contain the triggers for the experimental condition 1 and 2.
# The rows in column 18 (resp. 19) are filled with zeros, except at the timestamp corresponding to the beginning of the block
# for condition 1 (resp. 2), when the row gets a value of one.

'Data info:'
# In each session the experimental period is around 100 seconds long, hence around 50,000 samples as captured at 512Hz.
# Each session contains 10 blocks, 5 eyes open and 5 closed.
# Apprently the data has alrady been grounded to AFZ, just need to pre-process at zero.
# POSSIBLY restrict the EEG elec secltion to Occipital sensors.
# Bewaer aware of the exhaustion level /  possibly restrict to 1st 6 blocks, 3 eyes open / closed.
num_blocks = 10
num_subjects = 20
x = pb.is_odd(2)
print('IS ODD OUTPUT: ', x)

'References:'
# https://zenodo.org/record/2348892
# https://github.com/plcrodrigues/py.ALPHA.EEG.2017-GIPSA

# Load Alpha Waves Data from Mat Files.
direc = './Data/AlpData/'
eeg_files = os.listdir(direc)
print(eeg_files)

for i in range(num_subjects):
    # Load Subject File.
    sub_file = direc + eeg_files[i + 1]
    print('===========SUB FILE: ', sub_file)
    mat = scipy.io.loadmat(sub_file)['SIGNAL']
    print(np.shape(mat))

    # Data Category Extraction.
    stamps = mat[:, 0]
    elec = mat[:, 1:16]
    trig1 = mat[:, 17]  # EYES CLOSED - HIGH ALPHA
    trig2 = mat[:, 18]  # EYES OPEN - LOW ALPHA
    print('STAMPS DIMS: ', np.shape(stamps))
    print('ELEC DIMS: ', np.shape(elec))
    print('TRIG1 DIMS: ', np.shape(trig1))
    print('TRIG2 DIMS: ', np.shape(trig2))

    # Data Pre-Processing.
    'Alp Dataset: TURN INTO DATALOADER / FORMATTER /  HAVE DATA SWITCH:'
    # Electrodes for the ALP Dataset: FP1, FP2, FC5, FC6, FZ, T7, CZ, T8, P7, P3, PZ, P4, P8, O1, OZ, O2.
    eeg = pb.prepro(elec, samp_ratekHz=0.5, zero='ON', ext='no-Ref', elec=[13, 14, 15],
                    filtH='OFF', hlevel=1, filtL='OFF', llevel=10,
                    notc='OFF', notfq=50, ref_ind=-7, ref='OFF', avg='ON')
    print('EEG POST PREPRO DIMS: ', eeg.shape)

    # Grab block indices.
    ind_1 = np.where(trig1 == 1)
    ind_1 = ind_1[0][:]
    ind_2 = np.where(trig2 == 1)
    ind_2 = ind_2[0][:]
    print('Ind 1: ', ind_1)
    print('Ind 2: ', ind_2)
    if ind_1[0] - ind_2[0] > 0:
        print('EYES OPEN 1st | ', 'Ind 2: ', ind_2[0], ' smaller than ', 'Ind 1:', ind_1[0])
        ind_ = ind_1
        ind_1 = ind_2
        ind_2 = ind_
        ind_switch = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        eyes = '----EYES OPEN 1st: '
    else:
        print('EYES CLOSED 1st | ', 'Ind 1: ', ind_1[0], ' smaller than ', 'Ind 2: ', ind_2[0])
        ind_switch = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        eyes = '----EYES CLOSED 1st: '

    ind_ = np.sort(np.append(ind_1, ind_2))
    print('SLICE INDICES: ', ind_)
    # Grab indices data dimension differences.
    time_ranges = np.diff(ind_)
    print('Time Range Diffs: ', time_ranges)

    cl_agg_values = None
    cl_agg_rel_pow = None
    op_agg_values = None
    op_agg_rel_pow = None

    for j in range(num_blocks):
        if j != num_blocks - 1:
            block = eeg[ind_[j]:ind_[j + 1], :]
        elif j == num_blocks - 1:
            block = eeg[ind_[j]:-1, :]
        print(eyes, ind_switch)
        print('--------BLOCK: ', j + 1, '| DIMS: ', np.shape(block))
        '----Sub-Band Analysis----'
        'Band Power Reporting'
        # Option to report absolute band power, with each frequency band as a percentage of the total power of the signal.
        # This is called the relative band power. | Ref : https://raphaelvallat.com/bandpower.html'
        values, rel_pow = pb.band_power_plots(block, sing_plt='OFF', plotter='REL')
        values = np.expand_dims(values, axis=1)
        rel_pow = np.expand_dims(rel_pow, axis=1)
        if ind_switch[j] == 0:
            if j == 0 or cl_agg_values is None:
                cl_agg_values = values
                cl_agg_rel_pow = rel_pow
                print('Ind Switch Value is Closed 1st')
            else:
                cl_agg_values = np.append(cl_agg_values, values, axis=1)
                cl_agg_rel_pow = np.append(cl_agg_rel_pow, rel_pow, axis=1)
        elif ind_switch[j] == 1:
            if j == 0 or op_agg_values is None:
                op_agg_values = values
                op_agg_rel_pow = rel_pow
                print('Ind Switch Value is Open 1st')
            else:
                op_agg_values = np.append(op_agg_values, values, axis=1)
                op_agg_rel_pow = np.append(op_agg_rel_pow, rel_pow, axis=1)
print('AGG CL VALUES: \n', cl_agg_values)
print('AGG OP VALUES: \n', op_agg_values)
print('AGG CL REL POW: \n', cl_agg_rel_pow)
print('AGG OP REL POW: \n', op_agg_rel_pow)

# pb.sub_band_chunk_plot(bloc, divs=2, pow_disp='REL', plotter='OFF')
