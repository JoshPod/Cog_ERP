# System import
import sys
import numpy as np


def dict_bash_kwargs():
    """
    This function returns a dictionary containing the keyword arguments given in bash as
    a dictionary for better handling. The arguments in bash should have the following
    form: argument1=something . If there are ANY spaces there, the computer will interpret
    it as different arguments. One can also put strings like arg="here we can use spaces".
    """

    args = []
    for i in range(len(sys.argv) - 1):
        arg = str(sys.argv[i+1]).split("=")
        args.append(arg)
    return dict(args)


def rowcol_paradigm():
    """
    This function defines a list containing the rowcol paradigm
    of BCI spellers to make it easier to decypher using only
    indices for the columns and rows from 1 to 6 since we have
    36 characters.
    """
    char1 = ["A", "B", "C", "D", "E", "F"]
    char2 = ["G", "H", "I", "J", "K", "L"]
    char3 = ["M", "N", "O", "P", "Q", "R"]
    char4 = ["S", "T", "U", "V", "W", "X"]
    char5 = ["Y", "Z", "0", "1", "2", "3"]
    char6 = ["4", "5", "6", "7", "8", "9"]
    return [char1, char2, char3, char4, char5, char6]


def dataset_probe(dataset):
    print("! DATASET: !")

    feat = dataset["features"]
    rc = dataset["rowcol"]
    flg = dataset["flags"]

    # Check the dataset's characteristics and print them out
    print("Length of features, rowcol and flags: {0}, {1}, {2}".format(
        feat.shape, len(rc), len(flg)))
    print("Length of the first 5 feature arrays:")
    i = 0
    for i in range(5):
        print("\t Data item number {0} has length {1}".format(i, len(feat[i])))
        print("\t \t and rowcol {0} and flag {1}".format(rc[i], flg[i]))
    print("\t Last data item number has length {0}".format(len(feat[-1])))
    print("Each item of the flags has a type {0}".format(type(flg[0])))
    print("\n")


def preprocess_erp(erp_array):
    """
    This function is used to change the format of the ERP data from the
    dataset used to train LDA and networks. It takes the (#channels
    + rowcol + flag) x # features array and turns it into 3 lists, that
    contain different lists for different packets of row/column augmentations.

    INPUT:
        erp_array: An array shape 11 x # data points, being channels 10 the
            rowcol indicator (6 rows and 6 columns ==> 1 to 12) and the channel
            11 the trigger indicator (whether the augmented rowcol is the one
            the user is focusing on (1) or not (0)).
    OUTPUT:
        Dictionary, containing three lists of the same length (number of
            sequences). First one contains lists with the features
    """

    # Initialize the lists that are going to hold all the arrays and values
    features = []
    rowcol = []
    flags = []

    # Start the iterations
    first_index = 0
    for i in range(erp_array.shape[1]):
        # If there's data in the temp list and it is a different rowcol, store it as a feature vector
        if i+1 == erp_array.shape[1] or erp_array[9, i] != erp_array[9, i+1]:
            # Save features of this chunk to list
            features.append(erp_array[0:9, first_index:i+1])
            rowcol.append(erp_array[9, i])
            flags.append(erp_array[10, i])

            # First index for next chunk is next index
            first_index = i+1

    # Here we standarise the features' vectors to have the same
    # normal lengths, because we can find anomalous vectors.

    # The first element is outside of the experiment, giving just baseline information
    #   As such, we delete it
    del(features[0])
    del(rowcol[0])
    del(flags[0])

    # Define standard lengths
    std_len_1 = features[0].shape[1]    # Flash
    std_len_2 = features[1].shape[1]    # Inter Stimuli Interval

    # Check which elements have length equal to the standar length
    index = 0
    while not index >= len(features):
        iter_len = features[index].shape[1]
        # Reshape the elements after sequences to have the same shapes as ISI arrays
        if iter_len != std_len_1 and iter_len != std_len_2:
            if features[index-1].shape[1] == std_len_1:
                features[index] = features[index][:, :std_len_2]
            elif features[index-1].shape[1] == std_len_2:
                features[index] = features[index][:, :std_len_1]
        index += 1

    # Compacting and making feature vectors for the training and testing
    # Start iterating over all the items in the lists
    iter_ = 0
    while not iter_+1 >= len(features):
        # The flatten does the channel concatenation
        features[iter_] = np.append(
            features[iter_], features[iter_ + 1], axis=1).flatten()
        del(features[iter_ + 1])
        del(rowcol[iter_ + 1])
        del(flags[iter_ + 1])
        iter_ += 1

    # Transform into an array
    features = np.asarray(features)
    rowcol = np.asarray(rowcol)
    flags = np.asarray(flags)
    # Return a dictionary with the feature vectors and labels.
    return {"features": features, "rowcol": rowcol, "flags": flags}


def save_sequence(file_name, aug_shuffle, prediction_list, final_prediction, fix_pos, fix_order):
    """
    This function is intended to help save all the information from the order of the
    augmentations and the ground truth to a file. The format can be changed since,
    right now, the format is intended to be easy to read by humans, but can be
    a bit messy in data analysis

    Input:
        name: The name of the file's name the data must be saved to
        aug_shuffle: List with lists of the order of the agumentations.
        prediction_list: List containing the preditions made by the model that
            processes the EEG signal.
        final_prediction: Value containing the position of the final prediction
            in the trial.

    Output:
        No output.

    """
    # Open a file with the given name
    file_object = open(file_name, "w")
    # Now for each sequence
    for s in range(len(aug_shuffle)):
        file_object.write("Sequence {0} has random sequence {1}. Predicted {2}. Cue was {3} \n".format(
            s+1, aug_shuffle[s], prediction_list[s], fix_pos))
    # For the whole trial
    file_object.write("Final Prediction was {0}, Cue was {1}, Cue  order was {2}.\n".format(
        final_prediction, fix_pos, fix_order))


def save_labels(trial_number, direc, file_name, fix_pos, flash_order):
    # Open a file with the given name
    'Trial Cue/ Target'
    'Randomised Position of Cue throughout trial'
    # For the whole trial
    labels = flash_order

    if trial_number <= 9:
        namer = '000'
    elif trial_number > 9 and trial_number <= 99:
        namer = '00'
    elif trial_number > 99 and trial_number <= 999:
        namer = ''
    fin_name = direc+namer+file_name
    np.savez(fin_name, labels, flash_order)


'''
----------------------- Randomisation

r_nums: Generates shuffled list of nums range(num_emoji)
consec_res: Evaluates list for consecutive elements Y/N = 1/0
seq_res: Evaluates for consecutive elements across adjacent lists
non_con_gen: Produces lists (range(num_emoji)) of non-consec elements
non_consec_aug_shuf: Produces n iterations of non_consec lists
num_iter: number of iterations of non-consec rand aug lists pre-gen'd

'''


def rand_nums(num_emoji):
    x = np.arange(num_emoji)
    np.random.shuffle(x)
    x = np.expand_dims(x, axis=1)
    return x


def consec_check(arr):
    res = 0
    for i in range(len(arr)-1):
        if arr[i] == arr[i+1]+1:
            res = 1
        if arr[i] == arr[i+1]-1:
            res = 1
        if arr[i] == arr[i-1]+1:
            res = 1
        if arr[i] == arr[i-1]-1:
            res = 1
    return res


def consec_seq(this_val, next_val):
    res = 0
    # Check if same value
    if this_val == next_val:
        res = 1
    # Check if 1 more.
    if this_val == next_val+1:
        res = 1
    # Check if 1 less.
    if this_val == next_val-1:
        res = 1
    # print(res)
    return res


def seq_gen(num_emoji, num_iter):
    # num_iter = number of lists you want to output.
    for i in range(num_iter):
        # print('------Seq: ', i+1)
        if i == 0:
            conseq = rand_nums(num_emoji)
            while (consec_check(conseq) == 1):
                # print('Working...', conseq)
                conseq = rand_nums(num_emoji)
            # print('Success: ', conseq)
        else:
            x = rand_nums(num_emoji)
            print('Conseq DIMS: ', np.shape(conseq))
            while (consec_check(x) == 1):
                print('Working...', x)
                x = rand_nums(num_emoji)
            conseq = np.append(conseq, x, axis=1)
            print('Success: ', x)
    # print('Final Seq DIMS: ', np.shape(conseq), 'Final Seqs: \n', conseq)
    return conseq


def list_gen(num_emoji, num_seqs, num_trials, num_iter):
    print('List_Gen Working...')
    conseq = seq_gen(num_emoji, num_iter)

    for i in range(num_trials*num_seqs):
        if i == 0:
            fin_con = conseq[:, 0]
            fin_con = np.expand_dims(fin_con, axis=1)
            # print('CONSEQ DIMS: ', np.shape(conseq))
            # print('1st FIN_CON DIMS: ', np.shape(fin_con))
        if i != 0:
            this_val = fin_con[-1, i-1]
            # print('This VAL: ', this_val)
            rand_val = np.random.randint(num_iter, size=1)
            # print('RAND VAL: ', rand_val)
            x = conseq[:, rand_val]
            next_val = x[0]

            while(consec_seq(this_val, next_val) == 1):
                rand_val = np.random.randint(num_iter, size=1)
                x = conseq[:, rand_val]
                next_val = x[0]
                # print('Working...', 'this val: ', this_val, 'next val: ', next_val)

            print('Success!', 'this val: ', this_val, 'next val: ', next_val)
            fin_con = np.append(fin_con, x, axis=1)
    print('FIN CON DIMS: ', np.shape(fin_con), 'FIN_CON: \n', fin_con)
    fin_con = np.reshape(fin_con, (num_emoji, num_seqs, num_trials), order='F')
    # return fin_con
    return fin_con
    ''' Example
    # list_gens = 1
    # if list_gens == 1:
    #     # num_emoji, num_seqs, num_trials, num_iterations of non_con_sec lists.
    #     x = list_gen(7, 5, 4, 100)
    #     print('Final Non-Consecutive Emoji / List DIMS: ', np.shape(x))
    #     print('Final Non-Consecutive Emoji / List DIMS: \n')
    #     print('1st Seq: ', x[:, 0, 1])
    #     print('2nd Seq: ', x[:, 1, 1])
    #     print('3rd Seq: ', x[:, 2, 1])
    #     print('4th Seq: ', x[:, 3, 1])
    #     print('5th Seq: ', x[:, 4, 1]) '''


def stamp_check(data):
    stamps = data[:, -1]
    reference = stamps[0]
    stamps = (stamps-reference)*1000
    trial_dur = stamps[-1]
    return(trial_dur)


def loc_shuffle(num_emoji, num_trials):
    import random
    # Get balanced, shuffled P300 aug_list.
    a = np.ones(np.int(num_trials/2))
    b = np.zeros(np.int(num_trials/2)-1)
    c = np.append(a, b)
    print('a: ', a, 'b: ', b, 'c: ', c)
    random.seed(a=5)
    np.random.shuffle(c)
    c = np.append(0, c)
    if ((num_emoji * num_trials) % 2) != 0:
        c = np.append(c, 0)
    c = c.astype(int)
    return c


def folder_gen(data_filename, labels_filename):
    import os
    # Create Data Directory.
    if not os.path.exists(data_filename):
        os.makedirs(labels_filename)
        print('Directory: ', data_filename, ' Created ')
    else:
        print('Directory: ', data_filename, ' Exists')
    # Create Labels Directory.
    if not os.path.exists(labels_filename):
        os.makedirs(labels_filename)
        print('Directory: ', labels_filename, ' Created ')
    else:
        print('Directory: ', labels_filename, ' Exists')


def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.1):
    'Ref: https://makersportal.com/blog/2018/8/14/real-time-graphing-in-python'
    import matplotlib.pyplot as plt

    # use ggplot style for more sophisticated visuals
    plt.style.use('ggplot')
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data), np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1
