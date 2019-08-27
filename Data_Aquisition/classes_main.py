# Networking imports
from pylsl import StreamInlet, resolve_stream
# Visual imports
from psychopy import visual, core, clock
# General imports
import os
import sys
import glob
import numpy as np
from datetime import datetime
import platform
if platform.architecture()[1][:7] == 'Windows':
    from win32api import GetSystemMetrics
# Marker imports
import time
import random
from pylsl import StreamInfo, StreamOutlet

# Pytorch imports
# from torch.utils.data import Dataset


class Stimuli(object):
    '''
    Class used as a container for the stimuli of the experiment. The advantage
    of using this object mainly comes in the form of using a single draw() method
    to update all of them and being able to see which stimuli the experiment has.

    METHODS:
        __init__(): Create a list for the items and the labels
        add(stimulus, label): Add an stimulus to the lists with given label
        draw(): Draw all the stimuli on a opened PsychoPy window
        draw_int(imin, imax): Draw SOME stimuli (in the slice of imin:imax)
        see(): Check labels
        swap(pos1, pos2): Swap to stimuli and their labels

    ATTRIBUTES:
        self.items: Contains the stimuli
        self.labels: Contains stimuli's labels
    '''
    # When the object initializes, it creates an empty list to contain stimuli

    def __init__(self):
        self.items = []
        self.labels = []

    # Method to add stimuli
    def add(self, stimulus, label):
        if type(stimulus) == type([]):
            self.items.extend(stimulus)
            self.labels.extend(label)
        else:
            self.items.append(stimulus)
            self.labels.append(label)

    # Method to update all stimuli 'simultaneously'
    def draw(self):
        for i in range(len(self.items)):
            self.items[i].draw()

    # Draw some stimuli in an interval given by imin-imax
    def draw_int(self, imin, imax):
        for i in range(len(self.items[imin:imax])):
            self.items[imin + i].draw()

    # Draw one stimulus (fixation)
    def draw_one(self, stim):
        self.items[stim].draw()

    # See which stimuli are contained
    def see(self):
        print('Labels (in order): {0}'.format(self.labels))

    # Swap the place of two stimuli, since the drawing is done from first to last
    def swap(self, pos1, pos2):
        self.items[pos1], self.items[pos2] = self.items[pos2], self.items[pos1]
        self.labels[pos1], self.labels[pos2] = self.labels[pos2], self.labels[pos1]

    # Invert the order of the stimuli
    def invert(self):
        self.items = self.items[::-1]
        self.labels = self.labels[::-1]

    # Put an stimulus first in the list
    def first(self, position):
        for i in range(position):
            self.swap(i, position)

    # Put an stimulus lastin the list
    def last(self, position):
        self.invert()
        self.first(-position)
        self.invert()


class LslMarker(object):

    '''
    First create a new stream info (here we set the name to MyMarkerStream,
    the content-type to Markers, 1 channel, irregular sampling rate,
    and string-valued data) The last value would be the locally unique
    identifier for the stream as far as available, e.g.
    program-scriptname-subjectnumber (you could also omit it but interrupted
    connections wouldn't auto-recover). The important part is that the
    content-type is set to 'Markers', because then other programs will know how
    to interpret the content.

    stream_info = Name / Type / Channel Count / Nominal SRate / Format / Stream ID.

    '''

    def __init__(self, **stream_info):
        self.marker_gen(**stream_info)

    def marker_gen(self, **stream_info):
        # Put the known information of the stream in a tuple. It is better to know as much
        # as possible if more than one kit is running LSL at the same time.
        stream_info_list = []
        for key, val in stream_info.items():
            stream_info_list.append(val)
            # print('Val: ', val)

        # The StreamInfo object stores the declaration of a data stream.
        self.info = StreamInfo(*stream_info_list)
        # print('Self INFO: ', self.info)

        # Create Outlet.
        self.outlet = StreamOutlet(self.info)
        # print('Outlet OG: ', self.outlet)

    def constant_mark(self, outlet):
        # Send constant stream of markers.
        while True:
            # pick a sample to send an wait for a bit
            markernames = ['0', '1', '2', '3', '4', '5', '6']
            marker_choice = [random.choice(markernames)]
            outlet.push_sample(marker_choice)
            time_step = 1
            time.sleep(time_step)

    def push(self, **kwargs):
        '''
        Pushes emoji label into marker stream for time tagging of onset and offset of each ERP event.
        marker_ind must be in **kwargs.

        INPUT:
            kwargs: Extra specifications for the data push from the stream

        OUTPUT:
            the label marker into the marker_stream
        '''

        marker = 'Inside'
        self.outlet.push_sample([marker])


class LslStream(object):
    '''
    This class creates the basic connection between the computer and a Lab Streaming
    Layer data stream. With it connecting is made simpler and pulling and processing
    information directly is made trivial.

    METHODS:
        __init__(**stream_info): Initiates a connection when the class is called
        connect(**stream_info): Connects to a data stream in the network given
                defined by the keyword args
        pull(**kwargs): Pulls a sample from the connected data stream
        chunk(**kwargs): Pulls a chunk of samples from the data stream

    ATTRIBUTES:
        streams: List of found LSL streams in the network
        inlet: Stream inlet used to pull data from the stream
        metainfo: Metadata from the stream
    '''

    def __init__(self, **stream_info):
        self.connect(**stream_info)

    def connect(self, **stream_info):
        '''
        This method connects to a LSL data stream. It accepts keyword arguments that define
        the data stream we are searching. Normally this would be (use keywords given between
        quotes as key for the argument) 'name' (e.g. 'Cognionics Quick-20'), 'type' (e.g. 'EEG'),
        'channels' (e.g. 8), 'freq' (from frequency, e.g. 500), 'dtype' (type of data, e.g.
        'float32'), 'serialn' (e.g. 'quick_20').

        After receiving the information of the stream, the script searches for it in the network
        and resolves it, and then connects to it (or the first one in case there are many, that's
        the reason why one has to be as specific as possible if many instances of LSL are being used
        in the lab). It prints some of the metadata of the data stream to the screen so the user
        can check if it is right, and returns the inlet to be used in other routines.

        INPUT:
            **kwargs: Keyword arguments defining the data stream

        RELATED ATTRIBUTES:
            streams, inlet, metainfo
        '''
        # Put the known information of the stream in a tuple. It is better to know as much
        # as possible if more than one kit is running LSL at the same time.
        stream_info_list = []
        for key, val in stream_info.items():
            stream_info_list.append(key)
            stream_info_list.append(val)

        # Resolve the stream from the lab network
        self.streams = resolve_stream(*stream_info_list)

        # Create a new inlet to read from the stream
        self.inlet = StreamInlet(self.streams[0])

        # Get stream information (including custom meta-data) and break it down
        self.metainfo = self.inlet.info()

    def pull(self, **kwargs):
        '''
        This method pulls data from the connected stream (using more information
        for the pull as given by kwargs).

        INPUT:
            kwargs: Extra specifications for the data pull from the stream

        OUTPUT:
            the data from the stream
        '''
        # Retrieve data from the data stream
        return self.inlet.pull_sample(**kwargs)

    def chunk(self, **kwargs):
        '''
        This method pulls chunks. Uses sames formating as .pull
        '''
        # chunk, timestamp = self.inlet.pull_chunk(**kwargs)
        return self.inlet.pull_chunk(**kwargs)


class LslBuffer(object):
    '''
    This class works like a buffer, or an enhanced list to store data temporally.
    It also stores the data in files when erasing it so you don't lose it but
    you don't lose RAM either.

    METHODS:
        __init__: Create the buffer
        add: Add data from LSL stream (formatted as such)
        take_old: Obtain the oldest part of data and erase it from the buffer
        take_new: Obtain the newest part of data and erase it from the buffer
        flag: Return a bool value indicating if the buffer has a certain size
        clear: Clear the buffer
        save: Save certain buffer data to a file
        zip: Take all the files saved and put them into a single .npz file


    ATTRIBUTES:
        self.items: A list with the data and the timestamps as the last column
        self.save_names: A list with the names of the files used for saving
    '''

    def __init__(self):
        self.items = []
        self.save_names = []    # A string with the names of the savefiles

    def add(self, new):
        data = new[0]
        stamps = new[1]
        for i in range(len(data)):  # Runs over all the moments (time points)
            # Timestamps become another column on the list
            data[i].append(stamps[i])

        self.items.extend(data)

    def take_old(self, ammount, delete=False, **kwargs):
        ''' Take the oldest data in the buffer. Has an option to remove the
        taken data from the buffer. '''

        # Save data to file
        if 'filename' in kwargs:
            self.save(imax=ammount, filename=kwargs['filename'])
        else:
            self.save(imax=ammount)

        # Delete data taken if asked
        if delete == True:
            return_ = self.items[:ammount]
            self.items = self.items[ammount:]
            return return_
        else:
            return self.items[:ammount]

    def take_new(self, ammount, delete=False, **kwargs):
        ''' Take the newest data in the buffer. Has an option to remove the
        taken data from the buffer. '''

        # Save data to file
        if 'filename' in kwargs:
            self.save(imin=ammount, filename=kwargs['filename'])
        else:
            self.save(imin=ammount)

        # Delete data taken if asked
        if delete == True:
            return_ = self.items[-ammount:]
            self.items = self.items[:ammount]
            return return_
        else:
            return self.items[-ammount:]

    def flag(self, size):
        # True if buffer bigger or equal than given size
        return len(self.items) >= size

    def clear(self, names=False):
        self.items = []
        if names == True:
            self.save_names = []

    def save(self, **kwargs):
        '''
        Save part of the buffer to a .npy file

        Arguments:
            imin (kwarg): First index of slice (arrays start with index 0)
            imax (kwarg): Last index of slice (last item will be item imax-1)
            filename (kwarg): Name of the file. Default is buffered_<date and time>
            timestamped (kwarg): Whether or not to timestamp a custom filename. Default is True
        '''

        time_string = datetime.now().strftime('%y%m%d_%H%M%S%f')
        if 'filename' in kwargs:
            if 'timestamp' in kwargs and kwargs['timestamp'] == False:
                file_name = kwargs['filename']
            else:
                file_name = kwargs['filename'] + time_string
        else:
            file_name = 'buffered_' + time_string

        # Save the name to the list of names
        direc = './Data/P_3Data/'
        file_name = direc + file_name
        self.save_names.append(file_name)

        # Save data to file_name.npy file
        if 'imin' in kwargs and 'imax' in kwargs:
            imin = kwargs['imin']
            imax = kwargs['imax']
            np.save(file_name, self.items[imin:imax])
        elif 'imin' in kwargs:
            imin = kwargs['imin']
            np.save(file_name, self.items[imin:])
        elif 'imax' in kwargs:
            imax = kwargs['imax']
            np.save(file_name, self.items[:imax])
        else:
            np.save(file_name, self.items)

    def zip(self, compress=False):
        '''
        Takes all the saved .npy files and turns them into a
        zipped (and compressed if compress = True) .npz file.

        Arguments:
            compress: True if want to use compressed version of
                zipped file.
        '''
        arrays = []
        for name in self.save_names:
            arrays.append(np.load(name + '.npy'))
            os.remove(name + '.npy')

        if compress == False:
            np.savez(self.save_names[0], *arrays)
        else:
            np.savez_compressed(self.save_names[0], *arrays)


class EmojiStimulus(object):
    ''' This object is created to handle every aspect of the visual representation
    of the emoji speller stimulus. It is created to simplify its use in other scripts
    making the readability skyrocket (due to reasons like: not having 200 lines on a
    main script)

    METHODS:
        __init__: Initialises the window and the emoji images and places everything where
            it is supposed to go. Also initialises the augmentation (white rectangle).
            Accepts scalings (window_scaling, motion_scaling, stimulus_scaling) as keyword
            arguments to change the relative size of those parameters with respect to the
            screen size.
        quit: Closes the PsychoPy's window and quits the PsychoPy's core
        experiment_setup: Set-up an experiment with all the neede parameters. Please,
            refer to that method's documentation to see all the arguments and usage.
        shuffle: Create a new random array for random augmentation order
        play_emoji: Draw an augmentation for the emoji in the given position by the
            shuffle array.
        play_sequence: Play an entire sequence of augmentations in the order given
            by the shuffle array
        play: Play the estimuli as set up.


    ATTRIBUTES:
        self.window: The window object of PsychoPy
        self.stimuli: The stimuli object (class defined in this file)
            containing all the stimuli from PsychoPy.
        self.num_emoji: Number of emoji images found
        self.emoji_size: Size of the emoji (in px)
        self.imXaxis: Positions of the emoji along the X axis.
        self.pres_dur: Duration of initial presentation of stimuli
        self.aug_dur: Duration of the augmentations
        self.aug_wait: Time between augmentations
        self.iseqi: Inter Sequence Interval duration
        self.num_seq: Number of sequences per trial
        self.sequence_duration: Time duration of each sequence
        self.aug_shuffle: Shuffled list indicating which emoji is going
            to augment in each sequence.
    '''

    def __init__(self, *kwargs):
        # Get monitor dimensions directly from system and define window
        try:    # For those cases in which user is not using Windows
            monitor_dims = np.array([GetSystemMetrics(0),
                                     GetSystemMetrics(1)])  # Monitor dimensions (px)
        except:
            monitor_dims = np.array([1920, 1080])

        # Monitor refresh rate in Hz
        refresh_rate = 60
        # Number of frames per ms
        min_refresh = ((1000 / refresh_rate) / 100)
        # Window Initialization.
        if 'Full' in kwargs:
            window_scaling = 1
        else:
            window_scaling = 0.5
        # Window dimensions (px).
        window_dims = window_scaling * monitor_dims

        '------------Stimuli Parameters Images'
        # Stimulus scaling parameter.
        if 'Large' in kwargs:
            stimulus_scaling = 0.20
        elif 'Medium' in kwargs:
            stimulus_scaling = 0.10
        elif 'Small' in kwargs:
            stimulus_scaling = 0.05
        else:
            stimulus_scaling = 0.20
        # Dimensions of the stimuli
        self.stimulus_dim = np.round(window_dims[0] * stimulus_scaling)
        # Dimension of emoji.
        self.emoji_size = self.stimulus_dim / 2
        # Create window ONLY when actual experiment begins.
        if 'Exp' in kwargs:
            self.window = visual.Window(
                window_dims, monitor='testMonitor', units='deg')

        # Stimuli holder
        self.stimuli = Stimuli()

        def stim_lister(location, aug):
            # Get a list with the path to the emoticon image files
            stim_path_list = glob.glob(location)
            # Iterate over them to create the stimuli and the labels corresponding to the filename
            for i in range(len(stim_path_list)):
                # Unpack the path string to get just filename without file format
                label = stim_path_list[i].split('\\')[1].split('.')[0]
                # Create the stimuli
                self.stimuli.add(visual.ImageStim(
                    win=self.window, image=stim_path_list[i], units='pix', size=self.emoji_size), label)
            # If stim_list used for emoji set self variables.
            if aug == []:
                self.emoji_path_list = stim_path_list
                self.num_emoji = len(self.emoji_path_list)
            # If stim_list used for aug set self variables.
            if aug == 1:
                self.aug_path_list = stim_path_list

        if 'Flash' in kwargs:
            print('STIM TYPE: ___________INVERT___________')
            '------------Emoticon Images'
            location = 'SVGs\Exp_Stim\Flh\\*.png'
            stim_lister(location, aug=[])
            '------------Cueing Stimuli'
            self.stimuli.add(visual.Circle(win=self.window, units='pix', radius=self.emoji_size / 10,
                                           edges=32, fillColor=[1, 1, 1], lineColor=[-1, -1, -1]), 'circWhite')
            '------------Augmentation Stimuli'
            self.stimuli.add(visual.Rect(win=self.window, units='pix', width=self.emoji_size * 1.5,
                                         height=self.emoji_size * 1.5, fillColor=[1, 1, 1], lineColor=[0, 0, 0]), 'rectWhite')
            # Position across x-axis
            emoji_pos = window_dims[0] * 0.9
            self.imXaxis = np.linspace(
                0 - emoji_pos / 2, 0 + emoji_pos / 2, self.num_emoji)
            for i in range(self.num_emoji):
                self.stimuli.items[i].pos = (self.imXaxis[i], 0)

        elif 'Invert' in kwargs:
            print('STIM TYPE: ___________INVERT___________')
            '------------Emoticon Images'
            location = 'SVGs\Exp_Stim\Inv\\*.png'
            stim_lister(location, aug=[])
            '------------Augment Images'
            aug_loc = 'SVGs\Exp_Stim\Inv\Inverted\\*.png'
            stim_lister(aug_loc, aug=1)
            '------------Cueing Stimuli'
            self.stimuli.add(visual.Circle(win=self.window, units='pix', radius=self.emoji_size / 10,
                                           edges=32, fillColor=[1, 1, 1], lineColor=[-1, -1, -1]), 'circWhite')
            # Positioning across x-axis
            emoji_pos = window_dims[0] * 0.9
            targ_pos = np.linspace(0 - emoji_pos / 2, 0 + emoji_pos / 2, self.num_emoji)
            targ_pos = np.append(targ_pos, targ_pos)
            self.imXaxis = targ_pos

            for i in range(len(targ_pos)):
                self.stimuli.items[i].pos = (self.imXaxis[i], 0)

        # Print Important Stimuli Info
        if 'Details' and 'Exp' in kwargs:
            # print('KWARGS: ', kwargs)
            print('____Key Stimuli Info____')
            print('Monitor dimensions: {0}'.format(monitor_dims))
            print('Min refresh rate: {0} ms'.format(min_refresh))
            print('self.stimuli INFO: ', np.shape(self.stimuli.items), self.stimuli.items)
            print('Emoji path list: ', self.emoji_path_list)
        if 'Invert' in kwargs:
            print('Aug path list: ', self.aug_path_list)

    def quit(self):
        self.window.close()
        core.quit()

    def experiment_setup(self, pres_duration, aug_duration, aug_wait,
                         inter_seq_interval, seq_number, num_trials,
                         inter_trial_int, cue_interval, aug_list):
        '''
        Set-up an emoji stimuli experiment.

        All the units are SI units unless specified.

        Arguments:
            pres_duration: Duration of initial stimuli presentation
            aug_duration: Duration of the augmentation on screen
            aug_wait: Temporal distance between augmentations
            inter_seq_interval: Time between sequences
            seq_number: Number of sequences
            num_trials: Number of trials
            per_augmentations: Percentage (/100) of augmented squares per block

        '''
        # Save experiment parameters in object
        self.pres_dur = pres_duration
        self.aug_dur = aug_duration
        self.aug_wait = aug_wait
        self.iseqi = inter_seq_interval
        self.num_seq = seq_number
        self.num_trials = num_trials
        self.itrint = inter_trial_int
        self.cue_int = cue_interval
        self.aug_non_con = aug_list

        # Compute the duration of the experiment and get the timing of the events
        self.sequence_duration = ((self.aug_dur + self.aug_wait) * self.num_emoji) + self.iseqi
        # augmentation_times = np.linspace(0, self.sequence_duration, self.num_emoji + 1)[:self.num_emoji]

        # Create sequence randomisation array
        self.cue_shuffle()

    def shuffle(self):
        # Randomisation for augmentations
        aug_shuffle = np.arange(
            self.num_emoji * self.num_seq).reshape(self.num_emoji, self.num_seq)
        for i in range(self.num_seq):
            aug_shuffle[:, i] = np.arange(0, self.num_emoji, 1)
            np.random.shuffle(aug_shuffle[:, i])
        self.aug_shuffle = aug_shuffle

    def cue_shuffle(self):
        # Randomisation of the fixation cue shuffle.
        cue_list = []
        for i in range(int(self.num_trials / self.num_emoji)):
            tracker = np.arange(self.num_emoji)
            if i == 0:
                cue_list = tracker
            else:
                cue_list = np.append(cue_list, tracker)
        cue_list = np.resize(cue_list, (self.num_trials))
        np.random.shuffle(cue_list)
        self.fix_shuffle = cue_list

    def play_emoji_inv(self, e, s, t):
        ''' Draw emoji augmentation from sequence s and emoji e'''
        # Draw fixation
        fix_dis = self.emoji_size / 2
        if s == 0 and e == 0:
            # Initialization Period for 1st Sequence 1st Emoji.
            clock.wait(self.pres_dur)
            # Position and Draw Cue.
            self.stimuli.items[-1].pos = (
                self.imXaxis[self.fix_shuffle[t]], -fix_dis)
            self.stimuli.draw_one(-1)
            self.window.flip()
            clock.wait(self.cue_int)
            # Position and Draw Cue.
            self.stimuli.items[-1].pos = (
                self.imXaxis[self.fix_shuffle[t]], -fix_dis)
            self.stimuli.draw_one(-1)
            # Draw Emoji
            self.stimuli.draw_int(0, self.num_emoji)
            # Flip Screen, wait as it's the first cue to appear in sequence.
            # print('Cue Time')
            self.window.flip()
            clock.wait(self.cue_int)
        # Position and Draw Cue.
        self.stimuli.items[-1].pos = (
            self.imXaxis[self.fix_shuffle[t]], -fix_dis)
        self.stimuli.draw_one(-1)
        # Draw emoji.
        self.stimuli.draw_int(0, self.num_emoji)

        'Aug List Technique'
        # Position and Invert Emoji
        em_pos = self.aug_non_con[e, s, t] + self.num_emoji
        self.stimuli.items[em_pos].pos = (
            self.imXaxis[self.aug_non_con[e, s, t]], 0)
        self.stimuli.draw_one(em_pos)
        # Window flip
        self.window.flip()
        # Wait the aug_dur time
        clock.wait(self.aug_dur)
        # Position and Draw Cue.
        self.stimuli.items[-1].pos = (
            self.imXaxis[self.fix_shuffle[t]], -fix_dis)
        self.stimuli.draw_one(-1)
        # Draw just the emoji.
        self.stimuli.draw_int(0, self.num_emoji)
        # Window flip
        self.window.flip()
        # Pause aug_wait time
        clock.wait(self.aug_wait)

    def play_emoji_fl(self, e, s, t):
        ''' Draw emoji augmentation from sequence s and emoji e'''
        # Draw fixation
        fix_dis = self.emoji_size / 2
        if s == 0 and e == 0:
            # Initialization Period for 1st Sequence 1st Emoji.
            clock.wait(self.pres_dur)
            # Position and Draw Cue.
            self.stimuli.items[-2].pos = (
                self.imXaxis[self.fix_shuffle[t]], -fix_dis)
            self.stimuli.draw_one(-2)
            self.window.flip()
            clock.wait(self.cue_int)
            # Position and Draw Cue.
            self.stimuli.items[-2].pos = (
                self.imXaxis[self.fix_shuffle[t]], -fix_dis)
            self.stimuli.draw_one(-2)
            # Draw Emoji
            self.stimuli.draw_int(0, self.num_emoji)
            # Flip Screen, wait as it's the first cue to appear in sequence.
            self.window.flip()
            clock.wait(self.cue_int)
        # Position and Draw Cue.
        self.stimuli.items[-2].pos = (
            self.imXaxis[self.fix_shuffle[t]], -fix_dis)
        self.stimuli.draw_one(-2)
        # Draw emoji.
        self.stimuli.draw_int(0, self.num_emoji)

        'Aug List Technique'
        # Position and Flash Emoji
        em_pos = -1
        self.stimuli.items[em_pos].pos = (
            self.imXaxis[self.aug_non_con[e, s, t]], 0)
        self.stimuli.draw_one(em_pos)
        # Window flip
        self.window.flip()
        # Wait the aug_dur time
        clock.wait(self.aug_dur)

        # Position and Draw Cue.
        self.stimuli.items[-2].pos = (
            self.imXaxis[self.fix_shuffle[t]], -fix_dis)
        self.stimuli.draw_one(-2)
        # Draw just the emoji.
        self.stimuli.draw_int(0, self.num_emoji)
        # Window flip
        self.window.flip()
        # Pause aug_wait time
        clock.wait(self.aug_wait)

    def play_seq(self, s, t, aug):
        ''' Play sequence number s as aug_shuffle is ordered '''
        if aug == 'Invert':
            for e in range(self.num_emoji):
                self.play_emoji_inv(e, s, t)
            clock.wait(self.iseqi)
        if aug == 'Flash':
            for e in range(self.num_emoji):
                self.play_emoji_fl(e, s, t)
            clock.wait(self.iseqi)


def erp_code():
    # class ERPDataset(Dataset):
    #     '''
    #     Previously used to load and preprocess OPEN source data from Guger 2009, use as
    #     DataLoader skeleton for input into analysis.
    #     '''
    #
    #     def __init__(self, filepath):
    #         # Use load method to load data
    #         self.load(filepath)
    #
    #         # Use the process function
    #         self.preprocess()
    #
    #     def __getitem__(self, index):
    #         ''' Gives an item from the training data '''
    #         return self.train_data[index]
    #
    #     def __len__(self):
    #         return self.train_data.shape[1] + self.test_data.shape[1]
    #
    #     def load(self, filepath):
    #         # This line is mainly to clean the format using only the filepath
    #         data = loadmat(filepath)[filepath.split('\\')[-1].split('.')[-2]][0, 0]
    #
    #         # Extract the train and test data from the void object
    #         self.train_data = data['train']
    #         self.test_data = data['test']
    #
    #     def preprocess(self):
    #         # Use the preprocessing function on both sets of data
    #         self.train_data = preprocess_erp(self.train_data)
    #         self.test_data = preprocess_erp(self.test_data)
    x = []
    return x
