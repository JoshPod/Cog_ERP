# Networking imports
from pylsl import StreamInlet, resolve_stream
# Visual imports
from psychopy import visual, core, clock
# General imports
import glob
import numpy as np
import platform
if platform.architecture()[1][:7] == 'Windows':
    from win32api import GetSystemMetrics
# Marker imports
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
    """
    First create a new stream info (here we set the name to MyMarkerStream,
    the content-type to Markers, 1 channel, irregular sampling rate,
    and string-valued data) The last value would be the locally unique
    identifier for the stream as far as available, e.g.
    program-scriptname-subjectnumber (you could also omit it but interrupted
    connections wouldn't auto-recover). The important part is that the
    content-type is set to 'Markers', because then other programs will know how
    to interpret the content.

    stream_info = Name / Type / Channel Count / Nominal SRate / Format / Stream ID.
    e.g. 'MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536'

    """

    def __init__(self, **kwargs):
        self.marker_gen(**kwargs)

    def marker_gen(self, **kwargs):

        # Put the known information of the stream in a tuple. It is better to know as much
        # as possible if more than one kit is running LSL at the same time.
        kwargs_list_info = []
        for key, val in kwargs.items():
            kwargs_list_info.append(val)

        name = kwargs_list_info[0]
        type = kwargs_list_info[1]
        channel_count = kwargs_list_info[2]
        nominal_srate = kwargs_list_info[3]
        channel_format = kwargs_list_info[4]
        source_id = kwargs_list_info[5]

        print('Stream Info List: ', name, type, channel_count, nominal_srate,
              channel_format, source_id)

        # The StreamInfo object stores the declaration of a data stream.
        self.info = StreamInfo(name, type, channel_count, nominal_srate,
                               channel_format, source_id)
        # Create Outlet.
        self.outlet = StreamOutlet(self.info)

    def push(self, **kwargs):
        """
        Pushes emoji label into marker stream for time tagging of onset and offset of each ERP event.
        marker_ind must be in kwargs.

        INPUT:
            kwargs: Extra specifications for the data push from the stream

        OUTPUT:
            the label marker into the marker_stream
        """

        kwargs_marker = []
        for key, val in kwargs.items():
            kwargs_marker.append(val)

        marker = kwargs_marker[0]
        self.outlet.push_sample([marker])

    def outlet_del(self):
        self.outlet.__del__()


class LslStream(object):
    """
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
    """

    def __init__(self, **stream_info):
        self.connect(**stream_info)

    def connect(self, **stream_info):
        """
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
        """
        # Put the known information of the stream in a tuple. It is better to know as much
        # as possible if more than one kit is running LSL at the same time.
        stream_info_list = []
        for key, val in stream_info.items():
            stream_info_list.append(key)
            stream_info_list.append(val)

        # Resolve the stream from the lab network
        self.streams = resolve_stream(*stream_info_list)

        # Create a new inlet to read from the stream
        self.inlet = StreamInlet(self.streams[0], max_buflen=938)

        # Get stream information (including custom meta-data) and break it down
        self.metainfo = self.inlet.info()

    def pull(self, **kwargs):
        """
        This method pulls data from the connected stream (using more information
        for the pull as given by kwargs).

        INPUT:
            kwargs: Extra specifications for the data pull from the stream

        OUTPUT:
            the data from the stream
        """
        # Retrieve data from the data stream
        return self.inlet.pull_sample(**kwargs)

    def init_pull(self, **kwargs):
        """
        This serves as a sacrificial initialization pull to get the streaming going.
        This method pulls data from the connected stream (using more information
        for the pull as given by kwargs).

        INPUT:
            kwargs: Extra specifications for the data pull from the stream

        OUTPUT:
            the data from the stream
        """

        print('Marker Stream Initializarion Pull: ', self.inlet.pull_sample(**kwargs))

    def chunk(self, **kwargs):
        """
        This method pulls chunks. Uses sames formating as .pull
        """
        # chunk, timestamp = self.inlet.pull_chunk(**kwargs)
        return self.inlet.pull_chunk(**kwargs)

    def mark_check(self, **kwargs):
        # Available Markers.
        ava_markers = self.inlet.samples_available()
        print('Available Markers: ', ava_markers)

    def inlet_del(self, **kwargs):
        self.inlet.__del__()


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
        self.emoji_duration: Time duration of one emoji augmentation
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
            print('STIM TYPE: ___________FLASH___________')
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
        self.emoji_duration = self.aug_dur + self.aug_wait
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

    def play_emoji_inv(self, e, s, t, marker_outlet, marker_inlet, eeg_stream, imp_stream, ammount):
        ''' Draw emoji augmentation from sequence s and emoji e'''
        # Draw fixation
        fix_dis = self.emoji_size / 2
        if t == 0 and s == 0 and e == 0:
            # Initialization Period for 1st Trial, 1st Sequence, 1st Emoji.
            clock.wait(self.pres_dur)
        if s == 0 and e == 0:
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
        'Data Start : EEG / IMP.'
        if e == 0:
            # Ensure you only start collecting at 1st emoji, after cue.
            eeg, eeg_time = eeg_stream.inlet.pull_chunk(max_samples=np.int(ammount * 1.1))
            imp, imp_time = imp_stream.inlet.pull_chunk(max_samples=np.int(ammount * 1.1))
        'Marker Start.'
        # Marker for labelling and time-stamping.
        marker_rand = np.array2string(self.aug_non_con[e, s, t])
        # Push and Pull Marker Denoting Start of Emoji augmentation.
        marker_outlet.push(marker=marker_rand)
        sample1, timestamp1 = marker_inlet.pull(timeout=1)
        if e == 0:
            print('Start Marker Time: ', timestamp1)
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
        'Marker End.'
        # Push and Pull Marker Denoting Start of Emoji augmentation.
        marker_outlet.push(marker=marker_rand)
        sample2, timestamp2 = marker_inlet.pull(timeout=1)
        if e == 6:
            print('End Marker Time: ', timestamp2)
        return sample1, timestamp1, sample2, timestamp2

    def play_emoji_fl(self, e, s, t, marker_outlet, marker_inlet, eeg_stream, imp_stream, ammount):
        ''' Draw emoji augmentation from sequence s and emoji e'''
        # Draw fixation
        fix_dis = self.emoji_size / 2
        if t == 0 and s == 0 and e == 0:
            # Initialization Period for 1st Trial, 1st Sequence, 1st Emoji.
            clock.wait(self.pres_dur)
        if s == 0 and e == 0:
            # Position and Draw Cue.
            self.stimuli.items[-2].pos = (
                self.imXaxis[self.fix_shuffle[t]], -fix_dis)
            self.stimuli.draw_one(-2)
            # Draw Emoji
            self.stimuli.draw_int(0, self.num_emoji)
            # Flip Screen, wait as it's the first cue to appear in sequence.
            self.window.flip()
            clock.wait(self.cue_int)
        'Data Start : EEG / IMP.'
        if e == 0:
            # Ensure you only start collecting at 1st emoji, after cue.
            eeg, eeg_time = eeg_stream.inlet.pull_chunk(max_samples=np.int(ammount * 1.1))
            imp, imp_time = imp_stream.inlet.pull_chunk(max_samples=np.int(ammount * 1.1))
        'Marker Start.'
        # Marker for labelling and time-stamping.
        marker_rand = np.array2string(self.aug_non_con[e, s, t])
        # Push and Pull Marker Denoting Start of Emoji augmentation.
        marker_outlet.push(marker=marker_rand)
        sample1, timestamp1 = marker_inlet.pull(timeout=1)
        if e == 0:
            print('Start Marker Time: ', timestamp1)
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
        'Marker End.'
        # Push and Pull Marker Denoting Start of Emoji augmentation.
        marker_outlet.push(marker=marker_rand)
        sample2, timestamp2 = marker_inlet.pull(timeout=1)
        if e == 6:
            print('End Marker Time: ', timestamp2)
        return sample1, timestamp1, sample2, timestamp2

    def play_seq(self, s, t, aug, marker_outlet, marker_inlet, eeg_stream, imp_stream, ammount):
        ''' Play sequence number s as aug_shuffle is ordered '''
        # Set Marker Label and Marker Timestamp variables.
        seq_sample1 = []
        seq_timestamp1 = []
        seq_sample2 = []
        seq_timestamp2 = []
        if aug == 'Invert':
            for e in range(self.num_emoji):
                sample1, timestamp1, sample2, timestamp2 = self.play_emoji_inv(
                    e, s, t, marker_outlet, marker_inlet, eeg_stream, imp_stream, ammount)
                # Marker Variables.
                seq_sample1 = np.append(seq_sample1, sample1)
                seq_timestamp1 = np.append(seq_timestamp1, timestamp1)
                seq_sample2 = np.append(seq_sample2, sample2)
                seq_timestamp2 = np.append(seq_timestamp2, timestamp2)
            'Data End'
            # Ensure you only finish collecting at 7th emoji, after cue augmentations,
            # before ITIs and allow for final P300 waveform delay to play through.
            clock.wait(self.iseqi)
            eeg, eeg_time = eeg_stream.inlet.pull_chunk(max_samples=ammount)
            imp, imp_time = imp_stream.inlet.pull_chunk(max_samples=ammount)
            # EEG vs Marker Time Check.
            eeg = np.asarray(eeg)
            eeg_time = np.asarray(eeg_time)
            print('----EEG Chunk DIMS: ', eeg.shape)
            print('----EEG Chunk Time DIMS: ', eeg_time.shape)
            print('----EEG Chunk Time Start: ', eeg_time[0], 'EEG Chunk Time End: ', eeg_time[-1])
            print('----Diff between EEG Chunk and Start Timestamps: ',
                  eeg_time[0] - seq_timestamp1[0])
            # Imp vs Marker Time Check.
            imp = np.asarray(imp)
            imp_time = np.asarray(imp_time)
            print('----IMP Chunk DIMS: ', imp.shape)
            print('----IMP Chunk Time DIMS: ', imp_time.shape)
            print('----IMP Chunk Time Start: ', imp_time[0], 'IMP Chunk Time End: ', imp_time[-1])
            print('----Diff between IMP Chunk and Start Timestamps: ',
                  imp_time[0] - seq_timestamp1[0])
            clock.wait(self.iseqi)
        if aug == 'Flash':
            for e in range(self.num_emoji):
                'RUN STIMULUS.'
                sample1, timestamp1, sample2, timestamp2 = self.play_emoji_fl(
                    e, s, t, marker_outlet, marker_inlet, eeg_stream, imp_stream, ammount)
                # Marker Variables.
                seq_sample1 = np.append(seq_sample1, sample1)
                seq_timestamp1 = np.append(seq_timestamp1, timestamp1)
                seq_sample2 = np.append(seq_sample2, sample2)
                seq_timestamp2 = np.append(seq_timestamp2, timestamp2)
            'Data End'
            # Ensure you only finish collecting at 7th emoji, after cue augmentations,
            # before ITIs and allow for final P300 waveform delay to play through.
            clock.wait(self.iseqi)
            eeg, eeg_time = eeg_stream.inlet.pull_chunk(max_samples=ammount)
            imp, imp_time = imp_stream.inlet.pull_chunk(max_samples=ammount)
            # Data vs Marker Time Check
            eeg = np.asarray(eeg)
            eeg_time = np.asarray(eeg_time)
            print('----EEG Chunk DIMS: ', eeg.shape)
            print('----EEG Chunk Time DIMS: ', eeg_time.shape)
            print('----EEG Chunk Time Start: ', eeg_time[0], 'EEG Chunk Time End: ', eeg_time[-1])
            print('----Diff between EEG Chunk and Start Timestamps: ',
                  eeg_time[0] - seq_timestamp1[0])
            # Imp vs Marker Time Check.
            imp = np.asarray(imp)
            imp_time = np.asarray(imp_time)
            print('----IMP Chunk DIMS: ', imp.shape)
            print('----IMP Chunk Time DIMS: ', imp_time.shape)
            print('----IMP Chunk Time Start: ', imp_time[0], 'IMP Chunk Time End: ', imp_time[-1])
            print('----Diff between IMP Chunk and Start Timestamps: ',
                  imp_time[0] - seq_timestamp1[0])
            clock.wait(self.iseqi)
        return seq_sample1, seq_timestamp1, seq_sample2, seq_timestamp2, eeg, eeg_time, imp, imp_time


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
