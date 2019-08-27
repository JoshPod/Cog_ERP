
import numpy as np
import ProBoy as pb
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

# Adapted Dataloader for P300 Data.
# 'E://P300_Project/Seg_Data/seg_data3.npz'
seg = './Seg_Data/seg_data3.npz'
database = np.load(seg)
# Arr_0 = eeg data | Samples x Channels x Emoji x Trials
# Arr_1 = labels | Flash Index for Cued Target Emoji for each trial.
# Arr_2 = grounds | ground truth label emoji location.
# Arr_3 = targets | flash target indices.
# Arr_4 = pres | presentation order of flashes.
# Arr_5 = exp | experiment type.
# Arr_6 = sub | subject tag.
# Arr_7 = ses | session tag.
# Arr_8 = seord | session order.
# Arr_9 = desc | data description of level (spatial position of NP3 comparisons).
'Print description of data:'
print('--------------------\n', database['arr_9'], '\n--------------------')

# Extract labels.
label = database['arr_1']
label = label.astype(int)

# Data Features Dimesions.
data = database['arr_0']
num_chans = data.shape[1]
num_samps = data.shape[0]
num_trls = data.shape[2]
num_ses = len(np.unique(database['arr_7']))
num_subs = len(np.unique(database['arr_6']))

# Reshaping Data.
print('OG Data DIMS: ', data.shape, 'Num Samps: ', num_samps,
      'Num Chans: ', num_chans, 'Num Trials: ', num_trls, 'Num Sessions: ', num_ses, 'Num Subjects: ', num_subs)
print('Pre-Reshape Data Dims: ', data.shape)
plt1 = data[:, 0, 0]
# Transform: Samples / Channels / Trials -> Trials / Samples / Channels.
data = np.swapaxes((np.swapaxes(data, 0, 1)), 0, 2)
print('Post-Reshape Data Dims: ', data.shape)
# Post Reshape Dimesnsion Reassignment.
num_chans = data.shape[2]
num_samps = data.shape[1]
num_trls = data.shape[0]

# Sub / Exp / Session Order Extraction.
exp = database['arr_5']
sub = database['arr_6']
seord = database['arr_8']
ses_tag = database['arr_7']
num_trl_per_sess = np.int(np.count_nonzero(ses_tag == 1) / 2)
print('Num Trials Per Session: ', num_trl_per_sess)

'--------------------------------------------------------------------'
'Data Selection Keys for Performance Trouble Shooting.'
# Exp_q @ [0, 1] = both FLASH (0) and Inversion (1) are included.
exp_q = [0, 1]
'ENSURE NUM SUBS IS UP-TO-DATE.'
# Sub_q @  ['1':'13'] = all 13 subjects are extracted.
# '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'
sub_q = ['1']
# Seord_q @ [1, 2, 3, 4] = all sessions are included.
seord_q = [1]  # , 2, 3, 4
'Data Selection'
all = 1  # If all = 1, just grab all the values.
data, label, exp_, sess_, sub_ = pb.expSubSessParser(
    data, label, all, exp, sub, seord, num_trls, exp_q, sub_q, seord_q)
print('Post Extraction Data DIMS: ', data.shape, 'Post Extraction Label Dims: ', label.shape)
'--------------------------------------------------------------------'

# Plot Checking.
plotcheck = 1
for i in range(1):
    if plotcheck == 1:
        # Ensure correct reshaping.
        plt2 = data[0, :, num_chans - 1] + 1
        plt.plot(plt1)
        plt.plot(plt2)
        plt.title('Plot Checking with +1 Offset to Allow Visual Confirmation of reshaping.')
        plt.show()
    if plotcheck == 2:
        # Plot average across each session.
        ses_tag = database['arr_7']

'-----SESSION AVERAGE PLOTTING-----'
plot_avg = 1
if plot_avg == 1:
    print('AVERAGE PLOTTING ACTIVE:')
    unq, p3_indices, np3_indices = pb.sess_plot(data, label, sess_, num_trl_per_sess)
    print('UNQ DIMS: ', unq.shape, unq)
    print('P3 Indices DIMS: ', p3_indices.shape, p3_indices)
    print('NP3 Indices DIMS: ', np3_indices.shape, np3_indices)


'-------Data Manipulation Options--------'
rand_data = 0
if rand_data == 1:
    print('PURE DATA RANDOMISATION ACTIVE:')
    # Data Features Dimesions.
    data = pb.rand_data(data, num_trls, num_samps, num_chans)
    print('Post-Rand Data Dims: ', data.shape)
elif rand_data == 2:
    print('UNIFROM DATA RANDOMISATION ACTIVE:')
    # Data Features Dimesions.
    data = pb.uni_100(data, label, num_trls, num_samps, num_chans,
                      n_con=0.01, zer_m=1, plotter=0)
    print('Post-Rand Data Dims: ', data.shape)


'----Data Normalization----'
norm_data = 1
if norm_data == 1:
    # Plots Init.
    plt.subplot(2, 1, 1)
    pre_pl = data[5, :, :]
    time_axis = pb.simp_temp_axis(pre_pl, 0.5)
    plt.plot(time_axis, pre_pl)
    plt.title('Pre-Normalization')
    # Scaling.
    data = pb.scaler2D(data)
    post_pl = data[5, :, :]
    time_axis = pb.simp_temp_axis(post_pl, 0.5)
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, post_pl)
    plt.title('Post-Normalization')
    plt.show()


'--------------------------------------'
'-------FINAL RESHAPE IMPORTANT--------'
'--------------------------------------'
# Option to execute the LDA.
lin_ex = 1
if lin_ex == 1:
    # Amends to Shape: Trials x Channels x Samples
    data = data.swapaxes(1, 2)
    print('Fin Data DIMS: ', data.shape)
    print('Fin Label DIMS: ', label.shape)

    # Linear Discriminant Analysis
    # Hyper-parameters
    covest = Covariances()  # 'oas'
    ts = TangentSpace()
    # n_split == 43, as we have 43 sessions, however they are random shuffled anyway.
    cv = StratifiedShuffleSplit(n_splits=20, random_state=5, test_size=0.1, train_size=0.9)  #
    print('Strat cv: ', cv)

    clf = make_pipeline(covest, ts, LDA(shrinkage='auto', solver='eigen'))  # print('CLF: ', clf)
    accuracy = cross_val_score(clf, data, label, cv=cv, verbose=1)
    print('Accuracy', accuracy)
    print('Mean Accuracy', "%f" % np.mean(accuracy))
    print('Standard Deviation', "%f" % np.std(accuracy))

'----------------------------------------------------------------------------------'
examp = 0
if examp == 1:
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import make_blobs
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    n_train = 20  # samples for training
    n_test = 200  # samples for testing
    n_averages = 50  # how often to repeat classification
    n_features_max = 75  # maximum number of features
    step = 4  # step size for the calculation

    def generate_data(n_samples, n_features):
        """Generate random blob-ish data with noisy features.

        This returns an array of input data with shape `(n_samples, n_features)`
        and an array of `n_samples` target labels.

        Only one feature contains discriminative information, the other features
        contain only noise.
        """
        X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

        # add non-discriminative features
        if n_features > 1:
            X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
        return X, y

    acc_clf1, acc_clf2 = [], []
    n_features_range = range(1, n_features_max + 1, step)
    for n_features in n_features_range:
        score_clf1, score_clf2 = 0, 0
        for _ in range(n_averages):
            X, y = generate_data(n_train, n_features)
            clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
            clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X, y)

            X, y = generate_data(n_test, n_features)
            score_clf1 += clf1.score(X, y)
            score_clf2 += clf2.score(X, y)

        acc_clf1.append(score_clf1 / n_averages)
        acc_clf2.append(score_clf2 / n_averages)

    features_samples_ratio = np.array(n_features_range) / n_train

    plt.plot(features_samples_ratio, acc_clf1, linewidth=2,
             label="Linear Discriminant Analysis with shrinkage", color='navy')
    plt.plot(features_samples_ratio, acc_clf2, linewidth=2,
             label="Linear Discriminant Analysis", color='gold')

    plt.xlabel('n_features / n_samples')
    plt.ylabel('Classification accuracy')

    plt.legend(loc=1, prop={'size': 12})
    plt.suptitle('Linear Discriminant Analysis vs. \
    shrinkage Linear Discriminant Analysis (1 discriminative feature)')
    plt.show()
