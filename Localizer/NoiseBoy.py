import numpy as np
import matplotlib.pyplot as plt
# Data Analysis Imports
import sys
sys.path.insert(0, '../../Data_Analysis')
import ProBoy as pb

# Load Data.
database = np.load('./Data/BackUps/RandRoot/rand_root_data.npz')
desc = database['arr_2']
p3 = database['arr_0']
np3 = database['arr_1']
# Expand Dimenions.
p3 = np.expand_dims(p3, axis=1)
np3 = np.expand_dims(np3, axis=1)
print(desc, '| p3: ', p3.shape, '| np3: ', np3.shape)
# Generate noise aug place-holders.
noise_p3 = []
noise_np3 = []
# Num Augmented Trials.
num_trials = 200
for i in range(num_trials - 1):
    # Generate Noise Signal
    # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
    a = 0          # a is the mean of the normal distribution you are choosing from
    b = 10         # b is the standard deviation of the normal distribution
    c = len(p3)    # c is the number of elements you get in array noise
    noise = np.random.normal(a, b, c)
    noise = np.expand_dims(noise, axis=1)
    # Add Noise to P3 and NP3 templates.
    no_p = np.add(np.copy(p3), np.copy(noise))
    no_n = np.add(np.copy(np3), np.copy(noise))
    # Append Noise Augmented P3 and NP3 arrays.
    if i == 0:
        noise_p3 = no_p
        noise_np3 = no_n
    noise_p3 = np.append(noise_p3, no_p, axis=1)
    noise_np3 = np.append(noise_np3, no_n, axis=1)
    # Plot OG vs noised signals.
    plotter = 0
    if plotter == 1:
        plt.subplot(221)
        plt.plot(p3)
        plt.subplot(222)
        plt.plot(noise_p3[:, i])
        plt.subplot(223)
        plt.plot(np3)
        plt.subplot(224)
        plt.plot(noise_np3[:, i])
        plt.show()
# Print Aug Data Dims.
print('noise_p3: ', noise_p3.shape, '| noise_np3: ', noise_np3.shape)
# Append P3 and NP3 data.
aug_data = np.append(noise_p3, noise_np3, axis=1)
print('aug_data: ', aug_data.shape)
# Generate Aug Data Labels
'P3 == 0, NP3 == 1'
p_lab = np.zeros(num_trials)
n_lab = np.ones(num_trials)
labels = np.append(p_lab, n_lab)
labels = labels.astype(int)
print('labels: ', labels.shape, labels[0:5], labels[-5:-1])
# Save Aug Data.
noise_info = '| standard deviation of noise: {0}'.format(b)
desc = np.append(desc, noise_info)
np.savez('./Data/BackUps/RandData/rand_data.npz', aug_data, labels, desc)

'-----DOWNSAMPLING-----'
# Downsample Signal.
down_samp = 0
if down_samp == 1:
    pb.down_S(aug_data, factor=25, plotter=1, verbose=1)

'---2D TSNE Analysis---'
tsneR = 1
if tsneR == 1:
    pb.tSNE_2D(aug_data, labels, n_components=None,
               perplexities=None, learning_rates=None, scaler='min')

'---3D TSNE Analysis---'
plot_3D = 0
if plot_3D == 1:
    data_3D = np.swapaxes(aug_data, 0, 1)
    print('data_3D dims: ', data_3D.shape)
    pb.tSNE_3D(data_3D, labels, n_components=3, init='pca',
               perplexity=10, learning_rate=10, scaler='min', multi=2, verbose=1)

'---LDA Analysis---'
ldaR = 0
if ldaR == 1:
    pb.lda_(aug_data, labels, split=None, div=None,
            num_comp=None, meth='eigen', scaler='min', covmat=None, verbose=1)
    print('LDA NOISE LEVEL: ', b)

'---LOGREG Analysis---'
log_regR = 1
if log_regR == 1:
    pb.log_reg(aug_data, labels, n_splits=2, train_size=0.9, random_state=2,
               solver='lbfgs', multi_chan='OFF', penalty='l2', max_iter=1000,
               cross_val_acc=1, covmat=1, verbose=1)

'---DATA PARSING---'
data_sss = 0
if data_sss == 1:
    X_train, y_train, X_test, y_test = pb.data_parsing(aug_data, labels, n_splits=2,
                                                       train_size=0.9, random_state=2,
                                                       multi_chan='OFF')
