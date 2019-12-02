import numpy as np
import ProBoy as pb
import matplotlib.pyplot as plt

# AGGBOY, takes data across-subjects from CrossData, to build aggregate array, then performs standard analysis.
'---LDA Paramters---'
num_comp = 2
'---Data Prep---'
subject = ['0001', '0002', '0003', '0004', '0005']
# agg_X = np.zeros(200, np.int(40*5))
for i in range(len(subject)):
    database = np.load('./SegData/Em_3/database{0}.npz'.format(subject[i]))
    X = database['arr_0']
    X = np.squeeze(X)
    X = np.swapaxes(X, 0, 1)
    print('X type: ', type(X), 'X DIMS: ', X.shape)
    # Spatial Labels = Arr_2 | Temporal Labels = Arr_3 | Binary Labels = Arr_4
    y = database['arr_4']
    # Interger labels.
    i_y = pb.int_labels(np.copy(y))
    print('y type: ', type(y), 'y DIMS: ', y.shape)
    # Aggregation
    if i == 0:
        agg_X = X
        agg_y = y
        agg_i_y = i_y
    else:
        agg_X = np.append(agg_X, X, axis=0)
        agg_y = np.append(agg_y, y)
        agg_i_y = np.append(agg_i_y, i_y)
# Data Info
print('AGG_X: ', agg_X.shape)
print('AGG_X: ', agg_y.shape)
print('AGG_X: ', agg_i_y.shape)

'----Class-Balancing----'
class_bal = 0
if class_bal == 1:
    # i.e. if the y string labels are configured to use the binary labels.
    agg_X, agg_y, agg_i_y = pb.class_balance(agg_X, agg_y, balance=1, verbose=1)
    agg_X = np.swapaxes(agg_X, 0, 1)

'---LOGREG Analysis---'
log_regR = 1
if log_regR == 1:
    log_X = np.swapaxes(agg_X, 0, 1)
    pb.log_reg(log_X, agg_i_y, scaler='min', n_splits=2, train_size=0.9, random_state=2,
               solver='lbfgs', multi_chan='OFF', penalty='l2', max_iter=10000,
               cross_val_acc=1, covmat=1, verbose=1)

# Save Aggregate Matrix.
'---P3 vs NP3 Plots---'
plot_versus = 0
if plot_versus == 1:
    import random
    # Preform p3 and np3 agregate arrays.
    p3 = []
    np3 = []
    # Balance P3 and NP3 sample sizes.
    id_p3 = np.where(agg_y == '0')[0][:]
    id_np3 = np.where(agg_y == '1')[0][:]
    random.shuffle(id_np3)
    print(id_p3[0], len(id_p3), id_p3[0:10])
    print(id_np3[0], len(id_np3), id_np3[0:10])
    # Aggregate p3 and np3 signals together.
    for i in range(len(id_p3)):
        p_eeg = np.expand_dims(agg_X[id_p3[i], :], axis=1)
        np_eeg = np.expand_dims(agg_X[id_np3[i], :], axis=1)
        if i == 0:
            p3 = p_eeg
            np3 = np_eeg
        else:
            p3 = np.append(p3, p_eeg, axis=1)
            np3 = np.append(np3, np_eeg, axis=1)
    # Average signals across the trials for p3 and np3
    p3_avg = np.average(p3, axis=1)
    np3_avg = np.average(np3, axis=1)
    print('agg_i_y dims: ', agg_i_y.shape)
    print('p3 dims: ', p3.shape, 'np3 dims: ', np3.shape,
          'p3_avg dims: ', p3_avg.shape, 'np3_avg dims: ', np3_avg.shape)
    # Average Plots.
    x_axis = pb.simp_temp_axis(p3_avg, 0.5)
    plt.plot(x_axis, p3_avg)
    plt.plot(x_axis, np3_avg)
    plt.title('Averages: P3 vs NP3')
    plt.show()

# Save Aggregate Matrix.
saveR = 0
if saveR == 1:
    np.savez('./CrossData/database_Em_7', agg_X, agg_y, agg_i_y)
