import numpy as np
import ProBoy as pb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Log Reg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
# Dataswitcher.
dataswitch = 2

if dataswitch == 0:
    from sklearn import datasets
    from sklearn.datasets import load_iris
    num_comp = 4
    # Original Data.
    iris = datasets.load_iris()
    X = iris.data
    print('X Iris Example: ', X[0, :])
    y = iris.target
    target_names = iris.target_names
    print('X type: ', type(X), 'X DIMS: ', X.shape)
    print('y DIMS: ', y.shape)
    print('Target Names DIMS: ', type(target_names), target_names.shape, target_names)
    # Plotting.
    colors = ['navy', 'turquoise', 'darkorange']
    if dataswitch == 1:
        # Random Data.
        X = np.random.random_sample((150, 4)) - 5
elif dataswitch == 2:
    '---LDA Paramters---'
    num_comp = 2
    '---Data Prep---'
    database = np.load('./SegData/database.npz')
    X = database['arr_0']
    print('X type: ', type(X), 'X DIMS: ', X.shape)
    X = np.squeeze(X)
    X = np.swapaxes(X, 0, 1)
    # Spatial Labels = Arr_2 | Temporal Labels = Arr_3 | Binary Labels = Arr_4
    y = database['arr_4']
    # Interger labels.
    i_y = pb.int_labels(np.copy(y))
    print('y type: ', type(y), 'y DIMS: ', y.shape, y[0:20])
    '---Plotting Parameters---'
    target_names = np.asarray(['0', '1'])  # , '2', '3', '4', '5', '6'
    num_targ = len(np.unique(y))
    target_names = target_names[0:num_targ]
    print('Target Names DIMS: ', type(target_names), target_names.shape, target_names)
    colors = ['b', 'g']  # , 'r', 'c', 'm', 'y', 'k'
    colors = colors[0:num_targ]
    print('DATABASE: ', database, 'X: ', X.shape, 'y: ', y.shape)

'----Class-Balancing----'
class_bal = 1
if class_bal == 1:
    # i.e. if the y string labels are configured to use the binary labels.
    X, y, i_y = pb.class_balance(X, y, balance=2, verbose=1)
    X = np.swapaxes(X, 0, 1)

'-----DownSampling-----'
# Downsample Signal.
down_samp = 0
if down_samp == 1:
    pb.down_S(X, factor=2, plotter=0, verbose=1)

'-----PCA-----'
pca_ex = 0
if pca_ex == 1:
    # Plotting
    colors_n = np.arange(len(target_names))
    pca = PCA(n_components=num_comp, whiten=True, svd_solver='randomized')
    X_r = pca.fit(X).transform(X)
    # X_r fitted comparison plots.
    for i in range(num_comp - 1):
        plt.scatter(X_r[:, 0], X_r[:, i + 1], s=5, color=colors[i])
    plt.show()
    print('X_r DIMS: ', X_r.shape, X_r)
    # Percentage of variance explained for each components
    print('explained variance ratio (first {0} components): {1}'.format(
        str(num_comp), str(pca.explained_variance_ratio_)))
    # Plots
    plt.figure()
    lw = 2
    for color, i, target_names in zip(colors, colors_n, target_names):

        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color,
                    alpha=.8, lw=lw)  # , label = target_names
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA Analysis')
    plt.show()

'---LOGREG Analysis---'
log_regR = 1
if log_regR == 1:
    log_X = np.swapaxes(X, 0, 1)
    pb.log_reg(log_X, i_y, scaler='min', n_splits=2, train_size=0.9, random_state=2,
               solver='lbfgs', multi_chan='OFF', penalty='l2', max_iter=10000,
               cross_val_acc=1, covmat=1, verbose=1)

'---LDA Analysis---'
ldaR = 0
if ldaR == 1:
    X = np.swapaxes(X, 0, 1)
    pb.lda_(X, i_y, split=5, div=0.9,
            num_comp=None, meth='eigen', scaler='min', covmat=1, verbose=1)

'---2D TSNE Analysis---'
tsneR = 0
if tsneR == 1:
    X = np.swapaxes(X, 0, 1)
    print('tsne X Dims: ', X.shape)
    pb.tSNE_2D(X, i_y, n_components=None,
               perplexities=None, learning_rates=None, scaler='min')

'---3D TSNE Analysis---'
plot_3D = 0
if plot_3D == 1:
    pb.tSNE_3D(X, n_components=3, init='pca',
               perplexity=60, learning_rate=1000, scaler='min', multi=None, verbose=1)

'---P3 vs NP3 Plots---'
plot_versus = 0
if plot_versus == 1:
    import random
    # Preform p3 and np3 agregate arrays.
    p3 = []
    np3 = []
    # Balance P3 and NP3 sample sizes.
    id_p3 = np.where(y == '0')[0][:]
    id_np3 = np.where(y == '1')[0][:]
    random.shuffle(id_np3)
    print(id_p3[0], len(id_p3), id_p3[0:10])
    print(id_np3[0], len(id_np3), id_np3[0:10])
    # Aggregate p3 and np3 signals together.
    for i in range(len(id_p3)):
        p_eeg = np.expand_dims(X[id_p3[i], :], axis=1)
        np_eeg = np.expand_dims(X[id_np3[i], :], axis=1)
        if i == 0:
            p3 = p_eeg
            np3 = np_eeg
        else:
            p3 = np.append(p3, p_eeg, axis=1)
            np3 = np.append(np3, np_eeg, axis=1)
    # Average signals across the trials for p3 and np3
    p3_avg = np.average(p3, axis=1)
    np3_avg = np.average(np3, axis=1)
    print('i_y dims: ', i_y.shape)
    print('p3 dims: ', p3.shape, 'np3 dims: ', np3.shape,
          'p3_avg dims: ', p3_avg.shape, 'np3_avg dims: ', np3_avg.shape)
    # Average Plots.
    x_axis = pb.simp_temp_axis(p3_avg, 0.5)
    plt.plot(x_axis, p3_avg)
    plt.plot(x_axis, np3_avg)
    plt.title('Averages: P3 vs NP3')
    plt.show()
    # # All Plots.
    # plt.plot(x_axis, p3)
    # plt.show()
    # plt.plot(x_axis, np3)
    # plt.show()


chain_pca_log = 0
# makepipeline
# if chain_pca_log == 1:
# from sklearn.pipeline import make_pipeline as Pipeline
# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier()),
# ])
# predicted = pipeline.fit(X_train, y_train).predict(X_train, y_train)
# # Now evaluate all steps on test set
# predicted = pipeline.predict(X_test, y_test)
