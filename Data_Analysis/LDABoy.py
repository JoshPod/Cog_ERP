import numpy as np
import ProBoy as pb
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Log Reg
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
# # Iris Data

# Dataswitcher.
dataswitch = 3

if dataswitch == 0:
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
    # Data Pathway Extraction.
    X, y, target_names = pb.lda_loc_extract(
        '..//Data_Aquisition/Data/LocData/J4/', num_trials=100, verbose=1, norm=0)
    # Plotting.
    colors = ['navy', 'darkorange']
elif dataswitch == 3:
    '---LDA Paramters---'
    num_comp = 2
    '---Data Prep---'
    database = np.load('./SegData/database.npz')
    X = database['arr_0']
    print('X type: ', type(X), 'X DIMS: ', X.shape)
    X = np.squeeze(X)
    X = np.swapaxes(X, 0, 1)
    # Spatial Labels = Arr_2 | Temporal Labels = Arr_3 | Binary Labels = Arr_4
    y = database['arr_2']
    print('y type: ', type(y), 'y DIMS: ', y.shape, y[0:20])
    # Interger labels.
    i_y = pb.int_labels(y)
    '---Plotting Parameters---'
    target_names = np.asarray(['0', '1'])  # , '2', '3', '4', '5', '6'
    num_targ = len(np.unique(y))
    target_names = target_names[0:num_targ]
    print('Target Names DIMS: ', type(target_names), target_names.shape, target_names)
    colors = ['b', 'g']  # , 'r', 'c', 'm', 'y', 'k'
    colors = colors[0:num_targ]
    print('DATABASE: ', database, 'X: ', X.shape, 'y: ', y.shape)

# Plotting
colors_n = np.arange(len(target_names))

'Data splitting for test and train'
sss = StratifiedShuffleSplit(n_splits=2, train_size=0.85, random_state=2)
sss.get_n_splits(X, y)
print('SSS Split: ', sss)


'------------------DownSampling.'
# Downsample Signal.
down_samp = 0
down_factor = 2
if down_samp == 1:
    from scipy import signal
    num_trials = np.shape(X)[0]
    num_samps = np.shape(X)[1]
    factor = np.int(num_samps / down_factor)
    print('Factor: ', factor)

    re_X = np.zeros((num_trials, factor))
    for i in range(num_trials):
        re_X[i, :] = signal.resample(X[i, :], factor)
    print('re_X DIMS: ', re_X.shape)
    # Downsampling plots confirmation.
    plt.subplot(211)
    plt.plot(X[0, :])
    plt.subplot(212)
    plt.plot(re_X[0, :])
    plt.show()
    # Assign to X
    X = re_X

'------------------Multi-Chan Train/ Test Parsing.'
multi_chan = 0
if multi_chan == 0:
    for train_index, test_index in sss.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
else:
    X_train = X[:, 0:150]
    X_test = X[:, 150:]
    y_train = y[0: 150]
    y_test = y[150:]


'-----PCA-----'
pca_ex = 0
if pca_ex == 1:
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

'-----LOG-----'
log_ex = 0
if log_ex == 1:
    clf = LogisticRegression(random_state=2, solver='lbfgs',
                             multi_class='ovr', penalty='l2').fit(X_train, y_train)
    predictions = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)
    print('Actual Labels: ', y_test)
    print('Predictions:   ', predictions)
    print('Predict Probabilities: \n', pred_proba)
    print('Score: ', score)
    '---------------------------------------------------------------------------------------'
    cross_val_acc = 1
    if cross_val_acc == 1:
        # Accuracy via cross val.
        clf = LogisticRegression(random_state=2, solver='sag',
                                 multi_class='multinomial', penalty='l2', max_iter=1000).fit(X, y)
        cv = StratifiedShuffleSplit(n_splits=5, train_size=0.85, test_size=0.15, random_state=2)
        accuracy = cross_val_score(clf, X, y, cv=cv)
        print('Accuracy', accuracy)
        print('Mean Accuracy: ', np.mean(accuracy) * 100, '%')
        print('Standard Deviation', "%f" % np.std(accuracy))

'---2D TSNE Analysis---'
tsneR = 0
if tsneR == 1:
    X = np.swapaxes(X, 0, 1)
    print('tsne X Dims: ', X.shape)
    pb.tSNE_2D(X, i_y, n_components=None,
               perplexities=None, learning_rates=None, scaler='min')

'---LDA Analysis---'
ldaR = 1
if ldaR == 1:
    X = np.swapaxes(X, 0, 1)
    pb.lda_(X, i_y, split=5, div=0.9,
            num_comp=None, meth='eigen', scaler='standard', covmat=1, verbose=1)


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
