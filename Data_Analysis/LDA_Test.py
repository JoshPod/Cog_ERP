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
dataswitch = 2

if dataswitch == 0:
    # Original Data.
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    print('X DIMS: ', X.shape, '\n', X[0:4, :])
    print('y DIMS: ', y.shape, y[0:4])
    print('Names DIMS: ', target_names.shape, target_names[0:4])
    # Plotting.
    colors = ['navy', 'turquoise', 'darkorange']
    if dataswitch == 1:
        # Random Data.
        X = np.random.random_sample((150, 4)) - 5
elif dataswitch == 1:
    # EEG Dataloading.
    seg = './Seg_Data/seg_data3.npz'
    database = np.load(seg)
    data = database['arr_0']
    # data = pb.scaler2D(data)
    # Swap Axes
    data = np.squeeze(data)
    X = data
    X = np.swapaxes(X, 0, 1)
    label = database['arr_1']
    y = label.astype(int)
    print('DATA DIMS: ', data.shape)
    target_names = np.array(['P300', 'NP300'])
    # Info.
    print('X DIMS: ', X.shape, '\n', X[50:54, 0:4])
    print('y DIMS: ', y.shape, y[0:4])
    print('Names DIMS: ', target_names.shape, target_names[0:4])
    # Plotting.
    colors = ['navy', 'darkorange']
elif dataswitch == 2:
    # Data Pathway Extraction.
    X, y, target_names = pb.lda_loc_extract(
        '..//Data_Aquisition/Data/LocData/J4/', num_trials=100, verbose=1, norm=0)
    # Plotting.
    colors = ['navy', 'darkorange']

# Plotting
colors_n = np.arange(len(target_names))

'Data splitting for test and train'
sss = StratifiedShuffleSplit(n_splits=2, train_size=0.85, random_state=2)
sss.get_n_splits(X, y)
print(sss)

for train_index, test_index in sss.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


'-----PCA-----'
pca_ex = 0
if pca_ex == 1:
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    print('X_r DIMS: ', X_r.shape)
    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))
    # Plots
    plt.figure()
    lw = 2
    for color, i, target_names in zip(colors, colors_n, target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_names)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA Analysis')
    plt.show()

'-----LDA-----'
lda_ex = 1
if lda_ex == 1:
    meth = 'eigen'
    clf = LinearDiscriminantAnalysis(solver=meth, shrinkage='auto',
                                     n_components=2, ).fit(X_train, y_train)
    # Performance.
    if meth == 'eigen':
        print('Explained Covariance Ratio of Components \n: ', clf.explained_variance_ratio_)
    predictions = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)
    print('Size of Test Sample: ', len(y_test))
    print('Actual Labels: ', y_test)
    print('Predictions:   ', predictions)
    print('Predict Probabilities: \n', np.round(pred_proba, decimals=2))
    print('Score: ', score)
    # Plots.
    'Sort plots for iterations improvements.'

'-----LOG-----'
log_ex = 0
if log_ex == 1:
    # X, y = load_iris(return_X_y=True)
    # # Binary Mod for Iris Dataset.
    # y1 = np.zeros(np.int(len(y) / 2))
    # y2 = np.ones(np.int(len(y) / 2))
    # y = np.append(y1, y2)

    clf = LogisticRegression(random_state=2, solver='lbfgs',
                             multi_class='multinomial', penalty='l2').fit(X_train, y_train)
    predictions = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)
    print('Actual Labels: ', y_test)
    print('Predictions:   ', predictions)
    print('Predict Probabilities: \n', pred_proba)
    print('Score: ', score)
    '---------------------------------------------------------------------------------------'
    # Accuracy via cross val.
    clf = LogisticRegression(random_state=2, solver='lbfgs',
                             multi_class='multinomial', penalty='l2').fit(X, y)
    cv = StratifiedShuffleSplit(n_splits=5, train_size=0.85, test_size=0.15, random_state=2)
    accuracy = cross_val_score(clf, X, y, cv=cv)
    print('Accuracy', accuracy)
    print('Mean Accuracy: ', np.mean(accuracy) * 100, '%')
    print('Standard Deviation', "%f" % np.std(accuracy))
