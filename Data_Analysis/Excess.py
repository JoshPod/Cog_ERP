import numpy as np
import ProBoy as pb
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Dataswitcher.
dataswitch = 0

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
elif dataswitch == 2:
    # EEG Dataloading.
    seg = './Seg_Data/seg_data3.npz'
    database = np.load(seg)
    data = database['arr_0']
    data = np.squeeze(data)
    X = np.swapaxes(data, 0, 1)
    label = database['arr_1']
    y = label.astype(int)
    print('DATA DIMS: ', data.shape)
    target_names = np.array(['P300', 'NP300'])
    # Info.
    print('X DIMS: ', X.shape, '\n')
    print('y DIMS: ', y.shape)
    print('Names DIMS: ', target_names.shape, target_names[0:4])
    # Plotting.
    colors = ['navy', 'darkorange']

# Plotting
colors_n = np.arange(len(target_names))

'-----PCA-----'
pca_ex = 1
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
    for color, i, target_name in zip(colors, colors_n, target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')
    plt.show()

'-----LDA-----'
lda_ex = 1
if lda_ex == 1:
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=2)
    X_r2 = lda.fit(X, y).transform(X)
    print('X_r2 DIMS: ', X_r2.shape, X_r2[0:4, 0])
    # Plots
    plt.figure()
    for color, i, target_name in zip(colors, colors_n, target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of IRIS dataset')
    plt.show()
