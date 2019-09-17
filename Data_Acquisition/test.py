# zipr test.py
import os
import numpy as np


def zipr(direc, ext, keyword, full_ext, compress=False):
    '''
    Takes all the saved .npy files and turns them into a
    zipped (and compressed if compress = True) .npz file.

    Arguments:
        direc = get data directory.
        ext = select your file delimiter.
        keyword = unique filenaming word/phrase.
        compress: True if want to use compressed version of
            zipped file.
    '''
    'Example: '
    #

    files = [i for i in os.listdir(direc) if os.path.splitext(i)[1] == ext]
    _files = []
    for j in range(len(files)):
        if files[j].find(keyword) != -1:
            if full_ext != 1:
                _files.append(files[j])
            if full_ext == 1:
                _files.append(direc + files[j])
    filename = _files[0][0:12] + '.npz'

    arrays = []
    for i in range(len(_files)):
        arrays.append(np.load(_files[i]))
        os.remove(_files[i])

    print('Arrays: ', arrays)
    filename = _files[0][0:-4] + '.npz'

    if compress is False:
        np.savez(filename, *arrays)
    else:
        np.savez_compressed(filename, *arrays)


data_direc = './Data/P_3Data/'
zipr(data_direc, ext='.npy', keyword='volt', full_ext=1, compress=False)
