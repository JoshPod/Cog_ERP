import numpy as np
import ProBoy as pb
import matplotlib.pyplot as plt

'''

1) Load good database into SegData, either L1/L2 OR M1/M1.
2) Re-run, save, then load the localizer model generated prior to the respective exp session.
3) Reshape eeg data array into the Samples x Sequences x Trials.
4) For each sequence input each emoji segment into the LDA, saving the predictions.
5) Compare prediction results for each emoji event to find the event with the highest P3-likeness.
6) Compare this final prediction to the actual answer.

'''

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
