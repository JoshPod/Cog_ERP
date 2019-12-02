import numpy as np
from functions import imp_check

x = np.load('E://P300_Project/Back_Ups/11-04-2019/Data_Aquisition/Data/Labels/0001_trial_labels.npz')
print(x['arr_0'])
print(x['arr_1'])

'Order of presentation, indexed by the ground truth gives you a numeric '

x = np.load('./Data/P_3Data/Em_3/Labels/0001_trial_labels.npz')
print(x['arr_0'].shape)
print(x['arr_1'].shape)
