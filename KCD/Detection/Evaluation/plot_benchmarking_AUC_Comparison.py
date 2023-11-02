import os
import numpy as np
import matplotlib.pyplot as plt

# set the plt backend to TKAgg
plt.switch_backend('TKAgg')

folder = '/Users/mcgoug01/Downloads/ROCs'
# extract the names of files in the folder that end with .npy
files = [f for f in os.listdir(folder) if f.endswith('.npy')]

# extract modelname, epochs, pred_window, and reading from files above
# file have the following name structure TileModel_ROC_TileModel_<modelname>_large_<epochs>_<pred_window>_<reading>.npy

modelnames = []
AUCs = []
for f in files:
    modelnames.append(f.split('_')[3])
    #evaluate AUCs
    data = np.load(os.path.join(folder, f))
    AUCs.append(np.trapz(data[:,1], 1-data[:,0]))

#average the AUCs by modelname
unique_modelnames = np.unique(modelnames)
AUCs_avg = []
for i in range(len(unique_modelnames)):
    AUCs_avg.append(np.mean(np.array(AUCs)[np.array(modelnames) == unique_modelnames[i]]))

# create a list of colours for unique modelnames
colours = ['b','g','r','c','m','y','k','w']
plot_colours = colours[:len(unique_modelnames)]

# loop through model names and AUCS and print
for i in range(len(unique_modelnames)):
    print(unique_modelnames[i] + ' {:.3f}'.format(AUCs_avg[i]))
# plot the AUCs with a categorical x-axis with modelnames, with a scatter plot y axis is AUC
plt.scatter(unique_modelnames, AUCs_avg,c=plot_colours)
plt.ylabel('AUC')
plt.xlabel('Model Name',fontsize=15)
plt.tick_params(axis='x', which='major', pad=2, labelsize=15)
plt.tick_params(axis='y', which='major', pad=2, labelsize=15)
plt.ylim(0.5,1.0)
plt.show()