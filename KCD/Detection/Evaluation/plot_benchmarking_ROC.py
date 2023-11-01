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
epochs = []
pred_windows = []
readings = []
AUCs = []
for f in files:
    modelnames.append(f.split('_')[3])
    epochs.append(f.split('_')[5])
    pred_windows.append(f.split('_')[6])
    readings.append(f.split('_')[7].split('.')[0])

    #evaluate AUCs
    data = np.load(os.path.join(folder, f))
    AUCs.append(np.trapz(data[:,1], 1-data[:,0]))
#plot the ROC curves for each modelname, epochs, pred_window, and reading on the same plot
#make same modelname curves the same color

# populate list of colours based on len of unique modelnames
colours = ['b','g','r','c','m','y','k','w']
unique_modelnames = np.unique(modelnames)
plot_colour = {}
for i in range(len(unique_modelnames)):
    plot_colour[unique_modelnames[i]] = colours[i]

# populate list of linestyles based on len of unique reading
linestyles = ['-', '--', '-.', ':']
unique_readings = np.unique(readings)
plot_linestyle = {}
for i in range(len(unique_readings)):
    plot_linestyle[unique_readings[i]] = linestyles[i]

#plot the ROC curves for each modelname, and reading on the same plot
for i in range(len(files)):
    # load the data
    data = np.load(os.path.join(folder, files[i]))
    # plot the data
    plt.plot(1-data[:,1], data[:,0], label = modelnames[i] + ' {:.3f}'.format(AUCs[i]), color = plot_colour[modelnames[i]], linestyle = plot_linestyle[readings[i]])
plt.legend()
plt.show()