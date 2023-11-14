import os
import numpy as np
import matplotlib.pyplot as plt

# set the plt backend to TKAgg
plt.switch_backend('TKAgg')

folder = '/Users/mcgoug01/Downloads/CV_ROCs'
# extract the names of files in the folder that end with .npy
files = [f for f in os.listdir(folder) if f.endswith('.npy')]
files.sort(reverse=True)
temp=files[1]
files[1]=files[2]
files[2]=temp

temp = files[-1]
files[-1] = files[-3]
files[-3] = temp

temp = files[-2]
files[-2] = files[-1]
files[-1] = temp

# extract modelname, epochs, pred_window, and reading from files above
# file have the following name structure TileModel_ROC_TileModel_<modelname>_large_<epochs>_<pred_window>_<reading>.npy

modelnames = []
AUCs = []
is_3d = []
for f in files:

    if 'GNN' in f:
        modelnames.append('GNN')
        is_3d.append(0)
    elif 'MLP' in f:
        modelnames.append('MLP')
        is_3d.append(0)
    elif 'Ensemble' in f:
        modelnames.append('Shape Ensemble')
        is_3d.append(0)
    elif 'PatchModel' in f:
        is_3d.append(2)
        if 'RESNEXT' in f:
            modelnames.append('ResNeXt 3D')
        else:
            modelnames.append('EffNet 3D')
    elif 'TileModel' in f:
        is_3d.append(1)
        if 'RESNEXT' in f:
            modelnames.append('ResNeXt 2D')
        else:
            modelnames.append('EffNet 2D')
    else:
        assert False


    #evaluate AUCs
    data = np.load(os.path.join(folder, f))
    AUCs.append(np.trapz(data[:,1], 1-data[:,0]))
#plot the ROC curves for each modelname, epochs, pred_window, and reading on the same plot
#make same modelname curves the same color

# populate list of styles based on len of unique modelnames
styles = ['-', '--',  ':']
unique_modeltypes = np.unique(is_3d)
plot_styles = {}
for i in range(len(unique_modeltypes)):
    plot_styles[unique_modeltypes[i]] = styles[i]

# populate list of colours based on len of unique modelnames
colours = ['b', 'r', 'gainsboro', 'darkgray', 'black']
unique_modelnames = np.unique(modelnames)
plot_colour = {}
colour_count = 2
for i in range(len(unique_modelnames)):
    if 'EffNet' in unique_modelnames[i]:
        plot_colour[unique_modelnames[i]] = colours[0]
        continue
    elif 'ResNeXt' in unique_modelnames[i]:
        plot_colour[unique_modelnames[i]] = colours[1]
        continue
    plot_colour[unique_modelnames[i]] = colours[colour_count]
    colour_count+=1

fig = plt.figure(figsize=(8,5))
#plot the ROC curves for each modelname, and reading on the same plot
for i in range(len(files)):
    # load the data
    f = files[i]
    data = np.load(os.path.join(folder, f))

    if 'GNN' in f:
        mn= 'GNN'
    elif 'MLP' in f:
        mn = 'MLP'
    elif 'Ensemble' in f:
        mn = 'Shape Ensemble'
    elif 'PatchModel' in f:
        if 'RESNEXT' in f:
            mn = 'ResNeXt 3D'
        else:
            mn = 'EffNet 3D'
    elif 'TileModel' in f:
        if 'RESNEXT' in f:
            mn = 'ResNeXt 2D'
        else:
            mn = 'EffNet 2D'

    lab = mn + ' ({:.3f})'.format(AUCs[i])
    # plot the data
    plt.plot((1-data[:,1])*100, data[:,0]*100,
             label = lab, color = plot_colour[mn],
             linestyle = plot_styles[is_3d[i]])
plt.legend(ncol=2)
plt.xlabel('100 - Specificity / %',fontsize=14)
plt.ylabel('Sensitivity / %',fontsize=14)
plt.ylim([-0,102])
plt.xlim([-2,100])


plt.savefig('/Users/mcgoug01/Desktop/ROC.png',dpi=300)
plt.show()