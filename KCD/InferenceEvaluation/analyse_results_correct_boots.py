import pandas
import numpy as np
import matplotlib.pyplot as plt
#import 1d gaussian filter
from scipy.ndimage import gaussian_filter1d
from sklearn.utils import resample
import warnings
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")

resnext_3d = pandas.read_csv('/Users/mcgoug01/Downloads/Data/inference/coreg_ncct_split_0/test_set/combined_results_resnext.csv')
effnet_3d = pandas.read_csv('/Users/mcgoug01/Downloads/Data/inference/coreg_ncct_split_0/test_set/combined_results_effnet.csv')
shape = pandas.read_csv('/Users/mcgoug01/Downloads/Data/inference/coreg_ncct_split_0/test_set/combined_results_shape.csv')
tile2d = pandas.read_csv('/Users/mcgoug01/Downloads/Data/inference/coreg_ncct_split_0/test_set/combined_results_2d.csv')

#drop unnamed
resnext_3d = resnext_3d.drop([column for column in resnext_3d.columns if 'unnamed' in column.lower()],axis=1)
effnet_3d = effnet_3d.drop([column for column in effnet_3d.columns if 'unnamed' in column.lower()],axis=1)
shape = shape.drop([column for column in shape.columns if 'unnamed' in column.lower()],axis=1)
tile2d = tile2d.drop([column for column in tile2d.columns if 'unnamed' in column.lower()],axis=1)


# combine the predictions and labels
resnext_3d['model'] = 'ResNeXt 3D'
effnet_3d['model'] = 'EffNet 3D'
shape['model'] = 'Shape Ensemble'
tile2d['model'] = 'EffNet 2D'
tile2d_OR = shape.copy()
tile2d_OR['prediction'] = tile2d['prediction'].astype(int) | shape['prediction'].astype(int)
tile2d_OR['model'] = 'EffNet 2D OR Shape'
combined = pandas.concat([resnext_3d, effnet_3d, shape, tile2d,tile2d_OR], axis=0)

specificities = {}
# for each model, calculate the TN, FP, FN, and TP overall
for df in [resnext_3d,effnet_3d,shape,tile2d,tile2d_OR]:
    TN = df[(df['prediction']==0)&(df['label']==0)].shape[0]
    FP = df[(df['prediction']==1)&(df['label']==0)].shape[0]
    FN = df[(df['prediction']==0)&(df['label']==1)].shape[0]
    TP = df[(df['prediction']==1)&(df['label']==1)].shape[0]

    # calculate the sensitivity and specificity for each model
    sens = 100*TP/(TP+FN)
    spec = 100*TN/(TN+FP)
    specificities[df['model'].values[0]] = spec
    print('{} sensitivity: {:.2f}%, specificity: {:.2f}%'.format(df['model'].values[0],sens,spec))

# generate bootstrap samples of combined data
n_iterations = 5000
n_size = int(len(combined) * 0.8)
radii = np.arange(0,45,5)


size_wise_results = []
for i in range(n_iterations):

    # prepare bootstrap sample
    sample = resample(combined, n_samples=n_size, replace=True)
    # store diameter, label, and prediction for each model
    resnext_3d_sample = sample[sample['model'] == 'ResNeXt 3D']
    effnet_3d_sample = sample[sample['model'] == 'EffNet 3D']
    shape_sample = sample[sample['model'] == 'Shape Ensemble']
    tile2d_sample = sample[sample['model'] == 'EffNet 2D']
    tile2dOR_sample = sample[sample['model'] == 'EffNet 2D OR Shape']

    # calculate TP, FP, FN, TN at size thresholds
    for j,(df, model) in enumerate(zip([resnext_3d_sample,effnet_3d_sample,shape_sample,tile2d_sample,tile2dOR_sample],
                                       ['ResNeXt 3D','EffNet 3D','Shape Ensemble','EffNet 2D','EffNet 2D OR Shape'])):
        prev_rad = 0
        prev_prev_rad= 0
        counter = 0

        for radius in radii:
            prev_prev_threshold = (4/3)*np.pi*(prev_prev_rad**3)
            mid_radius = (radius+prev_prev_rad)/2
            if radius == radii[-1]:
                radius=1000
            size_threshold = (4/3)*np.pi*(radius**3)
            size_df = df[(df['size'] < size_threshold) & (df['size'] > prev_prev_threshold)]
            entry = {'model':model, 'diameter': mid_radius*2}
            healthy_df = size_df[size_df['label']==0]
            entry['size_threshold'] = size_threshold
            entry['TP'] = size_df[size_df['prediction']==1].shape[0]
            entry['FP'] = healthy_df[healthy_df['prediction']==1].shape[0]
            entry['FN'] = size_df[size_df['prediction']==0].shape[0]
            entry['TN'] = healthy_df[healthy_df['prediction']==0].shape[0]
            prev_prev_rad= prev_rad
            prev_rad = radius
            counter+=1
            try:
                entry['Sensitivity'] = 100*entry['TP']/(entry['TP']+entry['FN'])
            except ZeroDivisionError:

                continue
            size_wise_results.append(entry)

size_wise_bootstrapped_df = pandas.DataFrame(size_wise_results)
plt.switch_backend('TkAgg')
fig = plt.figure(figsize=(8,5))

# LIST OF MATPLOTLIB named COLORS with variable darkening
colours = ['red','black','blue']

# for each model, calculate the mean and standard deviation of sensitivity at each diameter
for model,c in zip(['ResNeXt 3D', 'Shape Ensemble', 'EffNet 2D'],
                colours[-4:]):
    model_df = size_wise_bootstrapped_df[size_wise_bootstrapped_df['model']==model]
    model_df['total'] = model_df['TP'].copy()+model_df['FN'].copy() + model_df['FP'].copy() + model_df['TN'].copy()

    # calculate the mean and standard deviation of sensitivity at each diameter
    mean_sensitivity = []
    std_sensitivity = []
    mean_total = []
    for diameter in np.unique(model_df['diameter']):
        diameter_df = model_df[model_df['diameter']==diameter]
        mean_sensitivity.append(np.mean(diameter_df['Sensitivity']))
        std_sensitivity.append(np.std(diameter_df['Sensitivity']))
        mean_total.append(np.mean(diameter_df['total']))

    # divide std by sqrt(n) to get standard error
    std_error = np.array(std_sensitivity)/np.sqrt(np.array(mean_total))

    # find the 95% confidence interval
    ci = 1.96*std_error
    upper = np.array(mean_sensitivity)+ci
    lower = np.array(mean_sensitivity)-ci

    # plot the mean sensitivity with error bars
    plt.plot(np.unique(model_df['diameter']), mean_sensitivity,
             label=model+' (specificity: {:.2f}%)'.format(specificities[model]),color=mcolors.to_rgb(c))

    plt.fill_between(np.unique(model_df['diameter']), lower,
                     upper, alpha=0.2, label=None,color=mcolors.to_rgb(c))
plt.legend()
plt.xlabel('Diameter / mm',fontsize=14)
plt.ylabel('Sensitivity / %',fontsize=14)
plt.savefig('/Users/mcgoug01/Desktop/size_wise_sensitivity.png',dpi=300)
plt.show()



