import pandas
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.utils import resample

resnext_3d = pandas.read_csv('/Users/mcgoug01/Downloads/Data/inference/coreg_ncct_split_0/test_set/combined_results_resnext.csv')
effnet_3d = pandas.read_csv('/Users/mcgoug01/Downloads/Data/inference/coreg_ncct_split_0/test_set/combined_results_effnet.csv')

#drop unnamed
resnext_3d = resnext_3d.drop([column for column in resnext_3d.columns if 'unnamed' in column.lower()],axis=1)
resnext_3d = resnext_3d.drop([column for column in resnext_3d.columns if 'unnamed' in column.lower()],axis=1)


#calculate TP, FP, FN, TN at size thresholds
radii = np.arange(0,50,2.5)
size_wise_results = []
#calculate specificity from the combined results

for df, model in zip([resnext_3d,effnet_3d],['ResNeXt 3D','EffNet 3D']):
    prev_rad = 0
    prev_prev_rad= 0
    counter = 0
    size_wise_results.append(
        {'model': model, 'diameter': 0, 'size_threshold': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'Sensitivity': 0})
    for radius in radii:
        size_threshold = (4/3)*np.pi*(radius**3)
        prev_prev_threshold = (4/3)*np.pi*(prev_prev_rad**3)
        print(radius,prev_rad,prev_prev_rad)
        size_df = df[(df['size']<size_threshold) & (df['size']>prev_prev_threshold)]
        mid_radius = (radius+prev_prev_rad)/2

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
            entry['Sensitivity'] = entry['TP']/(entry['TP']+entry['FN'])
        except ZeroDivisionError:

            continue
        size_wise_results.append(entry)
    TN = df[(df['label']==0) & (df['prediction']==0)].shape[0]
    FP = df[(df['label']==0) & (df['prediction']==1)].shape[0]
    FN = df[(df['label']==1) & (df['prediction']==0)].shape[0]
    specificity = TN/(TN+FP)
    print('specificity is {}'.format(specificity))

plt.switch_backend('TkAgg')
size_results_df = pandas.DataFrame(size_wise_results)
#plot sensitivity against diameter for each model
fig, ax = plt.subplots()

xnew = np.linspace(0,80,20)
bootstrap_samples = 1000

for model,m,c in zip(['ResNeXt 3D','EffNet 3D'],['o','^'],['red','blue']):
    model_df = size_results_df[size_results_df['model']==model]
    model_df['total'] = model_df['TP'].copy()+model_df['FN'].copy() + model_df['FP'].copy() + model_df['TN'].copy()
    samples = model_df.total.values
    x = model_df['diameter'].values
    y = model_df['Sensitivity'].values * 100

    samples = model_df.total.values
    lowess_lines = np.zeros((bootstrap_samples, len(x)))
    for i in range(bootstrap_samples):

        sample_indices = resample(np.arange(len(x)), replace=True, n_samples=len(x))
        sample_x, sample_y = x[sample_indices], np.array(y)[sample_indices]
        lowess_lines[i, :] = lowess(sample_y, sample_x, frac=0.3)[:, 1]

    lowess_standard_dev = np.std(lowess_lines, axis=0)
    lowess_SE = lowess_standard_dev / np.sqrt(samples+1e-5)
    lowess_SE=np.where(lowess_SE>100,lowess_SE[1],lowess_SE)
    # Computing the lower and upper bounds of the 95% confidence interval
    lowess_mean = np.mean(lowess_lines,axis=0)
    # find the 95% confidence interval
    lowess_lower = lowess_mean - 1.96 * lowess_SE
    lowess_lower = np.where(lowess_lower<0,0,lowess_lower)
    lowess_upper = lowess_mean + 1.96 * lowess_SE
    lowess_upper = np.where(lowess_upper>100,100,lowess_upper)


    if model == 'ResNeXt 3D':
        ax.scatter(model_df['diameter'], model_df['Sensitivity'] * 100, s=40, marker=m, color=c, label=None, alpha=0.5)
        ax.plot(x, lowess_mean, c=c, alpha=1, label=model)
        ax.fill_between(x, lowess_lower, lowess_upper, alpha=0.2,color=c,label='ResNeXt 95% CI')
ax.legend()
ax.set_xlabel('Diameter (mm)')
ax.set_ylabel('Sensitivity / %')
plt.show(block=True)




