import nibabel as nib
import numpy as np
import os
import torch
from scipy.ndimage.morphology import binary_fill_holes
import scipy.ndimage as spim
from skimage.measure import regionprops
from KCD.Detection.Preprocessing.ObjectFiles import object_creation_utils as ocu
from skimage.segmentation import watershed

# we wan to find the optimum confidence threshold for determining the presence of cancerous voxel labels,
# and the optimum size threshold for determining the presence of a cancerous region
# we will do this by finding the threshold that maximises the dice score on a dataset

cancer_infp = os.path.join('/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions_nii/masked_test_set/[2 2 2]mm_cont')
kidney_infp = os.path.join('/bask/projects/p/phwq4930-renal-canc/data/seg_data/predictions_nii/test_set/[4 4 4]mm')
cancer_label = os.path.join('/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/test_set/cect_labels')
confidence_thresholds = np.append(np.arange(0,0.1,0.01),np.arange(0.1, 0.9, 0.1))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.9,0.98,0.01))
confidence_thresholds = np.append(confidence_thresholds,np.arange(0.98,1.005,0.005))

print(confidence_thresholds)
vol = 400

results =[]

#index 0 is left kidney, index 1 is right kidney
test_labels = {'20.nii.gz':[0,0],
                '74.nii.gz':[0,0],
                '131.nii.gz':[0,0],
                '132.nii.gz':[0,0],
                '133.nii.gz':[0,0],
                '135.nii.gz':[0,0],
                '136.nii.gz':[0,0],
                '137.nii.gz':[0,0],
                '138.nii.gz':[0,0],
                '139.nii.gz':[0,0],
                '189.nii.gz':[0,0],
                '191.nii.gz':[0,0],
                '396.nii.gz':[0,0],
                '397.nii.gz':[0,0],
                '398.nii.gz':[0,0],
                '767.nii.gz':[0,0],
                '772.nii.gz':[0,0],
                'Rcc_002.nii.gz':[0,1],
                'Rcc_005.nii.gz':[1,0],
                'Rcc_009.nii.gz':[0,1],
                'Rcc_010.nii.gz':[1,0],
                'Rcc_012.nii.gz':[1,0],
                'Rcc_018.nii.gz':[0,1],
                'Rcc_021.nii.gz':[0,1],
                'Rcc_022.nii.gz':[0,1],
                'Rcc_024.nii.gz':[0,1],
                'Rcc_026.nii.gz':[0,1],
                'Rcc_029.nii.gz':[1,0],
                'Rcc_048.nii.gz':[1,0],
                'Rcc_056.nii.gz':[1,0],
                'Rcc_063.nii.gz':[1,0],
                'Rcc_065.nii.gz':[1,0],
                'Rcc_070.nii.gz':[1,0],
                'Rcc_073.nii.gz':[1,0],
                'Rcc_077.nii.gz':[0,1],
                'Rcc_079.nii.gz':[0,1],
                'Rcc_080.nii.gz':[1,0],
                'Rcc_086.nii.gz':[0,1],
                'Rcc_091.nii.gz':[1,0],
                'Rcc_092.nii.gz':[1,0],
                'Rcc_094.nii.gz':[0,1],
                'Rcc_097.nii.gz':[1,0],
                'Rcc_098.nii.gz':[0,1],
                'Rcc_105.nii.gz':[1,0],
                'Rcc_106.nii.gz':[0,1],
                'Rcc_109.nii.gz':[1,0],
                'Rcc_110.nii.gz':[1,0],
                'Rcc_112.nii.gz':[1,0],
                'Rcc_119.nii.gz':[1,0],
                'Rcc_130.nii.gz':[0,1],
                'Rcc_133.nii.gz':[1,0],
                'Rcc_135.nii.gz':[0,1],
                'Rcc_137.nii.gz':[0,1],
                'Rcc_139.nii.gz':[0,1],
                'Rcc_159.nii.gz':[1,0],
                'Rcc_163.nii.gz':[1,0],
                'Rcc_165.nii.gz':[1,0],
                'Rcc_169.nii.gz':[0,0],
                'Rcc_175.nii.gz':[0,1],
                'Rcc_184.nii.gz':[0,1],
                'Rcc_187.nii.gz':[0,1],
                'Rcc_191.nii.gz':[1,0],
                'Rcc_196.nii.gz':[1,0],
                'Rcc_202.nii.gz':[0,1]}

#only load files that end in .nii.gz
for conf in confidence_thresholds:
    tp, fp, fn, tn = 0, 0, 0, 0
    stp, sfn = 0, 0 # for small cancers (vol < 33510mm cubed, or 4189 voxels)
    for file in [path for path in os.listdir(cancer_infp) if path.endswith('.nii.gz')]:
        if file == 'Rcc_036.nii.gz': continue
        case_positive_counted = False
        cancer_inf = nib.load(os.path.join(cancer_infp, file))
        kidney_inf = nib.load(os.path.join(kidney_infp, file))
        cancer_lb = nib.load(os.path.join(cancer_label, file))


        left_label,right_label = test_labels[file]
        # #find spacing
        spacing = cancer_inf.header['pixdim'][1:4]

        #resize all to 2mm spacing using torch
        cancer_inf = torch.from_numpy(cancer_inf.get_fdata()).unsqueeze(0).unsqueeze(0)/1000
        kidney_inf = torch.from_numpy(kidney_inf.get_fdata()).unsqueeze(0).unsqueeze(0)
        cancer_lb = torch.from_numpy(cancer_lb.get_fdata()).unsqueeze(0).unsqueeze(0)
        cancer_inf = torch.nn.functional.interpolate(cancer_inf, scale_factor=tuple(spacing/2), mode='trilinear').squeeze()
        kidney_inf = torch.nn.functional.interpolate(kidney_inf, scale_factor=tuple(spacing/2), mode='nearest').squeeze()
        cancer_lb = torch.nn.functional.interpolate(cancer_lb, scale_factor=tuple(spacing/2), mode='nearest').squeeze()
        # print(kidney_inf.shape)
        kidney_inf = torch.flip(kidney_inf, dims=[2])

        # if largest cancer region in cancer_lb is less than 33510mm cubed, or 4189 voxels, than small_cancer = True
        # all none addenbrookes cases should be left == right == 0, i.e. no cancer.
        small_cancer = False
        if right_label==0 and left_label==0: small_cancer = True
        elif not file.startswith('Rcc'): assert False
        elif (cancer_lb==1).sum() < 4189: small_cancer = True




        #find cancerous voxels
        cancer_voxels = cancer_inf > conf
        # fill holes in cancerous voxels
        cancer_voxels = binary_fill_holes(cancer_voxels)
        #filter out regions with less than vol voxels using regionprops and spim
        cancer_voxels = spim.label(cancer_voxels)[0]
        cancer_regions = np.array([region for region in regionprops(cancer_voxels) if region.area > vol])

        # reconstruct cancerous voxels from regions over vol
        cancer_voxels = np.zeros_like(cancer_voxels)
        for region in cancer_regions:
            cancer_voxels[region.coords[:,0],region.coords[:,1],region.coords[:,2]] = 2

        # algorithm: start with kidney inferences - if 2 distinct regions, then simply apply left and right labels, and use cancer_voxels
        # to check answers. if not 2 regions, then check if left and right labels are the same - if so, apply same label to all
        # regions, and use cancer_voxels to check answers. if not, then
        #we can check if 2 of the cancer voxel regions overlap with the ct label - if it does, we presume these
        # are the left and right kidneys, and apply the left and right labels to them. if not, then we have a problem
        #and we need to manually check the inference

        # loop through all cancer predicted regions without a corresponding kidney prediction - if they
        # predict cancer, its a false positive
        for region in regionprops(spim.label(cancer_voxels == 2)[0]):
            inf_region = kidney_inf[region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]]
            if inf_region.max() == 0:
                fp += 1

        # loop through all cancerous labels - checking if true or false positive
        positions = ['right' if region.centroid[0]<kidney_inf.shape[1]//2 else 'left' for region in regionprops(spim.label(kidney_inf == 1)[0])]
        ordered_positions = [left_label if pos == 'left' else right_label for pos in positions]
        for pos,lb,region in zip(positions,ordered_positions,regionprops(spim.label(kidney_inf == 1)[0])):
            prediction_region = cancer_voxels[region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]]

            if prediction_region.max() < 2:
                if lb == 0:
                    tn += 1
                else:
                    fn += 1
                    if small_cancer: sfn += 1
            else:
                # there can only be one true positive region per case -
                # if there are two positive labels, that is a result of incorrect labelling that happens
                # in the event of a false-positive KIDNEY infernece - ie: a bit of the liver has been labelled
                # as kidney, then the kidney on the liver side is also cancerous, so both the erroneous liver section
                # and the cancerous kidney are labelled as cancerous. in this case, we only want to count maximum
                # one true positive, so we only count the first one we find. anything else, we count as a false positive
                if lb == 0:
                    fp += 1
                else:
                    if not case_positive_counted:
                        tp += 1
                        if small_cancer: stp += 1
                        case_positive_counted = True
                    else:
                        fp += 1

    entry = {'confidence_threshold': conf, 'size_threshold': vol, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
             'small_tp': stp, 'small_fn': sfn}
    sensitivity = tp/(tp+fn+1e-6)
    specificity = tn/(tn+fp+1e-6)
    precision = tp/(tp+fp+1e-6)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    entry['sensitivity'] = sensitivity
    entry['small_sensitivity'] = stp/(stp+sfn+1e-6)
    entry['specificity'] = specificity
    entry['precision'] = precision
    entry['small_precision'] = stp/(stp+fp+1e-6)
    entry['accuracy'] = accuracy
    entry['small_accuracy'] = (stp+tn)/(stp+tn+fp+sfn)
    entry['dice'] = 2*tp/(2*tp+fp+fn+1e-6)
    entry['small_dice'] = 2*stp/(2*stp+fp+sfn+1e-6)

    print(entry)
    results.append(entry)

import pandas
import matplotlib.pyplot as plt
path = '/bask/projects/p/phwq4930-renal-canc/data'

df = pandas.DataFrame(results)
df.to_csv(os.path.join(path, 'results_400_test.csv'))

fpr = 100*(1-df['specificity']).copy()
sens = 100*df['sensitivity'].copy()
fpr = np.append(fpr,0)
sens = np.append(sens,0)

ssens = 100*df['small_sensitivity'].copy()
ssens = np.append(ssens,0)

AUC = np.abs(np.trapz(sens,fpr)/1e4)
print('All cancer AUC: {}'.format(AUC))
sAUC = np.abs(np.trapz(ssens,fpr)/1e4)
print('Small cancer AUC: {}'.format(sAUC))

# plot ROC curve
plt.figure()
plt.plot(fpr,sens,label='test (area = %0.3f)' % AUC)
plt.plot(fpr,ssens,label='test small cancers (area = %0.3f)' % sAUC)
plt.ylabel('Sensitivity (%)')
plt.xlabel('1 - Specificity (%)')
plt.title('2-Stage Segmentation-based Detection Test ROC')
plt.legend(loc="lower right")
plt.savefig(os.path.join(path, 'roc_test.png'))
plt.show()