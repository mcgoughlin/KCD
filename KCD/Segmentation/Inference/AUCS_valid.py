import pandas as pd
import numpy as np
import os
import matplotlib

df = pd.read_csv('/Users/mcgoug01/Downloads/validation_data/results.csv', index_col=0)

ROCs = []
# create an ROC for each size_threshold
for size in df['size_threshold'].unique():
    entry = {'size_threshold':size,
             0.0:(1,1)}
    for confidence in df['confidence_threshold'].unique():
        # get the TP, TN, FP, FN for this size and confidence threshold
        TP = df.loc[(df['size_threshold']==size) & (df['confidence_threshold']==confidence)]['tp'].values[0]
        TN = df.loc[(df['size_threshold']==size) & (df['confidence_threshold']==confidence)]['tn'].values[0]
        FP = df.loc[(df['size_threshold']==size) & (df['confidence_threshold']==confidence)]['fp'].values[0]
        FN = df.loc[(df['size_threshold']==size) & (df['confidence_threshold']==confidence)]['fn'].values[0]

        print(confidence, TP, TN, FP, FN)
        # calculate the TPR and FPR for this size and confidence threshold
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        # add the TPR and FPR to the entry
        entry[confidence] = (FPR, TPR)
    entry[1.0] = (0,0)
    ROCs.append(entry)

# plot the ROCs
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.figure()
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
for ROC in ROCs:
    FPR,TPR= np.array(list(ROC.values())[1:]).T
    print('AUC:',np.abs(np.trapz(TPR,FPR)))
    size_threshold = ROC['size_threshold']
    plt.plot(FPR,TPR,label=f'size_threshold={size_threshold}')

plt.legend()
plt.show()