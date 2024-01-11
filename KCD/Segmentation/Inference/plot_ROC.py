import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import numpy as np
import os
import pandas as pd

path_to_results = '/Users/mcgoug01/Downloads/test_data/results_finer.csv'
df = pd.read_csv(path_to_results)
df = df[df['size_threshold']==400]

fpr = 100*(1-df['specificity'])
sens = 100*df['sensitivity']
fpr = np.append(fpr,0)
sens = np.append(sens,0)

AUC = np.abs(np.trapz(sens,fpr)/1e4)
print(AUC)

#append (0,0) and (100,100) to x and y

# plot ROC curve
plt.figure()
plt.plot(fpr,sens,label='Validation (area = %0.3f)' % AUC)
plt.ylabel('Sensitivity (%)')
plt.xlabel('1 - Specificity (%)')
plt.title('2-Stage Segmentation-based Detection ROC')
plt.legend(loc="lower right")
plt.show()

