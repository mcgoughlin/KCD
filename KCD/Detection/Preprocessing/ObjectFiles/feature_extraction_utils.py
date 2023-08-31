import numpy as np

def get_hist(intensity_data,range_=(-20,80)):
    binned,names = np.histogram(intensity_data,range = range_,density= True)
    return binned, names

def generate_features(case,statistic,kidney,index,intensities,
                      is_labelled=False,cancer_vols=None,cyst_vols=None,
                      canc2kid=None,cyst2kid=None):
    entry = {}
    entry['case'] = case
    entry['volume'] = statistic[0]
    entry['convexity'] = statistic[1]
    entry['maj_dim'] = statistic[2]
    entry['min_dim'] = statistic[3]
    entry['eigvec1'] = statistic[4]
    entry['eigvec2'] = statistic[5]
    entry['eigvec3'] = statistic[6]
    entry['position'] = kidney
    for bin_, name in zip(*get_hist(intensities)):
        name = 'intens'+str(name)
        entry[name] = bin_
    
    canc_count,cyst_count = 0,0
    print('feu.generate_features',case,kidney,canc_count,cyst_count)
    if is_labelled:
        for j in range(len(cancer_vols)):
            if canc2kid[j]==index: 
                entry['cancer_{}_vol'.format(canc_count)] = cancer_vols[j]
                canc_count+=1
            
        for j in range(len(cyst_vols)):
            if cyst2kid[j]==index: 
                entry['cyst_{}_vol'.format(cyst_count)] = cyst_vols[j]
                cyst_count+=1

    return entry

