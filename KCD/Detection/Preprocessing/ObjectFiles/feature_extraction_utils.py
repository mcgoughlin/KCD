import numpy as np

def get_hist(intensity_data,range_=(-20,80)):
    binned,names = np.histogram(intensity_data,range = range_,density= True)
    return binned, names
def generate_mass_labels(entry,mass_vols,
                         mass2kid,index,mass='cancer',upto=10):
    mass_count=0
    for i in range(upto):
        if mass_count>=len(mass_vols):entry['{}_{}_vol'.format(mass,i)] = 0
        elif mass2kid[mass_count]==index: 
            entry['{}_{}_vol'.format(mass,mass_count)] = mass_vols[mass_count]
            mass_count+=1
        else:entry['{}_{}_vol'.format(mass,i)] = 0
        
    return entry
    
def generate_features(case,statistic,curvatures,kidney,index,intensities,
                      is_labelled=False,cancer_vols=None,cyst_vols=None,
                      canc2kid=None,cyst2kid=None,upto=10):
    entry = {}
    entry['case'] = case
    entry['position'] = kidney
    entry['volume'] = statistic[0]
    entry['convexity'] = statistic[1]
    entry['maj_dim'] = statistic[2]
    entry['min_dim'] = statistic[3]
    entry['eigvec1'] = statistic[4]
    entry['eigvec2'] = statistic[5]
    entry['eigvec3'] = statistic[6]

    for bin_, name in zip(*get_hist(curvatures,range_=(-0.5,0.5))): entry['curv'+str(name)] = bin_
    for bin_, name in zip(*get_hist(intensities)):entry['intens'+str(name)] = bin_

    if is_labelled:
        entry = generate_mass_labels(entry,cancer_vols,canc2kid,index,mass='cancer',upto=upto)
        entry = generate_mass_labels(entry,cyst_vols,cyst2kid,index,mass='cyst',upto=upto)

    return entry