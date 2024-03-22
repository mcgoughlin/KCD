from tcia_utils import nbia
import pandas as pd
import os

collections = nbia.getCollections()

CT_series_list = []

save_home = '/Users/mcgoug01/Downloads/TCIA'

# tutorial on how to use the NBIA API: https://github.com/kirbyju/TCIA_Notebooks/blob/main/TCIA_REST_API_Downloads.ipynb

# subject_ID - patientID - person
#series - date for a specific person
#study - specific scan taken on a specific study (on a specific date for a specific patient)

#I think we only want to download one study per patient. Maybe we can allow multiple studies, if they are
#seperated by more than 10 years.


for coldict in collections:
    col = coldict['Collection']

    if 'L' in col:
        continue
    if 'PET' in col:
        continue
    if 'MR' in col:
        continue
    if 'ACRIN' in col:
        continue
    if 'C4KC-KiTS' in col:
        continue
    if 'phantom' in col.lower():
        continue
    if 'breast' in col.lower():
        continue
    if 'cmb-crc' in col.lower():
        continue
    if 'cmb-pca' in col.lower():
        continue
    if 'covid' in col.lower():
        continue
    if 'adrenal-acc' in col.lower():
        continue
    if 'cmb-gec' in col.lower():
        continue
    if 'cc-tumor-heterogeneity' == col.lower():
        continue
    if 'cptac-ccrcc' == col.lower():
        continue

    print(col)

    CT_series = nbia.getSeries(collection=col,
                            modality="CT",format='df')

    if isinstance(CT_series, pd.DataFrame):
        mod_sav = save_home + '/CT/' + col
        if os.path.exists(mod_sav):
            print('CT {} already downloaded'.format(col))
        else:
            os.makedirs(mod_sav)
        CT_series = CT_series.sort_values(by='ImageCount',ascending=False)

        for study in CT_series['StudyInstanceUID'].unique():
            study_series = CT_series[CT_series.StudyInstanceUID == study]
            for patient in study_series['PatientID'].unique():
                patient_series = study_series[study_series.PatientID == patient]
                if len(patient_series) <2:
                    continue
                print('Downloading {} series for patient {}'.format(len(patient_series),patient))
                print('Series Descriptions: ' + str(patient_series['SeriesDescription'].unique()))
                for series_desc in patient_series['SeriesDescription'].unique():
                    print('Downloading series: ' + str(series_desc))
                    series = patient_series[patient_series.SeriesDescription == series_desc]
                    pat_save = mod_sav + '/' + str(patient) + '/' + study + '/' + str(series_desc).replace('/','')
                    if not os.path.exists(pat_save):
                        os.makedirs(pat_save)
                    else:continue
                    nbia.downloadSeries(series,number=5,path=pat_save,input_type='df')
                    CT_series_list.append(series)
    print()

CT_all = pd.concat(CT_series_list,axis=0)