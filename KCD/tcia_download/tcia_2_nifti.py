from KCD.utils import dcm2nii
import os

if __name__ == '__main__':
    home = '/Users/mcgoug01/Downloads/TCIA/CT'

    meta,fold = os.path.split(home)
    save_home = os.path.join(meta, fold + '-NIFTI')
    cect_folder = os.path.join(meta, 'cect_nii')
    ncct_folder = os.path.join(meta, 'ncct_nii')

    if not os.path.exists(save_home):
        os.makedirs(save_home)
    if not os.path.exists(cect_folder):
        os.makedirs(cect_folder)
    if not os.path.exists(ncct_folder):
        os.makedirs(ncct_folder)

    for ds in [folder for folder in os.listdir(home) if os.path.isdir(os.path.join(home, folder))]:
        print(ds)
        ds_path, ds_save = (os.path.join(home, ds),
                            os.path.join(save_home, ds))

        if not os.path.exists(ds_save):
            os.makedirs(ds_save)

        patients = [folder for folder in os.listdir(ds_path)
                    if os.path.isdir(os.path.join(ds_path, folder))]

        for patient in patients:
            patient_path, patient_save = (os.path.join(ds_path, patient),
                                          os.path.join(ds_save, patient))

            if not os.path.exists(patient_save):
                os.makedirs(patient_save)

            studies = [folder for folder in os.listdir(patient_path)
                       if os.path.isdir(os.path.join(patient_path, folder))]

            for study_count,study in enumerate(studies):
                study_path, study_save = (os.path.join(patient_path, study),
                                          os.path.join(patient_save, str(study_count)))

                if not os.path.exists(study_save):
                    os.makedirs(study_save)

                # check both phases are present, if not skip
                if not all([os.path.exists(os.path.join(study_path, phase)) for phase in ['ce','nc']]):
                    continue

                for phase in ['ce','nc']:
                    phase_path, phase_save = (os.path.join(study_path, phase),
                                              os.path.join(study_save, phase+'.nii.gz'))

                    if os.path.exists(phase_save):
                        print('Already converted {}'.format(phase_save))
                        continue

                    dcm_folders = [folder for folder in os.listdir(phase_path)
                                if os.path.isdir(os.path.join(phase_path, folder))]

                    if len(dcm_folders) == 0:
                        print('No folders found in {}'.format(phase_path))
                        continue
                    else:
                        dcm_fold = dcm_folders[0]


                    dcm_path = os.path.join(phase_path, dcm_fold)

                    if not os.path.exists(os.path.split(phase_save)[0]):
                        os.makedirs(os.path.split(phase_save)[0])
                    try:
                        dcm2nii.convert_directory(dcm_path, phase_save, compression=True,
                                                  reorient=True)
                    except:
                        print('Failed to convert directory')
                        continue
                    else:
                        print('Converted {} to {}'.format(dcm_path, phase_save))

                #check both phases have been converted
                if not all([os.path.exists(os.path.join(study_save, phase+'.nii.gz')) for phase in ['ce','nc']]):
                    print('Failed to convert both phases')
                    continue
                else:
                    print('Converted both phases')
                    print()

                #copy ce.nii.gz to cect_folder, nc to ncct folder, under the filename '<patient>_<study_number>.nii.gz'
                filename = '{}_{}.nii.gz'.format(patient, study_count)
                cect_path = os.path.join(cect_folder, filename)
                ncct_path = os.path.join(ncct_folder, filename)

                for phase in ['ce','nc']:
                    phase_path = os.path.join(study_save, phase+'.nii.gz')
                    if phase == 'ce':
                        os.system('cp {} {}'.format(phase_path, cect_path))
                    else:
                        os.system('cp {} {}'.format(phase_path, ncct_path))