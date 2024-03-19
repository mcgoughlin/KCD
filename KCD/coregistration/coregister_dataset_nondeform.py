import SimpleITK as sitk
import os
from utils import start_plot, end_plot, plot_values, display_images, display_images_with_alpha, update_multires_iterations
import matplotlib.pyplot as plt

# linear coreg with https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/60_Registration_Introduction.html


def linear_coreg(moving,fixed,initial_transform,
                 moving_mask=True,fixed_mask=True):
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=1000,
                                                      convergenceMinimumValue=1e-8, convergenceWindowSize=30)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2, 1,])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if moving_mask :
        mmask = sitk.Cast(sitk.IntensityWindowing(moving, windowMinimum=-100, windowMaximum=300),sitk.sitkUInt8)
        registration_method.SetMetricMovingMask(mmask)
    if fixed_mask:
        fmask = sitk.Cast(sitk.IntensityWindowing(fixed, windowMinimum=-100, windowMaximum=300),sitk.sitkUInt8)
        registration_method.SetMetricFixedMask(fmask)

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                                                  sitk.Cast(moving, sitk.sitkFloat32))

    return final_transform, registration_method

if __name__ == '__main__':



    ncct_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/kits_ncct/unseen'
    cect_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/kits23_nooverlap/images'
    save_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/kits_ncct/registered_nondeform'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # ncct images have the following name structure: KiTS_XXXXX.nii.gz
    # cect images have the following name structure: case_XXXXX.nii.gz
    # where XXXXX is the case number - matching case numbers correspond to the same patient

    # identify the matching ncct and cect images
    ncct_images = [f for f in os.listdir(ncct_dir) if f.endswith('.nii.gz')]
    cect_images = [f for f in os.listdir(cect_dir) if f.endswith('.nii.gz')]

    # create a dictionary to store the matching ncct and cect images
    matching_images = {}
    for ncct_image in ncct_images:
        for cect_image in cect_images:
            if ncct_image[5:10] == cect_image[5:10]:
                matching_images[ncct_image] = cect_image

    for ncct_image, cect_image in matching_images.items():
        nc_fp = os.path.join(ncct_dir, ncct_image)
        ce_fp = os.path.join(cect_dir, cect_image)

        # Read the images
        fixed_image = sitk.ReadImage(ce_fp, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(nc_fp, sitk.sitkFloat32)

        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        #
        # moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0,
        #                                  moving_image.GetPixelID())
        #
        linear_transform, reg_method = linear_coreg(moving_image,fixed_image,initial_transform)
        print('Final metric value: {0}'.format(reg_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(reg_method.GetOptimizerStopConditionDescription()))

        moving_resampled = sitk.Resample(moving_image, fixed_image, linear_transform, sitk.sitkLinear, -1000.0,
                                         moving_image.GetPixelID())


        # save the moving image
        save_fp = os.path.join(save_dir, ncct_image)
        print(f'Saving the registered image to {save_fp}')
        sitk.WriteImage(moving_resampled, save_fp)

