import SimpleITK as sitk
import os
from utils import start_plot, end_plot, plot_values, display_images, display_images_with_alpha, update_multires_iterations
import matplotlib.pyplot as plt

# linear coreg with https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/60_Registration_Introduction.html
def resample_volume(volume_path, interpolator = sitk.sitkLinear, new_spacing = [1.5,1.5,1.5]):
    volume = sitk.ReadImage(volume_path, sitk.sitkFloat32)
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), -1000,
                         volume.GetPixelID())

def linear_coreg(moving,fixed,initial_transform,
                 moving_mask=True,fixed_mask=True):
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=1000,
                                                      convergenceMinimumValue=1e-7, convergenceWindowSize=30)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1,])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if moving_mask :
        registration_method.SetMetricMovingMask( (moving>-150) & (moving<-20))
    if fixed_mask:
        registration_method.SetMetricFixedMask( (fixed>-150) & (fixed<-20))

    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)

    final_transform = registration_method.Execute(fixed,
                                                  moving)

    return final_transform, registration_method

def nonlinear_coreg(moving,fixed,
                    moving_mask=True,fixed_mask=True):
    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical spacing we want for the control grid.
    grid_physical_spacing = [15.0, 15.0, 15.0]  # A control point every 15mm
    image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size / grid_spacing + 0.5) \
                 for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]

    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                         transformDomainMeshSize=mesh_size, order=3)

    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    if moving_mask:
        registration_method.SetMetricMovingMask( (moving>-150) & (moving<-20))
    if fixed_mask:
        registration_method.SetMetricFixedMask( (fixed>-150) & (fixed<-20))


    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.3, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-7, convergenceWindowSize=20)

    final_transform = registration_method.Execute(fixed, moving)

    return final_transform, registration_method


if __name__ == '__main__':



    ncct_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/kits_ncct/unseen'
    cect_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/kits23_nooverlap/images'
    save_dir = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/kits_ncct/registered_justfat'

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

        # resample the moving image and the fixed image to a common spacing
        moving_image = resample_volume(nc_fp, interpolator = sitk.sitkLinear)
        fixed_image = resample_volume(ce_fp, interpolator = sitk.sitkLinear)

        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

        linear_transform, reg_method = linear_coreg(moving_image,fixed_image,initial_transform)
        print('Final metric value: {0}'.format(reg_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(reg_method.GetOptimizerStopConditionDescription()))

        moving_resampled = sitk.Resample(moving_image, fixed_image, linear_transform, sitk.sitkLinear, -1000.0,
                                         moving_image.GetPixelID())

        if reg_method.GetMetricValue() < 100:
            print(f'Linear registration successful for {ncct_image}')
            # save the moving image
            save_fp = os.path.join(save_dir, ncct_image)
            print(f'Saving the registered image to {save_fp}')
            sitk.WriteImage(moving_resampled, save_fp)
            continue
        else:
            print(f'Attempting nonlinear registration')

        # apply nonlinear coregistration
        nonlinear_transform, reg_method = nonlinear_coreg(moving_resampled,fixed_image)

        print('Final metric value: {0}'.format(reg_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(reg_method.GetOptimizerStopConditionDescription()))

        resampled_moving = sitk.Resample(moving_resampled, fixed_image, nonlinear_transform, sitk.sitkLinear, -1000.0,
                                         moving_resampled.GetPixelID())

        # save the moving image
        save_fp = os.path.join(save_dir, ncct_image)
        print(f'Saving the registered image to {save_fp}')
        sitk.WriteImage(resampled_moving, save_fp)

