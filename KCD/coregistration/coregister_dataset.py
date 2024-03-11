import SimpleITK as sitk
import os
from utils import start_plot, end_plot, plot_values, display_images, display_images_with_alpha, update_multires_iterations
import matplotlib.pyplot as plt

# linear coreg with https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/60_Registration_Introduction.html


def linear_coreg(moving,fixed,initial_transform,
                 moving_mask=True,fixed_mask=True):
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-7, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1,])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if moving_mask :
        mmask = sitk.Cast(sitk.IntensityWindowing(moving, windowMinimum=-200, windowMaximum=350),sitk.sitkUInt8)
        moving = sitk.Mask(moving, mmask)
    if fixed_mask:
        fmask = sitk.Cast(sitk.IntensityWindowing(fixed, windowMinimum=-200, windowMaximum=350),sitk.sitkUInt8)
        fixed = sitk.Mask(fixed, fmask)

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

def nonlinear_coreg(moving,fixed,initial_transform,
                    moving_mask=True,fixed_mask=True):
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetInterpolator(sitk.sitkBSpline)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-7, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if moving_mask :
        mmask = sitk.Cast(sitk.IntensityWindowing(moving, windowMinimum=-200, windowMaximum=350),sitk.sitkUInt8)
        moving = sitk.Mask(moving, mmask)
    if fixed_mask:
        fmask = sitk.Cast(sitk.IntensityWindowing(fixed, windowMinimum=-200, windowMaximum=350),sitk.sitkUInt8)
        fixed = sitk.Mask(fixed, fmask)

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

def bspline_coreg(moving, fixed, initial_transform=None,
                  moving_mask=True, fixed_mask=True):
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=20)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.3)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=2000,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=20)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if moving_mask :
        mmask = sitk.Cast(sitk.IntensityWindowing(moving, windowMinimum=-200, windowMaximum=350),sitk.sitkUInt8)
        moving = sitk.Mask(moving, mmask)
    if fixed_mask:
        fmask = sitk.Cast(sitk.IntensityWindowing(fixed, windowMinimum=-200, windowMaximum=350),sitk.sitkUInt8)
        fixed = sitk.Mask(fixed, fmask)

    # Use BSpline transformation.
    transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacement_field_filter.SetSize(fixed.GetSize())
    transform_to_displacement_field_filter.SetOutputSpacing(fixed.GetSpacing())
    transform_to_displacement_field_filter.SetOutputOrigin(fixed.GetOrigin())
    transform_to_displacement_field_filter.SetOutputDirection(fixed.GetDirection())

    if not initial_transform:
        initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(fixed.GetDimension()))

    displacement_field = transform_to_displacement_field_filter.Execute(initial_transform)
    bspline_transform = sitk.DisplacementFieldTransform(displacement_field)
    bspline_transform.SetInterpolator(sitk.sitkLinear)

    registration_method.SetInitialTransform(bspline_transform, inPlace=False)


    final_transform = registration_method.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                                                  sitk.Cast(moving, sitk.sitkFloat32))

    return final_transform, registration_method


if __name__ == '__main__':



    ncct_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/kits_ncct/unseen'
    cect_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/kits23/images'

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
        fixed_image = sitk.ReadImage(nc_fp, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(ce_fp, sitk.sitkFloat32)

        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

        moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0,
                                         moving_image.GetPixelID())

        linear_transform, reg_method = linear_coreg(moving_image,fixed_image,initial_transform)
        print('Final metric value: {0}'.format(reg_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(reg_method.GetOptimizerStopConditionDescription()))

        # apply nonlinear coregistration
        nonlinear_transform, reg_method = bspline_coreg(moving_image,fixed_image,linear_transform)

        print('Final metric value: {0}'.format(reg_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(reg_method.GetOptimizerStopConditionDescription()))

        resampled_moving = sitk.Resample(moving_image, fixed_image, nonlinear_transform, sitk.sitkBSpline, 0.0,
                                         moving_image.GetPixelID())

        # extract arrays of the images and plot them
        fixed_image_array = sitk.GetArrayFromImage(fixed_image)
        moving_image_array = sitk.GetArrayFromImage(resampled_moving)

        # display the images
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(fixed_image_array[30, :, :], cmap='gray')
        plt.title('Fixed Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(moving_image_array[30, :, :], cmap='gray')
        plt.title('Moving Image')
        plt.axis('off')
        plt.show()

        break

