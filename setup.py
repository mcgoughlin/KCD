from setuptools import setup,find_packages

setup(
    name='KCD_early',
    packages=find_packages('KCD_src', exclude=['test']),
    install_requires=['torch','dgl','matplotlib','nibabel','bpy',
              'numpy','opencv-python','pandas','pydicom',
              'scikit-image','scikit-learn','scipy','numpy-stl',
              'simpleitk','torchvision','pymeshfix','rt_utils'],
    python_requires='>=3.10',
    description='Python Package for the early detection of renal cancer',
    version='0.1',
    url='https://github.com/mcgoughlin/KCD',
    author='bmcgough',
    author_email='billy.mcgough1@hotmail.com',
    keywords=['pip','pytorch','cancer']
    )