from setuptools import setup,find_packages

setup(
    name='KCD_early',
    packages=find_packages('KCD_src', exclude=['test']),
    # install_requires=['pytorch==1.13','dgl==1.0.2','matplotlib==3.6.2','nibabel==5.0.0',
    #           'numpy==1.23.5','opencv-python==4.7.0.68','pandas==1.5.3','pydicom==2.3.1',
    #           'scikit-image==0.19.3','scikit-learn==1.2.2','scipy==1.10.1',
    #           'simpleitk==2.2.1','torchvision==0.15.2'],
    install_requires=['torch','dgl','matplotlib','nibabel','bpy; python version<3.8',
              'numpy','opencv-python','pandas','pydicom','stl',
              'scikit-image','scikit-learn','scipy','numpy-stl',
              'simpleitk','torchvision','pymeshfix','open3d'],
    python_requires=['>=3.10'],
    description='Python Package for the early detection of renal cancer',
    version='0.1',
    url='https://github.com/mcgoughlin/KCD',
    author='bmcgough',
    author_email='billy.mcgough1@hotmail.com',
    keywords=['pip','pytorch','cancer']
    )