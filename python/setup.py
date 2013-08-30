#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

setup(  name='swoc_framework',
        version='0.1.0',
        author='Loy van Beek',
        author_email='loy.vanbeek@gmail.com',
        scripts=['images.py'],
        url='https://github.com/yol/swoc-framework',
        license='LICENSE.txt',
        description='Sketch recognition',
        long_description="Recognize black and white sketches using HoG descriptors and other computer vision and machine learning methods",
        install_requires=["matplotlib", "skimage", "numpy"]
)