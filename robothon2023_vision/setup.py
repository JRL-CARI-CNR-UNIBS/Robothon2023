#! /usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['robothon2023_vision'],
    package_dir={'yolov6': '~/projects/third_parties_ws/src/software_modules/YOLOv6'}
)
setup(**d)
