# -*- coding: utf-8 -*-

__author__    = "Christian Richter"
__copyright__ = "Copyright 2019, TU Dresden"
__license__   = "GPL"
__credits__   = ["Christian Richter"]
__email__     = "christian.richter1@tu-dresden.de"
__project__   = "FmiRL"
__version__   = "0.1.0"


from setuptools import setup, find_packages


setup(
    name="fmirl",
    version='0.1.0',
    description="Using Functional Mockup Units for Reinforcement Learning with OpenAI Gym",
    author="Christian Richter",
    author_email='christian.richter1@tu-dresden.de',
    url="https://tu-dresden.de/bft",
    license='GNU GPL',
    packages=['fmirl', 'fmirl.agents', 'fmirl.envs'],
    include_package_data=True,
    platforms='all',
    install_requires=[
        'pyfmi',
        'gym',
        'numpy',
    ]
)