from distutils.core import setup
import numpy as np

setup(
    name = 'hilbert-coresets',
    version='0.1',
    description="Hilbert coresets for approximate Bayesian inference",
    author='Trevor Campbell',
    author_email='tdjc@mit.edu',
    url='https://github.com/trevorcampbell/hilbert-coresets/',
    packages=['hilbertcoresets'],
    install_requires=['numpy'],
    keywords = ['coreset', 'hilbert', 'frank-wolfe'],
    platforms='ALL',
)
