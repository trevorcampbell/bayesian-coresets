from distutils.core import setup
import numpy as np

setup(
    name = 'hilbert-coresets',
    version='0.1',
    description="Coresets in a Hilbert space",
    author='Trevor Campbell',
    author_email='tdjc@mit.edu',
    url='https://github.com/trevorcampbell/hilbert-coresets/',
    packages=['hilbertcoresets'],
    package_data={'python' : ['*.so']},
    install_requires=['numpy'],
    keywords = ['coreset', 'hilbert', 'frank-wolfe'],
    platforms='ALL',
    scripts=['compile.sh'],
)
