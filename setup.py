from setuptools import setup
from setuptools.command.install import install
from pip.commands.install import logger
import subprocess


class build_so(install):
  def run(self):
    retcode = subprocess.call(['g++', '-std=c++17', '-pthread', '-DNDEBUG', '-fwrapv', '-Ofast', '-fopenmp', '-Werror', '-Wall', '-fno-strict-aliasing', '-Wdate-time', '-D_FORTIFY_SOURCE=2', '-fstack-protector-strong', '-Wformat', '-Werror=format-security', '-fPIC', 'hilbertcoresets/gigasearch.cpp', '-shared', '-o', 'hilbertcoresets/libgigasearch.so']) 
    if retcode != 0:
      raise Exception('g++: Compile of gigasearch.cpp failed')
    return install.run(self)

setup(
    name = 'hilbertcoresets',
    version='0.5',
    description="Hilbert coresets for approximate Bayesian inference",
    author='Trevor Campbell',
    author_email='tdjc@mit.edu',
    url='https://github.com/trevorcampbell/hilbert-coresets/',
    packages=['hilbertcoresets'],
    install_requires=['numpy', 'scipy'],
    keywords = ['Bayesian', 'inference', 'coreset', 'Hilbert', 'Frank-Wolfe', 'greedy', 'geodesic'],
    platforms='ALL',
    cmdclass={'install':build_so},
    package_data={'hilbertcoresets': ['libgigasearch.so']}
)
