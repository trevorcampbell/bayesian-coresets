from setuptools import setup
from setuptools.command.install import install
from pip.commands.install import logger
import subprocess


class build_so(install):
  def run(self):
    retcode = subprocess.call(['g++', '-std=c++11', '-pthread', '-DNDEBUG', '-fwrapv', '-Ofast', '-fopenmp', '-Werror', '-Wall', '-Wextra', '-fno-strict-aliasing', '-D_FORTIFY_SOURCE=2', '-fstack-protector-strong', '-Werror=format-security', '-fPIC', 'bayesiancoresets/gigasearch.cpp', '-shared', '-o', 'bayesiancoresets/libgigasearch.so']) 
    if retcode != 0:
      raise Exception('g++: Compile of gigasearch.cpp failed')
    return install.run(self)

setup(
    name = 'bayesiancoresets',
    version='0.6',
    description="Coresets for approximate Bayesian inference",
    author='Trevor Campbell',
    author_email='tdjc@mit.edu',
    url='https://github.com/trevorcampbell/bayesian-coresets/',
    packages=['bayesiancoresets'],
    install_requires=['numpy', 'scipy'],
    keywords = ['Bayesian', 'inference', 'coreset', 'Hilbert', 'Frank-Wolfe', 'greedy', 'geodesic'],
    platforms='ALL',
    cmdclass={'install':build_so},
    package_data={'bayesiancoresets': ['libgigasearch.so']}
)
