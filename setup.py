from setuptools import setup
from setuptools.command.install import install
from pip.commands.install import logger
import subprocess


class build_so(install):
  def run(self):
    retcode = subprocess.call(['g++', '-std=c++17', '-pthread', '-DNDEBUG', '-fwrapv', '-O3', '-Wall', '-fno-strict-aliasing', '-Wdate-time', '-D_FORTIFY_SOURCE=2', '-fstack-protector-strong', '-Wformat', '-Werror=format-security', '-fPIC', 'hilbertcoresets/captree.cpp', '-shared', '-o', 'hilbertcoresets/libcaptreec.so']) 
    if retcode != 0:
      raise Exception('g++: Compile of captreec.cpp failed')
    return install.run(self)

setup(
    name = 'hilbertcoresets',
    version='0.3',
    description="Hilbert coresets for approximate Bayesian inference",
    author='Trevor Campbell',
    author_email='tdjc@mit.edu',
    url='https://github.com/trevorcampbell/hilbert-coresets/',
    packages=['hilbertcoresets'],
    install_requires=['numpy', 'scipy'],
    keywords = ['Bayesian', 'inference', 'coreset', 'Hilbert', 'Frank-Wolfe', 'greedy', 'geodesic'],
    platforms='ALL',
    #libraries=[libcaptreec],
    #cmdclass={'install':build_so_first, 'build_clib':build_so},
    cmdclass={'install':build_so},
    package_data={'hilbertcoresets': ['libcaptreec.so']}
)
