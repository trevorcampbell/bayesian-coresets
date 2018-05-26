from setuptools import setup

setup(
    name = 'bayesiancoresets',
    version='0.8',
    description="Coresets for approximate Bayesian inference",
    author='Trevor Campbell',
    author_email='tdjc@mit.edu',
    url='https://github.com/trevorcampbell/bayesian-coresets/',
    packages=['bayesiancoresets'],
    install_requires=['numpy', 'scipy'],
    keywords = ['Bayesian', 'inference', 'coreset', 'Hilbert', 'Frank-Wolfe', 'greedy', 'geodesic'],
    platforms='ALL',
)
