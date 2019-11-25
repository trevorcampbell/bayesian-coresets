from setuptools import setup, find_packages

setup(
    name = 'bayesiancoresets',
    version='0.9.1',
    description="Coresets for approximate Bayesian inference",
    author='Trevor Campbell',
    author_email='trevor@stat.ubc.ca',
    url='https://github.com/trevorcampbell/bayesian-coresets/',
    #packages=['bayesiancoresets'],
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    keywords = ['Bayesian', 'inference', 'coreset', 'sparse', 'variational inference', 'Riemann',  'Hilbert', 'Frank-Wolfe', 'greedy', 'geodesic'],
    platforms='ALL',
)
