from setuptools import setup, find_packages

def open_requirements(path):
    with open(path) as f:
        requires = [
            r.split('/')[-1] if r.startswith('git+') else r
            for r in f.read().splitlines()]
    return requires

requires = open_requirements('requirements.txt')

with open('README.md', mode='r') as readme:
    long_description = readme.read()

setup(
    name             = 'xenodiffusionscope',
    version          = '0.0.1',
    packages=find_packages(exclude=['tests*']),
    author           = 'Ricardo Peres',
    author_email     = 'rperes@physik.uzh.ch',
    description      = 'A Python package to simulate the Xenoscope LXe TPC light readout.',
    url              = 'https://github.com/ricmperes/XenoDiffusionScope',
    include_package_data = True,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    classifiers      = ['Development Status :: 4 - Beta',
                        'Natural Language :: English',
                        'Programming Language :: Python :: 3.8',
                        'Programming Language :: Python :: 3.9',
                        'Programming Language :: Python :: 3.10',
                        'Operating System :: POSIX :: Linux',
                        'Operating System :: MacOS',
                        'Operating System :: Microsoft :: Windows',
                        'Intended Audience :: Science/Research',
                        'License :: OSI Approved :: Apache Software License',
                        'Topic :: Scientific/Engineering :: Physics'],
    install_requires = requires,
)