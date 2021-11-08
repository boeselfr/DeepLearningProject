from setuptools import setup, find_packages

setup(
    name='splicing',
    version='0.0.1',
    description='Splicing tools',
    author='Frederic Boesel, Clement Guerner, Anej Svete',
    packages=find_packages(),
    install_requires=[
                      'numpy',
                      'pandas',
                      'matplotlib',
                      ],

    classifiers=[
        'Development Status :: 1 - Highly unstable research development ;)',
        'Programming Language :: Python :: 3',
    ],
)
