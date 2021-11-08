from setuptools import setup, find_packages

setup(
    name='graph_splice',
    version='0.0.1',
    description='Graph Splicing prediction Package',
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
