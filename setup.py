import os

from setuptools import setup, find_packages

setup(
    name='cody',
    version='1.0.0',
    description='Counterfactual Explanations for Deep Graph Models on Dynamic Graphs',
    author='Daniel Gomm',
    author_email='daniel.gomm@cwi.nl',
    url='https://github.com/daniel-gomm/CoDy',
    keywords='explanation, gnn, tgnn, counterfactual, temporal graph neural network, dynamic graph',
    packages=find_packages(),
    install_requires=[
        'numpy==1.25.2',
        'pandas==2.0.1',
        'torch==2.0.1',
        'tqdm==4.65.0',
        'ipython',
        'pyarrow',
        f"TGN @ file://localhost/{os.getcwd()}/submodules/tgn/",
        f"TTGN @ file://localhost/{os.getcwd()}/submodules/ttgn/"
    ]
)
