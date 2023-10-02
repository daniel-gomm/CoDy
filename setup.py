import os

from setuptools import setup, find_packages

setup(
    name='CFTGNNExplainer',
    version='0.0.1',
    description='Counterfactual explanations for graph neural networks on dynamic graphs',
    author='Daniel Gomm',
    author_email='daniel.gomm@student.kit.edu',
    url='https://github.com/daniel-gomm/CFTGNNExplainer',
    keywords='explanation, gnn, tgnn, counterfactual',
    packages=find_packages(),
    install_requires=[
        'numpy==1.25.2',
        'pandas==2.0.1',
        'torch==2.0.1',
        'tqdm==4.65.0',
        'ipython',
        f"TGN @ file://localhost/{os.getcwd()}/submodules/tgn/",
        f"TTGN @ file://localhost/{os.getcwd()}/submodules/ttgn/"
    ]
)
