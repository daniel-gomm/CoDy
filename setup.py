from setuptools import setup, find_packages

setup(
    name='CFTGNNExplainer',
    version='0.0.1',
    description='Counterfactual explanations for graph neural networks on dynamic graphs',
    author='Daniel Gomm',
    author_email='daniel.gomm@student.kit.edu',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'tqdm'
    ],
)
