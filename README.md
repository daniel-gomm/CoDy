# CF-TGNNExplainer - Counterfactual explanations for predictions by Temporal Graph Neural Networks

This repository contains the code for CF-TGNNExplainer. This README details how the project is structured, how the 
project is used and extensible, and how to reproduce the results.

## 1. Install CFTGNNExplainer
To use the project or to reproduce the results you need to first install the CF-TGNNExplainer package.

To install the package run the following:

```bash
pip install -e .
```

From the source directory of this repository. This will install the package in editable mode.

## 2. Reproduce the results
The easiest way to reproduce the CF-TGNNExplainer results is to follow this guide and use the provided bash scripts. 
Alternatively you can also run the pyton scripts in the [/scripts](./scripts) directory for each step directly, giving
you more freedom over the parameters.

### 2.1. Prepare the datasets
First, the datasets are transformed into the appropriate format for training the TGN model.
1. Download the datasets you want to investigate. For example from the 
[stanford website](http://snap.stanford.edu/jodie/#datasets) for the Reddit, Wikipedia, MOOC and LastFM datasets.
2. Place the dataset as .csv file into the [/resources/datasets/raw](./resources/datasets/raw) directory.
3. From the [/scripts](./scripts) directory run the following command: 
```shell 
bash preprocess_data.bash
```
This automatically preprocesses all datasets in the [/resources/datasets/raw](./resources/datasets/raw) directory and 
puts the results into the [/resources/datasets/processed](./resources/datasets/processed) directory.

### 2.2. Train the TGN Model
Next, a TGN model is trained for each dataset. To do the training run the following command from the 
[/scripts](./scripts) directory:

```shell
bash train_tgn_model.bash DATASET-NAME
```
Replace ``DATASET-NAME`` with the name of the dataset on which you want to train the TGN model, e.g., 'reddit', 
'wikipedia', etc.

### 2.3. Train the PGExplainer baseline model
Next, train the PGExplainer model, serving as baseline in the evaluation. Run the following command from the 
[/scripts](./scripts) directory:
```shell
bash train_pg_explainer.bash DATASET-NAME
```
Replace ``DATASET-NAME`` with the name of the dataset on which you want to train the TGN model, e.g., 'reddit', 
'wikipedia', etc.

### 2.3. Train the sampler model
Next, train the sampler model. For this run the following from the [/scripts](./scripts) directory:
```shell
bash train_sampler.bash DATASET-NAME
```
Replace ``DATASET-NAME`` with the name of the dataset on which you want to train the TGN model, e.g., 'reddit', 
'wikipedia', etc.

### 2.4. Run the evaluation
To conduct the evaluation for a given dataset run the following command from the [/scripts](./scripts) directory:
```shell
bash run_evaluation.bash DATASET-NAME
```
Replace ``DATASET-NAME`` with the name of the dataset on which you want to train the TGN model, e.g., 'reddit', 
'wikipedia', etc.
