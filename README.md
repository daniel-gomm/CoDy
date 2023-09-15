# CF-TGNNExplainer - Counterfactual explanations for predictions by Temporal Graph Neural Networks

## Install CFTGNNExplainer

To install the package run the following:

```bash
pip install -e .
```

From the source directory of this repository. This will install the package in editable mode.

## Prepare dataset
To prepare the datasets for usage download them into a directory of your choosing and then run the following to 
preprocess the data:

```bash
python CFTGNNExplainer/data/preprocess_dataset.py -f <path-to-dataset> -t <target-directory> --bipartite
```

You can omit the ```--bipartite``` flag if the dataset is not bipartite.


## Train the TGN model

Run the following command to train the TGN model:

```bash
python train_tgnn.py \
--dataset <path-to-dataset-folder> \
--model_path <path-where-to-store-model> \
--bipartite \
--cuda \
--epochs 50
```
