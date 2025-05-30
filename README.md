# CoDy - Counterfactual Explainer for Models on Dynamic Graphs

This repository contains the code for CoDy. This README details how the project is structured, how the 
project is used and extensible, and how to reproduce the results.

Contents:
1. [Install CoDy](#1-install-cody)
2. [Project Structure](#2-project-structure)
3. [Run the evaluation](#3-run-the-evaluation)
   1. [Prepare datasets](#31-prepare-datasets)
   2. [Train the TGN model](#32-train-the-tgn-model)
   3. [Train the PGExplainer model](#33-train-the-pgexplainer-model)
   4. [Run evaluation experiments](#34-run-evaluation-experiments)
   5. [Postprocess the evaluation results](#35-postprocess-the-evaluation-results)
4. [Development guide](#4-development-guide)


## 1. Install CoDy
The first step towards installing the CoDy repository is to download the repository and its submodules. To do this run 
the following command to clone and initialize the repository:

```bash
git clone --recurse-submodules https://github.com/daniel-gomm/CoDy.git
```
Next, you need to install the CoDy package. Run the following command from the root directory of this repository to 
install the CoDy package using pip:

```bash
pip install -e .
```

Providing the `-e`-flag installs the package in editable mode, meaning that changes to the code translate to the package
without re-installation. If you are not interested in modifying the code, you can omit the `-e`-flag.

> Note: Original experiments were conducted on Python 3.11

## 2. Project Structure
The project is structured into multiple subdirectories. These are listed and described in the following:

- [cody](./cody): Contains the CoDy package, including the code for the explanation methods, interfaces to TGNNs and 
data preprocessing and handling.
- [resources](./resources): Initially empty. This folder holds all the data that results from running the project, like
model checkpoints, evaluation results, and preprocessed datasets. Raw datasets are placed in the 
[datasets/raw](./resources/datasets/raw) subfolder.
- [scripts](./scripts): Contains an array of different scripts that are used to interact with the CoDy package. For 
example, code to run the evaluation.
- [submodules](./submodules): This folder contains the implementations of TGNNs as separate git repositories. By 
default, it contains two seperate implementations of the [TGN model](https://github.com/twitter-research/tgn).

## 3. Run the evaluation

### 3.1. Prepare Datasets
The first step to running the experiments is the preparation of the datasets. The datasets can be downloaded from the 
following sites:

- [Wikipedia](http://snap.stanford.edu/jodie/#datasets)
- [UCI-Messages/UCI-Forums](https://toreopsahl.com/datasets/)

The raw dataset files should have the same format as the Wikipedia dataset; that is: 

- First column: Source node ids
- Second column: Target node ids
- Third column: UNIX timestamp
- Fourth column: State label (not necessary for link prediction task)
- Fifth column and onwards: Comma seperated list of edge features

The UCI datasets do not have this form by default. To make the conversion easier, the 
[format_uci_data.py](./scripts/format_uci_data.py) is provided. First download the dataset to a file from the website 
mentioned above, for example for the UCI-Messages dataset by running:

```shell
curl http://opsahl.co.uk/tnet/datasets/OCnodeslinks.txt > UCI-Messages.txt
```

Then use the script to convert the downloaded file to an appropriate .csv file by running:

```shell
python format_uci_data.py --input UCI-Messages.txt --output ../resources/datasets/raw/ucim.csv
```

Adjust the ``--input`` and the ``--output`` parameters accordingly.

Place the correctly formatted datasets as .csv files in the [/resources/datasets/raw](./resources/datasets/raw) directory.

To prepare the raw datasets for the usage run the following command from the [/scripts](./scripts) directory:

Wikipedia/UCI-Forums:
```shell 
bash preprocess_data.bash DATASET-NAME --bipartite
```

Replace ``DATASET-NAME`` with the name of the dataset that should be processed, e.g., 'wikipedia' or 'uci_forums'. The name 
corresponds to the file name without extension, i.e., if the wikipedia dataset is named 'wikipedia.csv', pass 'wikipedia'
as the ``DATASET-NAME``.

UCI-Messages:

```shell 
bash preprocess_data.bash DATASET-NAME
```

For the UCI-Messages dataset the ``--bipartite`` flag is omitted, as the dataset describes a unipartite graph.


### 3.2. Train the TGNN Models

Next, a TGNN model is trained for each dataset. To do the training run the following command from the 
[/scripts](./scripts) directory:

```shell
bash train_tgnn_model.bash MODEL-TYPE DATASET-NAME --bipartite
```

Replace ``MODEL-TYPE`` with the type of the model you want to evaluate, e.g., 'TGAT' or 'TGN'.

Replace ``DATASET-NAME`` with the name of the dataset on which you want to train the TGN model, e.g., 'uci', 
'wikipedia', etc.

Only provide the ``--bipartite`` flag if the underlying dataset is a bipartite graph (Wikipedia/UCI-Forums), else
omit the ``--bipartite`` flag from the command.

### 3.3. Train the PGExplainer model
T-GNNExplainer relies on a pretrained navigator that is realized as dynamic adaptation of PGExplainer. Thus, this
navigator component has to be trained prior to evaluating T-GNNExplainer. To do so run the follwing command from the 
[/scripts](./scripts) directory:

```shell
bash train_pg_explainer.bash MODEL-TYPE DATASET-NAME --bipartite
```

Replace ``MODEL-TYPE`` with the type of the model you want to evaluate, e.g., 'TGAT' or 'TGN'.

Replace ``DATASET-NAME`` with the name of the dataset on which you want to train the PGExplainer model, e.g., 'uci', 
'wikipedia', etc.

Only provide the ``--bipartite`` flag if the underlying dataset is a bipartite graph (Wikipedia/UCI-Forums), else
omit the ``--bipartite`` flag from the command.

### 3.4. Run evaluation experiments

With all these prerequisites out of the way you can now run the experiments themselves. The experiments are run for each
explanation method (T-GNNExplainer, GreDyCF, CoDy), for each dataset, for each correct/incorrect setting 
(correct predictions only/incorrect predictions only), and for each selection policy (random, temporal, spatio-temporal, 
local-gradient) separately. For convenience, all selection strategies can be automatically evaluated in parallel from a 
single script. An additional feature of the evaluation is that it can be interrupted by Keyboard Interruption or by the
maximum processing time. When the evaluation is interrupted before it is finished, the intermediary results are saved. 
The evaluation automatically resumes from intermediary results.

To run the evaluation in one experimental setting, run the following command from the [/scripts](./scripts) directory:

Evaluate explanations for correct predictions only:

```shell
bash evaluate.bash MODEL-TYPE DATASET-NAME EXPLAINER-NAME SELECTION-NAME TIME-LIMIT --bipartite
```

Replace ``MODEL-TYPE`` with the type of the model you want to evaluate, e.g., 'TGAT' or 'TGN'.

Replace ``DATASET-NAME`` with the name of the dataset on which you want to train the PGExplainer model, e.g., 'uci', 
'wikipedia', etc.

Replace ``EXPLAINER-NAME`` with the explainer you want to evaluate. Options are ``tgnnexplainer``, ``greedy``, ``cody``.

Replace ``SELECTION-NAME`` with the selection policy that you want to evaluate. The options are ``random``, 
``temporal``, ``spatio-temporal``, ``local-gradient``, and ``all``. Use the ``all`` option to efficiently evaluate the
different selection strategies with caching between selection strategies.
**Do not provide a `SELECTION-NAME`` argument when evaluating T-GNNExplainer**

Replace ``TIME-LIMIT`` with an integer number that sets a limit on the maximum time that the evaluation runs before 
concluding in minutes. The evaluation can be resumed from that state at a later time.

Only provide the ``--bipartite`` flag if the underlying dataset is a bipartite graph (Wikipedia/UCI-Forums), else
omit the ``--bipartite`` flag from the command.

As an example, to run the evaluation of CoDy for all selection strategies, with a time limit of 240 minutes and the
bipartite wikipedia dataset, the following command is used:

```shell
bash evaluate.bash wikipedia cody all 240 --bipartite
```

To run the evaluation for incorrect predictions only you can use the same options with another bash script:

```shell
bash evaluate_incorrect_predictions_only.bash MODEL-TYPE DATASET-NAME EXPLAINER-NAME SELECTION-NAME TIME-LIMIT --bipartite
```

### 3.5. Postprocess the evaluation results

To finish the evaluation and enrich the evaluation datasets with further information two further scripts are employed.

**Enrich Factual Explanations:**

The fist step is to add the prediction that is achieved when removing the events in the explanation produced by 
T-GNNExplainer from the input data (to evaluate fidelity+/whether the explanation is counterfactual). To add this 
information, use the [evaluate_factual_subgraphs.py](/scripts/evaluate_factual_subgraphs.py) script from the 
[/scripts](./scripts) directory. Use it in the following way:

```shell
python evaluate_factual_subgraphs.py 
  -d ../resources/datasets/processed/wikipedia 
  --cuda 
  --bipartite 
  --model ../resources/models/wikipedia/wikipedia-wikipedia.pth 
  --type TGAT
  --results ../resources/results/wikipedia/tgnnexplainer/results_wikipedia_tgnnexplainer.parquet
```

This command adds the information to the results for correct predictions only for the wikipedia dataset.

**Add Fidelity- information:**

The second step is to analyze how much the events in the explanation influence the prediction. For this, use the 
[evaluate_fidelity_minus.py](/scripts/evaluate_fidelity_minus.py) script from the [/scripts](./scripts) directory. Use it
in the following way:

```shell
python evaluate_fidelity_minus.py
  -d ../resources/datasets/processed/wikipedia
  --cuda
  --bipartite
  --model ../resources/models/wikipedia/wikipedia-wikipedia.pth
  --type TGAT
  --results ../results/wikipedia/greedy
  --all_files_in_dir
```

This adds the fidelity- information to all the results saved in the _results/wikipedia/greedy_ directory. Rerun the 
command for each of the explainers and each of the datasets to add the information to all results.


## 4. Development Guide

### 4.1. Adding a new target model

To add a new target model you need to implement a new wrapper that acts as a bridge between the functionalities of
the explainers and the target model. To add a new wrapper create a new class that extends the 
[TGNNWrapper](./cody/implementations/connector.py) class. Make sure to override all the necessary functions of this 
class. As reference you can take a look at the implementation of the wrapper for the TGN model: 
[TGNWrapper](./cody/implementations/tgn.py)

Implementing a new explanation method like this will work with the explanation methods, however, to run the evaluation
for this target model, the seperated evaluation scripts have to be developed.


### 4.2. Add a new explanation method

A new explanation method is implemented as a new class that extends the [Explainer](./cody/explainer/base.py) class. 
Make sure to overrride the functions appropriately. It may be useful to use the [TreeNode](./cody/explainer/base.py)
class as basis for a search tree.

As inspiration you can have a look at the implementations of [CoDy](./cody/explainer/cody.py) or 
[GreeDyCF](./cody/explainer/greedy.py).

You can also add a new selection policy by extending the [SelectionStrategy](./cody/selection.py) class.
