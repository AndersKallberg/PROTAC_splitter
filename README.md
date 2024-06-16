# PROTAC_splitter

![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

## Project description

A machine learning tool for identifying the substructures of PROTACs. Based on Pytorch and graph neural networks, investigates multiple architectures to predict the substructures. The best architecture predicts the boundary bonds / links / edges between the substructures. This repository also contains notebooks to create training, validation and test sets, as well as a notebook to run a hyperparameter optimization. A detailed description of the model architectures will be available through the master's thesis *Machine Learning for Structural Predictions of PROTACs*.

The three architectures which were investigated are outlined here:

**Node prediction** - Predicts the individual atoms/nodes of the molecule to be one of the three classes: warhead, linker, or E3-ligand
![node_pred_architecture](https://github.com/AndersKallberg/PROTAC_splitter/assets/143714806/409d63b8-17f1-47f1-9ffe-4cc9046b46dc)

**Boundary node prediction** - Predicts the individual atoms/nodes of the molecule to be one of the three classes: Warhead-Linker boundary node, non-boundary node, E3-Linker boundary node.
![boundary_node_architecture](https://github.com/AndersKallberg/PROTAC_splitter/assets/143714806/270992f8-e13a-4151-b448-cbcad2d8490e)

**Boundary bond prediction** - Predicts the individual bonds/links of the molecule to be one of the three classes: Warhead-Linker boundary bond, non-boundary bond, E3-Linker boundary bond. The atoms belonging to a splittable bond were fed into a feed-forward networks to predict the location and type of bond. Splittable bonds were defined as non-ring bonds, and bonds that do not connect a single atom.
![boundary_bond_architecture](https://github.com/AndersKallberg/PROTAC_splitter/assets/143714806/3a2c8658-01d8-4b54-9bd5-2d3f9be10ce5)


## Dependencies

Python and Linux were used, and the necessary dependencies are found in "environment.yml".

Alternatively:

- conda install conda-forge::networkx
- conda install conda-forge::rdkit
- conda install conda-forge::seaborn
- conda install conda-forge::scikit-learn
- conda install conda-forge::pytorch 
- conda install conda-forge::pytorch_geometric
- conda install conda-forge::optuna
- conda install conda-forge::scipy   

## How to run the code

1. Run the notebook data_curation_augmentation_splitting.ipynb. It will create all training, validation and test sets.

2. Run the notebook hyperparameter_optimization.ipynb - Note down the optimal hyperparameters (Best identified hyperparameters are presented in *Machine Learning for Structural Predictions of PROTACs*).

3. Input optimal hyperparameters in the notebook gnn_full_pipeline.ipynb and train the model. This will create protacsplitter_params.pt which can be loaded.
3.1 Evaluation metrics and plots are displayed in gnn_full_pipeline.ipynb 

4. An example to how to load and use the model is found in use_the_PROTAC_splitter.ipynb
