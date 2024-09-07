# CMCSG Model Implementation

This repository contains the implementation of the CMCSG model, as described in the research titled "A multi-task prediction method based on neighborhood structure embedding and signed graph representation learning to infer the relationship between circRNA, miRNA, and cancer". The CMCSG model is specifically designed for predicting circRNA-miRNA-cancer interactions, incorporating both **neighborhood structure embedding** and **signed graph representation learning** for feature extraction and prediction.

## Prerequisites

To run the CMCSG model, you need to install the following dependencies:

- **Python 3.7**: The required Python version.
- **Numpy 1.21.5**: A fundamental package for numerical computing in Python.
- **Scikit-learn 1.0.2**: A machine learning library used for tasks such as classification, regression, and clustering.
- **Torch 1.10.2**: The PyTorch framework, used for deep learning and neural network operations.

### Installation

You can install the required dependencies using the following command:

```bash
pip install numpy==1.21.5 scikit-learn==1.0.2 torch==1.10.2
```

## Model Architecture

The CMCSG model is divided into two main modules:

1. **Neighborhood Structure Embedding Module**: 
   - This module extracts both **local** and **global** features from the network.
   - **Local features** are computed using the `main.py` script.
   - **Global features** are extracted through the `struc2vec_flight.py` script, which implements the Struc2Vec algorithm for embedding the entire network.

2. **Feature Extraction Based on Signed Graph Representation Learning**:
   - This module uses **Signed Graph Attention Networks (SIGAT)** to learn from both positive and negative relationships within the graph.
   - It improves prediction accuracy by modeling the network's signed interactions between nodes, which in this case are circRNA, miRNA, and cancer-related entities.

## Data

The CMCN (circRNA-miRNA-cancer network) dataset is stored in the `data/` directory. Users can customize the dataset by following the provided format within the `data/` folder. This enables users to run the model on their own datasets while maintaining compatibility with the current setup.

### Data Customization

To use custom datasets, please place your files in the `data/` directory and ensure the format aligns with the structure of the CMCN data provided. You may need to adjust the paths in the scripts to point to your data files as needed.

## Key Scripts

1. **`main.py`**:
   - Implements the **local feature extraction** method for the neighborhood structure embedding module. 
   - This script analyzes local node interactions within the circRNA-miRNA-cancer network and generates relevant feature representations.

   To execute the local feature extraction process, run the following command:
   
   ```bash
   python main.py
   ```

2. **`struc2vec_flight.py`**:
   - Handles **global feature extraction** using the Struc2Vec algorithm, which captures the overall structure of the network and the relationships between distant nodes.
   - Struc2Vec helps in identifying deeper patterns in the network by embedding the graph structure.

   To execute global feature extraction, run the following command:
   
   ```bash
   python struc2vec_flight.py
   ```

3. **`sigat.py`**:
   - Implements the **signed graph representation learning** using **Signed Graph Attention Networks (SIGAT)**.
   - This component captures both positive and negative interactions in the network and helps improve the predictive power of the model by learning complex signed relationships.

   To run the signed graph learning module, execute:
   
   ```bash
   python sigat.py
   ```

## Execution Workflow

To fully run the CMCSG model and predict circRNA-miRNA-cancer interactions, follow these steps:

1. **Step 1**: Extract local features by running `main.py`.
   ```bash
   python main.py
   ```

2. **Step 2**: Extract global features by running `struc2vec_flight.py`.
   ```bash
   python struc2vec_flight.py
   ```

3. **Step 3**: Perform signed graph representation learning by executing `sigat.py`.
   ```bash
   python sigat.py
   ```

## Contact

This research is currently unpublished, and access to related data and models requires authorization from the authors. For access to the datasets or further information, please contact the lead author at:

**Email**: xinfei106@gmail.com

---

This README provides a detailed guide for the setup and execution of the CMCSG model, outlining each step necessary for running the model on either the provided CMCN dataset or customized datasets. Please follow the instructions carefully, and for any additional queries or permissions, contact the author directly.
