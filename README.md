# AgentBind-brain #
AgentBind is a machine-learning framework for analyzing gene-regulatory regions in the human genome and identifying genomic bases with strong effects on regulatory activities. This code repository contains code for the classification + visualization experiments.

Preprint paper: [**click here**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8009085/)

Please cite: \
`Zheng, A., Shen, Z., Glass, C., and Gymrek, M. Deep learning predicts regulatory impact of variants in cell-type-specific enhancers in brain. bioRxiv (2022).`

## System Requirement & Installation ##
Code in this repository has been tested on CentOS Linux 7 (core) with Python (v3.7.4) in the Anaconda (v4.11.0) environment. Please make sure you have installed the following python libraries before you run the code.

### python libraries ###
Our code requires external python libraries including tensorflow v2.3.0 , biopython v1.71, numpy v1.18.5, six v1.15.0, and scikit-image v0.21.3. You can install them with the conda package manager.


## Data Download ##
Input data of the experiments can be downloaded using this link: [**download**](https://drive.google.com/file/d/12mrLk9Ci7u2tKB8kuqldGXE9ghAzpbUk/view?usp=sharing)


## Data Preparation ##
**data_prep.py** is a python script that prepares training/validation/test data for the deep learning models and converts DNA sequences into the one-hot encoding format.

**Required parameters:**
* --path: path to the data folder that you download from our Google Drive link.

To run this script, you can simply execute:
```
python data_prep.py --path {your-data-path}/storage
```

## Run Experiments ##
**run_experiment.py** is a python script that trains a deep learning model for each brain cell type and annotates all the bases in the input data with importance scores.

**Required parameters:**
* --path: path to the data folder that you download from our Google Drive link.

To run this script, you can simply execute:
```
python run_experiment.py --path {your-data-path}/storage
```

Once the program finishes, you can find the classification results in folder `{your-data-path}/storage/experiments/results_extended_coordconv_sliding/{cell-type}/category_0/`. And the annotation results can be found in folder`{your-data-path}/storage/experiments/results_extended_coordconv_sliding/{cell-type}/annotations/`

This python program "run_experiment.py" takes ~24-48 hours to complete. If you need the Grad-CAM annotation scores only, you can directly download them here: [**download**](https://drive.google.com/file/d/1jvfIXezcihy21sf0AWrVQ99ycRWj5EJj/view?usp=sharing)

## Contact ##
For questions on usage, please open an issue, submit a pull request, or contact Melissa Gymrek (mgymrek AT ucsd.edu) or An Zheng (anz023 AT eng.ucsd.edu).
