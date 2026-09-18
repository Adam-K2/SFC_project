# SFC - LSTM nebo GRU - porovnání s vícevrstvým perceptronem, 1D konvoluční sítí nebo transformerem při zpracování sekvenčních dat README
## Overview
This repository contains code and documentaion for this project.  


## Folder structure
```
project/
├── doc/
└── src/
```

### doc/
Contains:
- Project documentation in pdf format 

### src/
Contains source code of the project. More detailed description in src folder.

### environment.yml
Source file with all libraries, ready to set up the project with anaconda with install.sh script.

### install.sh
Script for automatic installation of environment.

### requirements.txt
This file has the libraries, which could be installed by pip, but i would recommend using environment.yml.

# Instalation
## Automatic with script
For the automatic instalation run the following command:
```
source install.sh
```
This will install miniconda and create the environment.  
To activate it:
```
conda activate sfc_proj
```

If you have conda installed then it would be easier to run this command:
```
conda env create --file environment.yml --name sfc_proj
```

# Usage
## Train and evaluate
Each task includes two Python scripts that handle both training and evaluation.

- **Regression:** `energy.py`, `temperature.py`  
- **Sentiment:** `imdb.py`, `sst5.py`

Run any task with:

```bash
python energy.py
python temperature.py
python imdb.py
python sst5.py
```

## Demo scripts
These scripts are right inside the src folder. For **demo_sentiment.py**:
```
demo_sentiment.py [--dataset] [imdb|sst5] [--model] [mlp|cnn|gru|transformer]
```
Without arguments it run with default imdb dataset and mlp model.

For **demo_regression.py**:
```
demo_regression.py [--dataset] [temperature|energy] [--model] [mlp|cnn|gru|transformer]
```
Without arguments it run with default temperature dataset and mlp model. 