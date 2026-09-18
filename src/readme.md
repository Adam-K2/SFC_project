# Source folder README
## Overview
This repository contains all the source code for sentiment classification task and regression task.

## Folder structure
```
project/
├── regression_task/
└── sentiment_classification/
```

### regression_task/
Contains the code for training and evaluating the regression model.
Pretrained model weights are not included in the repository due to size limitations and can be downloaded through the provided Google Drive link.  

### sentiment_classification/
Contains the code for training and evaluating the sentiment classification models.
Model weights are also available through the Google Drive download link.

This folder additionally includes a Jupyter notebook used for experiments with pretrained word embeddings. These experiments were executed and evaluated in Google Colab.

### demo_regression.py
A demonstration script that loads the best pretrained regression model and performs evaluation on the test set.
It also generates plots where applicable (e.g., prediction vs. target comparison).
Plots that require training history (training vs. validation loss) are only available when running the full training pipeline inside the regression_task/ folder.

### demo_sentiment.py
Equivalent demo script for the sentiment classification task.
It loads the best pretrained model and runs evaluation on the test dataset.