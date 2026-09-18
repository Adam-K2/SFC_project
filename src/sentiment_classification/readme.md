# Sentiment classification folder README
## Overview  
This folder contains all source code related to the **sentiment_classification** in the project.  
Two separate tasks are included:

- **imdb.py** – sentiment task for imdb dataset  
- **sst5.py** – sentiment task for sst5 dataset  

Both scripts support **training and evaluation**.

---

## Running the scripts  
You can run each task directly:

```bash
python imdb.py
python sst5.py
```
Training for imdb is generally pretty slow, depending on the hardware on cpu it can possibly take up to 2 hours. On gpu estimated max 30 minutes.
Training of sst5 is pretty fast for mlp, cnn. GRU is bit longer maybe 5 minutes and transformer is something about 10 minutes.

## Pretrained models
Pretrained model checkpoints can be downloaded from the provided Google Drive (**google_drive.txt**) link due to repository size limitations.