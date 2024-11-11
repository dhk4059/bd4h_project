# bd4h_project

This project is currently a work-in-progress.

This purpose of this project is to replicate the procedure and results of the 2018 paper: **A disease inference method based on symptom extraction and bidirectional Long Short Term Memory networks**, available at https://ieeexplore.ieee.org/document/8621182.

## Datasets

The datasets used are located in the `data` folder, as follows:
- **summaries.csv**: contains GenAI-generated patient IDs and discharge summaries
- **diagnoses.csv**: contains GenAI-generated patient IDs and disease diagnoses
- **symptoms.csv**: contains the symptoms extracted by MetaMap for each patient ID


## Running `run_metamap.py`

`run_metamap.py` is responsible for preprocessing the datasets to extract symptoms from written reports on patient symptoms.

### Dependencies 
To run `run_metamap.py`, MetaMap2020 is required and can be found at: https://lhncbc.nlm.nih.gov/ii/tools/MetaMap.html. The Java Runtime Environment (JRE) is required to run MetaMap.

Furthermore, the python package `pymetamap` must be installed as well: https://github.com/AnthonyMRios/pymetamap.

`run_metamap.py` was tested and run on MetaMap2020's Linux distribution.

### How to Run

To run the file, run the command:
```bash
python run_metamap.py
```

This assumes that the [necessary MetaMap servers](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/README.html) have been launched already.

To run the file and also start the servers inside the script, run the command:
```bash
python run_metamap.py -s
```


## Running `main.ipynb`

`main.ipynb` is responsible for training and evaluating the relevant word embeddings + BiLSTMs discussed in the paper.

### Dependencies 
The following python packages are required:
- `nltk`
- `numpy`
- `pandas`
- `scikit-learn`
- `gensim`
- `tensorflow`


### How to Run
Upon running the file in Google Colab, you will be asked to upload the datasets. Upload the datasets from the `data` folder into the base directory of the Colab notebook.

Afterwards, simply choose `Run all`.



