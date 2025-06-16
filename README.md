# Feature Scaling Induced Data Leakage Quantification in Machine-learning Based Voice Pathology Detection

This repository contains the code for the paper "Feature Scaling Induced Data Leakage Quantification in Machine-learning Based Voice Pathology Detection" 
by Jan Vrba, Jakub Steinbach, Tomáš Jirsa and Noriyasu Homma. DOI: XYZ

## Requirements 
Used libraries and software
- Python 3.13.2
- see requiretemnts.txt for all dependencies
- we recommend using virtual environment and using `pip install -r requirements.txt` to install all requirements

Used setup for experiments
- AMD Ryzen 9 5900X
- 112 GB RAM
- 1TB SSD hard drive
- Ubuntu 24.04.2 LTS

## Dataset preparation
The SVD dataset is not included in this repository due to the license reason, but it can be downloaded from publicly
available website. Please, follow the instructions in our repository [available here](https://github.com/aailab-uct/Automated-Robust-and-Reproducible-Voice-Pathology-Detection/blob/main/README.md).

Once the `features.csv` file is generated, place it into the `data` folder, run `flatten_features.py` to generate the `flattened_features.csv` file.

For any following work, we assume following directory structure:

```
vpd_scaling_leakage_study
└───data
    │   features.csv
    │   flattened_features.csv
    │   voiced_features_8000_fft.csv
```

## Reproducing the results

After data preparation, run `main.py` to run the calculations. The results will be saved in the form of json files in four folders named `XXX_results` for randomly splitted data and `XXX_results_stratified` for stratified data split. Note that `XXX` represents the database name.

You can utilize the `result_tables.ipynb` notebook to generate results in the form of Tables 3 to 7. Similarly, you can utilize the `permutation_test_bias.ipynb` notebook to conduct the permutation test of statistical significance of bias for each dataset-transformer-model-split combination.


