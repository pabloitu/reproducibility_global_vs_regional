# Reproducing comparative test results for globally and regionally calibrated seismicity models for California, New Zealand, and Italy.
This repository provides the code, data, and additional resources to fully reproduce the comparative and consistency test results for the Global Earthquake Activity Rate (GEAR1) model and nineteen regional time-invariant seismicity models for California, New Zealand, and Italy reported in Bayona et al. (in review). The experiment needs 5GB in disk to run and takes about one hour on a modern desktop computer if the number of simulations per forecast and per test (except for the Poisson and Negative Binomial Distribution (NBD) number N-tests) is set to 1000.

## Code description
The Python scripts needed to run this forecast experiment can be found in the `code` directory of this repository. This folder contains the `download_data.py`, which downloads forecast files, earthquake catalogs, and additional data from [Zenodo](https://zenodo.org/record/7086053#.YySPZGzMJz0), and the `reproducibility_global_vs_regional.py` file, which runs the computations and creates the figures presented in the manuscript. Finally, the `run_all.sh` file, in the top-level directory, is a shell script that runs the entire experiment by only typing `bash ./run_all.sh`.

## Software dependencies

python= 3.8.6

numpy= 1.19.2 

pycsep=0.6.0 

## Further software requirements
To run this reproducibility software package, the user must have a pycsep environment installed on her/his/their machine ('tsr-gr' in this example). The easiest way to install pycsep is using `conda`; however, pycsep can also be installed using `pip` or built from source (see the [Documentation on how to install pyCSEP](https://docs.cseptesting.org/getting_started/installing.html)).

```
conda create -n tsr-gr
conda activate tsr-gr
conda install --channel conda-forge numpy=1.19.2 pycsep=0.6.0
```

In addition, the user must have access to a Unix shell with python3 and the `requests` library included. If this is not the case, she/he/they can install the library using:

```
conda install requests
```

Furthermore, the user must install `zenodo_get` to download the forecast files from Zenodo. This library can be installed using:

```
pip3 install zenodo_get
```

## Running instructions
These instructions assume that the user is "within" the (e.g. tsr-gr) environment, with python3 and the `request` library already installed. Thus, running this experiment is as easy as typing:

```
git clone https://github.com/bayonato89/reproducibility_global_vs_regional.git
cd reproduciblity_global_vs_regional/forecasts
zenodo_get 7116221
cd ..
bash ./run_all.sh
```
