echo 'Downloading data from Zenodo'
echo '============================'
python ./code/download_data.py

echo ''
echo 'Running the computations'
echo '============================'
python ./code/reproducibility_global_vs_regional.py
