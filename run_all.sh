echo 'Unzipping forecast files'
echo '============================'
unzip ./forecasts/access_forecasts.zip

echo ''
echo 'Running the computations'
echo '============================'
python ./code/reproducibility_global_vs_regional3.py
