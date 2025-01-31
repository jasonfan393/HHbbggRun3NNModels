# Cornell Mjj Regressor 
Currently includes mjj_regressor model evaluator to be applied upon v2-style HiggsDNA HHbbgg parquets.

Set up necessary conda environment with 
```
conda install -f environment.yml
conda activate mjj_env
```
Run with:
```
python mjj_regressor_evaluator.py --input_location (Directory containing higgsdna parquets) --output_location test_dir
```

Currently does not included the training (which was done in a separate jupyter notebook), this will be added soon.
