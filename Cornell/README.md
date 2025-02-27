# Run3 HHbbgg Mjj Regressor 
Currently includes mjj_regressor model evaluator to be applied upon v2-style HiggsDNA HHbbgg parquets.

Set up necessary conda environment with 
```
conda install -f environment.yml
conda activate mjj_env
```
## mjj trainer
Contains a version of the regressor training. Also generates plots for measuring performance of DNN. Runs on lxplus, but is slow and typically run on a local system with a decent GPU
Run with default settings with:
```
python mjj_trainer.py
```
Plots may be run standalone (with a pre-generated model) with the --plotsOnly option
## mjj regressor evaluator 
takes HiggsDNA parquets (that have PNet info added) and applies extra columns from the regressor. 
Run with:
```
python mjj_regressor_evaluator.py --input_location (Directory containing higgsdna parquets) --output_location test_dir
```

