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
The latest model is trained with the following arguments:
```
python mjj_trainer.py --model mjj_model_2022_MET --vars configs/variables_mjj_alljets_MET.json --out_dir plots --year 2022
python mjj_trainer.py --model mjj_model_2023_MET --vars configs/variables_mjj_alljets_MET.json --out_dir plots --year 2023
```
Plots may be run standalone (with a pre-generated model) with the --plotsOnly option
--doGridSearch option performs a randomized search for hyperparameter tuning, but can be slow on lxplus
## mjj regressor evaluator 
takes HiggsDNA parquets (that have PNet info added) and applies extra columns from the regressor. For now this runs over all merged HiggsDNA parquets in some given folder, but will be updated to accept a json input. 

Run with:

```
python mjj_regressor_evaluator.py --input_location (Directory containing higgsdna parquets) --output_location test_dir
```
Example (latest production on lxplus):

```
python mjj_regressor_evaluator.py --input_location /eos/user/e/evourlio/HiggsDNA_v3Production/ --output_location output_dir --vars configs/variables_mjj_alljets_MET.json --model mjj_model_2022_MET --year 2022
python mjj_regressor_evaluator.py --input_location /eos/user/e/evourlio/HiggsDNA_v3Production/ --output_location output_dir --vars configs/variables_mjj_alljets_MET.json --model mjj_model_2023_MET --year 2023
```
