## Original model: 
CNN with residual blocks<br>
input: pixel map<br>
output: 3 classes classification


## Multi-output model
This model is trained to predict flavour (NC, nu_e CC, nu_mu CC), number of protons (0, 1, 2, >3) and number of charged pions (0, 1+).

If running the multi-output (3output) script, you will need to look up "CHANGE PATH" in the code to see where to change hard-coded paths to proper directories of stored data, saved model, etc. 
There is an option as well to load up pre-saved weights, if needed. 
Currently this code relies on truth information being pre-stored (and pre-selected <4 GeV) in a dataframe .pkl file. 

Ultimately, you will run a script that looks something like:
```
python Rice_ResNet_atmo_3output.py --num_epochs 25 --learning_rate 3e-4 --batch_size 64 --pixel_map_size 200 --listname 'full_statistics' --test_name 'ResNet_20240204' --path_checkpoint '/home/sophiaf/CNN/20240204/'
```
See the full code and argument list to observe what these variables mean.
