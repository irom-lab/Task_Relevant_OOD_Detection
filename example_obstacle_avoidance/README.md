# Task-Driven Out-of-Distribution Detection

This code uses the following:
- pytorch ≥ 1.4.0
- matplotlib
- scipy

If you are using Anaconda, you can run the following commands to create an environment and install the necessary packages.
```
conda create -n todd
conda activate todd
conda install pytorch=1.4.0 -c pytorch
pip install matplotlib scipy
```
## Swing example
#### Data
Go to [this link](https://drive.google.com/file/d/1zpqZbxp-7z3HOktoru5qvEx4ah7FBHBM/view?usp=sharing) and download the zip file of all the data used in this example. Unzip the folder and place the contents into `./matlab_gen_data/data/` of this repository. Alternatively, using Matlab, run `./matlab_gen_data/gen_data.m` to generate all training and testing data necessary for this example. 

#### Training
You may skip this step and use the pre-trained network in the `weights` folder. This network was deployed on the hardware example. Alternatively,
run the following commands to train the network and compute the PAC-Bayes bound:
```
python train.py --step 0  # trains prior
python train.py --step 1  # trains posterior
python train.py --step 2  # computes PAC-Bayes bound
```
Weights and bound values are saved into the `weights` folder. If you train your own network, change `policy_path` in `plot.py` to match `policy_path` in `train_swing.py`.

#### Testing
Run `python plot.py` to generate the plots that were used in this example.