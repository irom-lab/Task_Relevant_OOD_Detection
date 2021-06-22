# Task-Driven Out-of-Distribution Detection


## Grasping Example





## Vision-Based Obstacle Avoidance
The code uses the following:
- python = 3.8
- pytorch = 1.7.1
- matplotlib
- scipy

If you are using Anaconda, you can run the following commands to create an environment and install the necessary packages.
```
conda create -n todd python=3.8
conda activate todd
conda install pytorch=1.7.1 -c pytorch
pip install matplotlib scipy
```

#### Data
Go to [this link](https://drive.google.com/file/d/1zpqZbxp-7z3HOktoru5qvEx4ah7FBHBM/view?usp=sharing) and download the zip file of all the data used in this example. 
Unzip the folder and place the contents into `./obsavoid/matlab_gen_data/data/`. 
Alternatively, using Matlab, run `./obsavoid/matlab_gen_data/gen_data.m` to generate all training and testing data necessary for this example. 

#### Training
You may skip this step and use the pre-trained network in the `./obsavoid/weights` folder. 
This pre-trained network was deployed on the hardware setup. 
Alternatively, run the following commands to train the network and compute the PAC-Bayes bound:
```
python train_obsavoid.py --step 0  # trains prior
python train_obsavoid.py --step 1  # trains posterior
python train_obsavoid.py --step 2  # computes PAC-Bayes bound
```
Weights and bound values are saved into the `obsavoid/weights` folder. 
If you train your own network, change `policy_path` in `oodd_obsavoid.py` to match `policy_path` in `train_obsavoid.py`.

#### Testing
Run `python oodd_obsavoid.py` to generate the plots and results that were used in this example.
