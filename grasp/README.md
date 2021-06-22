# Robotic Grasping

**Description:** This folder includes the code for: (i) training a Franka Panda arm to grasp mugs in PyBullet, and (ii) OOD detection when the manipulator encounters mugs with poses that are statistically different from the poses in the training distribution and bowls instead of mugs.

## Dependencies
1. [PyTorch](https://pytorch.org/)
2. [CVXPY](https://www.cvxpy.org/) 
3. [MOSEK](https://www.mosek.com/)
4. [PyBullet](https://pybullet.org/)

## Training and Test Data
The training and test data was generated from the [ShapeNet dataset](https://shapenet.org/).

## Training
Training can be performed by running `trainGrasp_es.py` as follows:
```(python)
python trainGrasp_es.py train_config
```
where `train_config` refers to the `train_config.json` file which includes the path the training dataset of mugs as well as other hyperparameters. Pre-trained weights are provided in `Weights/` folder.

**Note:** Since we cannot share the ShapeNet dataset, the training code cannnot be executed out of the box.

#### Testing
Run `python oodd_obsavoid.py` to generate the plots and results that were used in this example.
