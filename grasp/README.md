# Robotic Grasping

**Description:** This folder includes the code for: (i) training a Franka Panda arm to grasp mugs in PyBullet, and (ii) OOD detection when the manipulator encounters mugs with poses that are statistically different from the poses in the training distribution and bowls instead of mugs.

## Dependencies
1. [PyTorch](https://pytorch.org/)
2. [CVXPY](https://www.cvxpy.org/) 
3. [MOSEK](https://www.mosek.com/)
4. [PyBullet](https://pybullet.org/)
5. [Ray](https://ray.io/)

## Training and Test Data
The training and test data was generated from the [ShapeNet dataset](https://shapenet.org/).

## Training
Training can be performed by running `trainGrasp_es.py` as follows:
```python
python trainGrasp_es.py train_config
```
where `train_config` refers to the `train_config.json` file which includes the path the training dataset of mugs as well as other hyperparameters. Pre-trained weights are provided in `Weights/` folder.

**Note:** Since we cannot share the ShapeNet dataset, the training code cannnot be executed out of the box.

## Compute the derandomized PAC-Bayes bound
Run `python grasp_test.py` to compute the [derandomized PAC-Bayes](https://arxiv.org/pdf/2102.08649.pdf) bound.

## OOD Detection Results
Executing `plot_mug_pos_results.py` generates the plot for shifts in the distribution of poses for the mugs; (Fig. 2(b) in the paper). Executing `plot_mug_bowl_results.py` generates the plot that compares the OOD indicators between mug picking and bowl picking.

Since the rollouts require the ShapeNet dataset, for convenience, the costs for the rollouts are provided in the `results/` folder. By setting `load=True` in `plot_mug_pos_results.py` and `plot_mug_bowl_results.py`, the saved costs are used to compute the OOD indicators.
