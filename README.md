# Task-Driven Out-of-Distribution Detection for Robot Learning
This repository contains code for the out-of-distribution (OOD) detection approach presented in the paper titled: Task-Driven Out-of-Distribution Detection with Statistical Guarantees for Robot Learning. 

**Description:** OOD detection refers to the task of detecting when a robot is operating in environments that are drawn from a different distribution than the environments used to train the robot. We leverage Probably Approximately Correct (PAC)-Bayes theory in order to train a policy with a guaranteed bound on performance on the training distribution. Our key idea for OOD detection relies on the following intuition: violation of the performance bound on test environments provides evidence that the robot is operating OOD. This idea is formalized via statistical techniques based on p-values and concentration inequalities. The resulting approach (i) provides guaranteed confidence bounds on OOD detection, and (ii) is task-driven and sensitive only to changes that impact the robot’s performance.

The code for computing the p-values and the confidence intervals is available in this file: [ood_detect.py](./ood_detect.py).

This approach is applied on the tasks of robotic grasping and navigation of a drone through obstacle fields. Code for these robotic tasks and instructions to run it is available in the following folders:
1. [Grasp](./grasp)
2. [Navigate](./navigate)
