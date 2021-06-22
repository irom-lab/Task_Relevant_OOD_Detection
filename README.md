# Task-Driven Out-of-Distribution Detection for Robot Learning
This repository contains code for the out-of-distribution (OOD) detection approach presented in the paper titled: Task-Driven Out-of-Distribution Detection with Statistical Guarantees for Robot Learning. 

OOD detection refers to the task of detecting when a robot is operating in environments that are drawn from a different distribution than the environments used to train the robot. Our key idea for OOD detection relies on the following intuition: violation of the performance bound on test environments provides evidence that the robot is operating OOD. This idea is formalized via statistical techniques based on p-values and concentration inequalities. The resulting approach (i) provides guaranteed confidence bounds on OOD detection, and (ii) is task-driven and sensitive only to changes that impact the robot’s performance.

The code for computing the p-values and the confidence intervals is available in this file: [ood_detect.py](https://github.com/irom-lab/Task_Relevant_OOD_Detection/blob/include-grasp/ood_detect.py).

This approach is applied on the tasks of robotic grasping and navigation of a drone through obstacle fields. Details of the code for the robotic tasks can be found in the README of the following folders:
1. [Grasp](https://github.com/irom-lab/Task_Relevant_OOD_Detection/tree/include-grasp/grasp)
2. [Navigate](https://github.com/irom-lab/Task_Relevant_OOD_Detection/tree/include-grasp/navigate)
