#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from grasp_test import evaluate_grasp
import pickle
from util.plotter import plot_compare_methods

import sys
sys.path.insert(0, '..')
from utils.util_oodd import ood_confidence, ood_p_value

if __name__ == "__main__":
    
    np.random.seed(1)
    bound = 0.1
    print("PAC-Bound:", bound)
    
    config_file = "configs/config.json"
    grasper = evaluate_grasp(config_file)
    
    load = True

    """ Distribution shift in mug location """

    if load:
        with open("results/emp_cost_mug_pos.txt", "rb") as fp:   #Pickling
            emp_cost_test = pickle.load(fp)
        p_all = np.load("results/p_ood_mug_pos.npy")
        violation_all = np.load("results/delta_C_mug_pos.npy")
    else:
        x_lim_list = [[0.45, 0.55], [0.4, 0.6], [0.35, 0.65], [0.3, 0.7], [0.25, 0.75], [0.2, 0.8]]
        y_lim_list = [[-0.05, 0.05],[-0.1, 0.1],[-0.15, 0.15],[-0.2, 0.2],[-0.25, 0.25], [-0.3, 0.3]]
        num_seeds = 20
        
        p_all = []
        violation_all = []
        emp_cost_test = []
        for (x_lim,y_lim) in zip(x_lim_list, y_lim_list):
            # print((x_lim,y_lim))
            p_ood_detect = []
            conf_ood_detect = []
            emp_cost = 0
            p_list = []
            violation_list = []
            for seed in range(num_seeds):
                _, cost, _ = grasper.test_policy_derandomized(
                    numObjs=10, 
                    obj_folder="geometry/mugs/SNC_v4_mug_xs/", 
                    x_lim=x_lim, 
                    y_lim=y_lim, 
                    gui=False,
                    obj_seed=seed
                )
                p_val = 1 - ood_p_value(cost, bound)
                p_list.append(1 - p_val)
                _, violation = ood_confidence(cost, bound, deltap=0.04)
                violation_list.append(violation)
                emp_cost += np.mean(cost)
            p_all.append(p_list)
            violation_all.append(violation_list)
            emp_cost_test.append(emp_cost/num_seeds)
            
        p_all = np.array(p_all).transpose()
        violation_all = np.array(violation_all).transpose()
        
        np.save("results/p_ood_mug_pos.npy", p_all)
        np.save("results/delta_C_mug_pos.npy", violation_all)
        
        with open("results/emp_cost_mug_pos.txt", "wb") as fp:   #Pickling
            pickle.dump(emp_cost_test, fp)
        
    ys = [p_all, violation_all+0.95]
    plot_compare_methods(np.array(emp_cost_test), ys, legend = ["$1 - p$", "$\Delta C + 0.95$"])