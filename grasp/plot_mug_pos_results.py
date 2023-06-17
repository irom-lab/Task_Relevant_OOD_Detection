#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from grasp_test import evaluate_grasp
import pickle
from util.plotter import plot_compare_methods

import sys
sys.path.insert(0, '..')
from ood_detect import ood_confidence, ood_p_value

if __name__ == "__main__":
    
    np.random.seed(1)
    bound = 0.1
    print("PAC-Bound:", bound)
    
    config_file = "configs/config.json"
    grasper = evaluate_grasp(config_file)
    
    load = True

    """ Distribution shift in mug location """

    emp_cost_sim_file = "results/emp_cost_mug_pos.txt"
    cost_all_sim_file = "results/cost_all_mug_pos.txt"
    sim_output_file = "results/mug_pos_indicators.png"

    emp_cost_hw_file = "results/hardware_experiments/emp_cost_mug_pos_hw.txt"
    cost_all_hw_file = "results/hardware_experiments/cost_all_mug_pos_hw.txt"
    hw_output_file = "results/hardware_experiments/mug_pos_indicators_hw.png"

    if load:
        with open(emp_cost_sim_file, "rb") as fp:   #Pickling
            emp_cost_test = pickle.load(fp)
        with open(cost_all_sim_file, "rb") as fp:   #Pickling
            cost_all = pickle.load(fp)
    else:
        x_lim_list = [[0.45, 0.55], [0.4, 0.6], [0.35, 0.65], [0.3, 0.7], [0.25, 0.75], [0.2, 0.8]]
        y_lim_list = [[-0.05, 0.05],[-0.1, 0.1],[-0.15, 0.15],[-0.2, 0.2],[-0.25, 0.25], [-0.3, 0.3]]
        num_seeds = 20
        
        emp_cost_test = []
        cost_all = []
        for (x_lim,y_lim) in zip(x_lim_list, y_lim_list):
            # print((x_lim,y_lim))
            p_ood_detect = []
            conf_ood_detect = []
            emp_cost = 0
            cost_list = []
            for seed in range(num_seeds):
                _, cost, _ = grasper.test_policy_derandomized(
                    numObjs=10, 
                    obj_folder="geometry/mugs/SNC_v4_mug_xs/", 
                    x_lim=x_lim, 
                    y_lim=y_lim, 
                    gui=False,
                    obj_seed=seed
                )
                cost_list.append(cost)
                emp_cost += np.mean(cost)
            emp_cost_test.append(emp_cost/num_seeds)
            cost_all.append(cost_list)

        with open(emp_cost_sim_file, "wb") as fp:   #Pickling
            pickle.dump(emp_cost_test, fp)
        with open(cost_all_sim_file, "wb") as fp:   #Pickling
            pickle.dump(cost_all, fp)

    p_all = []
    Delta_C_all = []
    for cost_list in cost_all:
        p_list = []
        violation_list = []
        for cost in cost_list:
            p_val = 1 - ood_p_value(cost, bound)
            p_list.append(1 - p_val)
            _, violation = ood_confidence(cost, bound, deltap_A=0.04)
            violation_list.append(violation)
        p_all.append(p_list)
        Delta_C_all.append(violation_list)
            
    p_all = np.array(p_all).transpose()
    Delta_C_all = np.array(Delta_C_all).transpose()
        
    ys = [p_all, Delta_C_all+0.95]
    plot_compare_methods(np.array(emp_cost_test), ys, legend = ["$1 - p_A$", "$\Delta C_A + 0.95$"], output_file=sim_output_file)