#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from grasp_test import evaluate_grasp
import pickle
from util.plotter import plot_object_type

import sys
sys.path.insert(0, '..')
from ood_detect import ood_confidence, ood_p_value

if __name__ == "__main__":
    
    np.random.seed(1)
    bound = 0.1
    print("PAC-Bound:", bound)
    
    config_file = "configs/config.json"
    grasper = evaluate_grasp(config_file)
    numObjs = 10
    cost_all = []
    
    load = True
    
    if load:
        with open("results/mug_bowl_costs.txt", "rb") as fp:   #Pickling
            cost_all = pickle.load(fp)

    else:
        """ Distribution shift in object type """
        p_val_list = []
        conf_list = []
        objects = ["mugs/SNC_v4_mug_xs/", "bowls_snc/"]
        for object_type in objects:
            _, cost_list, _ = grasper.test_policy_derandomized(
                numObjs=numObjs,
                obj_folder="geometry/"+object_type, 
                gui=False
            )
            cost_all.append(cost_list)
            
        with open("results/mug_bowl_costs.txt", "wb") as fp:   #Pickling
            pickle.dump(cost_all, fp)

    p_val_list = []
    Delta_C_list = []
    for cost_list in cost_all:
        p_val = []
        conf = []
        for i in range(1,len(cost_list)+1):
            p_val.append(ood_p_value(cost_list[:i], bound))
            _, violation = ood_confidence(cost_list[:i], bound, deltap=0.04)
            conf.append(violation)
        p_val_list.append(p_val)
        Delta_C_list.append(conf)
                    
    
    plot_object_type(
        list(range(1,numObjs+1)), 
        p_val_list, 
        Delta_C_list, 
        legend=["Mug: $1 - p$", 
                "Mug: $\Delta C + 0.95$", 
                "Bowl: $1 - p$", 
                "Bowl: $\Delta C + 0.95$", 
                ]
    )