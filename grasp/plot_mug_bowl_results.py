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
    
    load = True
    
    if load:
        with open("results/p_ood_mug_bowl.txt", "rb") as fp:   #Pickling
            p_val_list = pickle.load(fp)
        with open("results/delta_C_mug_bowl.txt", "rb") as fp:   #Pickling
            conf_list = pickle.load(fp)

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
            p_val = []
            conf = []
            for i in range(1,len(cost_list)+1):
                p_val.append(ood_p_value(cost_list[:i], bound))
                _, violation = ood_confidence(cost_list[:i], bound, deltap=0.04)
                conf.append(violation)
            p_val_list.append(p_val)
            conf_list.append(conf)
        
        with open("results/p_ood_mug_bowl.txt", "wb") as fp:   #Pickling
            pickle.dump(p_val_list, fp)
        with open("results/delta_C_mug_bowl.txt", "wb") as fp:   #Pickling
            pickle.dump(conf_list, fp)
    
    plot_object_type(
        list(range(1,numObjs+1)), 
        p_val_list, 
        conf_list, 
        legend=["Mug: $1 - p$", 
                "Mug: $\Delta C + 0.95$", 
                "Bowl: $1 - p$", 
                "Bowl: $\Delta C + 0.95$", 
                ]
    )