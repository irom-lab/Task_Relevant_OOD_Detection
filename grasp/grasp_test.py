#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import math
import torch
import numpy as np
from numpy import array
import json
from src.nn_grasp import PolicyNet
from src.grasp_rollout_env import GraspRolloutEnv
from src.pac_es import kl_inverse

import sys
sys.path.insert(0, '..')

class evaluate_grasp:
    
    def __init__(self, config_file):
        
        args = json.load(open(config_file))
        
        self.device = args['device']
        self.num_cpus = args['num_cpus']
        self.long_finger = args['long_finger']
        self.actor_pr_path = args['actor_pr_path']
        self.out_cnn_dim = args['out_cnn_dim']
        self.z_conv_dim = args['z_conv_dim']
        self.z_mlp_dim = args['z_mlp_dim']
        self.z_total_dim = self.z_conv_dim + self.z_mlp_dim
        self.obj_folder = args['obj_folder']
        
        info_file = args['info_file']
        info =	torch.load(info_file)
        emp_rate, bound, _, self.mu, self.logvar, self.seed_states = info['best_bound_data']
        self.sigma = (0.5*self.logvar).exp()

        # Load prior policy
        self.actor_pr = PolicyNet(
            input_num_chann=1,
            dim_mlp_append=0,
            num_mlp_output=5,
            out_cnn_dim=self.out_cnn_dim,
            z_conv_dim=self.z_conv_dim,
            z_mlp_dim=self.z_mlp_dim
        ).to(self.device)
        self.actor_pr.load_state_dict(torch.load(self.actor_pr_path, map_location=self.device))
        for name, param in self.actor_pr.named_parameters():
            param.requires_grad = False
        print(self.actor_pr.eval())
        
        self.rollout_env = GraspRolloutEnv(
            actor=self.actor_pr, 
            z_total_dim=self.z_total_dim,
            num_cpus=self.num_cpus,
            checkPalmContact=1,
            useLongFinger=self.long_finger
        )
        
    def _get_object_config(self, numObjs, obj_ind_list, obj_folder, x_lim=[0.45, 0.55], y_lim=[-0.05, 0.05], obj_seed=0):
        np.random.seed(obj_seed)
        obj_x = np.random.uniform(
            low=x_lim[0],
            high=x_lim[1], 
            size=(numObjs, 1)
        )
        obj_y = np.random.uniform(
            low=y_lim[0], 
            high=y_lim[1], 
            size=(numObjs, 1)
        )
        obj_yaw = np.random.uniform(low=-np.pi, high=np.pi, size=(numObjs, 1))
        if Path(obj_folder+"dim.npy").exists():
            z_data = np.load(obj_folder+"dim.npy")
            obj_z = np.expand_dims(z_data[obj_ind_list, 2]/2, axis=1)
            objOrn = np.hstack((np.pi/2*np.ones((numObjs, 1)),np.zeros((numObjs, 1)), obj_yaw))
        else:
            obj_z = 0.005*np.ones((numObjs, 1))
            objOrn = np.hstack((np.zeros((numObjs, 2)), obj_yaw))
        objPos = np.hstack((obj_x, obj_y, obj_z))
        # print(objPos)
        objPathInd = np.arange(0,numObjs)  # each object has unique initial condition -> one env
        objPathList = []
        for obj_ind in obj_ind_list:
            objPathList += [obj_folder + str(obj_ind) + '.urdf']
        return (objPos, objOrn, objPathInd, objPathList)

    def test_policy_derandomized(self, numObjs=10, seed=0, obj_seed=0, obj_folder="geometry/bowls/", x_lim=[0.4, 0.6], y_lim=[-0.05, 0.05], gui=False):
        # Generate new obstacles
        objPos, objOrn, objPathInd, objPathList = self._get_object_config(
            numObjs=numObjs, 
            obj_ind_list=list(range(numObjs)),
            obj_folder=obj_folder,
            x_lim=x_lim,
            y_lim=y_lim,
            obj_seed=obj_seed
        )

        torch.manual_seed(seed)
        # Config all test trials
        epsilons = torch.normal(
            mean=0.,
            std=1.,
            size=(1,self.z_total_dim)
        )
        zs = self.mu + self.sigma*epsilons
        estimate_cost, cost = self._execute_parallel_derandomized(zs, objPos, objOrn, objPathList, objPathInd, gui)
        return estimate_cost, cost, len(objPos)

    def _execute_parallel_derandomized(self, zs, objPos, objOrn, objPathList, objPathInd, gui):
        # Run test trials and get estimated true cost
        with torch.no_grad():  # speed up
            estimate_success_list = self.rollout_env.parallel_derandomized(
                zs=zs,
                objPos=objPos,
                objOrn=objOrn,
                objPathInd=objPathInd,
                objPathList=objPathList,
                gui=gui
            )
        cost = array([1-s for s in estimate_success_list])
        estimate_cost = np.mean(cost)
        return estimate_cost, cost
	
    def emp_cost_derandomized(self, seed=0, gui=False):

        # Load Allen's obstacles
        checkpoint = torch.load('Weights/model_5')
        objPos, objOrn, objPathInd, _ = checkpoint['trainEnvs']
        objPathList = ['geometry/mugs/SNC_v4_mug_xs/'+str(s)+'.urdf' for s in objPathInd]
    
        torch.manual_seed(seed)
        # Config all test trials
        epsilons = torch.normal(
            mean=0.,
            std=1.,
            size=(1,self.z_total_dim)
        )
        zs = self.mu + self.sigma*epsilons
        
        estimate_cost, cost = self._execute_parallel_derandomized(zs, objPos, objOrn, objPathList, objPathInd, gui)
        return estimate_cost, cost, len(objPos)
	
    def derandomized_pac_bayes(self, delta):
        emp_cost, _, N = self.emp_cost_derandomized()
        print(N)
        renyi = Renyidiv_gaussian(self.mu, self.logvar, torch.zeros_like(self.mu), torch.zeros_like(self.mu))
        # renyi = renyi_divergence(alpha, self.mu, self.sigma, torch.zeros_like(self.mu), torch.ones_like(self.sigma))
        reg = (renyi + np.log(2*(N**0.5)/(delta**3)))/(2*N)
        # print(emp_cost + np.sqrt(reg))
        derandom_kl_inv = kl_inverse(emp_cost, 2*reg)
        return derandom_kl_inv, emp_cost, renyi, N, reg

def Renyidiv_gaussian(mu1, logvar1, mu2, logvar2, a=2):
    mu1 = torch.flatten(mu1)  # make sure we are 1xd so torch functions work as expected
    logvar1 = torch.flatten(logvar1)
    mu2 = torch.flatten(mu2)
    logvar2 = torch.flatten(logvar2)

    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    vara = a * var2 + (1 - a) * var1

    finiteness_check = a * (1 / var1) + (1 - a) * (1 / var2)
    if torch.sum(finiteness_check > 0) < mu1.shape[0]:
        return torch.Tensor([float("Inf")])

    sum_logvara = torch.sum(torch.log(vara))
    sum_logvar1 = torch.sum(logvar1)
    sum_logvar2 = torch.sum(logvar2)

    r_div = (a/2) * torch.sum(((mu1 - mu2) ** 2) * vara)
    r_div -= 1 / (2*a - 2) * (sum_logvara - (1-a)*sum_logvar1 - a*sum_logvar2)
    return r_div    
    
if __name__=="__main__":
    config_file = "configs/config.json"
    grasper = evaluate_grasp(config_file)
    
    delta = 0.005 #(bound is 1-2*delta, so this will will give 99% confidence)
    kl_inv_PAC, emp_cost, _, _, _ = grasper.derandomized_pac_bayes(delta)
    print("kl_inv_bound:", kl_inv_PAC)