#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:08:40 2020

@author: Sushant Veer
"""

import json
import os

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def create_config(feedback, 
                  grad_method, 
                  num_func_eval,
                  num_envs_per_policy,
                  loss,
                  init_sd,
                  lr_mu,
                  lr_logvar,
                  seed,
                  delta_list):
    if grad_method[:5]=='coNES':
        for delta in delta_list:
            # config['num_itr'] = int(budget/(2*num_func_eval))
            config['vision'] = feedback[1]
            config['delta'] = delta
            config['grad_method'] = grad_method
            config['num_func_eval'] = num_func_eval
            config['seed'] = seed
            config['loss_func'] = loss
            config['lr_mu'] = lr_mu
            config['lr_logvar'] = lr_logvar
            config['init_sd'] = init_sd
            config['num_envs_per_policy'] = num_envs_per_policy
            config['save_file'] = feedback[0]+'_'+loss+'_'+grad_method+'_eval='+str(num_func_eval)+'_envs='+str(num_envs_per_policy)\
                                + '_init_sd='+str(init_sd)+'_lr_mu='+str(lr_mu)+'_lr_logvar='+str(lr_logvar)+'_seed='+str(seed)\
                                +'_delta='+str(delta)
            config_file_name = 'eval='+str(num_func_eval)+'_envs='+str(num_envs_per_policy)\
                             + '_init_sd='+str(init_sd)+'_lr_mu='+str(lr_mu)+'_lr_logvar='+str(lr_logvar)+'_seed='+str(seed)\
                             +'_delta='+str(delta)+'.json'
            json.dump(config, open('configs/'+feedback[0]+'/'+loss+'/'+grad_method+'/'+config_file_name, 'w'), indent=4)
    else:
        # config['num_itr'] = int(budget/(2*num_func_eval))
        config['vision'] = feedback[1]
        if grad_method=='CMA':
            config['method'] = grad_method
        config['grad_method'] = grad_method
        config['num_func_eval'] = num_func_eval
        config['seed'] = seed
        config['loss_func'] = loss
        config['lr_mu'] = lr_mu
        config['lr_logvar'] = lr_logvar
        config['init_sd'] = init_sd
        config['num_envs_per_policy'] = num_envs_per_policy
        config['save_file'] = feedback[0]+'_'+loss+'_'+grad_method+'_eval='+str(num_func_eval)+'_envs='+str(num_envs_per_policy)\
                            + '_init_sd='+str(init_sd)+'_lr_mu='+str(lr_mu)+'_lr_logvar='+str(lr_logvar)+'_seed='+str(seed)
        config_file_name = 'eval='+str(num_func_eval)+'_envs='+str(num_envs_per_policy)\
                         + '_init_sd='+str(init_sd)+'_lr_mu='+str(lr_mu)+'_lr_logvar='+str(lr_logvar)+'_seed='+str(seed)+'.json'
        json.dump(config, open('configs/'+feedback[0]+'/'+loss+'/'+grad_method+'/'+config_file_name, 'w'), indent=4)

if __name__ == '__main__':
    
    config = {
                "itr_start": 0,
                "timestep_start": 0,
                "num_itr": 5000,
                "num_params": 1000,
                "num_func_eval": 32,
                "num_envs_per_policy": 8,
                "RL": True,
                "vision": False,
                "loss_func": "",
                "init_sd": 1,
                "grad_method": "",
                "lr_mu": 1,
                "lr_logvar": 0.1,
                "seed": 0,
                "delta": 0,
                "num_cpu": 2,
                "num_gpu": 0,
                "num_sample_F_est": 8000,
                "load_weights": False,
                "load_weights_from": "",
                "load_optimizer": False,
                "logging": True,
                "save_file": "",
                "loss_min": 1e18,
                "timestep_max": 5e7
    }
    
    budget = 5e5
    grad_method_list = ['centered_utility','eNES_logvar_centered_utility','coNES_logvar_centered_utility','CMA']
    delta_list = [1000]
    seed_list = [0,1,2,3,4,5,6,7,8,9]
    feedback_list = [['state', False]]
    num_func_eval_list = [40]
    num_envs_per_policy_list = [1]
    loss_list = ['Half_Cheetah', 'Walker2D']
    lr_mu_list = [0.01]
    lr_logvar_list = [0.01]
    init_sd_list = [0.02]
    
    for feedback in feedback_list:
        make_dir('configs/'+feedback[0])
        for loss in loss_list:
            make_dir('configs/'+feedback[0]+'/'+loss)
            for init_sd in init_sd_list:
                for grad_method in grad_method_list:
                    make_dir('configs/'+feedback[0]+'/'+loss+'/'+grad_method)
                    for lr_mu in lr_mu_list:
                        for lr_logvar in lr_logvar_list:
                            for seed in seed_list:
                                for num_func_eval in num_func_eval_list:
                                    for num_envs_per_policy in num_envs_per_policy_list:
                                        create_config(
                                            feedback, 
                                            grad_method, 
                                            num_func_eval,
                                            num_envs_per_policy,
                                            loss,
                                            init_sd,
                                            lr_mu, 
                                            lr_logvar, 
                                            seed,
                                            delta_list
                                        )
                                        