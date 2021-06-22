clear;
close all;
folder = "data/";
addpath('functions');
load('matlab_files/prims.mat');
verbose = 0;

easy = 0;
hard = 0;
variable_difficulty = 0;
ir_shift = 0;
add_app = "";


%% Data for training prior, training posterior, and testing
num_envs = 10000;
num_obs = 9;
add_app = "_prior";
generate_data_cont

add_app = "_post";
generate_data_cont

num_envs = 50000;
add_app = "_test";
generate_data_cont
add_app = '';


%% hardware OOD data: easy env
num_obs = 6;
easy = 1;
num_envs = 1000;
add_app = "_easy_6";
generate_data_cont
easy = 0;

%% hardware OOD data: harder env
num_obs = 9;
hard = 1;
add_app = "_hard_9";
generate_data_cont
hard = 0;


%% Data for precision comparison plot
vds = [-9, -4, -3, -2, -1, 0.0001, 1, 2, 3, 3.5, 3.8, 3.9, 4, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.4, 9];
for variable_difficulty = vds
    disp(variable_difficulty)
    if variable_difficulty < 0
        easy = 1;
    else
        easy = 0;
    end
    num_obs = 9;
    num_envs = 2000;
    if variable_difficulty == 0.0001
        add_app = "_vd0";
    else
        add_app = "_vd" + string(variable_difficulty);
    end
    generate_data_cont
end
easy = 0;


%% Data for bound violation check
variable_difficulty = 4.12;
num_obs = 9;
num_envs = 50000;
add_app = '_vd' + string(variable_difficulty);
generate_data_cont
variable_difficulty = 0;


%% Data for task-irrelevant shift 
ir_shift = 1;
num_obs = 4;
num_envs = 2000;
add_app = '_ir_shift';
generate_data_cont
ir_shift = 0;



clear





