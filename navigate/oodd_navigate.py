import numpy as np
from numpy.lib.shape_base import dsplit
import warnings
import torch
import json
import sys
sys.path.insert(0, '..')
from models.models_swing import SPolicy
from utils.util_models import run_policy, load_data, load_weights
from utils.util import plot_wind, plot_cardinality, plot_compare_precision, plot_combined_detector
from ood_detect import ood_p_value_batch, ood_confidence_batch, ood_msp_batch, ood_maxlogit_batch
warnings.filterwarnings('ignore')

# policy_path = 'post_1'

policy_path = 'post_datasets_'
datasets = ["data100", "data300", "data500", "data750", "data1000"]
default_dataset = "data500"

def get_increasing_difficulty_data(dataset = 'data500'):
    OOD_data = {'OOD_vd-5': load_data(app='_vd-9', option='prim_cost', dataset = dataset),
                'OOD_vd-4': load_data(app='_vd-4', option='prim_cost', dataset = dataset),
                'OOD_vd-3': load_data(app='_vd-3', option='prim_cost', dataset = dataset),
                'OOD_vd-2': load_data(app='_vd-2', option='prim_cost', dataset = dataset),
                'OOD_vd-1': load_data(app='_vd-1', option='prim_cost', dataset = dataset),
                'ID_vd0': load_data(app='_vd0', option='prim_cost', dataset = dataset),
                'OOD_vd1': load_data(app='_vd1', option='prim_cost', dataset = dataset),
                'OOD_vd2': load_data(app='_vd2', option='prim_cost', dataset = dataset),
                'OOD_vd3': load_data(app='_vd3', option='prim_cost', dataset = dataset),
                'OOD_vd4': load_data(app='_vd3.5', option='prim_cost', dataset = dataset),
                'OOD_vd5': load_data(app='_vd3.8', option='prim_cost', dataset = dataset),
                'OOD_vd6': load_data(app='_vd3.9', option='prim_cost', dataset = dataset),
                'OOD_vd7': load_data(app='_vd4', option='prim_cost', dataset = dataset),
                'OOD_vd8': load_data(app='_vd4.05', option='prim_cost', dataset = dataset),
                'OOD_vd9': load_data(app='_vd4.1', option='prim_cost', dataset = dataset),
                'OOD_vd10': load_data(app='_vd4.15', option='prim_cost', dataset = dataset),
                'OOD_vd11': load_data(app='_vd4.2', option='prim_cost', dataset = dataset),
                'OOD_vd12': load_data(app='_vd4.25', option='prim_cost', dataset = dataset),
                'OOD_vd13': load_data(app='_vd4.3', option='prim_cost', dataset = dataset),
                'OOD_vd14': load_data(app='_vd4.4', option='prim_cost', dataset = dataset),
                'OOD_vd15': load_data(app='_vd9', option='prim_cost', dataset = dataset),
                }
    return OOD_data


def get_task_irrelevant_data(dataset = 'data500'):
    print(dataset)
    OOD_data = {
                'ID': load_data(app='_vd0', option='prim_cost', dataset = dataset),
                'OOD': load_data(app='_ir_shift', option='prim_cost', dataset = dataset),
                }
    return OOD_data


def get_hardware_data():
    data = json.load(open('hardware_data.json','r'))
    return data


def get_bound_validation_data(dataset = 'data500'):
    data = {'ID': load_data(app='_test', option='prim_cost', dataset = dataset),
            'OOD': load_data(app='_vd4.12', option='prim_cost', dataset = dataset),
            }
    return data


def get_train_data(dataset = 'data500'):
    data = load_data(app='_post', option='prim_cost', dataset = dataset)
    return data


def get_comparison_detections(policy, data, upper_bound, lower_bound, deltap_A = 0.04, deltap_B = 0.04, m=10, trials=20):
    ps = np.zeros((len(data), trials))
    cs = np.zeros((len(data), trials))
    msps = np.zeros((len(data), trials))
    maxlogits = np.zeros((len(data), trials))
    cs_wd = np.zeros((len(data), trials))
    ps_wd = np.zeros((len(data), trials))
    x = []
    for key in data:  # processing step to speed up trials
        depth_maps, prim_costs = data[key]

        model_output = policy(depth_maps)
        _, y_OOD = run_policy(policy, depth_maps, prim_costs)

        cost = y_OOD[:, 0].detach().numpy()  # policy(x_OOD) = y_OOD because that's the actual cost
        model_output = model_output.detach().numpy()
        data[key] = (cost, model_output)
        x.append(np.mean(cost))

    #     print(key, np.mean(cost))
    #
    # exit()

    for trial in range(trials):
        p_oods = []
        p_wds = [] 
        c_oods = []
        c_wds = [] 
        msp_oods = []
        maxlogit_oods = []
        for key in data:
            cost, model_output = data[key]  # model_output.detach().numpy()
            p = np.random.permutation(len(cost))
            cost = cost[p]
            model_output = model_output[p]

            p_ood = ood_p_value_batch(cost[:m], upper_bound, batch=True, ubound=True)
            p_oods.append(p_ood)
            # for WD detection
            p_wd = ood_p_value_batch(cost[:m], lower_bound, batch=True, ubound=False)
            p_wds.append(p_wd)
            c_ood = ood_confidence_batch(cost[:m], upper_bound, deltap_A=deltap_A, batch=True, ood_adverse=True)
            c_oods.append(c_ood)
            # false negative uses lower bound (for ood-benign detection)
            c_wd = ood_confidence_batch(cost[:m], lower_bound, deltap_B=deltap_B, batch=True, ood_adverse=False)
            c_wds.append(c_wd)

            msp_ood = ood_msp_batch(model_output[:m], batch=True)
            msp_oods.append(msp_ood)
            maxlogit_ood = ood_maxlogit_batch(model_output[:m], batch=True)
            maxlogit_oods.append(maxlogit_ood)

        ps[:, trial] = p_oods
        cs[:, trial] = c_oods
        msps[:, trial] = msp_oods
        maxlogits[:, trial] = maxlogit_oods
        cs_wd[:, trial] = c_wds
        ps_wd[:, trial] = p_wds

    return x, ps, cs, msps, maxlogits, cs_wd, ps_wd


def get_costs(policy, data, outputs): 
    x = []
    for key in data:  # processing step to speed up trials
        depth_maps, prim_costs = data[key]
        model_output = policy(depth_maps)
        _, y_OOD = run_policy(policy, depth_maps, prim_costs)

        cost = y_OOD[:, 0].detach().numpy()  # policy(x_OOD) = y_OOD because that's the actual cost
        model_output = model_output.detach().numpy()
        outputs[key] = (cost, model_output)
        x.append(np.mean(cost))

    return x


def get_hardware_detections(cost_data, upper_bound):
    ps = []
    cs = []
    for key in cost_data:
        cost = hardware_data[key]
        p = ood_p_value_batch(cost, upper_bound, batch=True)
        c = ood_confidence_batch(cost, upper_bound, batch=True)
        if 'w' in key or 'id' in key:
            ps.append(p)
            cs.append(c)
            print(key, np.mean(cost), p, c)
        else:
            print(key, np.mean(cost), p, c)
    return ps, cs


def get_detection_bound(policy, data, upper_bound, max_m=10, trials=10):
    id_depth_maps, id_prim_costs = data['ID']
    ood_depth_maps, ood_prim_costs = data['OOD']
    _, id_y = run_policy(policy, id_depth_maps, id_prim_costs)
    _, ood_y = run_policy(policy, ood_depth_maps, ood_prim_costs)
    CD = id_y[:, 0].detach().numpy()
    CDp = ood_y[:, 0].detach().numpy()
    cdpmcd = np.mean(CDp) - np.mean(CD)
    cost = CDp
    cs = np.zeros((max_m, trials))
    for trial in range(trials):
        p = np.random.permutation(len(cost))
        cost = cost[p]
        c_ood = ood_confidence_batch(cost[:max_m], upper_bound, deltap_A=0.09, batch=False)
        cs[:, trial] = c_ood

    return cs, cdpmcd


def get_thresholds(policy, data, m=10, trials=1000, max_fp=0.05):
    depth_maps, prim_costs = data
    _, y_OOD = run_policy(policy, depth_maps, prim_costs)
    cost = y_OOD[:, 0].detach().numpy()  # policy(x_OOD) = y_OOD because that's the actual cost

    model_output = policy(depth_maps)
    model_output = model_output.detach().numpy()
    p_ids = []
    c_ids = []
    msp_ids = []
    maxlogit_ids = []
    for trial in range(trials):
        p = np.random.permutation(len(model_output))
        model_output = model_output[p]
        
        p_id = ood_p_value_batch(cost[:m], upper_bound, batch=True)
        p_ids.append(p_id)
        c_id = ood_confidence_batch(cost[:m], upper_bound, batch=True)
        c_ids.append(c_id)

        msp_id = ood_msp_batch(model_output[:m], batch=True)
        msp_ids.append(msp_id)
        maxlogit_id = ood_maxlogit_batch(model_output[:m], batch=True)
        maxlogit_ids.append(maxlogit_id)

    c_threshold = np.percentile(c_ids, 100 - 100*max_fp)
    p_threshold = np.percentile(p_ids, 100 - 100*max_fp)
    msp_threshold = np.percentile(msp_ids, 100 - 100*max_fp)
    maxlogit_threshold = np.percentile(maxlogit_ids, 100 - 100*max_fp)

    return p_threshold, c_threshold, msp_threshold, maxlogit_threshold


fig1 = 1 # unchanged from CoRL paper
fig2 = 0 # unchanged from CoRl paper
fig3 = 1 # subset of fig4
fig4 = 1
fig5 = 1
fig6 = 1
fig7 = 1

gen_data = False # if False, just load in plot data 

# trials = 2
trials = 2000
lower_bound_trials = 500000
m = 50

# hardware results
if fig1:
    params = np.load('weights/post_1.npy')
    print("Params = {}".format(params))
    upper_bound = params[1] # C(overbar)(pi, s) = cost + sqrt(R)
    print("Upper Bound = {}".format(upper_bound))

    hardware_data = get_hardware_data()
    ps, cs = get_hardware_detections(hardware_data, upper_bound)
    plot_wind((ps, cs), (r'$1 - p_A$', r'$\Delta C_A+0.95$'))

# cardinality data
if fig2:
    policy = SPolicy()
    policy = load_weights(policy, policy_path+default_dataset)
    torch.manual_seed(2)
    policy.init_xi()

    params = np.load('weights/' + policy_path + default_dataset + '.npy')
    print("Params = {}".format(params))
    upper_bound = params[0][1] # C(overbar)(pi, s) = cost + sqrt(R)
    print("Upper Bound = {}".format(upper_bound))
    # needed for false negative
    lower_bound = params[0][2] # C(underbar)(pi, s) = cost - sqrt(R)
    print("Lower Bound = {}".format(lower_bound))

    data = get_bound_validation_data()
    cs, cdpmcd = get_detection_bound(policy, data, upper_bound, max_m=50, trials=lower_bound_trials)
    plot_cardinality(cdpmcd, cs)

# C_O, p_O for particular dataset
if fig3:
    OOD_data = get_increasing_difficulty_data(dataset = default_dataset)
    print("Got OOD Data")
    ID_data = get_train_data(dataset = default_dataset)
    print("Got ID Data")

    policy = SPolicy()
    policy = load_weights(policy, policy_path+default_dataset)
    torch.manual_seed(2)
    policy.init_xi()

    params = np.load('weights/' + policy_path + default_dataset + '.npy')
    print("Params = {}".format(params))
    upper_bound = params[0][1] # C(overbar)(pi, s) = cost + sqrt(R)
    print("Upper Bound = {}".format(upper_bound))
    # needed for false negative
    lower_bound = params[0][2] # C(underbar)(pi, s) = cost - sqrt(R)
    print("Lower Bound = {}".format(lower_bound))

    _, _, msp_threshold, maxlogit_threshold = get_thresholds(policy, ID_data, m=m, trials=trials, max_fp=0.05)
    print("Got Thresholds")
    x, ps, cs, msps, maxlogits, cs_W, ps_W = get_comparison_detections(policy, OOD_data, upper_bound, lower_bound, m=m, trials=trials)
    print("Got Comparison Detection")
    # proportion detected out of distribution
    ps = np.sum(ps >= 0.95, -1)/trials
    cs = np.sum(cs > 0, -1)/trials
    msps = np.sum(msps >= msp_threshold, -1)/trials
    maxlogits = np.sum(maxlogits >= maxlogit_threshold, -1)/trials
    cs_W = np.sum(cs_W > 0, -1)/trials

    plot_compare_precision(x - x[5], (ps, cs, msps, maxlogits),
                           (r'1-$p_{A}$', r'$\Delta C_{A}$', 'MSP', 'MaxLogit'),
                           ylabel="Proportion Detected OOD", app="_precision")

# combined detector bar graph for all datasets 
if fig4:
    # generate and plot data
    if gen_data:  
        ds = [] # correspond to the dataset
        fig_4_data = {} # to save data
        i = 0
        for dataset in datasets: 
            OOD_data = get_increasing_difficulty_data(dataset = dataset)
            d_thresh = dataset[4:]
            ds.append(d_thresh)
            print("Got OOD Data")
            # ID_data = get_train_data(dataset = dataset)
            # print("Got ID Data")
            # _, _, msp_threshold, maxlogit_threshold = get_thresholds(policy, ID_data, m=m, trials=trials, max_fp=0.05)
            # print("Got Thresholds")

            policy = SPolicy()
            policy = load_weights(policy, policy_path+dataset)
            torch.manual_seed(2)
            policy.init_xi()

            params = np.load('weights/' + policy_path + dataset + '.npy')
            print("Params = {}".format(params))
            upper_bound = params[0][1] # C(overbar)(pi, s) = cost + sqrt(R)
            print("Upper Bound = {}".format(upper_bound))
            # needed for false negative
            lower_bound = params[0][2] # C(underbar)(pi, s) = cost - sqrt(R)
            print("Lower Bound = {}".format(lower_bound))

            x, ps, cs, msps, maxlogits, cs_W, ps_W = get_comparison_detections(policy, OOD_data, upper_bound, lower_bound, m=m, trials=trials)
            print("Got Comparison Detection")
            
            # combined detector (confidence interval based) 
            prop_ood_a = np.sum(cs > 0, -1)/trials # OOD-adverse when C_fp > 0 
            prop_ood_b = np.sum(cs_W >= 0, -1)/trials # OOD-benign when C_fn >= 0
            prop_wd = 1 - prop_ood_a - prop_ood_b # other times unkown

            print("Confidence Interval")
            print(prop_ood_a)
            print(prop_ood_b)
            print(prop_wd)

            # combined detector (p-value based) 
            alpha_O = 0.05
            alpha_W = 0.05
            prop_ood_a_p = np.sum(ps >= 1-alpha_O, -1)/trials # OOD-adverse when C_fp > 0 
            prop_ood_b_p = np.sum(ps_W >= 1-alpha_W, -1)/trials # OOD-benign when C_fn >= 0
            prop_wd_p = 1 - prop_ood_a_p - prop_ood_b_p # other times indecipherable

            print("P-Value")
            print(prop_ood_a_p)
            print(prop_ood_b_p)
            print(prop_wd_p)

            # save data for the future to make it easier to edit plot style
            fig_4_data[d_thresh] = {'x':x-x[5], 'deltaC_ood_a':prop_ood_a, 'deltaC_ood_b':prop_ood_b, 'deltaC_wd':prop_wd, 
                                    'p_ood_a': prop_ood_a_p, 'p_ood_b': prop_ood_b_p, 'p_wd':prop_wd_p}
            
            np.save('plots/fig_4_data.npy', fig_4_data)
            print("Saved Plotting Data")

            plot_combined_detector(x - x[5], [prop_ood_a, prop_ood_b, prop_wd], 
                                (r'Proportion $OOD_{A}$: $\Delta C_{A} > 0$'+ r" ($\delta_{A} + \delta'_{A} = 0.05$)", 
                                r'Proportion $OOD_{B}$: $\Delta C_{B} \geq 0$'+r" ($\delta_{B} + \delta'_{B} = 0.05$)", 'Proportion WD'), 
                                xlabel= r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", ylabel = "Proportion of Environments",
                                app="combined_detector_CI_d{}".format(ds[i]), 
                                figtext= r"$d_{thresh} = $" + "{} mm".format(ds[i]))

            plot_combined_detector(x - x[5], [prop_ood_a_p, prop_ood_b_p, prop_wd_p], 
                                (r"Proportion $OOD_{A}$: $p_{A} \leq \alpha_A$"+ r" ($\alpha_{A} = 0.05$)", 
                                r"Proportion $OOD_{B}$: $p_{B} \leq \alpha_B$"+r" ($\alpha_{B} = 0.05$)", 'Proportion WD'), 
                                xlabel= r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", ylabel = "Proportion of Environments", 
                                app="combined_detector_p_d{}".format(ds[i]), 
                                figtext= r"$d_{thresh} = $" + "{} mm".format(ds[i]))
            i = i+1

    # load in data
    else: 
        d = np.load('plots/fig_4_data.npy', allow_pickle=True)
        d = d.item()
        for d_thresh, data in d.items(): 
            plot_combined_detector(data['x'], [data['deltaC_ood_a'], data['deltaC_ood_b'], data['deltaC_wd']], 
                                (r'Proportion $OOD_{A}$: $\Delta C_{A} > 0$'+ r" ($\delta_{A} + \delta'_{A} = 0.05$)", 
                                r'Proportion $OOD_{B}$: $\Delta C_{B} \geq 0$'+r" ($\delta_{B} + \delta'_{B} = 0.05$)", 'Proportion WD'), 
                                xlabel= r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", ylabel = "Proportion of Environments",
                                app="combined_detector_CI_d{}".format(d_thresh), 
                                figtext= r"$d_{thresh} = $" + "{} mm".format(d_thresh))
            plot_combined_detector(data['x'], [data['p_ood_a'], data['p_ood_b'], data['p_wd']], 
                                (r"Proportion $OOD_{A}$: $p_{A} \leq \alpha_A$"+ r" ($\alpha_{A} = 0.05$)", 
                                r"Proportion $OOD_{B}$: $p_{B} \leq \alpha_B$"+r" ($\alpha_{B} = 0.05$)", 'Proportion WD'), 
                                xlabel= r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", ylabel = "Proportion of Environments", 
                                app="combined_detector_p_d{}".format(d_thresh), 
                                figtext= r"$d_{thresh} = $" + "{} mm".format(d_thresh))

# varying confidence bounds for particular dataset
if fig5: 
    # generate and plot data
    if gen_data:
        dataset = 'data500'
        # define delta' values to experiment with 
        deltap_Os = [0.09, 0.14, 0.19, 0.24, 0.29, 0.34, 0.39, 0.44, 0.49, 0.54, 0.59, 0.64, 0.89]
        deltap_Ws = [0.09, 0.14, 0.19, 0.24, 0.29, 0.34, 0.39, 0.44, 0.49, 0.54, 0.59, 0.64, 0.09]

        fig_5_data = {} 

        for deltap_O, deltap_W in zip(deltap_Os, deltap_Ws):
            print("Delta Prime = {}".format(deltap_O))

            # get dataset and PAC-Bayes bounds 
            OOD_data = get_increasing_difficulty_data(dataset = dataset)
            print("Got OOD Data")

            policy = SPolicy()
            policy = load_weights(policy,policy_path+dataset)
            torch.manual_seed(2)
            policy.init_xi()
            # print(type(policy))

            params = np.load('weights/' + policy_path + dataset + '.npy')
            print("Params = {}".format(params))
            upper_bound = params[0][1] # C(overbar)(pi, s) = cost + sqrt(R)
            print("Upper Bound = {}".format(upper_bound))
            # needed for false negative
            lower_bound = params[0][2] # C(underbar)(pi, s) = cost - sqrt(R)
            print("Lower Bound = {}".format(lower_bound))

            x, ps, cs, msps, maxlogits, cs_W, ps_W = get_comparison_detections(policy, OOD_data, upper_bound, lower_bound, deltap_O, deltap_W, m=m, trials=trials)

            # combined detector (confidence interval based) 
            prop_ood_a = np.sum(cs > 0, -1)/trials # OOD-adverse when C_fp > 0 
            prop_ood_b = np.sum(cs_W >= 0, -1)/trials # OOD-benign when C_fn >= 0
            prop_wd = 1 - prop_ood_a - prop_ood_b # other times within distribution

            # save data for the future to make it easier to edit plot style
            fig_5_data[deltap_O] = {'x':x-x[5], 'deltaC_ood_a':prop_ood_a, 'deltaC_ood_b':prop_ood_b, 'deltaC_wd':prop_wd}
            np.save('plots/fig_5_data.npy', fig_5_data)
            print("Saved Plotting Data")

            plot_combined_detector(x - x[5], [prop_ood_a, prop_ood_b, prop_wd], 
                                (r'Proportion $OOD_{A}$: $\Delta C_{A} > 0$'+ r" ($\delta_{A} + \delta'_{A} = $" + "{:.2f})".format(deltap_O + 0.01), 
                                r'Proportion $OOD_{B}$: $\Delta C_{B} \geq 0$'+r" ($\delta_{B} + \delta'_{B} = $" + "{:.2f})".format(deltap_W + 0.01), 'Proportion WD'), 
                                xlabel= r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", ylabel = "Proportion of Environments",
                                app="combined_detector_CI_delta={}".format(deltap_O), 
                                figtext= r"$d_{thresh} = 500$" + "mm")

    # load in data
    else: 
        d = np.load('plots/fig_5_data.npy', allow_pickle=True)
        d = d.item()
        for deltap_O, data in d.items(): 
            deltap_W = deltap_O
            # for the skewed graph
            if deltap_O == 0.89: 
                deltap_W = 0.09
            plot_combined_detector(data['x'], [data['deltaC_ood_a'], data['deltaC_ood_b'], data['deltaC_wd']], 
                                (r'Proportion $OOD_A$: $\Delta C_{A} > 0$'+ r" ($\delta_{A} + \delta'_{A} = $" + "{:.2f})".format(deltap_O + 0.01), 
                                r'Proportion $OOD_B$: $\Delta C_{B} \geq 0$'+r" ($\delta_{B} + \delta'_{B} = $" + "{:.2f})".format(deltap_W + 0.01), 'Proportion WD'), 
                                xlabel= r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", ylabel = "Proportion of Environments",
                                app="combined_detector_CI_delta={}".format(deltap_O), 
                                figtext= r"$d_{thresh} = 500$" + "mm")

# OOD comparison to max logit and MSP for all d_thresh (used in fig 6, fig 11 and fig 12)
if fig6: 
    # generate and plot data
    if gen_data: 
        costs = []
        upper_bounds = []
        lower_bounds = []
        y_fp = []
        y_fn = [] 
        y_po = []
        y_pw = [] 
        ds = [] # correspond to the dataset
        fig_6_data = {} # to save plot data for each dataset
        fig_11_data = {} # to save plot data for comparison across d_thresh
        i = 0

        for dataset in datasets:
            print(dataset)
            d_thresh = dataset[4:]
            ds.append(d_thresh) # for the legend (encodes distance, d from obstacle)
            OOD_data = get_increasing_difficulty_data(dataset = dataset)
            print("Got OOD Data")
            ID_data = get_train_data(dataset = dataset)
            print("Got ID Data")
            # get policy for the dataset 
            policy = SPolicy()
            policy = load_weights(policy, policy_path+dataset)
            torch.manual_seed(2)
            policy.init_xi()
            # print(policy)

            params = np.load('weights/' + policy_path + dataset + '.npy')
            print("Params = {}".format(params))
            upper_bound = params[0][1] # C(overbar)(pi, s) = cost + sqrt(R)
            print("Upper Bound = {}".format(upper_bound))
            # needed for false negative
            lower_bound = params[0][2] # C(underbar)(pi, s) = cost - sqrt(R)
            print("Lower Bound = {}".format(lower_bound))

            _, _, msp_threshold, maxlogit_threshold = get_thresholds(policy, ID_data, m=m, trials=trials, max_fp=0.05)
            x, ps, cs, msps, maxlogits, cs_W, ps_W = get_comparison_detections(policy, OOD_data, upper_bound, lower_bound, m=m, trials=trials)
            costs.append(x - x[5])

            c_props = np.sum(cs > 0, -1)/trials
            # print(c_props)
            c_fneg_props = np.sum(cs_W >= 0, -1)/trials
            y_fp.append(c_props)
            y_fn.append(c_fneg_props)

            # plot fig 3 equivalent for every value of d
            msps = np.sum(msps >= msp_threshold, -1)/trials
            maxlogits = np.sum(maxlogits >= maxlogit_threshold, -1)/trials
            ps = np.sum(ps >= 0.95, -1)/trials
            ps_W = np.sum(ps_W >= 0.95, -1)/trials
            y_po.append(ps)
            y_pw.append(ps_W)

            # save data for the future to make it easier to edit plot style
            fig_6_data[d_thresh] = {'x':x-x[5], 'p_ood_a': ps, 'deltaC_ood_a':c_props, 
                                    'msp': msps, 'maxlogit': maxlogits}
            np.save('plots/fig_6_data.npy', fig_6_data)
            print("Saved Plotting Data")
            

            plot_compare_precision(x - x[5], (ps, c_props, msps, maxlogits),
                            (r'$1-p_A$', r'$\Delta C_{A}$', 'MSP', 'MaxLogit'), 
                            xlabel = r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", 
                            ylabel="Proportion Detected OOD-adverse", app="_precision_combined_d{}".format(ds[i]))
            i = i+1

        legend_fp = []
        legend_fn = []
        legend_fp_p = [] 
        legend_fn_p = [] 

        for i, d in enumerate(ds):
            legend_fp.append(r'$\Delta C_{A}$' + " " + r'($d_{thresh} = $' + '{})'.format(int(d)))
            legend_fp_p.append(r'$1-p_{A}$' + " " + r'($d_{thresh} = $' + '{})'.format(int(d)))
            legend_fn.append(r'$\Delta C_{B}$' + " " + r'($d_{thresh} = $' + '{})'.format(int(d)))
            legend_fn_p.append(r'$1-p_{B}$' + " " + r'($d_{thresh} = $' + '{})'.format(int(d)))

        # behavior of false negative and false positive detector over range of values of d_thresh 
        plot_compare_precision(costs, y_fp, legend_fp, ylabel = "Proportion Detected OOD-adverse", app = "_datasets_ood_CI", loc = 'upper left') 
        plot_compare_precision(costs, y_fn, legend_fn, ylabel = "Proportion Detected OOD-benign", app = "_datasets_wd_CI", loc = 'upper right') 

        # behavior of false negative and false positive detector over range of values of d_thresh 
        plot_compare_precision(costs, y_po, legend_fp_p, ylabel = "Proportion Detected OOD-adverse", app = "_datasets_ood_p", loc = 'upper left') 
        plot_compare_precision(costs, y_pw, legend_fn_p, ylabel = "Proportion Detected OOD-benign", app = "_datasets_wd_p", loc = 'upper right') 

        # save data
        fig_11_data['costs'] = costs
        fig_11_data['deltaC_ood_a'] = y_fp
        fig_11_data['deltaC_ood_a_legend'] = legend_fp
        fig_11_data['deltaC_ood_b'] = y_fn
        fig_11_data['deltaC_oob_b_legend'] = legend_fn
        fig_11_data['ps_ood_a'] = y_po
        fig_11_data['ps_ood_a_legend'] = legend_fp_p
        fig_11_data['ps_ood_b'] = y_pw
        fig_11_data['ps_ood_b_legend'] = legend_fn_p
        np.save('plots/fig_11_data.npy', fig_11_data)
        print("Saved Plotting Data")

    else: 
        # each value of d_thresh individually (fig 6, fig 12) 
        d = np.load('plots/fig_6_data.npy', allow_pickle=True)
        d = d.item()
        for d_thresh, data in d.items(): 
            plot_compare_precision(data['x'], (data['p_ood_a'], data['deltaC_ood_a'], data['msp'], data['maxlogit']),
                            (r'$1-p_A$', r'$\Delta C_{A}$', 'MSP', 'MaxLogit'), 
                            xlabel = r"Estimated $C_{\mathcal{D}'}(\pi) - C_{\mathcal{D}}(\pi)$", 
                            ylabel="Proportion Detected OOD-adverse", app="_precision_combined_d{}".format(d_thresh))
        
        # across d_thresh (fig 11)
        fig_11_data = np.load('plots/fig_11_data.npy', allow_pickle=True)
        fig_11_data = fig_11_data.item()

        # behavior of false negative and false positive detector over range of values of d_thresh 
        plot_compare_precision(fig_11_data['costs'], fig_11_data['deltaC_ood_a'], fig_11_data['deltaC_ood_a_legend'], 
                               ylabel = "Proportion Detected OOD-adverse", app = "_datasets_ood_a_CI", loc = 'upper left') 
        plot_compare_precision(fig_11_data['costs'], fig_11_data['deltaC_ood_b'], fig_11_data['deltaC_ood_b_legend'], 
                               ylabel = "Proportion Detected OOD-benign", app = "_datasets_ood_b_CI", loc = 'upper right') 

        # behavior of false negative and false positive detector over range of values of d_thresh 
        plot_compare_precision(fig_11_data['costs'], fig_11_data['ps_ood_a'], fig_11_data['ps_ood_a_legend'], 
                               ylabel = "Proportion Detected OOD-adverse", app = "_datasets_ood_a_p", loc = 'upper left') 
        plot_compare_precision(fig_11_data['costs'], fig_11_data['ps_ood_b'], fig_11_data['ps_ood_b_legend'], 
                               ylabel = "Proportion Detected OOD-benign", app = "_datasets_ood_b_p", loc = 'upper right')

# task irrelevant data
if fig7:
    dataset = 'data500'
    OOD_data = get_task_irrelevant_data(dataset = dataset)
    print("Got OOD Data")
    ID_data = get_train_data(dataset = dataset) 
    print("Got ID Data")
    # get policy for the dataset 
    policy = SPolicy()
    policy = load_weights(policy, policy_path+dataset)
    torch.manual_seed(2)
    policy.init_xi()
    # print(policy)

    params = np.load('weights/' + policy_path + dataset + '.npy')
    print("Params = {}".format(params))
    upper_bound = params[0][1] # C(overbar)(pi, s) = cost + sqrt(R)
    print("Upper Bound = {}".format(upper_bound))
    # needed for false negative
    lower_bound = params[0][2] # C(underbar)(pi, s) = cost - sqrt(R)
    print("Lower Bound = {}".format(lower_bound))

    print("d_thresh = {} mm".format(dataset[4:]))
    
    _, _, msp_threshold, maxlogit_threshold = get_thresholds(policy, ID_data, m=m, trials=trials, max_fp=0.05)
    x, ps, cs, msps, maxlogits, cs_wd, ps_wd = get_comparison_detections(policy, OOD_data, upper_bound, lower_bound, m=m, trials=trials)
    print('x: {}'.format(x))
    print("C_D' - C_D: {}".format(x[0] - x))
    ps = np.sum(ps >= 0.95, -1)/trials
    cs = np.sum(cs > 0, -1)/trials
    msps = np.sum(msps >= msp_threshold, -1)/trials
    maxlogits = np.sum(maxlogits >= maxlogit_threshold, -1)/trials
    print('Proportion Detected OOD (ps):[{:.5} {:.5}]'.format(ps[0], ps[1]))
    print('Proportion Detected OOD (cs):[{:.5} {:.5}]'.format(cs[0], cs[1]))
    print('Proportion Detected OOD (msps):[{:.5} {:.5}]'.format(msps[0], msps[1]))
    print('Proportion Detected OOD (maxlogits):[{:.5} {:.5}]'.format(maxlogits[0], maxlogits[1]))
