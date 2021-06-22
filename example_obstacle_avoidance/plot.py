import numpy as np
from models.models_swing import SPolicy
import warnings
import torch
import json
from utils.util_models import run_policy, load_data, load_weights
from utils.util import plot_wind, plot_cardinality, plot_compare_precision
from utils.util_oodd import p_value, confidence, msp, maxlogit
warnings.filterwarnings('ignore')

policy_path = 'post_1'


def get_increasing_difficulty_data():
    OOD_data = {'OOD_vd-5': load_data(app='_vd-9', option='prim_cost'),
                'OOD_vd-4': load_data(app='_vd-4', option='prim_cost'),
                'OOD_vd-3': load_data(app='_vd-3', option='prim_cost'),
                'OOD_vd-2': load_data(app='_vd-2', option='prim_cost'),
                'OOD_vd-1': load_data(app='_vd-1', option='prim_cost'),
                'ID_vd0': load_data(app='_vd0', option='prim_cost'),
                'OOD_vd1': load_data(app='_vd1', option='prim_cost'),
                'OOD_vd2': load_data(app='_vd2', option='prim_cost'),
                'OOD_vd3': load_data(app='_vd3', option='prim_cost'),
                'OOD_vd4': load_data(app='_vd3.5', option='prim_cost'),
                'OOD_vd5': load_data(app='_vd3.8', option='prim_cost'),
                'OOD_vd6': load_data(app='_vd3.9', option='prim_cost'),
                'OOD_vd7': load_data(app='_vd4', option='prim_cost'),
                'OOD_vd8': load_data(app='_vd4.05', option='prim_cost'),
                'OOD_vd9': load_data(app='_vd4.1', option='prim_cost'),
                'OOD_vd10': load_data(app='_vd4.15', option='prim_cost'),
                'OOD_vd11': load_data(app='_vd4.2', option='prim_cost'),
                'OOD_vd12': load_data(app='_vd4.25', option='prim_cost'),
                'OOD_vd13': load_data(app='_vd4.3', option='prim_cost'),
                'OOD_vd14': load_data(app='_vd4.4', option='prim_cost'),
                'OOD_vd15': load_data(app='_vd9', option='prim_cost'),
                }
    return OOD_data


def get_task_irrelevant_data():
    OOD_data = {
                'ID': load_data(app='_vd0', option='prim_cost'),
                'OOD': load_data(app='_ir_shift', option='prim_cost'),
                }
    return OOD_data


def get_hardware_data():
    data = json.load(open('hardware_data.json','r'))
    return data


def get_bound_validation_data():
    data = {'ID': load_data(app='_test', option='prim_cost'),
            'OOD': load_data(app='_vd4.12', option='prim_cost'),
            }
    return data


def get_train_data():
    data = load_data(app='_post', option='prim_cost')
    return data


def get_comparison_detections(policy, data, upper_bound, m=10, trials=20):
    ps = np.zeros((len(data), trials))
    cs = np.zeros((len(data), trials))
    msps = np.zeros((len(data), trials))
    maxlogits = np.zeros((len(data), trials))
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
        c_oods = []
        msp_oods = []
        maxlogit_oods = []
        for key in data:
            cost, model_output = data[key]  # model_output.detach().numpy()
            p = np.random.permutation(len(cost))
            cost = cost[p]
            model_output = model_output[p]

            p_ood = p_value(cost[:m], upper_bound, batch=True)
            p_oods.append(p_ood)
            c_ood = confidence(cost[:m], upper_bound, batch=True)
            c_oods.append(c_ood)
            msp_ood = msp(model_output[:m], batch=True)
            msp_oods.append(msp_ood)
            maxlogit_ood = maxlogit(model_output[:m], batch=True)
            maxlogit_oods.append(maxlogit_ood)

        ps[:, trial] = p_oods
        cs[:, trial] = c_oods
        msps[:, trial] = msp_oods
        maxlogits[:, trial] = maxlogit_oods

    return x, ps, cs, msps, maxlogits


def get_hardware_detections(cost_data, upper_bound):
    ps = []
    cs = []
    for key in cost_data:
        cost = hardware_data[key]
        p = p_value(cost, upper_bound, batch=True)
        c = confidence(cost, upper_bound, batch=True)
        if 'w' in key or 'id' in key:
            ps.append(p)
            cs.append(c)
            # print(key, np.mean(cost), p, c)
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
        c_ood = confidence(cost[:max_m], upper_bound, batch=False, deltap=0.09)
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

        p_id = p_value(cost[:m], upper_bound, batch=True)
        p_ids.append(p_id)
        c_id = confidence(cost[:m], upper_bound, batch=True)
        c_ids.append(c_id)

        msp_id = msp(model_output[:m], batch=True)
        msp_ids.append(msp_id)
        maxlogit_id = maxlogit(model_output[:m], batch=True)
        maxlogit_ids.append(maxlogit_id)

    c_threshold = np.percentile(c_ids, 100 - 100*max_fp)
    p_threshold = np.percentile(p_ids, 100 - 100*max_fp)
    msp_threshold = np.percentile(msp_ids, 100 - 100*max_fp)
    maxlogit_threshold = np.percentile(maxlogit_ids, 100 - 100*max_fp)

    return p_threshold, c_threshold, msp_threshold, maxlogit_threshold


policy = SPolicy()
load_weights(policy, policy_path)
torch.manual_seed(2)
policy.init_xi()  # will be the same as when computing the bound since we use same seed
params = np.load('weights/' + policy_path + '.npy')
upper_bound = params[1]

fig1 = 1
fig2 = 1
fig3 = 1
fig4 = 1

trials = 2000
lower_bound_trials = 500000
m = 10

if fig1:
    hardware_data = get_hardware_data()
    ps, cs = get_hardware_detections(hardware_data, upper_bound)
    plot_wind((ps, cs), (r'$1 - p$', r'$\Delta C+0.95$ '))

if fig2:
    data = get_bound_validation_data()
    cs, cdpmcd = get_detection_bound(policy, data, upper_bound, max_m=50, trials=lower_bound_trials)
    plot_cardinality(cdpmcd, cs)

if fig3:
    OOD_data = get_increasing_difficulty_data()
    ID_data = get_train_data()
    _, _, msp_threshold, maxlogit_threshold = get_thresholds(policy, ID_data, m=m, trials=trials, max_fp=0.05)
    x, ps, cs, msps, maxlogits = get_comparison_detections(policy, OOD_data, upper_bound, m=m, trials=trials)
    ps = np.sum(ps >= 0.95, -1)/trials
    cs = np.sum(cs >= 0, -1)/trials
    msps = np.sum(msps >= msp_threshold, -1)/trials
    maxlogits = np.sum(maxlogits >= maxlogit_threshold, -1)/trials
    plot_compare_precision(x - x[5], (ps, cs, msps, maxlogits),
                           (r'$1 - p$', r'$\Delta C$', 'MSP', 'MaxLogit',),
                           ylabel="Proportion detected as OOD", app="_precision")

if fig4:
    OOD_data = get_task_irrelevant_data()
    ID_data = get_train_data()
    _, _, msp_threshold, maxlogit_threshold = get_thresholds(policy, ID_data, m=m, trials=trials, max_fp=0.05)
    x, ps, cs, msps, maxlogits = get_comparison_detections(policy, OOD_data, upper_bound, m=m, trials=trials)
    print(x)
    ps = np.sum(ps >= 0.95, -1)/trials
    cs = np.sum(cs >= 0, -1)/trials
    msps = np.sum(msps >= msp_threshold, -1)/trials
    maxlogits = np.sum(maxlogits >= maxlogit_threshold, -1)/trials
    print(ps)
    print(cs)
    print(msps)
    print(maxlogits)
