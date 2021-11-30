import torch
import scipy.io
import os
import numpy as np

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_weights(policy, save_file_name=''):
    make_dir('weights')
    torch.save(policy.state_dict(), 'weights/' + save_file_name + '.pt')


def load_weights(policy, save_file_name=''):
    policy.load_state_dict(torch.load("weights/" + save_file_name + ".pt", map_location=torch.device('cpu')))
    return policy


def get_max_label_mismatch(pred, label):
    label_max_ind = torch.max(label, dim=1).indices
    pred_max_ind = torch.max(pred, dim=1).indices
    return sum(abs(label_max_ind - pred_max_ind) > 0).item()


def run_policy(policy, depth_maps, prim_costs, save_file_name=None):
    if save_file_name is not None:
        policy = load_weights(policy, save_file_name)

    num_envs = depth_maps.shape[0]

    y = torch.zeros((num_envs, 1))
    x = policy(depth_maps)

    # Identify best primitive for each environment
    prims = (x.max(dim=1).indices).tolist()

    for (j, prim) in enumerate(prims):
        y[j] = prim_costs[j, prim]

    return x.detach(), y.detach()


def set_weights(policy, weights, index):
    # Read the weight for policy numbered index from the weights dictionary and load on the net
    for (p1, p2) in zip(policy.parameters(), weights):
        p1.data = weights[p2][index, :]
    return policy


def get_weight_samples(num_policies, file_name):
    mu = torch.load(file_name, map_location=torch.device('cpu'))
    total_params = sum(mu[p].numel() for p in mu)
    torch.manual_seed(0)
    epsilon = torch.randn((num_policies, total_params))
    # Save all sampled weights in the weights dictionary
    weights = {}
    count = 0

    for p in mu:
        num_params_p = mu[p].numel()
        # Standard deviation is assumed to be 0.01
        weights[p] = mu[p].unsqueeze(0) + 0.01 * epsilon[:, count:count + num_params_p].view(
            [num_policies] + list(mu[p].shape))
        count += num_params_p

    return weights


def load_data(app='', option=None, dataset = "data500"):
    path = "./matlab_gen_data/data/"+dataset+"/"
    app = app + ".m"
    depth_maps = torch.Tensor(scipy.io.loadmat(path + "depth_maps" + app)['depth_maps'].astype('float32')).unsqueeze(1)

    if option is None:
        option = 'prim_cost'

    if option not in ['dist_softmax', 'prim_collision', 'prim_cost']:
        raise NotImplementedError

    cost_array = scipy.io.loadmat(path + option + app)[option].astype('float32')
    costs = torch.Tensor(cost_array)

    return depth_maps, costs
