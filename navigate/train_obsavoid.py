import copy
import torch
import numpy as np
from obsavoid.utils.util_models import load_weights, save_weights, load_data
from obsavoid.models.models_swing import SPolicy, NSPolicy
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--step', default=0)
args = parser.parse_args()

file_version = 'test'
prior_path = 'prior_' + str(file_version)
post_path = 'post_' + str(file_version)


class DataLoader():
    def __init__(self, data_dict, device, batch=16):
        self.device = device
        self.batch = batch
        self.data = {}
        self.position = {}
        for key in data_dict:
            file_name, option = data_dict[key]
            if option is None:
                option = 'prim_cost'
            self.data[key] = load_data(app=file_name, option=option)
            self.position[key] = 0

    def get_batch(self, dataset, shuffle=True, batch=None):
        x, y = self.data[dataset]

        if batch is None:
            batch = self.batch

        x_batch = torch.empty((batch, *list(x.shape[1:])))
        y_batch = torch.empty((batch, *list(y.shape[1:])))
        if shuffle:
            inds = np.random.randint(0, x.shape[0], (batch,))
        else:
            inds = np.arange(0, x.shape[0])

        for i, ind in enumerate(inds):
            x_batch[i] = x[ind]
            y_batch[i] = y[ind]

        return x_batch.detach().to(self.device), y_batch.detach().to(self.device)


step = int(args.step)
compute_prior = 1 if int(args.step) == 0 else 1
device = torch.device('cuda')
batch = 16
test_batch = 100


data_dict = {'prior': ('_prior', 'dist_softmax'),
             'post': ('_post', 'dist_softmax'),
             'post_prim_costs': ('_post', 'prim_cost'),
             'test': ('_test', 'prim_cost'),
             }

dl = DataLoader(data_dict, device, batch)
criterion = torch.nn.BCELoss().to(device)



if step == 0:
    print("Training prior")
    model = NSPolicy().to(device)
    prior = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elif step == 1:
    print("Training posterior")
    model = SPolicy().to(device)
    load_weights(model, prior_path)
    model.init_logvar(-6, -6)
    prior = copy.deepcopy(model)
    save_weights(prior, prior_path)
    # finetune_params = []
    # for name, p in model.named_parameters():
    #     if 'logvar' not in name:
    #         finetune_params.append(p)
    # optimizer = torch.optim.Adam(finetune_params, lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
else:
    print("Computing bound")
    model = SPolicy().to(device)
    prior = SPolicy().to(device)
    load_weights(model, post_path)
    load_weights(prior, prior_path)
    optimizer = None


N = 10000
m = torch.tensor(len(dl.data['post'][0]), dtype=torch.float)
delta = torch.tensor(0.005, dtype=torch.float)
np.random.seed(step)
torch.manual_seed(step)

losses = []
missmatch = []

if step < 2:
    for n in range(1, int(N)+1):

        if compute_prior:
            x, y = dl.get_batch('prior')
        else:
            x, y = dl.get_batch('post')

        model.init_xi()
        model_output = model(x)
        reg = 0

        if step == 1:
            r_div = model.calc_r_div(prior, device=device)
            # r_div = model.calc_kl_div(prior, device=device)
            num_term = torch.log(2 * torch.sqrt(m) / delta**3)
            reg += torch.sqrt(torch.div(r_div + num_term, 2 * m))
            # reg = 0

        loss = criterion(model_output, y)
        loss += reg

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if step == 1:
            model.project_logvar(prior)

        losses.append(float(loss.to('cpu')))

        print('Iteration: {}, avg loss: {:.7f}'.format(n, np.mean(losses)), end='\r')

        if n % 1000 == 0:
            print()
            x, y = dl.get_batch('test', batch=test_batch)
            model_output = model(x)
            prims = model_output.max(dim=1).indices
            emp_cost = 0
            for (j, prim) in enumerate(prims.tolist()):
                emp_cost += y[j, prim]
            emp_cost /= test_batch

            print(float(emp_cost))
            losses = []

            if step == 0:
                save_weights(model, save_file_name=prior_path)
            else:
                save_weights(model, save_file_name=post_path)

    print()

if step == 2:
    x, y = dl.data['post_prim_costs']
    x = x.to(device)
    # y = y.to(device)
    model.init_xi()
    model_output = model(x)

    prims = model_output.max(dim=1).indices.tolist()
    emp_cost = torch.zeros((1))
    for (j, prim) in enumerate(prims):
        emp_cost += y[j, prim]
    emp_cost /= len(x)
    emp_cost = float(emp_cost)

    r_div = model.calc_r_div(prior, device=device).to('cpu')
    log_term_r = torch.log(2 * torch.sqrt(m) / delta**3)
    Rg_r = float((r_div + log_term_r) / m)
    reg_pac_bayes_r = np.sqrt(Rg_r/2)

    print("Emp cost", emp_cost)
    # print("Maurer", emp_cost + reg_pac_bayes_k)
    # pac_bound = kl_inv_l(emp_cost, Rg_k)
    # print("KL-inv", pac_bound)
    print("Pointwise PAC-Bayes", emp_cost + reg_pac_bayes_r)

    np.save('obsavoid/weights/'+post_path+'.npy', (emp_cost, emp_cost + reg_pac_bayes_r))
