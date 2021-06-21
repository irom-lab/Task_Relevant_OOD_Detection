import numpy as np


def ood_p_value(cost, bound, ubound=True):
    violations = cost - bound if ubound else bound - cost
    violation = np.mean(violations)
    tau = max(violation, 0)
    m = len(cost)
    p_val = np.exp(-2 * m * (tau ** 2))
    return 1 - p_val


def ood_confidence(cost, bound, deltap, ubound=True):
    violations = cost - bound if ubound else bound - cost
    violation = np.mean(violations)
    m = len(cost)
    gamma = np.sqrt(np.log(1/deltap)/(2*m))
    if violation - gamma > 0:
        return True, violation - gamma
    else:
        return False, violation - gamma


def p_value(costs, bound, batch=False):
    ps = []
    for m in range(1,len(costs) + 1):
        p = ood_p_value(costs[:m], bound, ubound=True)
        ps.append(p)
    if batch:
        return ps[-1]
    else:
        return ps


def confidence(costs, bound, deltap=0.04, batch=False):
    ps = []
    for m in range(1, len(costs) + 1):
        p = ood_confidence(costs[:m], bound, deltap=deltap, ubound=True)[1]
        ps.append(p)
    if batch:
        return ps[-1]
    else:
        return ps


def msp(model_output, batch=False):
    msp_single = 1 - np.max(model_output, axis=-1)
    msp_ood = np.cumsum(msp_single) / [i + 1 for i in range(len(msp_single))]
    if batch:
        return msp_ood[-1]
    else:
        return msp_ood


def maxlogit(model_output, batch=False):
    maxlogit_output = -np.log(model_output / (1 - model_output))
    maxlogit_single = np.max(maxlogit_output, axis=-1)
    maxlogit_ood = -np.cumsum(maxlogit_single) / [i + 1 for i in range(len(maxlogit_single))]
    if batch:
        return maxlogit_ood[-1]
    else:
        return maxlogit_ood
