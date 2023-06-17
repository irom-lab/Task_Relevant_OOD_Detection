import numpy as np


def ood_p_value(cost, bound, ubound=True):
    violations = cost - bound if ubound else bound - cost
    violation = np.mean(violations)
    tau = max(violation, 0)
    m = len(cost)
    p_val = np.exp(-2 * m * (tau ** 2))
    return 1 - p_val


def ood_confidence(cost, bound, deltap_A, deltap_B=0.04, ood_adverse=True):
    # ood_adverse: cost - bound; ood_benign = bound - cost
    violations = cost - bound if ood_adverse else bound - cost
    violation = np.mean(violations)
    m = len(cost)
    gamma = 0
    if ood_adverse: 
        gamma = np.sqrt(np.log(1/deltap_A)/(2*m)) 
    else:
        gamma = np.sqrt(np.log(1/deltap_B)/(2*m))
    # gamma = 0
    if violation - gamma > 0:
        return True, violation - gamma
    else:
        return False, violation - gamma


def ood_p_value_batch(costs, bound, batch=False, ubound=True):
    ps = []
    for m in range(1,len(costs) + 1):
        p = ood_p_value(costs[:m], bound, ubound=ubound)
        ps.append(p)
    if batch:
        return ps[-1]
    else:
        return ps


def ood_confidence_batch(costs, bound, deltap_A=0.04, deltap_B=0.04, batch=False, ood_adverse=True):
    cs = []
    for m in range(1, len(costs) + 1):
        c = ood_confidence(costs[:m], bound, deltap_A=deltap_A, deltap_B=deltap_B, ood_adverse=ood_adverse)[1]
        cs.append(c)
    if batch:
        return cs[-1]
    else:
        return cs


def ood_msp_batch(model_output, batch=False):
    """Maximum Softmax Probability"""
    msp_single = 1 - np.max(model_output, axis=-1)
    msp_ood = np.cumsum(msp_single) / [i + 1 for i in range(len(msp_single))]
    if batch:
        return msp_ood[-1]
    else:
        return msp_ood


def ood_maxlogit_batch(model_output, batch=False):
    """MaxLogit"""
    maxlogit_output = -np.log(model_output / (1 - model_output))
    maxlogit_single = np.max(maxlogit_output, axis=-1)
    maxlogit_ood = -np.cumsum(maxlogit_single) / [i + 1 for i in range(len(maxlogit_single))]
    if batch:
        return maxlogit_ood[-1]
    else:
        return maxlogit_ood
