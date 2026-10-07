"""Verbatim reference copies (test oracle only) of official DrM code, github.com/XuGW-Kevin/DrM @ 989732d6:
``utils.weight_init``, ``utils.LinearOutputHook``, ``utils.cal_dormant_ratio``, ``utils.perturb`` and the target and
expectile computations of ``agents/drm_mw.py``. MIT License, Copyright (c) 2024 DrM authors."""

from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class LinearOutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)


def cal_dormant_ratio(model, *inputs, percentage=0.025):
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(*inputs)
    for module, hook in zip((module for module in model.modules() if isinstance(module, nn.Linear)), hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indices = (mean_output < avg_neuron_output * percentage).nonzero(as_tuple=True)[0]
                total_neurons += module.weight.shape[0]
                dormant_neurons += len(dormant_indices)
    for hook in hooks:
        hook.outputs.clear()
    for hook_handler in hook_handlers:
        hook_handler.remove()
    return dormant_neurons / total_neurons


def perturb(net, optimizer, perturb_factor):
    linear_keys = [name for name, mod in net.named_modules() if isinstance(mod, torch.nn.Linear)]
    new_net = deepcopy(net)
    new_net.apply(weight_init)
    for name, param in net.named_parameters():
        if any(key in name for key in linear_keys):
            noise = new_net.state_dict()[name] * (1 - perturb_factor)
            param.data = param.data * perturb_factor + noise
        else:
            param.data = net.state_dict()[name]
    optimizer.state = defaultdict(dict)
    return net, optimizer


def target_q(agent, next_obs, next_proprio, reward, discount, step):
    """drm_mw.DrMAgent.update_critic target (with the shared proprio input)."""
    with torch.no_grad():
        dist = agent.actor(next_obs, next_proprio, agent.stddev(step))
        next_action = dist.sample(clip=agent.stddev_clip)
        target_Q1, target_Q2 = agent.critic_target(next_obs, next_proprio, next_action)
        target_V_explore = torch.min(target_Q1, target_Q2)
        target_V_exploit = agent.value_predictor(next_obs, next_proprio)
        lambda_ = agent.target_lambda
        target_V = lambda_ * target_V_exploit + (1 - lambda_) * target_V_explore
        return reward + (discount * target_V)


def predictor_loss(agent, obs, proprio, action):
    """drm_mw.DrMAgent.update_predictor loss (with the shared proprio input)."""
    Q1, Q2 = agent.critic(obs, proprio, action)
    Q = torch.min(Q1, Q2)
    V = agent.value_predictor(obs, proprio)
    vf_err = V - Q
    vf_sign = (vf_err > 0).float()
    vf_weight = (1 - vf_sign) * agent.expectile + vf_sign * (1 - agent.expectile)
    return (vf_weight * (vf_err**2)).mean()
