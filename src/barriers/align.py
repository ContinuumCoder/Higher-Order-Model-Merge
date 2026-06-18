#!/usr/bin/env python3
"""
Lightweight weight-matching alignment for ResNet-20.

Implements the greedy weight matching algorithm from:
  Ainsworth et al., "Git Re-Basin: Merging Models modulo Permutation Symmetries" (2023)

For ResNet-20, the permutation symmetry is per-layer: we can permute the output
channels of each conv layer (and corresponding BN params, and input channels of
the next layer) without changing the function.

This is a simplified version that handles the ResNet-20 architecture specifically.
"""

import copy
from collections import OrderedDict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def get_permutation_spec_resnet20():
    """Return the permutation groups for ResNet-20.

    Each group is a list of (param_name, axis) pairs that must be permuted together.
    axis=0 means output channels, axis=1 means input channels.

    ResNet-20 structure:
      conv1 (3->16) -> bn1
      layer1: 3 blocks, each with conv1(16->16), bn1, conv2(16->16), bn2
      layer2: 3 blocks, first with conv1(16->32,s=2), bn1, conv2(32->32), bn2
              + shortcut conv(16->32,s=2), shortcut bn
      layer3: 3 blocks, first with conv1(32->64,s=2), bn1, conv2(64->64), bn2
              + shortcut conv(32->64,s=2), shortcut bn
      fc (64->10)
    """
    groups = []

    # IMPORTANT: In ResNet, backbone channels (conv1 output, conv2 output of each
    # block) are tied by residual connections and CANNOT be freely permuted in
    # isolation. The only freely permutable channels are the HIDDEN channels
    # within each block: conv1.output = conv2.input. These are not connected
    # to any residual path.

    # Layer1: 3 blocks, hidden channels = 16
    for block_idx in range(3):
        prefix = f'layer1.{block_idx}'
        groups.append({
            'size': 16,
            'output': [(f'{prefix}.conv1.weight', 0), (f'{prefix}.bn1.weight', 0),
                       (f'{prefix}.bn1.bias', 0), (f'{prefix}.bn1.running_mean', 0),
                       (f'{prefix}.bn1.running_var', 0)],
            'input': [(f'{prefix}.conv2.weight', 1)]
        })

    # Layer2: 3 blocks, hidden channels = 32
    for block_idx in range(3):
        prefix = f'layer2.{block_idx}'
        groups.append({
            'size': 32,
            'output': [(f'{prefix}.conv1.weight', 0), (f'{prefix}.bn1.weight', 0),
                       (f'{prefix}.bn1.bias', 0), (f'{prefix}.bn1.running_mean', 0),
                       (f'{prefix}.bn1.running_var', 0)],
            'input': [(f'{prefix}.conv2.weight', 1)]
        })

    # Layer3: 3 blocks, hidden channels = 64
    for block_idx in range(3):
        prefix = f'layer3.{block_idx}'
        groups.append({
            'size': 64,
            'output': [(f'{prefix}.conv1.weight', 0), (f'{prefix}.bn1.weight', 0),
                       (f'{prefix}.bn1.bias', 0), (f'{prefix}.bn1.running_mean', 0),
                       (f'{prefix}.bn1.running_var', 0)],
            'input': [(f'{prefix}.conv2.weight', 1)]
        })

    return groups


def compute_weight_matching_perm(sd_a, sd_b, group):
    """Find the permutation of group's output channels that best aligns sd_b to sd_a.

    Uses the weight matching criterion: minimize ||W_a - P W_b||^2
    which reduces to maximizing trace(W_a^T P W_b), solved by linear assignment.
    """
    size = group['size']

    # Build cost matrix from output weights
    cost = np.zeros((size, size))
    for item in group['output']:
        param_name, axis = item[0], item[1]
        if param_name not in sd_a or param_name not in sd_b:
            continue
        wa = sd_a[param_name].float().numpy()
        wb = sd_b[param_name].float().numpy()

        if axis == 0:
            wa_flat = wa.reshape(size, -1)
            wb_flat = wb.reshape(size, -1)
            cost += wa_flat @ wb_flat.T

    # Input weights of the next layer (conv or fc)
    for item in group.get('input', []):
        param_name, axis = item[0], item[1]
        if param_name not in sd_a or param_name not in sd_b:
            continue
        wa = sd_a[param_name].float().numpy()
        wb = sd_b[param_name].float().numpy()

        if axis == 1:
            other_dims = wa.shape[0]
            wa_flat = wa.reshape(other_dims, size, -1)
            wb_flat = wb.reshape(other_dims, size, -1)
            # Vectorized cost computation
            for i in range(size):
                for j in range(size):
                    cost[i, j] += np.sum(wa_flat[:, i, :] * wb_flat[:, j, :])

    # Special handling for conv->fc transition
    for item in group.get('input_fc', []):
        param_name, axis, spatial = item[0], item[1], item[2]
        if param_name not in sd_a or param_name not in sd_b:
            continue
        wa = sd_a[param_name].float().numpy()  # shape (out_features, in_features)
        wb = sd_b[param_name].float().numpy()
        # in_features = size * spatial, reshape to (out, size, spatial)
        out_features = wa.shape[0]
        wa_r = wa.reshape(out_features, size, spatial)
        wb_r = wb.reshape(out_features, size, spatial)
        for i in range(size):
            for j in range(size):
                cost[i, j] += np.sum(wa_r[:, i, :] * wb_r[:, j, :])

    # Solve linear assignment (maximize cost = minimize -cost)
    row_ind, col_ind = linear_sum_assignment(-cost)
    perm = col_ind  # perm[i] = which channel of B maps to channel i of A
    return perm


def apply_permutation(sd, group, perm):
    """Apply permutation to all parameters in the group."""
    sd = OrderedDict(sd)
    perm_torch = torch.tensor(perm, dtype=torch.long)
    size = group['size']

    # Permute output channels
    for item in group['output']:
        param_name, axis = item[0], item[1]
        if param_name not in sd:
            continue
        sd[param_name] = torch.index_select(sd[param_name], axis, perm_torch)

    # Permute input channels of next conv layer
    for item in group.get('input', []):
        param_name, axis = item[0], item[1]
        if param_name not in sd:
            continue
        sd[param_name] = torch.index_select(sd[param_name], axis, perm_torch)

    # Permute conv->fc transition (groups of spatial columns)
    for item in group.get('input_fc', []):
        param_name, axis, spatial = item[0], item[1], item[2]
        if param_name not in sd:
            continue
        w = sd[param_name]  # (out_features, size * spatial)
        out_features = w.shape[0]
        w_r = w.view(out_features, size, spatial)
        w_r = torch.index_select(w_r, 1, perm_torch)
        sd[param_name] = w_r.view(out_features, size * spatial)

    return sd


def align_models(sd_ref, sd_target):
    """Align sd_target to sd_ref using weight matching.

    Returns a new state dict for target that is permutation-aligned to ref.
    """
    groups = get_permutation_spec_resnet20()
    sd_aligned = OrderedDict({k: v.clone() for k, v in sd_target.items()})

    for group in groups:
        perm = compute_weight_matching_perm(sd_ref, sd_aligned, group)
        sd_aligned = apply_permutation(sd_aligned, group, perm)

    return sd_aligned


def align_all_to_reference(state_dicts, ref_idx=0):
    """Align all state dicts to a reference model.

    Args:
        state_dicts: list of state dicts
        ref_idx: index of the reference model

    Returns:
        list of aligned state dicts (reference is unchanged)
    """
    ref = state_dicts[ref_idx]
    aligned = []
    for i, sd in enumerate(state_dicts):
        if i == ref_idx:
            aligned.append(OrderedDict({k: v.clone() for k, v in sd.items()}))
        else:
            print(f"  Aligning model {i} to reference {ref_idx}...")
            aligned.append(align_models(ref, sd))
    return aligned
