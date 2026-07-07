# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# ---------------------------------------------------------------------------
# VENDORED, UNMODIFIED-LOGIC copy of the PaPaGei-S model architecture from
#   github.com/Nokia-Bell-Labs/papagei-foundation-model  (models/resnet.py)
# by Nokia Bell Labs (ICLR'25), re-distributed here under its original
# BSD-3-Clause-Clear license (notice retained above, per the license terms).
#
# Only the classes needed to instantiate `ResNet1DMoE` and load the pretrained
# `papagei_s.pt` state_dict are vendored (the trunk + its MoE heads). Unused
# training/plotting imports from the upstream file are dropped; the *network
# logic is byte-for-byte the authors'* so the loaded checkpoint reproduces the
# published model exactly. This module is used ONLY on the training/port side
# (fine-tuning on MPS + generating ground-truth embeddings for the NumPy-port
# parity gate). It is never imported by serving/feature-extraction code.
#
# Underlying 1-D ResNet design credited by the authors to Shenda Hong (2019).
# ---------------------------------------------------------------------------
"""PaPaGei-S (ResNet1DMoE) reference architecture — vendored for the port."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """extend nn.Conv1d to support SAME padding"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net


class MyMaxPool1dPadSame(nn.Module):
    """extend nn.MaxPool1d to support SAME padding"""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.max_pool(net)
        return net


class BasicBlock(nn.Module):
    """ResNet Basic Block (pre-activation)"""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, groups,
        downsample, use_bn, use_do, is_first_block=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=self.stride, groups=self.groups,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=1, groups=self.groups,
        )

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.max_pool(identity)

        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        out += identity
        return out


class ResNet1DMoE(nn.Module):
    """ResNet1D with two Mixture-of-Experts regression heads — PaPaGei-S.

    forward(x) -> (out_class, out_moe1, out_moe2, out_embedding). The pretrained
    feature extractor uses the 4th output (the 512-d pooled embedding); the port
    keeps ONLY that. The heads are loaded (so the checkpoint loads strict) but not
    used downstream.
    """

    def __init__(
        self, in_channels, base_filters, kernel_size, stride, groups, n_block,
        n_classes, n_experts=2, downsample_gap=2, increasefilter_gap=4,
        use_bn=True, use_do=True, verbose=False, use_projection=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_projection = use_projection
        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap
        self.n_experts = n_experts

        self.first_block_conv = MyConv1dPadSame(
            in_channels=in_channels, out_channels=base_filters,
            kernel_size=self.kernel_size, stride=1,
        )
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            is_first_block = i_block == 0
            downsample = i_block % self.downsample_gap == 1
            in_channels = (
                base_filters
                if is_first_block
                else int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
            )
            out_channels = (
                in_channels * 2
                if (i_block % self.increasefilter_gap == 0 and i_block != 0)
                else in_channels
            )
            tmp_block = BasicBlock(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=self.kernel_size, stride=self.stride, groups=self.groups,
                downsample=downsample, use_bn=self.use_bn, use_do=self.use_do,
                is_first_block=is_first_block,
            )
            self.basicblock_list.append(tmp_block)

        if self.use_projection:
            self.projector = nn.Sequential(
                nn.Linear(out_channels, 256), nn.BatchNorm1d(256),
                nn.ReLU(), nn.Linear(256, 128),
            )
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, n_classes)

        self.expert_layers_1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, out_channels // 2), nn.ReLU(),
                nn.Linear(out_channels // 2, 1),
            )
            for _ in range(self.n_experts)
        ])
        self.gating_network_1 = nn.Sequential(
            nn.Linear(out_channels, self.n_experts), nn.Softmax(dim=1),
        )
        self.expert_layers_2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, out_channels // 2), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(out_channels // 2, 1),
            )
            for _ in range(self.n_experts)
        ])
        self.gating_network_2 = nn.Sequential(
            nn.Linear(out_channels, self.n_experts), nn.Softmax(dim=1),
        )

    def embedding(self, x):
        """Trunk-only forward → the 512-d pooled embedding (no MoE heads run).

        This is the authors' forward truncated at `out.mean(-1)` — added so
        fine-tuning does not waste compute/gradient on the discarded heads. The
        network logic up to the embedding is byte-for-byte the authors'.
        """
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        for i_block in range(self.n_block):
            out = self.basicblock_list[i_block](out)
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        return out.mean(-1)

    def forward(self, x):
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        for i_block in range(self.n_block):
            out = self.basicblock_list[i_block](out)
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        out_class = self.projector(out) if self.use_projection else self.dense(out)

        expert_outputs_1 = torch.stack(
            [expert(out) for expert in self.expert_layers_1], dim=1
        )
        gate_weights_1 = self.gating_network_1(out)
        out_moe1 = torch.sum(gate_weights_1.unsqueeze(2) * expert_outputs_1, dim=1)

        expert_outputs_2 = torch.stack(
            [expert(out) for expert in self.expert_layers_2], dim=1
        )
        gate_weights_2 = self.gating_network_2(out)
        out_moe2 = torch.sum(gate_weights_2.unsqueeze(2) * expert_outputs_2, dim=1)

        return out_class, out_moe1, out_moe2, out


# Official PaPaGei-S configuration (README "Extracting Embeddings: Quick Start").
PAPAGEI_S_CONFIG: dict[str, int] = {
    "in_channels": 1,
    "base_filters": 32,
    "kernel_size": 3,
    "stride": 2,
    "groups": 1,
    "n_block": 18,
    "n_classes": 512,  # embedding dimension
    "n_experts": 3,
}

# Preprocessing contract the weights were pretrained under (README §Pre-process).
PAPAGEI_TARGET_FS_HZ = 125.0
PAPAGEI_SEGMENT_SECONDS = 10.0
PAPAGEI_SEGMENT_SAMPLES = 1250  # 125 Hz × 10 s


def build_papagei_s() -> ResNet1DMoE:
    """Instantiate PaPaGei-S with the published config (weights loaded separately)."""
    return ResNet1DMoE(**PAPAGEI_S_CONFIG)


def load_papagei_s_state_dict(model: ResNet1DMoE, checkpoint_path: str) -> ResNet1DMoE:
    """Load `papagei_s.pt`, stripping the DataParallel `module.` prefix (authors'
    `load_model_without_module_prefix`), and set eval mode."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_state_dict[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
