#!/usr/bin/env python3
"""用官方模型类（原生 LSTM/FSMN 算子）导出 pe/dur/postnet 的紧凑 ONNX。"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HW = Path(os.environ["HW_DIR"])
OUT = Path(os.environ["OUT_DIR"])


class W:
    def __init__(self, d):
        d = Path(d)
        self.m = json.load(open(d / "manifest.json"))
        self.data = {}
        for name, shape in self.m.items():
            self.data[name] = np.fromfile(d / (name + ".bin"), dtype=np.float32).reshape(shape)

    def t(self, name):
        return torch.tensor(self.data[name])


# ---- 官方 FSMN ----
class FeedForwardNet(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_out, 1, bias=False)

    def forward(self, x):
        o = F.relu(self.w_1(x.transpose(1, 2)))
        o = self.w_2(o)
        return o.transpose(1, 2)


class MemoryBlockV2(nn.Module):
    def __init__(self, d, filter_size, shift):
        super().__init__()
        left_padding = int(round((filter_size - 1) / 2))
        right_padding = int((filter_size - 1) / 2)
        if shift > 0:
            left_padding += shift
            right_padding -= shift
        self.lp, self.rp = left_padding, right_padding
        self.conv_dw = nn.Conv1d(d, d, filter_size, 1, 0, groups=d, bias=False)

    def forward(self, input):
        x = F.pad(input, (0, 0, self.lp, self.rp, 0, 0), mode="constant", value=0.0)
        o = self.conv_dw(x.transpose(1, 2)).transpose(1, 2)
        return o + input


class FsmnEncoderV2(nn.Module):
    def __init__(self, filter_size, fsmn_num_layers, input_dim, num_memory_units,
                 ffn_inner_dim, shift):
        super().__init__()
        if not isinstance(shift, list):
            shift = [shift] * fsmn_num_layers
        self.ffn_lst = nn.ModuleList()
        self.ffn_lst.append(FeedForwardNet(input_dim, ffn_inner_dim, num_memory_units))
        for _ in range(1, fsmn_num_layers):
            self.ffn_lst.append(FeedForwardNet(num_memory_units, ffn_inner_dim, num_memory_units))
        self.memory_block_lst = nn.ModuleList([
            MemoryBlockV2(num_memory_units, filter_size, shift[i])
            for i in range(fsmn_num_layers)
        ])

    def forward(self, input):
        x = input
        for ffn, mem in zip(self.ffn_lst, self.memory_block_lst):
            m = mem(ffn(x))
            if m.size(-1) == x.size(-1):
                m = m + x
            x = m
        return x


class Prenet(nn.Module):
    def __init__(self, in_units, prenet_units):
        super().__init__()
        self.fcs = nn.ModuleList()
        for in_dim, out_dim in zip([in_units] + prenet_units[:-1], prenet_units):
            self.fcs.append(nn.Linear(in_dim, out_dim))
            self.fcs.append(nn.ReLU())

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
        return x


def load_fsmn(fsmn, W, pre):
    for i, (ffn, mem) in enumerate(zip(fsmn.ffn_lst, fsmn.memory_block_lst)):
        ffn.w_1.weight.data = W.t(f"{pre}_ffn{i}_w1")[:, :, 0][:, :, None]
        ffn.w_1.bias.data = W.t(f"{pre}_ffn{i}_b1")
        ffn.w_2.weight.data = W.t(f"{pre}_ffn{i}_w2")[:, :, 0][:, :, None]
        mem.conv_dw.weight.data = W.t(f"{pre}_mem{i}_conv")


def load_lstm(lstm, w_ih, w_hh, b_ih, b_hh):
    lstm.weight_ih_l0.data = torch.tensor(w_ih)
    lstm.weight_hh_l0.data = torch.tensor(w_hh)
    lstm.bias_ih_l0.data = torch.tensor(b_ih)
    lstm.bias_hh_l0.data = torch.tensor(b_hh)


def load_lstm_bi(lstm, W, pre):
    load_lstm(lstm, W.data[f"{pre}_blstm_w_ih"], W.data[f"{pre}_blstm_w_hh"],
              W.data[f"{pre}_blstm_b_ih"], W.data[f"{pre}_blstm_b_hh"])
    lstm.weight_ih_l0_reverse.data = torch.tensor(W.data[f"{pre}_blstm_w_ih_r"])
    lstm.weight_hh_l0_reverse.data = torch.tensor(W.data[f"{pre}_blstm_w_hh_r"])
    lstm.bias_ih_l0_reverse.data = torch.tensor(W.data[f"{pre}_blstm_b_ih_r"])
    lstm.bias_hh_l0_reverse.data = torch.tensor(W.data[f"{pre}_blstm_b_hh_r"])


class PitchEnergy(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.pitch = nn.Module()
        self.pitch.fsmn = FsmnEncoderV2(41, 3, 96, 128, 256, [0, 0, 0])
        self.pitch.blstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.pitch.fc = nn.Linear(256, 1)
        load_fsmn(self.pitch.fsmn, W, "pitch")
        load_lstm_bi(self.pitch.blstm, W, "pitch")
        self.pitch.fc.weight.data = W.t("pitch_fc_w")
        self.pitch.fc.bias.data = W.t("pitch_fc_b")

        self.energy = nn.Module()
        self.energy.fsmn = FsmnEncoderV2(41, 3, 96, 128, 256, [0, 0, 0])
        self.energy.blstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.energy.fc = nn.Linear(256, 1)
        load_fsmn(self.energy.fsmn, W, "energy")
        load_lstm_bi(self.energy.blstm, W, "energy")
        self.energy.fc.weight.data = W.t("energy_fc_w")
        self.energy.fc.bias.data = W.t("energy_fc_b")

    def forward(self, var_in):
        x = var_in
        h = self.pitch.fsmn(x)
        h, _ = self.pitch.blstm(h)
        p = self.pitch.fc(h).squeeze(-1)
        h = self.energy.fsmn(x)
        h, _ = self.energy.blstm(h)
        e = self.energy.fc(h).squeeze(-1)
        return p, e


class Duration(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.prenet = Prenet(1, [128, 128])
        self.lstm = nn.LSTM(224, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 1)
        self.prenet.fcs[0].weight.data = W.t("dur_pre0_w")
        self.prenet.fcs[0].bias.data = W.t("dur_pre0_b")
        self.prenet.fcs[2].weight.data = W.t("dur_pre1_w")
        self.prenet.fcs[2].bias.data = W.t("dur_pre1_b")
        self.lstm.weight_ih_l0.data = W.t("dur_lstm_w_ih0")
        self.lstm.weight_hh_l0.data = W.t("dur_lstm_w_hh0")
        self.lstm.bias_ih_l0.data = W.t("dur_lstm_b_ih0")
        self.lstm.bias_hh_l0.data = W.t("dur_lstm_b_hh0")
        self.lstm.weight_ih_l1.data = W.t("dur_lstm_w_ih1")
        self.lstm.weight_hh_l1.data = W.t("dur_lstm_w_hh1")
        self.lstm.bias_ih_l1.data = W.t("dur_lstm_b_ih1")
        self.lstm.bias_hh_l1.data = W.t("dur_lstm_b_hh1")
        self.fc.weight.data = W.t("dur_fc_w")
        self.fc.bias.data = W.t("dur_fc_b")

    def forward(self, cond):
        # 非自回归变体（一次前向处理整句，与板端 NPU 单次调用一致）
        x0 = torch.zeros(cond.shape[0], cond.shape[1], 1, device=cond.device)
        pre = self.prenet(x0)
        xin = torch.cat([pre, cond], dim=-1)
        y, _ = self.lstm(xin)
        y = F.relu(self.fc(y).squeeze(-1))
        return y


class PostNet(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.fsmn = FsmnEncoderV2(41, 4, 80, 256, 512, [17, 17, 17, 17])
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc = nn.Linear(128, 80)
        load_fsmn(self.fsmn, W, "post")
        self.lstm.weight_ih_l0.data = W.t("post_lstm_w_ih")
        self.lstm.weight_hh_l0.data = W.t("post_lstm_w_hh")
        self.lstm.bias_ih_l0.data = W.t("post_lstm_b_ih")
        self.lstm.bias_hh_l0.data = W.t("post_lstm_b_hh")
        self.fc.weight.data = W.t("post_fc_w")
        self.fc.bias.data = W.t("post_fc_b")

    def forward(self, dec):
        x = self.fsmn(dec)
        y, _ = self.lstm(x)
        return self.fc(y) + dec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = W(HW)

    pe = PitchEnergy(w).eval()
    p, e = pe(torch.zeros(1, 128, 96))
    print("pe:", p.shape, e.shape)
    torch.onnx.export(pe, (torch.zeros(1, 128, 96),), str(OUT / "pitch_energy.onnx"),
                      input_names=["var_in"], output_names=["pitch", "energy"],
                      opset_version=17, dynamo=False)
    print("pitch_energy.onnx ok")

    dur = Duration(w).eval()
    ld = dur(torch.zeros(1, 22, 96))
    print("dur:", ld.shape)
    torch.onnx.export(dur, (torch.zeros(1, 22, 96),), str(OUT / "duration.onnx"),
                      input_names=["cond"], output_names=["log_dur"],
                      opset_version=17, dynamo=False)
    print("duration.onnx ok")

    post = PostNet(w).eval()
    o = post(torch.zeros(1, 128, 80))
    print("post:", o.shape)
    torch.onnx.export(post, (torch.zeros(1, 128, 80),), str(OUT / "postnet.onnx"),
                      input_names=["dec"], output_names=["output"],
                      opset_version=17, dynamo=False)
    print("postnet.onnx ok")


if __name__ == "__main__":
    main()
