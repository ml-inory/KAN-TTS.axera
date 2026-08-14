#!/usr/bin/env python3
"""为 enc/pitch_energy/duration/postnet/voc 生成真实业务校准数据（tar.gz，npy 带 batch 维）。"""
import tarfile
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "export" / "calib_data"
MAX_T, MAX_M = 128, 270


def pack(name, samples):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(samples):
        np.save(d / f"{i:04d}.npy", np.ascontiguousarray(s))
    with tarfile.open(OUT / f"{name}.tar.gz", "w:gz") as tar:
        for f in sorted(d.glob("*.npy")):
            tar.add(f, arcname=f.name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(TASK / "export/ref_am.npz", allow_pickle=False)
    utts = [{k[len(u) + 2:]: z[k] for k in z.files if k.startswith(u + "__")}
            for u in ["utt0", "utt1", "utt2", "utt3", "utt4", "utt5", "utt6", "utt7"]]

    # ---- am_enc ----
    enc_inputs = {n: [] for n in ["inputs_ling", "inputs_emo", "inputs_spk", "inputs_len"]}
    for d in utts:
        T = d["inputs_ling"].shape[1]
        ling = np.zeros((1, MAX_T, 4), np.int32); ling[:, :T] = d["inputs_ling"]
        emo = np.zeros((1, MAX_T), np.int32); emo[:, :T] = d["inputs_emo"]
        spk = np.zeros((1, MAX_T), np.int32); spk[:, :T] = d["inputs_spk"]
        enc_inputs["inputs_ling"].append(ling)
        enc_inputs["inputs_emo"].append(emo)
        enc_inputs["inputs_spk"].append(spk)
        enc_inputs["inputs_len"].append(np.array([T], np.int32))
    for n, samples in enc_inputs.items():
        pack(f"enc_{n}", samples)

    # ---- am_dec（用 utt0 的 4 个步态）----
    d0 = utts[0]
    mem = np.zeros((1, MAX_M, 160), np.float32)
    mem[:, : d0["memory"].shape[1]] = d0["memory"]
    M = d0["memory"].shape[1]
    xb = int(d0["x_band"][0])
    steps = [0, 10, 20, 30]
    dec_inputs = {n: [] for n in ["mel_frame", "memory_step", "memory", "x_k", "x_v",
                                  "step", "x_band", "h_band", "mem_len"]}
    frame = np.zeros((1, 1, 80), np.float32)
    for s in steps:
        if s >= M:
            continue
        st_k = np.zeros((1, 12, 8, MAX_M, 16), np.float32)
        st_v = np.zeros((1, 12, 8, MAX_M, 16), np.float32)
        for lidx in range(12):
            kk = d0[f"step_{s}_l{lidx}_k"]
            vv = d0[f"step_{s}_l{lidx}_v"]
            st_k[0, lidx, :, : s + 1, :] = kk[:, : s + 1, :]
            st_v[0, lidx, :, : s + 1, :] = vv[:, : s + 1, :]
        dec_inputs["mel_frame"].append(frame.astype(np.float32))
        dec_inputs["memory_step"].append(mem[:, s : s + 1].astype(np.float32))
        dec_inputs["memory"].append(mem.astype(np.float32))
        dec_inputs["x_k"].append(st_k.astype(np.float32))
        dec_inputs["x_v"].append(st_v.astype(np.float32))
        dec_inputs["step"].append(np.array([s], np.int32))
        dec_inputs["x_band"].append(np.array([xb], np.int32))
        dec_inputs["h_band"].append(np.array([xb], np.int32))
        dec_inputs["mem_len"].append(np.array([M], np.int32))
    for n, samples in dec_inputs.items():
        pack(f"dec_{n}", samples)

    # ---- pitch_energy（var_in (1,128,96)）----
    pe_samples = []
    for d in utts:
        T = d["var_in"].shape[1]
        v = np.zeros((1, 128, 96), np.float32)
        v[0, :T, :] = d["var_in"][0]
        pe_samples.append(v)
    pack("pe_var_in", pe_samples)

    # ---- duration（cond (1,22,96)）----
    dur_samples = []
    for d in utts:
        T = d["dur_cond"].shape[1]
        c = np.zeros((1, 22, 96), np.float32)
        c[0, :T, :] = d["dur_cond"][0]
        dur_samples.append(c)
    pack("dur_cond", dur_samples)

    # ---- postnet（dec 拼接 (1,128,80)，用 utt0 步态）----
    d0 = utts[0]
    M = d0["memory"].shape[1]
    dec_all = np.zeros((1, 128, 80), np.float32)
    for s in range(min(M, 42)):
        dec_all[0, s * 3:(s + 1) * 3, :] = d0[f"step_{s}_out"][0].reshape(3, 80)
    pack("postnet_dec", [dec_all])

    # ---- voc（分块，块长 200 帧，重叠 10 帧）----
    voc_mels = []
    for d in utts:
        T = d["valid"][0]
        mel_full = d["postnet_out"][0, :T].T  # (80,T)
        C, O = 200, 40
        start = 0
        while start < T:
            end = min(start + C, T)
            cs = max(0, start - O)
            chunk = np.zeros((1, 80, C), np.float32)
            chunk[:, :, : end - cs] = mel_full[:, cs:end]
            voc_mels.append(chunk)
            start = end
    pack("voc_mel", voc_mels)

    print("calib done:", sorted(p.name for p in OUT.glob("*.tar.gz")))


if __name__ == "__main__":
    main()
